import os
import cv2
import gc
import time
import glob
import json
import shutil
import torch
import numpy as np
import gradio as gr
import trimesh
from datetime import datetime

from scipy.spatial.transform import Rotation
import matplotlib
from pi3.utils.geometry import depth_edge

from loop_utils.config_utils import load_config
from pi_long import Pi_Long

# Optional GPU neighbor search via PyTorch3D
try:
    from pytorch3d.ops import knn_points as _knn_points
    _PT3D_AVAILABLE = True
except Exception:
    _PT3D_AVAILABLE = False


"""
Pi-Long Gradio demo
- Mimics Pi3 demo UX
- Adds controls for chunking, overlap, loop closure options, align method, and DBOW
- Supports video input with configurable FPS (default 1)
- Produces full-quality point clouds (no downsampling) and GLB visualization
"""


# ------------------------------
# Visualization helpers (ported from Pi3 demo)
# ------------------------------
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)

    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def predictions_to_glb(predictions: dict,
    conf_thres: float = 20.0,
    filter_by_frames: str = "All",
    show_cam: bool = True,
    dedup_enable: bool = True,
    dedup_radius: float = 0.001,
) -> trimesh.Scene:
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    selected_frame_idx = None
    if filter_by_frames not in ("all", "All"):
        try:
            selected_frame_idx = int(str(filter_by_frames).split(":")[0])
        except Exception:
            selected_frame_idx = None

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))
    images = predictions["images"]
    camera_poses = predictions.get("camera_poses", None)

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        if camera_poses is not None:
            camera_poses = camera_poses[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)

    if images.ndim == 4 and images.shape[1] == 3:  # NCHW
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = conf_thres / 100.0

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # Optional spatial deduplication (voxel-based)
    if dedup_enable and vertices_3d.size > 0 and dedup_radius is not None and dedup_radius > 0:
        try:
            keys = np.floor(vertices_3d / float(dedup_radius)).astype(np.int64)
            # pack 3 int64 into one view for uniqueness
            keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
            _, unique_idx = np.unique(keys_view, return_index=True)
            unique_idx.sort()
            vertices_3d = vertices_3d[unique_idx]
            colors_rgb = colors_rgb[unique_idx]
        except Exception as e:
            print(f"Dedup failed, proceeding without dedup: {e}")

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    if show_cam and camera_poses is not None:
        num_cameras = len(camera_poses)
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / max(1, num_cameras))
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.0)

    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    scene_3d.apply_transform(align_rotation)
    return scene_3d


# ------------------------------
# Data IO helpers
# ------------------------------
def handle_uploads(input_video, input_images, fps: float = 1.0, interval: int = -1):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(OUTPUTS_DIR, f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []

    # images
    if input_images is not None:
        if interval is not None and interval > 0:
            input_images = input_images[::interval]
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) and "name" in file_data else file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # video
    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) and "name" in input_video else input_video
        vs = cv2.VideoCapture(video_path)
        src_fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
        desired_fps = fps if (fps is not None and fps > 0) else 1.0
        frame_interval = max(1, int(round(src_fps / desired_fps)))
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
            count += 1
        vs.release()

    image_paths = sorted(image_paths)
    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images, fps, interval):
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images, fps=fps, interval=interval)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def reload_frames_in_place(target_dir, input_video, input_images, fps, interval):
    """
    Re-extract frames into the existing target_dir/images using current FPS/interval.
    Keeps the same working folder for consistent re-runs.
    """
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, None, None, "No working directory. Please upload first."

    images_root = os.path.join(target_dir, "images")
    os.makedirs(images_root, exist_ok=True)

    # Clear existing images
    for fn in os.listdir(images_root):
        fp = os.path.join(images_root, fn)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    # Reuse handle logic but into fixed images_root
    image_paths = []
    # Images
    if input_images is not None:
        if interval is not None and interval > 0:
            input_images = input_images[::interval]
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) and "name" in file_data else file_data
            dst_path = os.path.join(images_root, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)
    # Video
    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) and "name" in input_video else input_video
        vs = cv2.VideoCapture(video_path)
        src_fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
        desired_fps = fps if (fps is not None and fps > 0) else 1.0
        frame_interval = max(1, int(round(src_fps / desired_fps)))
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                image_path = os.path.join(images_root, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
            count += 1
        vs.release()

    image_paths = sorted(image_paths)
    return None, target_dir, image_paths, "Frames reloaded. Click 'Reconstruct'."


def build_frame_filter_choices(images_root: str):
    all_files = sorted(os.listdir(images_root)) if os.path.isdir(images_root) else []
    return [f"{i}: {filename}" for i, filename in enumerate(all_files)]


# ------------------------------
# Local refinement (frame-level SE3 point-to-point ICP with temporal smoothness)
# ------------------------------
def _kabsch_weighted(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted Kabsch: finds R, t minimizing \sum w_i ||R x_i + t - y_i||^2
    src, dst: (M, 3), w: (M,) non-negative
    Returns (R, t)
    """
    w = np.clip(w, 0.0, None)
    s = np.sum(w) + 1e-8
    mu_src = (w[:, None] * src).sum(axis=0) / s
    mu_dst = (w[:, None] * dst).sum(axis=0) / s
    X = src - mu_src
    Y = dst - mu_dst
    Xw = X * w[:, None]
    H = Xw.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def _so3_log(Rm: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rt
    return Rt.from_matrix(Rm).as_rotvec()


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rt
    return Rt.from_rotvec(omega).as_matrix()


def _blend_se3(Ra: np.ndarray, ta: np.ndarray, Rb: np.ndarray, tb: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    # slerp-like via log/exp
    wa = _so3_log(Ra)
    wb = _so3_log(Rb)
    w = (1.0 - alpha) * wa + alpha * wb
    R = _so3_exp(w)
    t = (1.0 - alpha) * ta + alpha * tb
    return R, t


def _kabsch_weighted_torch(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch weighted Kabsch. src, dst: (M, 3) on same device/dtype; w: (M,)
    Returns R(3,3), t(3,)
    """
    # Upcast to float32 for numerical stability of SVD
    dtype_in = src.dtype
    src32 = src.to(torch.float32)
    dst32 = dst.to(torch.float32)
    w32 = torch.clamp(w.to(torch.float32), min=0)
    s = torch.sum(w32) + 1e-8
    mu_src = (w32[:, None] * src32).sum(dim=0) / s
    mu_dst = (w32[:, None] * dst32).sum(dim=0) / s
    X = src32 - mu_src
    Y = dst32 - mu_dst
    Xw = X * w32[:, None]
    H = Xw.T @ Y
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    R32 = Vh @ U.T
    if torch.det(R32) < 0:
        Vh[-1, :] *= -1
        R32 = Vh @ U.T
    t32 = mu_dst - R32 @ mu_src
    return R32.to(dtype_in), t32.to(dtype_in)


def refine_frames_in_place(save_dir: str,
                           corr_radius: float = 0.03,
                           max_points_per_frame: int = 10000,
                           iters_per_frame: int = 3,
                           smooth_lambda: float = 0.2,
                           voxel_for_ref: float = 0.02,
                           huber_delta: float = 0.005,
                           trim_ratio: float = 0.2,
                           step_scale: float = 0.5) -> str:
    """
    Refines per-frame world points (and camera_poses if present) via weighted point-to-point ICP to a fused reference.
    Modifies predictions.npz in-place and returns an info log string.
    """
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        return f"No predictions.npz at {pred_path}"
    print(f"[Refine] Loading predictions: {pred_path}")
    loaded = np.load(pred_path)
    needed = ['points', 'conf']
    for k in needed:
        if k not in loaded:
            return f"predictions.npz missing key '{k}'"
    points = np.array(loaded['points'])  # (N, H, W, 3)
    conf = np.array(loaded['conf'])      # (N, H, W)
    images = np.array(loaded['images']) if 'images' in loaded else None
    camera_poses = np.array(loaded['camera_poses']) if 'camera_poses' in loaded else None

    N = points.shape[0]
    print(f"[Refine] Frames: {points.shape[0]}, corr_radius={corr_radius}, voxel_ref={voxel_for_ref}")
    # Build reference by voxel dedup
    all_pts = points.reshape(-1, 3)
    if voxel_for_ref > 0:
        keys = np.floor(all_pts / float(voxel_for_ref)).astype(np.int64)
        keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, uniq_idx = np.unique(keys_view, return_index=True)
        ref_pts = all_pts[uniq_idx]
    else:
        ref_pts = all_pts
    num_ref = ref_pts.shape[0]
    print(f"[Refine] Reference size (pre-cap): {num_ref}")
    if num_ref > 2_000_000:
        sel = np.random.choice(num_ref, 2_000_000, replace=False)
        ref_pts = ref_pts[sel]
        print(f"[Refine] Reference capped to {ref_pts.shape[0]}")

    from scipy.spatial import cKDTree
    # Fixed reference for cumulative metrics (created on first refine)
    fixed_ref_path = os.path.join(save_dir, 'fixed_reference.npy')
    try:
        if os.path.exists(fixed_ref_path):
            fixed_ref = np.load(fixed_ref_path)
        else:
            fixed_ref = ref_pts.copy()
            np.save(fixed_ref_path, fixed_ref)
    except Exception as e:
        fixed_ref = ref_pts.copy()
        print(f"[Refine] Fixed reference load/save issue: {e}; using current ref for metrics.")
    fixed_kdt = cKDTree(fixed_ref) if fixed_ref.shape[0] > 0 else None

    print(f"[Refine] Building KD-Tree...")
    kdt = cKDTree(ref_pts)

    # Helper to compute global mean NN error vs fixed reference (for cumulative trends)
    def _global_nn_error_vs_fixed(points_arr: np.ndarray, sample_cap: int = 200000) -> tuple[float, int]:
        if fixed_kdt is None:
            return 0.0, 0
        all_pts = points_arr.reshape(-1, 3)
        valid = np.isfinite(all_pts).all(axis=1)
        all_pts = all_pts[valid]
        if all_pts.shape[0] == 0:
            return 0.0, 0
        m = min(sample_cap, all_pts.shape[0])
        sel = np.random.choice(all_pts.shape[0], m, replace=False)
        Ps = all_pts[sel]
        d, _ = fixed_kdt.query(Ps, k=1)
        nz = np.isfinite(d) & (d > 1e-9)
        if nz.sum() == 0:
            return 0.0, 0
        return float(np.mean(d[nz])), int(nz.sum())

    # Run-start cumulative metric
    pre_global_err, pre_matches = _global_nn_error_vs_fixed(points)
    print(f"[Refine] Run start vs fixed_ref: mean_err={pre_global_err:.6f} (matches={pre_matches})")

    # Per-frame refinement
    total_mean_shift = 0.0
    total_frames_shifted = 0
    for i in range(N):
        Pi = points[i].reshape(-1, 3)
        Pi_orig = Pi.copy()
        Ci = conf[i].reshape(-1)
        valid = np.isfinite(Pi).all(axis=1)
        Ci = Ci * valid.astype(np.float32)
        # sample
        if Pi.shape[0] > max_points_per_frame:
            idx = np.random.choice(Pi.shape[0], max_points_per_frame, replace=False)
        else:
            idx = np.arange(Pi.shape[0])
        P = Pi[idx]
        W = Ci[idx]
        sel = W > 1e-6
        if sel.sum() < 100:
            print(f"[Refine][{i}] skipped (few valid points: {sel.sum()})")
            continue
        P = P[sel]; W = W[sel]
        # iterative NN and Kabsch
        R_acc = np.eye(3, dtype=np.float64)
        t_acc = np.zeros(3, dtype=np.float64)
        Q = P.copy()
        for it in range(iters_per_frame):
            d, j = kdt.query(Q, k=1, distance_upper_bound=corr_radius)
            mask = np.isfinite(d) & (d < corr_radius)
            if mask.sum() < 50:
                print(f"[Refine][{i}] iter {it}: few matches ({mask.sum()})")
                break
            # Use current transformed points (Q) for incremental Kabsch
            src = Q[mask]
            dst = ref_pts[j[mask]]
            # Exclude near-zero self-matches
            dists = np.linalg.norm(src - dst, axis=1)
            nz = dists > 1e-9
            src = src[nz]
            dst = dst[nz]
            w = W[mask][nz]
            if src.shape[0] < 50:
                print(f"[Refine][{i}] iter {it}: few nonzero matches ({src.shape[0]})")
                break
            # Pre-update error
            pre_err = float(np.mean(dists[nz])) if src.shape[0] > 0 else 0.0
            # Robust weights (Huber)
            if huber_delta is not None and huber_delta > 0:
                r = np.linalg.norm(src - dst, axis=1)
                huber_w = np.ones_like(r)
                large = r > huber_delta
                huber_w[large] = huber_delta / (r[large] + 1e-12)
                w = w * huber_w
            # Trimming (drop top trim_ratio residuals)
            if trim_ratio is not None and 0 < trim_ratio < 1 and src.shape[0] > 100:
                r = np.linalg.norm(src - dst, axis=1)
                thr = np.quantile(r, 1.0 - trim_ratio)
                keep = r <= thr
                src, dst, w = src[keep], dst[keep], w[keep]
            R_upd, t_upd = _kabsch_weighted(src, dst, w)
            # Backtracking line search on step size
            alpha = float(np.clip(step_scale, 0.0, 1.0))
            best_alpha = 0.0
            best_post = pre_err
            R_damp = np.eye(3); t_damp = np.zeros(3)
            while alpha > 1e-4:
                R_try, t_try = _blend_se3(np.eye(3), np.zeros(3), R_upd, t_upd, alpha)
                post_err = float(np.mean(np.linalg.norm(((R_try @ src.T).T + t_try) - dst, axis=1))) if src.shape[0] > 0 else 0.0
                if post_err < best_post:
                    best_post = post_err
                    best_alpha = alpha
                    R_damp, t_damp = R_try, t_try
                    break
                alpha *= 0.5
            # accumulate (apply only if we found a descent step)
            if best_alpha > 0:
                R_acc = R_damp @ R_acc
                t_acc = R_damp @ t_acc + t_damp
                Q = (R_damp @ Q.T).T + t_damp
            print(f"[Refine][{i}] iter {it}: matches={src.shape[0]}, pre_err={pre_err:.6f}, post_err={(best_post if best_alpha>0 else pre_err):.6f}, alpha={best_alpha:.3f}")

        # temporal smoothing (blend with previous frame correction)
        if i > 0:
            # derive prev incremental from updated vs original
            # here we simply blend the corrections
            R_prev = np.eye(3); t_prev = np.zeros(3)
            R_acc, t_acc = _blend_se3(R_prev, t_prev, R_acc, t_acc, 1.0 - np.clip(smooth_lambda, 0.0, 1.0))

        # apply to full frame
        Pi_new = (R_acc @ Pi.T).T + t_acc
        points[i] = Pi_new.reshape(points[i].shape)
        try:
            mean_shift = float(np.mean(np.linalg.norm(Pi_new - Pi_orig, axis=1)))
            print(f"[Refine][{i}] mean_shift={mean_shift:.6f}")
            total_mean_shift += mean_shift
            total_frames_shifted += 1
        except Exception:
            pass
        if camera_poses is not None and camera_poses.shape[0] == N:
            # camera pose maps local->world; apply left-multiplication by correction on world side
            T = camera_poses[i]
            T_upd = np.eye(4, dtype=T.dtype)
            T_upd[:3, :3] = R_acc
            T_upd[:3, 3] = t_acc
            camera_poses[i] = T_upd @ T
        print(f"[Refine][{i}] applied correction |R-I|_F={np.linalg.norm(R_acc - np.eye(3)):.3e}, |t|={np.linalg.norm(t_acc):.6f}")

    # Save back
    save_dict = dict(points=points, conf=conf)
    if images is not None:
        save_dict['images'] = images
    if camera_poses is not None:
        save_dict['camera_poses'] = camera_poses
    np.savez(pred_path, **save_dict)
    # quick post-save verification
    try:
        _chk = np.load(pred_path)
        print(f"[Refine] Saved predictions.npz. points.shape={_chk['points'].shape}")
    except Exception as e:
        print(f"[Refine] Verification load failed: {e}")
    # Run-end cumulative metric vs the same fixed_ref
    post_global_err, post_matches = _global_nn_error_vs_fixed(points)
    print(f"[Refine] Run end vs fixed_ref: mean_err={post_global_err:.6f} (matches={post_matches}), delta={(pre_global_err - post_global_err):.6f}")
    # Persist metrics history
    try:
        metrics_path = os.path.join(save_dir, 'refine_metrics.json')
        history = []
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                history = json.load(f)
        history.append({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'voxel_for_ref': float(voxel_for_ref),
            'corr_radius': float(corr_radius),
            'pre_mean_err': float(pre_global_err),
            'post_mean_err': float(post_global_err),
            'delta': float(pre_global_err - post_global_err),
            'matches_start': int(pre_matches),
            'matches_end': int(post_matches),
        })
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Refine] Failed to persist metrics: {e}")
    avg_shift = (total_mean_shift / max(1, total_frames_shifted))
    return f"Local refinement done: frames={N}, voxel_ref={voxel_for_ref}, corr_radius={corr_radius}, ref_pts={ref_pts.shape[0]}, avg_frame_shift={avg_shift:.6f}"


def refine_and_update_viz(enable_refine,
                          save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius,
                          corr_radius, max_points_per_frame, iters_per_frame, smooth_lambda, voxel_for_ref,
                          huber_delta, trim_ratio, step_scale):
    if not (bool(enable_refine) if enable_refine is not None else False):
        glb, log = update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example="False")
        return glb, f"Refine disabled. {log}"
    msg = refine_frames_in_place(
        save_dir,
        corr_radius=float(corr_radius) if corr_radius is not None else 0.03,
        max_points_per_frame=int(max_points_per_frame) if max_points_per_frame is not None else 10000,
        iters_per_frame=int(iters_per_frame) if iters_per_frame is not None else 3,
        smooth_lambda=float(smooth_lambda) if smooth_lambda is not None else 0.2,
        voxel_for_ref=float(voxel_for_ref) if voxel_for_ref is not None else 0.02,
        huber_delta=float(huber_delta) if huber_delta is not None else 0.005,
        trim_ratio=float(trim_ratio) if trim_ratio is not None else 0.2,
        step_scale=float(step_scale) if step_scale is not None else 0.5,
    )
    # Invalidate cached GLBs so refreshed visualization uses updated predictions
    try:
        for fn in os.listdir(save_dir):
            if fn.startswith('glbscene_') and fn.endswith('.glb'):
                os.remove(os.path.join(save_dir, fn))
    except Exception:
        pass
    glb, log = update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example="False")
    return glb, f"{msg}. {log}"


def piecewise_refine_in_place(save_dir: str,
                              voxel_size: float = 0.03,
                              corr_radius_factor: float = 1.5,
                              min_pairs_per_voxel: int = 50,
                              max_points_per_voxel: int = 5000,
                              workers: int = 0,
                              kd_workers: int = 1,
                              ref_voxel: float = 0.0) -> str:
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        return f"No predictions.npz at {pred_path}"
    print(f"[Piecewise] Loading predictions: {pred_path}")
    loaded = np.load(pred_path)
    if 'points' not in loaded or 'conf' not in loaded:
        return "predictions.npz missing points/conf"
    points = np.array(loaded['points'])
    conf = np.array(loaded['conf'])
    images = np.array(loaded['images']) if 'images' in loaded else None
    camera_poses = np.array(loaded['camera_poses']) if 'camera_poses' in loaded else None

    N = points.shape[0]
    all_pts = points.reshape(-1, 3)
    print(f"[Piecewise] Frames={N}, voxel={voxel_size}, factor={corr_radius_factor}")
    # Reference (no downsample here; use earlier cap if needed)
    ref_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if ref_voxel is not None and ref_voxel > 0:
        keys = np.floor(ref_pts / float(ref_voxel)).astype(np.int64)
        keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, uniq_idx = np.unique(keys_view, return_index=True)
        ref_pts = ref_pts[uniq_idx]
        print(f"[Piecewise] Ref voxel dedup: {ref_pts.shape[0]} points")
    from scipy.spatial import cKDTree
    kdt = cKDTree(ref_pts)
    radius = max(1e-6, voxel_size * corr_radius_factor)

    def voxel_keys(pts: np.ndarray, offset: float) -> np.ndarray:
        return np.floor((pts - offset) / voxel_size).astype(np.int64)

    def compute_voxel_transforms_for_frame(Pi: np.ndarray, Wi: np.ndarray, keys: np.ndarray) -> dict:
        # limit memory: work per unique key
        transforms = {}
        # Group indices by key
        # Pack keys to tuple for dict
        if Pi.shape[0] == 0:
            return transforms
        keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, first_idx, inverse = np.unique(keys_view, return_index=True, return_inverse=True)
        # Map inverse indices to lists of original indices in a streaming way
        buckets = [[] for _ in range(first_idx.shape[0])]
        for idx_pt, b in enumerate(inverse):
            buckets[b].append(idx_pt)
        from concurrent.futures import ThreadPoolExecutor

        def process_bucket(b_idx_and_list):
            b_idx, idx_list = b_idx_and_list
            if len(idx_list) < min_pairs_per_voxel:
                return None
            idx_arr = np.array(idx_list, dtype=np.int64)
            P = Pi[idx_arr]
            W = Wi[idx_arr]
            if P.shape[0] > max_points_per_voxel:
                sel = np.random.choice(P.shape[0], max_points_per_voxel, replace=False)
                P = P[sel]; W = W[sel]
            d, j = kdt.query(P, k=1, distance_upper_bound=radius, workers=kd_workers)
            mask = np.isfinite(d) & (d < radius)
            if mask.sum() < min_pairs_per_voxel:
                return None
            src = P[mask]
            dst = ref_pts[j[mask]]
            w = W[mask]
            R, t = _kabsch_weighted(src, dst, w)
            ktuple = (int(keys[idx_arr[0], 0]), int(keys[idx_arr[0], 1]), int(keys[idx_arr[0], 2]))
            return (ktuple, R, t)

        if workers and workers != 0:
            mw = os.cpu_count() or 4 if workers == -1 else int(workers)
            with ThreadPoolExecutor(max_workers=mw) as ex:
                for res in ex.map(process_bucket, enumerate(buckets), chunksize=16):
                    if res is not None:
                        ktuple, R, t = res
                        transforms[ktuple] = (R, t)
        else:
            for item in enumerate(buckets):
                res = process_bucket(item)
                if res is not None:
                    ktuple, R, t = res
                    transforms[ktuple] = (R, t)
        return transforms

    total_mean_shift = 0.0
    total_frames_shifted = 0
    for i in range(N):
        Pi = points[i].reshape(-1, 3)
        Pi_orig = Pi.copy()
        Wi = conf[i].reshape(-1)
        valid = np.isfinite(Pi).all(axis=1)
        Wi = Wi * valid.astype(np.float32)
        sel = Wi > 1e-6
        Pi = Pi[sel]; Wi = Wi[sel]
        if Pi.shape[0] == 0:
            print(f"[Piecewise][{i}] no valid points")
            continue
        # Two overlapping grids: offset 0 and voxel/2
        keys0 = voxel_keys(Pi, offset=0.0)
        keys1 = voxel_keys(Pi, offset=voxel_size * 0.5)
        print(f"[Piecewise][{i}] voxels0={np.unique(keys0.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])).shape[0]} | voxels1={np.unique(keys1.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])).shape[0]}")
        T0 = compute_voxel_transforms_for_frame(Pi, Wi, keys0)
        T1 = compute_voxel_transforms_for_frame(Pi, Wi, keys1)
        # apply per-point blended transforms
        Pi_new = Pi.copy()
        for idx_pt in range(Pi.shape[0]):
            k0 = (int(keys0[idx_pt, 0]), int(keys0[idx_pt, 1]), int(keys0[idx_pt, 2]))
            k1 = (int(keys1[idx_pt, 0]), int(keys1[idx_pt, 1]), int(keys1[idx_pt, 2]))
            R0, t0 = T0.get(k0, (np.eye(3), np.zeros(3)))
            R1, t1 = T1.get(k1, (np.eye(3), np.zeros(3)))
            # blend 50/50
            Rb, tb = _blend_se3(R0, t0, R1, t1, 0.5)
            Pi_new[idx_pt] = (Rb @ Pi[idx_pt]) + tb
        # scatter back into full frame
        full = points[i].reshape(-1, 3)
        full_sel = np.where(sel)[0]
        before = full.copy()
        full[full_sel] = Pi_new
        points[i] = full.reshape(points[i].shape)
        try:
            mean_shift = float(np.mean(np.linalg.norm(full - before, axis=1)))
            print(f"[Piecewise][{i}] mean_shift={mean_shift:.6f}")
            total_mean_shift += mean_shift
            total_frames_shifted += 1
        except Exception:
            pass
        print(f"[Piecewise][{i}] applied per-voxel corrections")

    # Save back
    save_dict = dict(points=points, conf=conf)
    if images is not None:
        save_dict['images'] = images
    if camera_poses is not None:
        save_dict['camera_poses'] = camera_poses
    np.savez(pred_path, **save_dict)
    try:
        _chk = np.load(pred_path)
        print(f"[Piecewise] Saved predictions.npz. points.shape={_chk['points'].shape}")
    except Exception as e:
        print(f"[Piecewise] Verification load failed: {e}")
    avg_shift = (total_mean_shift / max(1, total_frames_shifted))
    return f"Piecewise refinement done: frames={N}, voxel={voxel_size}, radius={radius}, workers={workers}, kd_workers={kd_workers}, ref_dedup={ref_voxel}, avg_frame_shift={avg_shift:.6f}"


def piecewise_refine_and_update_viz(enable_piecewise,
                                    save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius,
                                    voxel_size, corr_factor, min_pairs, max_pts, workers, kd_workers):
    if not (bool(enable_piecewise) if enable_piecewise is not None else False):
        glb, log = update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example="False")
        return glb, f"Piecewise disabled. {log}"
    msg = piecewise_refine_in_place(
        save_dir,
        voxel_size=float(voxel_size) if voxel_size is not None else 0.03,
        corr_radius_factor=float(corr_factor) if corr_factor is not None else 1.5,
        min_pairs_per_voxel=int(min_pairs) if min_pairs is not None else 50,
        max_points_per_voxel=int(max_pts) if max_pts is not None else 5000,
        workers=int(workers) if workers is not None else 0,
        kd_workers=int(kd_workers) if kd_workers is not None else 1,
    )
    glb, log = update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example="False")
    return glb, f"{msg}. {log}"


def piecewise_refine_in_place_gpu(save_dir: str,
                                  voxel_size: float = 0.03,
                                  corr_radius_factor: float = 1.5,
                                  min_pairs_per_voxel: int = 50,
                                  max_points_per_voxel: int = 5000,
                                  fp16: bool = True,
                                  ref_cap: int = 500000,
                                  cdist_chunk: int = 20000,
                                  ref_voxel: float = 0.0) -> str:
    if not torch.cuda.is_available():
        return "GPU piecewise unavailable (CUDA missing)."
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        return f"No predictions.npz at {pred_path}"
    print(f"[Piecewise-GPU] Loading predictions: {pred_path}")
    loaded = np.load(pred_path)
    if 'points' not in loaded or 'conf' not in loaded:
        return "predictions.npz missing points/conf"
    points_np = np.array(loaded['points'])
    conf_np = np.array(loaded['conf'])
    images = np.array(loaded['images']) if 'images' in loaded else None
    camera_poses = np.array(loaded['camera_poses']) if 'camera_poses' in loaded else None

    device = torch.device('cuda')
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    N = points_np.shape[0]
    # Build reference on CPU then move to GPU
    all_pts_np = points_np.reshape(-1, 3)
    valid_mask_np = np.isfinite(all_pts_np).all(axis=1)
    ref_pts_np = all_pts_np[valid_mask_np]
    if ref_voxel is not None and ref_voxel > 0:
        keys = np.floor(ref_pts_np / float(ref_voxel)).astype(np.int64)
        keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, uniq_idx = np.unique(keys_view, return_index=True)
        ref_pts_np = ref_pts_np[uniq_idx]
        print(f"[Piecewise-GPU] Ref voxel dedup: {ref_pts_np.shape[0]} points")
    if ref_cap is not None and ref_cap > 0 and ref_pts_np.shape[0] > ref_cap:
        sel = np.random.choice(ref_pts_np.shape[0], ref_cap, replace=False)
        ref_pts_np = ref_pts_np[sel]
        print(f"[Piecewise-GPU] Reference capped to {ref_pts_np.shape[0]}")
    ref_pts = torch.from_numpy(ref_pts_np).to(device=device, dtype=dtype)
    radius = max(1e-6, voxel_size * corr_radius_factor)
    print(f"[Piecewise-GPU] Frames={N}, voxel={voxel_size}, radius={radius}, ref_size={ref_pts.shape[0]}")

    def voxel_keys_torch(pts: torch.Tensor, offset: float) -> torch.Tensor:
        return torch.floor((pts - offset) / voxel_size).to(torch.int64)

    def _gpu_nn_query(P: torch.Tensor, ref: torch.Tensor, radius_val: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mask, idx) where mask indicates neighbors within radius and idx are NN indices.
        Tries PyTorch3D; falls back to chunked torch.cdist on GPU.
        """
        if _PT3D_AVAILABLE:
            knn = _knn_points(P[None].to(torch.float32), ref[None].to(torch.float32), K=1)
            d = torch.sqrt(torch.clamp(knn.dists.squeeze(0).squeeze(1), min=0))
            idx = knn.idx.squeeze(0).squeeze(1)
            mask = torch.isfinite(d) & (d <= radius_val)
            return mask, idx
        # Fallback: chunked cdist
        M = P.shape[0]
        chunk = min(int(cdist_chunk) if cdist_chunk is not None and cdist_chunk > 0 else 20000, M)
        idx_all = torch.empty((M,), device=P.device, dtype=torch.long)
        dmin_all = torch.full((M,), float('inf'), device=P.device, dtype=torch.float32)
        for s in range(0, M, chunk):
            e = min(M, s + chunk)
            d = torch.cdist(P[s:e].to(torch.float32), ref.to(torch.float32))  # (m, R)
            dmin, idx = torch.min(d, dim=1)
            idx_all[s:e] = idx
            dmin_all[s:e] = dmin
        mask = torch.isfinite(dmin_all) & (dmin_all <= radius_val)
        return mask, idx_all

    # Process frames one-by-one to reduce VRAM
    points_out = points_np.copy()

    total_mean_shift = 0.0
    total_frames_shifted = 0
    for i in range(N):
        Pi_full_np = points_np[i].reshape(-1, 3)
        Wi_full_np = conf_np[i].reshape(-1)
        valid_np = np.isfinite(Pi_full_np).all(axis=1)
        Wi_full_np = Wi_full_np * valid_np.astype(np.float32)
        sel_np = Wi_full_np > 1e-6
        if not np.any(sel_np):
            print(f"[Piecewise-GPU][{i}] no valid points")
            continue
        Pi = torch.from_numpy(Pi_full_np[sel_np]).to(device=device, dtype=dtype)
        Wi = torch.from_numpy(Wi_full_np[sel_np]).to(device=device, dtype=dtype)
        if Pi.shape[0] == 0:
            print(f"[Piecewise-GPU][{i}] no valid points")
            continue
        keys0 = voxel_keys_torch(Pi, 0.0)
        keys1 = voxel_keys_torch(Pi, voxel_size * 0.5)
        # pack keys to strings for grouping (CPU-safe via transfer of small arrays)
        k0_cpu = keys0.detach().cpu().numpy().view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        k1_cpu = keys1.detach().cpu().numpy().view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, inv0 = np.unique(k0_cpu, return_inverse=True)
        _, inv1 = np.unique(k1_cpu, return_inverse=True)
        # build buckets index lists on CPU
        buckets0 = [[] for _ in range(inv0.max() + 1)]
        for idx_pt, b in enumerate(inv0):
            buckets0[b].append(idx_pt)
        buckets1 = [[] for _ in range(inv1.max() + 1)]
        for idx_pt, b in enumerate(inv1):
            buckets1[b].append(idx_pt)

        def solve_bucket(idx_list: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if len(idx_list) < min_pairs_per_voxel:
                return None
            idx_t = torch.tensor(idx_list, device=device, dtype=torch.long)
            P = Pi.index_select(0, idx_t)
            W = Wi.index_select(0, idx_t)
            if P.shape[0] > max_points_per_voxel:
                sel2 = torch.randperm(P.shape[0], device=device)[:max_points_per_voxel]
                P = P.index_select(0, sel2)
                W = W.index_select(0, sel2)
            # NN to reference (PyTorch3D or torch.cdist fallback)
            mask, idx_ref = _gpu_nn_query(P, ref_pts, radius)
            if mask.sum().item() < min_pairs_per_voxel:
                return None
            src = P[mask]
            dst = ref_pts.index_select(0, idx_ref[mask])
            w = W[mask]
            R, t = _kabsch_weighted_torch(src, dst, w)
            return idx_t[mask], R, t

        # apply two overlapping grids
        Pi_new = Pi.clone()
        for idx_list in buckets0:
            res = solve_bucket(idx_list)
            if res is None:
                continue
            sub_idx, R, t = res
            Pi_new[sub_idx] = (Pi_new[sub_idx].to(torch.float32) @ R.T.to(torch.float32) + t.to(torch.float32)).to(dtype)
        for idx_list in buckets1:
            res = solve_bucket(idx_list)
            if res is None:
                continue
            sub_idx, R, t = res
            # 50/50 blend: average the transformed point (simple and effective)
            trans = (Pi.index_select(0, sub_idx).to(torch.float32) @ R.T.to(torch.float32) + t.to(torch.float32)).to(dtype)
            blended = (Pi_new.index_select(0, sub_idx) + trans) * 0.5
            Pi_new[sub_idx] = blended

        # scatter back to CPU array
        full_np = Pi_full_np.copy()
        full_np[sel_np] = Pi_new.detach().cpu().numpy()
        points_out[i] = full_np.reshape(points_out[i].shape)
        try:
            mean_shift = float(np.mean(np.linalg.norm(full_np - Pi_full_np, axis=1)))
            print(f"[Piecewise-GPU][{i}] mean_shift={mean_shift:.6f}")
            total_mean_shift += mean_shift
            total_frames_shifted += 1
        except Exception:
            pass
        print(f"[Piecewise-GPU][{i}] applied per-voxel corrections")

    # Save back to npz
    out_path = os.path.join(save_dir, 'predictions.npz')
    save_dict = dict(points=points_out, conf=conf_np)
    if images is not None:
        save_dict['images'] = images
    if camera_poses is not None:
        save_dict['camera_poses'] = camera_poses
    np.savez(out_path, **save_dict)
    try:
        _chk = np.load(out_path)
        print(f"[Piecewise-GPU] Saved predictions.npz. points.shape={_chk['points'].shape}")
    except Exception as e:
        print(f"[Piecewise-GPU] Verification load failed: {e}")
    avg_shift = (total_mean_shift / max(1, total_frames_shifted))
    return f"Piecewise-GPU refinement done: frames={N}, voxel={voxel_size}, radius={radius}, avg_frame_shift={avg_shift:.6f}"


def dispatch_piecewise_refine(enable_piecewise, use_gpu,
                              save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius,
                              voxel_size, corr_factor, min_pairs, max_pts, workers, kd_workers,
                              fp16, ref_cap, cdist_chunk, ref_voxel):
    if not (bool(enable_piecewise) if enable_piecewise is not None else False):
        glb, log = update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example="False")
        return glb, f"Piecewise disabled. {log}"
    if bool(use_gpu) and torch.cuda.is_available():
        msg = piecewise_refine_in_place_gpu(
            save_dir,
            voxel_size=float(voxel_size) if voxel_size is not None else 0.03,
            corr_radius_factor=float(corr_factor) if corr_factor is not None else 1.5,
            min_pairs_per_voxel=int(min_pairs) if min_pairs is not None else 50,
            max_points_per_voxel=int(max_pts) if max_pts is not None else 5000,
            fp16=bool(fp16) if fp16 is not None else True,
            ref_cap=int(ref_cap) if ref_cap is not None else 500000,
            cdist_chunk=int(cdist_chunk) if cdist_chunk is not None else 20000,
            ref_voxel=float(ref_voxel) if ref_voxel is not None else 0.0,
        )
    else:
        msg = piecewise_refine_in_place(
            save_dir,
            voxel_size=float(voxel_size) if voxel_size is not None else 0.03,
            corr_radius_factor=float(corr_factor) if corr_factor is not None else 1.5,
            min_pairs_per_voxel=int(min_pairs) if min_pairs is not None else 50,
            max_points_per_voxel=int(max_pts) if max_pts is not None else 5000,
            workers=int(workers) if workers is not None else 0,
            kd_workers=int(kd_workers) if kd_workers is not None else 1,
            ref_voxel=float(ref_voxel) if ref_voxel is not None else 0.0,
        )
    # Invalidate cached GLBs so refreshed visualization uses updated predictions
    try:
        for fn in os.listdir(save_dir):
            if fn.startswith('glbscene_') and fn.endswith('.glb'):
                os.remove(os.path.join(save_dir, fn))
    except Exception:
        pass
    glb, log = update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example="False")
    return glb, f"{msg}. {log}"


# ------------------------------
# Pi-Long pipeline integration
# ------------------------------
_MODEL_CACHE = {
    'pi3': None,
}


def unload_models():
    try:
        if _MODEL_CACHE['pi3'] is not None:
            del _MODEL_CACHE['pi3']
            _MODEL_CACHE['pi3'] = None
            torch.cuda.empty_cache()
            gc.collect()
        return "Models unloaded."
    except Exception as e:
        return f"Unload failed: {e}"


def run_pi_long_and_aggregate(target_dir: str,
                              conf: dict,
                              chunk_size: int,
                              overlap: int,
                              loop_enable: bool,
                              loop_chunk_size: int,
                              use_dbow: bool,
                              align_method: str,
                              keep_temps: bool = True,
                              zero_depth_edges: bool = True,
                              keep_models_loaded: bool = True) -> tuple[str, dict, list[str]]:
    """
    Runs Pi-Long with given overrides and aggregates predictions for visualization.
    Returns (save_dir, predictions_dict, frame_filter_choices)
    """
    save_dir = target_dir  # Use working folder as predictions folder
    images_root = os.path.join(save_dir, "images")
    if not os.path.isdir(images_root):
        raise ValueError(f"No images found at {images_root}")

    os.makedirs(save_dir, exist_ok=True)

    # Override config
    conf = dict(conf)  # shallow copy
    conf.setdefault('Model', {})
    conf['Model']['chunk_size'] = int(chunk_size)
    conf['Model']['overlap'] = int(overlap)
    conf['Model']['loop_enable'] = bool(loop_enable)
    conf['Model']['loop_chunk_size'] = int(loop_chunk_size)
    conf['Model']['useDBoW'] = bool(use_dbow)
    conf['Model']['align_method'] = str(align_method)
    conf['Model']['delete_temp_files'] = bool(not keep_temps)  # keep temps by default for viz

    # Full-quality point cloud settings
    conf.setdefault('Model', {}).setdefault('Pointcloud_Save', {})
    conf['Model']['Pointcloud_Save']['sample_ratio'] = 1.0
    conf['Model']['Pointcloud_Save']['conf_threshold_coef'] = 0.0

    # Clean previous outputs (keep images)
    try:
        to_remove_dirs = [
            '_tmp_results_unaligned', '_tmp_results_aligned', '_tmp_results_loop', 'pcd'
        ]
        for d in to_remove_dirs:
            p = os.path.join(save_dir, d)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        # Remove known files
        for fn in os.listdir(save_dir):
            if fn.startswith('glbscene_') and fn.endswith('.glb'):
                try:
                    os.remove(os.path.join(save_dir, fn))
                except Exception:
                    pass
        for fn in ['predictions.npz', 'camera_poses.txt', 'camera_poses.ply', 'sim3_opt_result.png']:
            fp = os.path.join(save_dir, fn)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: failed to clean previous outputs: {e}")

    print("Launching Pi-Long...")
    preloaded_model = _MODEL_CACHE.get('pi3')
    pi_long = Pi_Long(image_dir=images_root, save_dir=save_dir, config=conf, preloaded_model=preloaded_model, keep_model_loaded=keep_models_loaded)
    pi_long.run()
    # Ensure temporary files are kept for post-viz aggregation
    pi_long.close()
    if keep_models_loaded:
        _MODEL_CACHE['pi3'] = pi_long.model if hasattr(pi_long, 'model') else _MODEL_CACHE.get('pi3')
    del pi_long
    torch.cuda.empty_cache()
    gc.collect()

    # Aggregate predictions from aligned chunks
    aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
    if not os.path.isdir(aligned_dir):
        raise ValueError(f"Aligned results directory not found: {aligned_dir}")

    def _chunk_idx_from_path(p: str) -> int:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(stem.split('_')[-1])
        except Exception:
            return 0

    chunk_files = sorted(glob.glob(os.path.join(aligned_dir, 'chunk_*.npy')), key=_chunk_idx_from_path)
    if len(chunk_files) == 0:
        # Fallback: unaligned
        chunk_files = sorted(glob.glob(os.path.join(save_dir, '_tmp_results_unaligned', 'chunk_*.npy')), key=_chunk_idx_from_path)
    if len(chunk_files) == 0:
        raise ValueError("No chunk result files found.")

    points_list = []
    confs_list = []
    images_list = []
    z_list = []  # for depth edge suppression
    for cf in chunk_files:
        d = np.load(cf, allow_pickle=True).item()
        pts = d['points']  # [N, H, W, 3]
        cfv = d.get('conf')
        if cfv is None:
            cfv = np.ones(pts.shape[:-1], dtype=np.float32)
        elif cfv.ndim == 4 and cfv.shape[-1] == 1:
            cfv = cfv[..., 0]
        imgs = d.get('images')
        if imgs is None:
            # Create dummy colors if missing
            n, h, w, _ = pts.shape
            imgs = np.zeros((n, h, w, 3), dtype=np.float32)
        else:
            # NCHW -> NHWC
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = np.transpose(imgs, (0, 2, 3, 1))

        # collect z for edge suppression if requested and available
        if zero_depth_edges and 'local_points' in d:
            lp = d['local_points']
            if lp.ndim == 4 and lp.shape[-1] >= 3:
                z_list.append(lp[..., 2])

        points_list.append(pts)
        confs_list.append(cfv)
        images_list.append(imgs)

    points = np.concatenate(points_list, axis=0)
    confs = np.concatenate(confs_list, axis=0)
    images = np.concatenate(images_list, axis=0)

    # Zero confidence at depth edges (Pi3-like) if enabled
    if zero_depth_edges and len(z_list) > 0:
        z_all = np.concatenate(z_list, axis=0)
        try:
            edge_mask = depth_edge(torch.from_numpy(z_all).float(), rtol=0.03).cpu().numpy()
            confs[edge_mask] = 0.0
        except Exception as e:
            print(f"Depth edge suppression failed: {e}")

    # Load aligned camera poses (C2W) from txt if available
    camera_txt = os.path.join(save_dir, 'camera_poses.txt')
    camera_poses = None
    if os.path.exists(camera_txt):
        mats = []
        with open(camera_txt, 'r') as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) == 16:
                    mats.append(np.array(vals, dtype=np.float32).reshape(4, 4))
        if len(mats) == points.shape[0]:
            camera_poses = np.stack(mats, axis=0)

    predictions = dict(points=points, conf=confs, images=images)
    if camera_poses is not None:
        predictions['camera_poses'] = camera_poses

    # Persist predictions for re-viz
    np.savez(os.path.join(save_dir, 'predictions.npz'), **predictions)

    frame_choices = build_frame_filter_choices(images_root)
    return save_dir, predictions, frame_choices


# ------------------------------
# Gradio callbacks
# ------------------------------
def clear_fields():
    return None


# ------------------------------
# Export PLY with normals
# ------------------------------
def export_ply_with_normals(save_dir: str, filename: str, conf_thres: float, dedup_enable: bool, dedup_radius: float, normals_k: int, chunk_size: int):
    if not save_dir or not os.path.isdir(save_dir):
        return None, "No valid target directory found."
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        return None, f"No predictions.npz at {pred_path}"

    loaded = np.load(pred_path)
    for key in ["points", "conf", "images"]:
        if key not in loaded:
            return None, f"Missing key '{key}' in predictions.npz"
    points = np.array(loaded['points'])
    conf = np.array(loaded['conf'])
    images = np.array(loaded['images'])
    cams = None
    if 'camera_poses' in loaded:
        cams_np = np.array(loaded['camera_poses'])
        cams = cams_np[..., :3, 3]

    N, H, W, _ = points.shape
    P = points.reshape(-1, 3)
    C = conf.reshape(-1)
    RGB = (np.clip(images, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
    cam_pos = np.repeat(cams, H * W, axis=0) if cams is not None else np.zeros_like(P)

    thr = (conf_thres or 0.0) / 100.0
    mask = (C >= thr) & np.isfinite(P).all(axis=1) & (C > 1e-6)
    P = P[mask]
    RGB = RGB[mask]
    cam_pos = cam_pos[mask]
    if dedup_enable and dedup_radius is not None and float(dedup_radius) > 0:
        keys = np.floor(P / float(dedup_radius)).astype(np.int64)
        keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, uniq_idx = np.unique(keys_view, return_index=True)
        P = P[uniq_idx]
        RGB = RGB[uniq_idx]
        cam_pos = cam_pos[uniq_idx]
    M = P.shape[0]
    if M == 0:
        return None, "No points to export after filtering."

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    K = max(3, int(normals_k or 16))
    normals = np.zeros_like(P, dtype=np.float32)
    chunk = max(10000, int(chunk_size or 50000))
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        d, idx = tree.query(P[s:e], k=K+1)
        idx = idx[:, 1:]
        nbrs = P[idx]
        cent = nbrs.mean(axis=1, keepdims=True)
        X = nbrs - cent
        cov = np.einsum('mki,mkj->mij', X, X) / max(1, K-1)
        w, v = np.linalg.eigh(cov)
        n = v[:, :, 0]
        view_dir = (P[s:e] - cam_pos[s:e])
        sign = np.sign(np.sum(n * view_dir, axis=1, keepdims=True) + 1e-12)
        n = n * sign
        normals[s:e] = n.astype(np.float32)

    # Invert positions after normal processing
    P = -P

    out_path = os.path.join(save_dir, filename if filename else 'export_cloud.ply')
    try:
        with open(out_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {M}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property float nx\n')
            f.write('property float ny\n')
            f.write('property float nz\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(M):
                x, y, z = P[i]
                nx, ny, nz = normals[i]
                r, g, b = RGB[i]
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n')
    except Exception as e:
        return None, f"Failed to write PLY: {e}"
    return None, f"PLY exported: {out_path} (points={M}, k={K})"

def update_visualization(save_dir, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example):
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."
    if not save_dir or save_dir == "None" or not os.path.isdir(save_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."
    predictions_path = os.path.join(save_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    loaded = np.load(predictions_path)
    keys = ["images", "points", "conf", "camera_poses"]
    predictions = {}
    for k in keys:
        if k in loaded:
            predictions[k] = np.array(loaded[k])

    dedup_tag = f"_dedup{int(bool(dedup_enable))}_rad{(0.0 if dedup_radius is None else float(dedup_radius)):.6f}"
    dedup_tag = dedup_tag.replace('.', 'p')
    glbfile = os.path.join(
        save_dir,
        f"glbscene_{conf_thres}_{str(frame_filter).replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}{dedup_tag}.glb",
    )
    if not os.path.exists(glbfile):
        scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
            dedup_enable=bool(dedup_enable),
            dedup_radius=float(dedup_radius) if dedup_radius is not None else 0.001,
        )
        scene.export(file_obj=glbfile)
    return glbfile, "Updating Visualization"


def gradio_demo(target_dir,
                conf_thres=20.0,
                frame_filter="All",
                show_cam=True,
                dedup_enable=True,
                dedup_radius=0.001,
                chunk_size=60,
                overlap=30,
                loop_enable=True,
                loop_chunk_size=20,
                use_dbow=False,
                align_method='numba',
                keep_temps=True,
                zero_depth_edges=True,
                keep_models_loaded=True):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    base_conf = load_config(os.path.join("configs", "base_config.yaml"))
    images_root = os.path.join(target_dir, "images")
    frame_filter_choices = ["All"] + build_frame_filter_choices(images_root)

    print("Running Pi-Long pipeline...")
    save_dir, predictions, frame_choices = run_pi_long_and_aggregate(
        target_dir,
        conf=base_conf,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
        loop_chunk_size=loop_chunk_size,
        use_dbow=use_dbow,
        align_method=align_method,
        keep_temps=keep_temps,
        zero_depth_edges=zero_depth_edges,
        keep_models_loaded=keep_models_loaded,
    )

    dedup_tag = f"_dedup{int(bool(dedup_enable))}_rad{(0.0 if dedup_radius is None else float(dedup_radius)):.6f}"
    dedup_tag = dedup_tag.replace('.', 'p')
    glbfile = os.path.join(
        save_dir,
        f"glbscene_{conf_thres}_{str(frame_filter).replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}{dedup_tag}.glb",
    )
    scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
        dedup_enable=bool(dedup_enable),
        dedup_radius=float(dedup_radius) if dedup_radius is not None else 0.001,
    )
    scene.export(file_obj=glbfile)

    return glbfile, f"Reconstruction Success ({predictions['points'].shape[0]} frames). Waiting for visualization.", gr.Dropdown(choices=["All"] + frame_choices, value="All", interactive=True), save_dir

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()} | device: {device}")

    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(theme=theme) as demo:
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        target_dir_output = gr.Textbox(label="Working Dir", visible=False, value="None")
        save_dir_output = gr.Textbox(label="Save Dir (predictions)", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Upload Media")
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(file_count="multiple", label="Or Upload Images", interactive=True)
                    fps = gr.Number(1, label='Video FPS (frames/sec)', info="Sampling rate for video extraction.")
                    interval = gr.Number(None, label='Image Interval', info="Sampling interval for image lists; leave empty to use all images.")
                    # Reload frames button (re-extract frames with current FPS)
                    reload_btn = gr.Button("Reload Frames", variant="secondary", scale=1)

                image_gallery = gr.Gallery(
                    label="Image Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 2. View Reconstruction")
                log_output = gr.Markdown("Please upload media and click Reconstruct.")
                reconstruction_output = gr.Model3D(height=480, zoom_speed=0.5, pan_speed=0.5, label="3D Output")

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=3, variant="primary")
                    clear_btn = gr.ClearButton(scale=1)

                with gr.Group():
                    gr.Markdown("### 3. Adjust Visualization")
                    with gr.Row():
                        conf_thres = gr.Slider(minimum=0, maximum=100, value=20, step=0.1, label="Confidence Threshold (%)")
                        show_cam = gr.Checkbox(label="Show Cameras", value=True)
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                    with gr.Row():
                        dedup_enable = gr.Checkbox(label="Spatial Dedup (voxel)", value=True)
                        dedup_radius = gr.Number(0.001, label='Dedup Radius (scene units)')
                    # Refresh display with current visualization options
                    refresh_viz_btn = gr.Button("Refresh Visualization", variant="secondary", scale=1)

                with gr.Group():
                    gr.Markdown("### 4. Pi-Long Controls")
                    with gr.Row():
                        chunk_size = gr.Number(60, label='Chunk Size')
                        overlap = gr.Number(30, label='Overlap')
                        loop_enable = gr.Checkbox(label="Enable Loop Closure", value=True)
                        loop_chunk_size = gr.Number(20, label='Loop Half-Window')
                    with gr.Row():
                        use_dbow = gr.Checkbox(label="Use DBoW2 Loop Retrieval", value=False)
                        align_method = gr.Dropdown(choices=['numba', 'numpy'], value='numba', label='Align Method')
                        keep_temps = gr.Checkbox(label="Keep Temp Files (for re-viz)", value=True)
                        zero_edges = gr.Checkbox(label="Zero Depth Edges", value=True)
                        keep_models_loaded = gr.Checkbox(label="Keep Models Loaded", value=True)
                    # Unload button under the Keep Models Loaded checkbox
                    with gr.Row():
                        unload_btn = gr.Button("Unload Models", variant="secondary", scale=1)
                        unload_log = gr.Markdown(visible=False)
                        unload_btn.click(fn=unload_models, inputs=[], outputs=[unload_log])

                with gr.Group():
                    gr.Markdown("### 5. Local Refinement (Frame-level)")
                    with gr.Row():
                        enable_refine = gr.Checkbox(label="Enable Local Refinement", value=False)
                        refine_corr_radius = gr.Number(0.03, label='Correspondence Radius (m)')
                        refine_voxel_ref = gr.Number(0.02, label='Ref Voxel (m)')
                    with gr.Row():
                        refine_iters = gr.Number(3, label='Iters per Frame')
                        refine_max_pts = gr.Number(10000, label='Max Points per Frame')
                        refine_smooth = gr.Number(0.2, label='Temporal Smooth λ (0–1)')
                    with gr.Row():
                        refine_huber = gr.Number(0.005, label='Huber δ (m)')
                        refine_trim = gr.Number(0.2, label='Trim Ratio (0–1)')
                        refine_step = gr.Number(0.5, label='Step Scale (0–1)')
                    refine_btn = gr.Button("Refine Locally", variant="secondary")

                with gr.Group():
                    gr.Markdown("### 6. Piecewise Refinement (Voxel-wise, Overlapping)")
                    with gr.Row():
                        enable_piecewise = gr.Checkbox(label="Enable Piecewise Refinement", value=False)
                        pw_use_gpu = gr.Checkbox(label="Use GPU (PyTorch3D / torch.cdist)", value=True)
                        pw_voxel = gr.Number(0.03, label='Voxel Size (m)')
                        pw_corr_factor = gr.Number(1.5, label='Corr Radius Factor')
                    with gr.Row():
                        pw_min_pairs = gr.Number(50, label='Min Pairs per Voxel')
                        pw_max_pts = gr.Number(5000, label='Max Points per Voxel')
                        pw_workers = gr.Number(0, label='Threads (0=no threads, -1=all)')
                        pw_kd_workers = gr.Number(1, label='KD workers (-1=all)')
                    with gr.Row():
                        pw_fp16 = gr.Checkbox(label="GPU FP16", value=True)
                        pw_ref_cap = gr.Number(500000, label='Ref Cap (points)')
                        pw_cdist_chunk = gr.Number(20000, label='GPU cdist chunk')
                        pw_ref_voxel = gr.Number(0.02, label='Ref Voxel Dedup (m)')
                    piecewise_btn = gr.Button("Refine Piecewise", variant="secondary")

                with gr.Group():
                    gr.Markdown("### 7. Export")
                    with gr.Row():
                        export_filename = gr.Textbox(value='export_cloud.ply', label='PLY Filename')
                        normals_k = gr.Number(16, label='Normals k')
                        normals_chunk = gr.Number(50000, label='Normals chunk')
                    export_btn = gr.Button("Export PLY", variant="secondary")

        # Clear
        clear_btn.add([input_video, input_images, reconstruction_output, log_output, target_dir_output, save_dir_output, image_gallery, fps, interval])

        # Upload handlers
        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, fps, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, fps, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        # Reload frames handler (same as upload handlers)
        reload_btn.click(
            fn=reload_frames_in_place,
            inputs=[target_dir_output, input_video, input_images, fps, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

        # Reconstruct button
        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=lambda: "Loading and Reconstructing...", inputs=[], outputs=[log_output]
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                dedup_enable,
                dedup_radius,
                chunk_size,
                overlap,
                loop_enable,
                loop_chunk_size,
                use_dbow,
                align_method,
                keep_temps,
                zero_edges,
                keep_models_loaded,
            ],
            outputs=[reconstruction_output, log_output, frame_filter, save_dir_output],
        ).then(
            fn=lambda: "False", inputs=[], outputs=[is_example]
        )

        # Real-time visualization updates
        conf_thres.change(
            update_visualization,
            [save_dir_output, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example],
            [reconstruction_output, log_output],
        )
        frame_filter.change(
            update_visualization,
            [save_dir_output, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example],
            [reconstruction_output, log_output],
        )
        show_cam.change(
            update_visualization,
            [save_dir_output, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example],
            [reconstruction_output, log_output],
        )
        # Manual refresh button under dedup controls
        refresh_viz_btn.click(
            update_visualization,
            [save_dir_output, conf_thres, frame_filter, show_cam, dedup_enable, dedup_radius, is_example],
            [reconstruction_output, log_output],
        )

        # Local refinement (gated inside function by checkbox)
        refine_btn.click(
            refine_and_update_viz,
            inputs=[
                enable_refine,
                save_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                dedup_enable,
                dedup_radius,
                refine_corr_radius,
                refine_max_pts,
                refine_iters,
                refine_smooth,
                refine_voxel_ref,
                refine_huber,
                refine_trim,
                refine_step,
            ],
            outputs=[reconstruction_output, log_output],
        )

        # Piecewise refinement
        piecewise_btn.click(
            dispatch_piecewise_refine,
            inputs=[
                enable_piecewise,
                pw_use_gpu,
                save_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                dedup_enable,
                dedup_radius,
                pw_voxel,
                pw_corr_factor,
                pw_min_pairs,
                pw_max_pts,
                pw_workers,
                pw_kd_workers,
                pw_fp16,
                pw_ref_cap,
                pw_cdist_chunk,
                pw_ref_voxel,
            ],
            outputs=[reconstruction_output, log_output],
        )

        # Export PLY
        export_btn.click(
            export_ply_with_normals,
            inputs=[
                save_dir_output,
                export_filename,
                conf_thres,
                dedup_enable,
                dedup_radius,
                normals_k,
                normals_chunk,
            ],
            outputs=[reconstruction_output, log_output],
        )

    demo.queue(max_size=20).launch(show_error=True, share=True)