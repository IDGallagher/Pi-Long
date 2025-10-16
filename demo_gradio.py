import os
import cv2
import gc
import time
import glob
import json
import shutil
import random
import re
import sqlite3
import subprocess
import torch
import numpy as np
import gradio as gr
import trimesh
import traceback
from datetime import datetime

from scipy.spatial.transform import Rotation
import matplotlib
from PIL import Image
from pi3.utils.geometry import depth_edge

from loop_utils.config_utils import load_config
from pi_long import Pi_Long

# Optional GPU neighbor search via PyTorch3D
try:
    from pytorch3d.ops import knn_points as _knn_points
    _PT3D_AVAILABLE = True
except Exception:
    _PT3D_AVAILABLE = False

# Optional Open3D for mesh reconstruction
try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except Exception:
    _O3D_AVAILABLE = False


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


def _toast_info(msg: str):
    try:
        gr.Info(msg)
    except Exception:
        pass


def _toast_warn(msg: str):
    try:
        gr.Warning(msg)
    except Exception:
        pass


def _toast_error(msg: str):
    try:
        gr.Error(msg)
    except Exception:
        pass

def set_global_seed(seed: int):
    try:
        seed = int(seed)
    except Exception:
        return
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    x = np.clip(img.astype(np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    x = np.clip(img.astype(np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
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
        if not vs.isOpened():
            _toast_warn("Video could not be opened; skipping video frame extraction.")
            print(f"[Upload] Failed to open video: {video_path}")
        else:
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
                    frame_out = _correct_video_orientation(frame, vs, video_path)
                    image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                    cv2.imwrite(image_path, frame_out)
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
        if not vs.isOpened():
            _toast_warn("Video could not be opened for reload; skipping video frames.")
            print(f"[Reload] Failed to open video: {video_path}")
        else:
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
                    frame_out = _correct_video_orientation(frame, vs, video_path)
                    image_path = os.path.join(images_root, f"{video_frame_num:06}.png")
                    cv2.imwrite(image_path, frame_out)
                    image_paths.append(image_path)
                    video_frame_num += 1
                count += 1
            vs.release()

    image_paths = sorted(image_paths)
    return None, target_dir, image_paths, "Frames reloaded. Click 'Reconstruct'."


def append_video_to_images(target_dir, append_video, fps):
    """
    Append frames from a new video into the existing target_dir/images without resetting.
    Uses the given fps (frames/sec) to sample new frames and continues filename indices.
    """
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, None, None, "No working directory. Please upload first."

    images_root = os.path.join(target_dir, "images")
    os.makedirs(images_root, exist_ok=True)

    if append_video is None:
        return None, target_dir, sorted([os.path.join(images_root, f) for f in os.listdir(images_root) if f.lower().endswith((".png", ".jpg", ".jpeg"))]), "No video selected to append."

    video_path = append_video["name"] if isinstance(append_video, dict) and "name" in append_video else append_video
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        _toast_warn("Append video could not be opened; no frames appended.")
        print(f"[AppendVideo] Failed to open video: {video_path}")
        return None, target_dir, sorted([os.path.join(images_root, f) for f in os.listdir(images_root) if f.lower().endswith((".png", ".jpg", ".jpeg"))]), "Failed to open append video."
    src_fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
    desired_fps = fps if (fps is not None and fps > 0) else 1.0
    frame_interval = max(1, int(round(src_fps / desired_fps)))

    # Determine next index based on existing files
    existing = [fn for fn in os.listdir(images_root) if fn.lower().endswith((".png", ".jpg", ".jpeg"))]
    max_idx = -1
    for fn in existing:
        base = os.path.splitext(fn)[0]
        try:
            num = int(base)
            if num > max_idx:
                max_idx = num
        except Exception:
            pass
    next_idx = max_idx + 1 if max_idx >= 0 else len(existing)

    count = 0
    saved = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        if count % frame_interval == 0:
            frame_out = _correct_video_orientation(frame, vs, video_path)
            image_path = os.path.join(images_root, f"{next_idx:06}.png")
            try:
                cv2.imwrite(image_path, frame_out)
                next_idx += 1
                saved += 1
            except Exception:
                pass
        count += 1
    vs.release()

    image_paths = sorted([os.path.join(images_root, f) for f in os.listdir(images_root) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return None, target_dir, image_paths, f"Appended {saved} frames from video. Click 'Reconstruct'."


def build_frame_filter_choices(images_root: str):
    all_files = sorted(os.listdir(images_root)) if os.path.isdir(images_root) else []
    return [f"{i}: {filename}" for i, filename in enumerate(all_files)]


def _normalize_rotation(angle: float | int | None) -> int:
    if angle is None:
        return 0
    try:
        angle = float(angle)
    except Exception:
        return 0
    if not np.isfinite(angle):
        return 0
    angle = angle % 360.0
    candidates = [0, 90, 180, 270]
    best = min(candidates, key=lambda c: min(abs(angle - c), 360.0 - abs(angle - c)))
    diff = min(abs(angle - best), 360.0 - abs(angle - best))
    if diff > 6.0:
        best = int(round(angle / 90.0) * 90) % 360
    return int(best)


def _probe_video_rotation(video_path: str) -> int:
    if not video_path or not os.path.isfile(video_path):
        return 0
    # Try ffprobe if available
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=rotate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        rotate_str = (result.stdout or "").strip()
        if rotate_str:
            return _normalize_rotation(float(rotate_str))
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass
    except Exception as exc:
        print(f"[Orientation] ffprobe rotation query failed: {exc}")
    # Try MoviePy if installed
    try:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video_path)
        angle = float(getattr(clip, "rotation", 0.0))
        clip.close()
        return _normalize_rotation(angle)
    except Exception:
        pass
    return 0


def _heuristic_vertical_rotation(frame: np.ndarray) -> int:
    if frame is None:
        return 0
    h, w = frame.shape[:2]
    if h == 0 or w == 0 or w <= h:
        return 0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_val = max(5, int(gray.mean() * 0.1))
        _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return 0
        x, y, bw, bh = cv2.boundingRect(coords)
        if bw == 0 or bh == 0:
            return 0
        aspect = bw / float(bh)
        if aspect < 0.85 and (bh / max(1.0, bw)) > 1.1:
            return 90
        col_profile = (mask > 0).mean(axis=0)
        nonzero_cols = np.where(col_profile > 0.05)[0]
        if nonzero_cols.size > 0:
            span = nonzero_cols[-1] - nonzero_cols[0] + 1
            if span < 0.7 * h and (h / max(1.0, span)) > 1.1:
                return 90
    except Exception as exc:
        print(f"[Orientation] Heuristic rotation check failed: {exc}")
    return 0


def _crop_letterbox(frame: np.ndarray,
                    edge_threshold: float = 0.08,
                    min_margin_ratio: float = 0.02) -> tuple[np.ndarray, dict]:
    if frame is None or frame.size == 0:
        return frame, {}
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thr = max(4, int(np.percentile(gray, 5)))
    mask = gray > thr
    row_profile = mask.mean(axis=1)
    col_profile = mask.mean(axis=0)

    def find_bounds(profile, length):
        start = 0
        while start < length and profile[start] < edge_threshold:
            start += 1
        end = length - 1
        while end > start and profile[end] < edge_threshold:
            end -= 1
        return start, end

    top, bottom = find_bounds(row_profile, H)
    left, right = find_bounds(col_profile, W)

    top_margin = top
    bottom_margin = H - 1 - bottom
    left_margin = left
    right_margin = W - 1 - right

    crop_top = 0
    crop_bottom = H
    crop_left = 0
    crop_right = W

    if top_margin + bottom_margin > int(min_margin_ratio * H * 2):
        crop_top = max(0, top)
        crop_bottom = min(H, bottom + 1)

    if left_margin + right_margin > int(min_margin_ratio * W * 2):
        crop_left = max(0, left)
        crop_right = min(W, right + 1)

    crop_info = dict(
        original_height=H,
        original_width=W,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    if (crop_top == 0 and crop_bottom == H and crop_left == 0 and crop_right == W):
        return frame, crop_info

    cropped = frame[crop_top:crop_bottom, crop_left:crop_right]
    if cropped.size == 0:
        return frame, crop_info
    return cropped, crop_info


def _rescale_portrait(frame: np.ndarray,
                      crop_info: dict,
                      target_height: int | None = None) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    H, W = frame.shape[:2]
    if H == 0 or W == 0:
        return frame
    if target_height is None:
        target_height = crop_info.get("original_height", H)
    if target_height <= 0:
        target_height = H

    left_margin = crop_info.get("left_margin", 0)
    right_margin = crop_info.get("right_margin", 0)

    if H >= W and (left_margin + right_margin) > 0:
        scale = target_height / float(H)
        if scale <= 0:
            return frame
        target_width = int(round(W * scale))
        if target_width % 2 != 0:
            target_width += 1
        interpolation = cv2.INTER_CUBIC if scale > 1.01 else cv2.INTER_AREA
        frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
    return frame


_VIDEO_ROTATION_CACHE: dict[str, dict[str, object]] = {}


def _capture_cache_key(capture, video_path: str | None) -> str | None:
    if capture is not None:
        return f"cap_{id(capture)}"
    if video_path:
        try:
            return f"path_{os.path.abspath(video_path)}"
        except Exception:
            return f"path_{video_path}"
    return None


def _correct_video_orientation(frame: np.ndarray,
                               capture,
                               video_path: str | None = None,
                               rotation_hint: int | None = None) -> np.ndarray:
    if frame is None:
        return frame
    rotation = rotation_hint
    cache_key = _capture_cache_key(capture, video_path)
    if rotation is None and cache_key and cache_key in _VIDEO_ROTATION_CACHE:
        rotation = _VIDEO_ROTATION_CACHE[cache_key].get("rotation")
    if rotation is not None:
        rotation = _normalize_rotation(rotation)
    if rotation is None:
        rotation = 0
        prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        if capture is not None and prop is not None:
            try:
                orientation_val = capture.get(prop)
                if orientation_val is not None:
                    rotation = _normalize_rotation(orientation_val)
            except Exception as exc:
                print(f"[Orientation] CAP_PROP_ORIENTATION_META read failed: {exc}")
        if rotation == 0 and video_path:
            rotation = _probe_video_rotation(video_path)
        if rotation == 0:
            rotation = _heuristic_vertical_rotation(frame)
        rotation = _normalize_rotation(rotation)
        if cache_key:
            entry = _VIDEO_ROTATION_CACHE.setdefault(cache_key, {})
            entry["rotation"] = rotation
            if not entry.get("logged"):
                print(f"[Orientation] Detected rotation {rotation} deg for video {video_path}")
                entry["logged"] = True
    if rotation in (90, 270):
        h, w = frame.shape[:2]
        if h < w:
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotation = 0
            if cache_key:
                entry = _VIDEO_ROTATION_CACHE.setdefault(cache_key, {})
                entry["rotation"] = rotation
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    cropped, crop_info = _crop_letterbox(frame)
    if crop_info:
        frame = _rescale_portrait(cropped, crop_info, target_height=crop_info.get("original_height"))
    else:
        frame = cropped
    return frame


def _sanitize_subfolder_name(name: str, default: str = "colmap") -> str:
    if not isinstance(name, str):
        return default
    trimmed = name.strip()
    if not trimmed:
        return default
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", trimmed)
    sanitized = sanitized.strip("._")
    return sanitized or default


def _estimate_shared_intrinsics(points: np.ndarray,
                                camera_poses: np.ndarray,
                                max_samples_per_frame: int = 20000) -> tuple[float, float, float, float]:
    H, W = points.shape[1:3]
    if camera_poses is None or len(camera_poses) == 0:
        focal = float(max(H, W))
        return focal, focal, float(W * 0.5), float(H * 0.5)

    rng = np.random.default_rng(12345)
    n_frames = min(points.shape[0], camera_poses.shape[0])
    grid_v, grid_u = np.meshgrid(np.arange(H, dtype=np.float64),
                                 np.arange(W, dtype=np.float64),
                                 indexing='ij')
    u_flat = grid_u.reshape(-1)
    v_flat = grid_v.reshape(-1)

    x_norm_all = []
    y_norm_all = []
    u_all = []
    v_all = []

    for idx in range(n_frames):
        pts_world = points[idx].reshape(-1, 3)
        valid_mask = np.isfinite(pts_world).all(axis=1)
        if not np.any(valid_mask):
            continue
        pts_world = pts_world[valid_mask]
        u_vals = u_flat[valid_mask]
        v_vals = v_flat[valid_mask]

        c2w = camera_poses[idx]
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        pts_cam = (pts_world - t_c2w) @ R_c2w.T
        z = pts_cam[:, 2]
        depth_mask = np.abs(z) > 1e-6
        if not np.any(depth_mask):
            continue

        pts_cam = pts_cam[depth_mask]
        u_vals = u_vals[depth_mask]
        v_vals = v_vals[depth_mask]

        if max_samples_per_frame and pts_cam.shape[0] > max_samples_per_frame:
            sel = rng.choice(pts_cam.shape[0], size=int(max_samples_per_frame), replace=False)
            pts_cam = pts_cam[sel]
            u_vals = u_vals[sel]
            v_vals = v_vals[sel]

        x_norm = pts_cam[:, 0] / pts_cam[:, 2]
        y_norm = pts_cam[:, 1] / pts_cam[:, 2]

        x_norm_all.append(x_norm.astype(np.float64))
        y_norm_all.append(y_norm.astype(np.float64))
        u_all.append(u_vals.astype(np.float64))
        v_all.append(v_vals.astype(np.float64))

    if len(x_norm_all) == 0:
        focal = float(max(H, W))
        return focal, focal, float(W * 0.5), float(H * 0.5)

    x_norm_all = np.concatenate(x_norm_all)
    y_norm_all = np.concatenate(y_norm_all)
    u_all = np.concatenate(u_all)
    v_all = np.concatenate(v_all)

    A_u = np.stack([x_norm_all, np.ones_like(x_norm_all)], axis=1)
    A_v = np.stack([y_norm_all, np.ones_like(y_norm_all)], axis=1)

    sol_u, *_ = np.linalg.lstsq(A_u, u_all, rcond=None)
    sol_v, *_ = np.linalg.lstsq(A_v, v_all, rcond=None)

    fx, cx = sol_u
    fy, cy = sol_v

    if not np.isfinite(fx) or fx <= 0:
        fx = float(max(H, W))
    if not np.isfinite(fy) or fy <= 0:
        fy = float(max(H, W))
    if not np.isfinite(cx):
        cx = float(W * 0.5)
    if not np.isfinite(cy):
        cy = float(H * 0.5)

    return float(fx), float(fy), float(cx), float(cy)


_COLMAP_MAX_IMAGE_ID = 2**31 - 1
_COLMAP_DB_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {_COLMAP_MAX_IMAGE_ID}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB);
CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
"""


def _initialize_empty_colmap_database(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_COLMAP_DB_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _write_colmap_project_ini(colmap_dir: str,
                              image_path: str | None = None,
                              database_path: str | None = None):
    image_path = os.path.abspath(image_path if image_path is not None else os.path.join(colmap_dir, "images"))
    database_path = os.path.abspath(database_path if database_path is not None else os.path.join(colmap_dir, "database.db"))
    ini_path = os.path.join(colmap_dir, "project.ini")
    with open(ini_path, 'w', encoding='utf-8') as f:
        f.write("log_to_stderr=true\n")
        f.write("log_level=0\n")
        f.write("random_seed=0\n")
        f.write(f"database_path={database_path}\n")
        f.write(f"image_path={image_path}\n")
    return ini_path


def _build_colmap_entities(points: np.ndarray,
                           conf: np.ndarray,
                           images: np.ndarray,
                           image_sizes: list[tuple[int, int]],
                           conf_thres: float,
                           dedup_enable: bool,
                           dedup_radius: float,
                           max_points: int = 500_000) -> tuple[list, list]:
    if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = np.clip(images, 0.0, 1.0)
    images_uint8 = (images * 255).astype(np.uint8)

    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    total_frames = min(points.shape[0], len(image_sizes))
    if total_frames == 0:
        return [], []

    H, W = points.shape[1], points.shape[2]
    grid_v, grid_u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid_u = grid_u.reshape(-1)
    grid_v = grid_v.reshape(-1)

    rng = np.random.default_rng(1337)
    threshold = 0.0
    if conf_thres is not None:
        try:
            threshold = (float(conf_thres) or 0.0) / 100.0
        except Exception:
            threshold = 0.0

    sample_cap = None
    if max_points and total_frames > 0:
        sample_cap = max(1, int(max_points // total_frames))

    dedup_set = None
    dedup_radius = float(dedup_radius or 0.0)
    if dedup_enable and dedup_radius > 0:
        dedup_set = set()

    point_records = []
    image_measurements = [[] for _ in range(total_frames)]

    for img_idx in range(total_frames):
        pts_flat = points[img_idx].reshape(-1, 3)
        conf_flat = conf[img_idx].reshape(-1)
        colors_flat = images_uint8[img_idx].reshape(-1, 3)

        valid = np.isfinite(pts_flat).all(axis=1) & (conf_flat > 1e-6)
        if threshold > 0:
            valid &= conf_flat >= threshold

        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            continue

        if sample_cap is not None and valid_idx.size > sample_cap:
            valid_idx = rng.choice(valid_idx, size=sample_cap, replace=False)

        width, height = image_sizes[img_idx]
        sx = (width - 1) / max(1, W - 1)
        sy = (height - 1) / max(1, H - 1)

        for flat_idx in valid_idx:
            xyz = pts_flat[flat_idx]
            if dedup_set is not None:
                key = tuple(np.floor(xyz / dedup_radius).astype(np.int64))
                if key in dedup_set:
                    continue
                dedup_set.add(key)

            u_model = grid_u[flat_idx]
            v_model = grid_v[flat_idx]
            x_px = float(u_model * sx)
            y_px = float(v_model * sy)
            if not (np.isfinite(x_px) and np.isfinite(y_px)):
                continue

            point_id = len(point_records) + 1
            obs_idx = len(image_measurements[img_idx])
            image_measurements[img_idx].append((x_px, y_px, point_id))

            conf_val = float(np.clip(conf_flat[flat_idx], 0.0, 1.0))
            error = float(max(1e-6, 1.0 - conf_val))
            color = colors_flat[flat_idx]

            point_records.append((
                point_id,
                xyz.astype(np.float64),
                color.astype(np.uint8),
                error,
                [(img_idx + 1, obs_idx)],
            ))

            if max_points and len(point_records) >= max_points:
                break

        if max_points and len(point_records) >= max_points:
            break

    return point_records, image_measurements


def _c2w_to_colmap_pose(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    rot = Rotation.from_matrix(R_w2c)
    q_xyzw = rot.as_quat()
    q_colmap = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return q_colmap, t_w2c.astype(np.float64)


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
    _toast_info("Exporting PLY…")
    print(f"[ExportPLY] Start | save_dir={save_dir}, filename={filename}, conf%={conf_thres}, dedup={dedup_enable}, rad={dedup_radius}, k={normals_k}, chunk={chunk_size}")
    if not save_dir or not os.path.isdir(save_dir):
        _toast_error("No valid target directory found.")
        print("[ExportPLY][Error] Invalid save_dir")
        return None, "No valid target directory found."
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        _toast_error("No predictions file found. Run Reconstruct first.")
        print(f"[ExportPLY][Error] Missing predictions at {pred_path}")
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
    print(f"[ExportPLY] Filtered points: {P.shape[0]} (threshold={thr:.4f})")
    if dedup_enable and dedup_radius is not None and float(dedup_radius) > 0:
        keys = np.floor(P / float(dedup_radius)).astype(np.int64)
        keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
        _, uniq_idx = np.unique(keys_view, return_index=True)
        P = P[uniq_idx]
        RGB = RGB[uniq_idx]
        cam_pos = cam_pos[uniq_idx]
        print(f"[ExportPLY] Deduped points: {P.shape[0]} (radius={dedup_radius})")
    M = P.shape[0]
    if M == 0:
        _toast_warn("No points to export after filtering.")
        return None, "No points to export after filtering."

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    K = max(3, int(normals_k or 16))
    normals = np.zeros_like(P, dtype=np.float32)
    chunk = max(10000, int(chunk_size or 50000))
    print(f"[ExportPLY] Computing normals | M={M}, k={K}, chunk={chunk}")
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

    # Keep positions as-is to match mesh orientation

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
        _toast_error("PLY write failed.")
        print("[ExportPLY][Error] Write failed:\n" + traceback.format_exc())
        return None, f"Failed to write PLY: {e}"
    _toast_info("PLY export finished.")
    print(f"[ExportPLY] Finished | path={out_path}, points={M}")
    return None, f"PLY exported: {out_path} (points={M}, k={K})"


def export_colmap_dataset(save_dir: str,
                          folder_name: str,
                          conf_thres: float,
                          dedup_enable: bool,
                          dedup_radius: float):
    _toast_info("Exporting COLMAP dataset…")
    print(f"[ExportCOLMAP] Start | save_dir={save_dir}, folder={folder_name}, conf%={conf_thres}, dedup={dedup_enable}, rad={dedup_radius}")

    if not save_dir or not os.path.isdir(save_dir):
        _toast_error("No valid target directory found.")
        print("[ExportCOLMAP][Error] Invalid save_dir")
        return None, "No valid target directory found."

    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        _toast_error("No predictions file found. Run Reconstruct first.")
        print(f"[ExportCOLMAP][Error] Missing predictions at {pred_path}")
        return None, f"No predictions.npz at {pred_path}"

    try:
        data = np.load(pred_path)
    except Exception:
        _toast_error("Failed to load predictions.")
        print("[ExportCOLMAP][Error] Load predictions failed:\n" + traceback.format_exc())
        return None, "Failed to load predictions."

    required_keys = ["points", "conf", "images", "camera_poses"]
    for key in required_keys:
        if key not in data:
            _toast_error(f"Missing '{key}' in predictions.")
            print(f"[ExportCOLMAP][Error] Missing key '{key}' in predictions.")
            return None, f"Missing key '{key}' in predictions."

    points = np.array(data['points'])
    conf = np.array(data['conf'])
    images = np.array(data['images'])
    camera_poses = np.array(data['camera_poses'])

    if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
        images = np.transpose(images, (0, 2, 3, 1))

    total_frames = points.shape[0]
    if total_frames == 0:
        _toast_error("No frames available for COLMAP export.")
        print("[ExportCOLMAP][Error] No frames in predictions.")
        return None, "No frames available for COLMAP export."

    if camera_poses.shape[0] < total_frames:
        total_frames = camera_poses.shape[0]
        _toast_warn("Fewer camera poses than frames; truncating to pose count.")
        print("[ExportCOLMAP][Warn] camera_poses shorter than points; truncating.")

    H, W = points.shape[1], points.shape[2]

    folder_clean = _sanitize_subfolder_name(folder_name or "colmap")
    colmap_dir = os.path.join(save_dir, folder_clean)
    images_dir = os.path.join(colmap_dir, 'images')
    sparse_dir = os.path.join(colmap_dir, 'sparse', '0')

    try:
        if os.path.isdir(colmap_dir):
            shutil.rmtree(colmap_dir)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
    except Exception as e:
        _toast_error("Failed to prepare COLMAP directories.")
        print("[ExportCOLMAP][Error] Directory prep failed:\n" + traceback.format_exc())
        return None, f"Failed to prepare COLMAP directories: {e}"

    input_images_dir = os.path.join(save_dir, 'images')
    sorted_inputs = []
    if os.path.isdir(input_images_dir):
        sorted_inputs = sorted(
            [
                os.path.join(input_images_dir, f)
                for f in os.listdir(input_images_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))
            ]
        )
    if len(sorted_inputs) < total_frames:
        print(f"[ExportCOLMAP][Warn] Only {len(sorted_inputs)} source images found for {total_frames} frames. Falling back to cached frames where needed.")

    colmap_image_names = []
    image_sizes = []
    for idx in range(total_frames):
        dest_name = f"frame_{idx:06d}.png"
        dest_path = os.path.join(images_dir, dest_name)
        img_written = False
        width = None
        height = None
        if idx < len(sorted_inputs):
            src = sorted_inputs[idx]
            try:
                with Image.open(src) as img_pil:
                    img_rgb = img_pil.convert('RGB')
                    width, height = img_rgb.size
                    img_rgb.save(dest_path)
                img_written = True
            except Exception:
                print(f"[ExportCOLMAP][Warn] Failed to convert source image {src}; falling back to cached tensor.")

        if not img_written:
            frame_img = images[idx]
            if frame_img.ndim == 3 and frame_img.shape[0] == 3 and frame_img.shape[-1] != 3:
                frame_img = np.transpose(frame_img, (1, 2, 0))
            frame_img = np.clip(frame_img, 0.0, 1.0)
            frame_uint8 = (frame_img * 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_uint8)
            width, height = pil_img.size
            pil_img.save(dest_path)

        if width is None or height is None:
            try:
                with Image.open(dest_path) as check_img:
                    width, height = check_img.size
            except Exception:
                width = W
                height = H

        colmap_image_names.append(dest_name)
        image_sizes.append((int(width), int(height)))

    points_subset = points[:total_frames]
    conf_subset = conf[:total_frames]
    images_subset = images[:total_frames]
    camera_subset = camera_poses[:total_frames]

    fx, fy, cx, cy = _estimate_shared_intrinsics(points_subset, camera_subset)
    print(f"[ExportCOLMAP] Estimated intrinsics: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    point_records, image_measurements = _build_colmap_entities(
        points_subset,
        conf_subset,
        images_subset,
        image_sizes,
        conf_thres,
        dedup_enable,
        dedup_radius,
    )
    num_points = len(point_records)
    if num_points == 0:
        _toast_warn("No valid 3D points passed filtering; exporting cameras only.")
        print("[ExportCOLMAP][Warn] No valid 3D points after filtering.")

    cameras_txt = os.path.join(sparse_dir, 'cameras.txt')
    try:
        with open(cameras_txt, 'w', encoding='utf-8') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            f.write(f"1 PINHOLE {int(W)} {int(H)} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")
    except Exception:
        _toast_error("Failed to write cameras.txt.")
        print("[ExportCOLMAP][Error] cameras.txt write failed:\n" + traceback.format_exc())
        return None, "Failed to write cameras.txt."

    images_txt = os.path.join(sparse_dir, 'images.txt')
    try:
        with open(images_txt, 'w', encoding='utf-8') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {total_frames}, mean observations per image: 0\n")
            for idx in range(total_frames):
                qvec, tvec = _c2w_to_colmap_pose(camera_subset[idx])
                f.write(
                    f"{idx + 1} "
                    f"{qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
                    f"{tvec[0]:.8f} {tvec[1]:.8f} {tvec[2]:.8f} "
                    f"1 {colmap_image_names[idx]}\n"
                )
                measurements = image_measurements[idx] if idx < len(image_measurements) else []
                if measurements:
                    pts2d_str = " ".join(f"{obs[0]:.6f} {obs[1]:.6f} {obs[2]}" for obs in measurements)
                    f.write(f"{pts2d_str}\n")
                else:
                    f.write("\n")
    except Exception:
        _toast_error("Failed to write images.txt.")
        print("[ExportCOLMAP][Error] images.txt write failed:\n" + traceback.format_exc())
        return None, "Failed to write images.txt."

    points3d_txt = os.path.join(sparse_dir, 'points3D.txt')
    try:
        with open(points3d_txt, 'w', encoding='utf-8') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {num_points}\n")
            for point_id, xyz, color, error, track in point_records:
                track_str = ""
                if track:
                    track_str = " " + " ".join(f"{img_id} {pt_idx}" for img_id, pt_idx in track)
                f.write(
                    f"{point_id} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} {error:.6f}{track_str}\n"
                )
    except Exception:
        _toast_error("Failed to write points3D.txt.")
        print("[ExportCOLMAP][Error] points3D.txt write failed:\n" + traceback.format_exc())
        return None, "Failed to write points3D.txt."

    db_path = os.path.join(colmap_dir, "database.db")
    _write_colmap_project_ini(colmap_dir, image_path=images_dir, database_path=db_path)
    _initialize_empty_colmap_database(db_path)

    message = f"COLMAP dataset exported: {colmap_dir} ({total_frames} frames, {num_points} points)."
    _toast_info("COLMAP export finished.")
    print(f"[ExportCOLMAP] Finished | dir={colmap_dir}, frames={total_frames}, points={num_points}")
    return None, message


def export_nerfstudio_dataset(save_dir: str,
                              folder_name: str):
    _toast_info("Exporting Nerfstudio dataset…")
    if not save_dir or not os.path.isdir(save_dir):
        _toast_error("No valid target directory found.")
        return None, "No valid target directory found."

    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        _toast_error("No predictions file found. Run Reconstruct first.")
        return None, f"No predictions.npz at {pred_path}"

    try:
        data = np.load(pred_path)
    except Exception:
        _toast_error("Failed to load predictions.")
        print("[ExportNerfstudio][Error] Load predictions failed:\n" + traceback.format_exc())
        return None, "Failed to load predictions."

    required_keys = ["points", "images", "camera_poses"]
    for key in required_keys:
        if key not in data:
            _toast_error(f"Missing key '{key}' in predictions.")
            return None, f"Missing key '{key}' in predictions."

    points = np.array(data["points"])
    images = np.array(data["images"])
    camera_poses = np.array(data["camera_poses"])
    conf = np.array(data["conf"]) if "conf" in data else None

    if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
        images = np.transpose(images, (0, 2, 3, 1))

    total_frames = camera_poses.shape[0]
    if points.shape[0] < total_frames:
        total_frames = points.shape[0]
    if images.shape[0] < total_frames:
        total_frames = images.shape[0]

    if total_frames == 0:
        _toast_error("No frames available for Nerfstudio export.")
        return None, "No frames available for Nerfstudio export."

    folder_clean = _sanitize_subfolder_name(folder_name or "nerfstudio")
    ns_dir = os.path.join(save_dir, folder_clean)
    images_dir = os.path.join(ns_dir, "images")
    try:
        if os.path.isdir(ns_dir):
            shutil.rmtree(ns_dir)
        os.makedirs(images_dir, exist_ok=True)
    except Exception as exc:
        _toast_error("Failed to prepare Nerfstudio directories.")
        print("[ExportNerfstudio][Error] Directory prep failed:\n" + traceback.format_exc())
        return None, f"Failed to prepare Nerfstudio directories: {exc}"

    source_images_root = os.path.join(save_dir, "images")
    source_image_paths = []
    if os.path.isdir(source_images_root):
        source_image_paths = sorted([
            os.path.join(source_images_root, fn)
            for fn in os.listdir(source_images_root)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))
        ])

    image_sizes = []
    written_files = []
    for idx in range(total_frames):
        dest_name = f"{idx:06}.png"
        dest_path = os.path.join(images_dir, dest_name)
        written = False
        width = height = None

        if idx < len(source_image_paths):
            src = source_image_paths[idx]
            try:
                with Image.open(src) as img_pil:
                    img_rgb = img_pil.convert("RGB")
                    width, height = img_rgb.size
                    img_rgb.save(dest_path)
                    written = True
            except Exception:
                written = False

        if not written:
            frame_img = images[idx]
            if frame_img.ndim == 3 and frame_img.shape[-1] == 3:
                frame_uint8 = (np.clip(frame_img, 0, 1) * 255).astype(np.uint8)
            else:
                _toast_error("Invalid image array shape for Nerfstudio export.")
                return None, "Invalid image array shape for Nerfstudio export."
            try:
                pil_img = Image.fromarray(frame_uint8)
                width, height = pil_img.size
                pil_img.save(dest_path)
                written = True
            except Exception:
                written = False

        if not written:
            _toast_error(f"Failed to write frame {idx} image.")
            return None, f"Failed to write frame {idx} image."

        image_sizes.append((height, width))
        written_files.append(dest_name)

    points_subset = points[:total_frames]
    camera_subset = camera_poses[:total_frames]
    fx, fy, cx, cy = _estimate_shared_intrinsics(points_subset, camera_subset)

    height = image_sizes[0][0]
    width = image_sizes[0][1]

    frames = []
    denom = max(1, total_frames - 1)
    for idx in range(total_frames):
        transform = camera_subset[idx].tolist()
        frames.append({
            "file_path": f"images/{written_files[idx]}",
            "transform_matrix": transform,
            "time": float(idx / denom),
        })

    metadata = {
        "camera_model": "OPENCV",
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "w": int(width),
        "h": int(height),
        "frames": frames,
    }

    transforms_path = os.path.join(ns_dir, "transforms.json")
    try:
        with open(transforms_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        _toast_error("Failed to write transforms.json.")
        print("[ExportNerfstudio][Error] transforms.json write failed:\n" + traceback.format_exc())
        return None, "Failed to write transforms.json."

    message = f"Nerfstudio dataset exported: {ns_dir} ({total_frames} frames)."
    _toast_info("Nerfstudio export finished.")
    print(f"[ExportNerfstudio] Finished | dir={ns_dir}, frames={total_frames}")
    return None, message


def export_quads_unreal(save_dir: str,
                        filename: str,
                        conf_thres: float,
                        dedup_enable: bool,
                        dedup_radius: float,
                        normals_k: int,
                        normals_chunk: int,
                        quad_size: float):
    _toast_info("Exporting Quads for Unreal…")
    print(f"[ExportQuads] Start | save_dir={save_dir}, filename={filename}, conf%={conf_thres}, dedup={dedup_enable}, rad={dedup_radius}, k={normals_k}, chunk={normals_chunk}, size={quad_size}")
    if not save_dir or not os.path.isdir(save_dir):
        _toast_error("No valid target directory found.")
        print("[ExportQuads][Error] Invalid save_dir")
        return None, "No valid target directory found."
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        _toast_error("No predictions file found. Run Reconstruct first.")
        print(f"[ExportQuads][Error] Missing predictions at {pred_path}")
        return None, f"No predictions.npz at {pred_path}"

    try:
        loaded = np.load(pred_path)
        for key in ["points", "conf", "images"]:
            if key not in loaded:
                return None, f"Missing key '{key}' in predictions.npz"
        points = np.array(loaded['points'])
        conf = np.array(loaded['conf'])
        images = np.array(loaded['images'])
        cams = None
        if 'camera_poses' in loaded:
            try:
                cams_np = np.array(loaded['camera_poses'])
                cams = cams_np[..., :3, 3]
            except Exception:
                cams = None
    except Exception:
        _toast_error("Failed to load predictions.")
        print("[ExportQuads][Error] Load predictions failed:\n" + traceback.format_exc())
        return None, "Failed to load predictions."

    try:
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
        print(f"[ExportQuads] Filtered points: {P.shape[0]} (threshold={thr:.4f})")
        if dedup_enable and dedup_radius is not None and float(dedup_radius) > 0:
            keys = np.floor(P / float(dedup_radius)).astype(np.int64)
            keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
            _, uniq_idx = np.unique(keys_view, return_index=True)
            P = P[uniq_idx]
            RGB = RGB[uniq_idx]
            cam_pos = cam_pos[uniq_idx]
            print(f"[ExportQuads] Deduped points: {P.shape[0]} (radius={dedup_radius})")
        M = P.shape[0]
        if M == 0:
            _toast_warn("No points to export after filtering.")
            return None, "No points to export after filtering."

        # Compute oriented normals (kNN PCA; sign toward camera if available)
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        K = max(3, int(normals_k or 16))
        chunk = max(10000, int(normals_chunk or 50000))
        normals = np.zeros((M, 3), dtype=np.float32)
        print(f"[ExportQuads] Computing normals | M={M}, k={K}, chunk={chunk}")
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

        try:
            size = float(quad_size or 0.001)
        except Exception:
            size = 0.001
        half = float(size) * 0.5

        # Build tangent/bitangent per point (vectorized)
        up = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), M, axis=0)
        z_abs = np.abs(normals[:, 2])
        up[z_abs > 0.9] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tangent = np.cross(up, normals)
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(tangent_norm, 1e-8)
        bitangent = np.cross(normals, tangent)
        bitangent_norm = np.linalg.norm(bitangent, axis=1, keepdims=True)
        bitangent = bitangent / np.maximum(bitangent_norm, 1e-8)

        # Offsets for the 4 quad corners in plane (CCW winding for +normal)
        o0 = (-half * tangent) + (-half * bitangent)
        o1 = ( half * tangent) + (-half * bitangent)
        o2 = ( half * tangent) + ( half * bitangent)
        o3 = (-half * tangent) + ( half * bitangent)
        V = np.stack([P + o0, P + o1, P + o2, P + o3], axis=1).reshape(-1, 3)

        # Duplicate normals and colors for each corner
        VN = np.repeat(normals, repeats=4, axis=0)
        # Encode point index in vertex color alpha (normalized 0..1 mapped to 0..255)
        idx_norm = (np.arange(M, dtype=np.float64) / max(1, M - 1))
        A = np.clip(np.round(idx_norm * 255.0), 0, 255).astype(np.uint8)
        # Convert sRGB -> linear for Unreal-friendly vertex colors
        try:
            RGB_lin = (_srgb_to_linear((RGB.astype(np.float32) / 255.0)) * 255.0).astype(np.uint8)
        except Exception:
            RGB_lin = RGB
        RGBA = np.zeros((M, 4), dtype=np.uint8)
        RGBA[:, :3] = RGB_lin
        RGBA[:, 3] = A
        VC = np.repeat(RGBA, repeats=4, axis=0)

        # Faces (two triangles per quad)
        base = (np.arange(M, dtype=np.int64) * 4)
        F = np.empty((2 * M, 3), dtype=np.int64)
        F[0::2] = np.stack([base + 0, base + 1, base + 2], axis=1)
        F[1::2] = np.stack([base + 0, base + 2, base + 3], axis=1)

        # UVs for circle mask: (0,0),(1,0),(1,1),(0,1)
        uv_quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        UV = np.repeat(uv_quad[None, :, :], repeats=M, axis=0).reshape(-1, 2)

        # Build mesh and export GLB or PLY depending on extension
        try:
            tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
            # Provide vertex normals explicitly for stable shading (optional for masked unlit)
            try:
                tm.vertex_normals = VN
            except Exception:
                pass
            from trimesh.visual.texture import SimpleMaterial
            tm.visual = trimesh.visual.texture.TextureVisuals(uv=UV, material=SimpleMaterial())
            # For PLY viewers, use sRGB colors; for GLB, viewer applies sRGB automatically.
            tm.visual.vertex_colors = VC
            out_path = os.path.join(save_dir, filename if filename else f'export_quads_{size:.6f}.ply')
            ext = os.path.splitext(out_path)[1].lower()
            if ext in ['.glb', '.gltf']:
                scene = trimesh.Scene(tm)
                scene.export(out_path)
            elif ext == '.ply':
                # Ensure PLY has per-vertex colors (uint8 RGBA). Drop UVs since most PLY viewers ignore them.
                try:
                    tm.visual = trimesh.visual.ColorVisuals(vertex_colors=VC)
                except Exception:
                    pass
                tm.export(out_path)
            else:
                out_path = os.path.splitext(out_path)[0] + '.ply'
                try:
                    tm.visual = trimesh.visual.ColorVisuals(vertex_colors=VC)
                except Exception:
                    pass
                tm.export(out_path)
        except Exception:
            _toast_error("Mesh export failed.")
            print("[ExportQuads][Error] Export failed:\n" + traceback.format_exc())
            return None, "Failed to export mesh for quads."
    except Exception:
        _toast_error("Quad export failed.")
        print("[ExportQuads][Error] Failed:\n" + traceback.format_exc())
        return None, "Quad export failed."

    _toast_info("Quad export finished.")
    print(f"[ExportQuads] Finished | path={out_path}, quads={M}, verts={V.shape[0]}, faces={F.shape[0]}")
    return None, f"Quads exported: {out_path} (points={M}, size={size} m). VertexColor=RGBA, A=index_norm; UV0=(0..1) for circle mask."


def reconstruct_mesh_poisson(save_dir: str,
                             conf_thres: float,
                             dedup_enable: bool,
                             dedup_radius: float,
                             depth: int,
                             scale: float,
                             trim_q: float,
                             smooth_iters: int,
                             color_k: int,
                             color_from_full: bool,
                             proj_color: bool,
                             proj_k: int,
                             proj_angle_pow: float):
    _toast_info("Reconstructing mesh…")
    print(f"[Mesh] Start | save_dir={save_dir}, conf%={conf_thres}, dedup={dedup_enable}, rad={dedup_radius}, depth={depth}, scale={scale}, trim_q={trim_q}, smooth={smooth_iters}, color_k={color_k}")
    if not save_dir or not os.path.isdir(save_dir):
        _toast_error("No working directory. Upload and reconstruct first.")
        print("[Mesh][Error] Invalid save_dir")
        return None, "No valid target directory found. Please upload and reconstruct first."
    if not _O3D_AVAILABLE:
        _toast_error("Open3D not installed.")
        print("[Mesh][Error] Open3D missing")
        return None, "Open3D is not installed. Please install 'open3d' to use Mesh Reconstruction."
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        _toast_error("No predictions found. Run Reconstruct first.")
        print(f"[Mesh][Error] Missing predictions at {pred_path}")
        return None, f"No predictions.npz at {pred_path}. Please run Reconstruct first."

    try:
        loaded = np.load(pred_path)
        for key in ["points", "conf", "images"]:
            if key not in loaded:
                return None, f"Missing key '{key}' in predictions.npz"
        points = np.array(loaded['points'])
        conf = np.array(loaded['conf'])
        images = np.array(loaded['images'])
        cam_center = None
        cam_centers_all = None
        if 'camera_poses' in loaded:
            try:
                cams_np = np.array(loaded['camera_poses'])
                cam_centers_all = cams_np[..., :3, 3]
                cam_center = np.mean(cam_centers_all, axis=0)
            except Exception:
                cam_center = None
    except Exception as e:
        _toast_error("Failed to load predictions.")
        print("[Mesh][Error] Load predictions failed:\n" + traceback.format_exc())
        return None, f"Failed to load predictions: {e}"

    N, H, W, _ = points.shape
    P = points.reshape(-1, 3)
    C = conf.reshape(-1)
    RGB = (np.clip(images, 0, 1) * 255).astype(np.float32).reshape(-1, 3) / 255.0
    frames_flat = np.repeat(np.arange(N), H * W)

    thr = (conf_thres or 0.0) / 100.0
    mask = (C >= thr) & np.isfinite(P).all(axis=1) & (C > 1e-6)
    P = P[mask]
    RGB = RGB[mask]
    frames_full = frames_flat[mask]
    # Preserve a high-detail color source before any geometry deduplication
    P_color_full = P.copy()
    RGB_color_full = RGB.copy()
    print(f"[Mesh] Filtered points: {P.shape[0]} (threshold={thr:.4f})")
    if dedup_enable and dedup_radius is not None and float(dedup_radius) > 0:
        try:
            keys = np.floor(P / float(dedup_radius)).astype(np.int64)
            keys_view = keys.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
            _, uniq_idx = np.unique(keys_view, return_index=True)
            uniq_idx.sort()
            P = P[uniq_idx]
            RGB = RGB[uniq_idx]
            print(f"[Mesh] Deduped points: {P.shape[0]} (radius={dedup_radius})")
        except Exception as e:
            print(f"Mesh dedup failed, proceeding without dedup: {e}")

    if P.shape[0] < 1000:
        _toast_warn("Too few points for Poisson. Try lowering threshold or adding frames.")
        return None, f"Too few points after filtering ({P.shape[0]}). Increase frames or lower threshold."

    # Build Open3D point cloud for geometry (deduped)
    print("[Mesh] Building Open3D point cloud…")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(RGB, 0, 1).astype(np.float64))
    try:
        t_norm0 = time.time()
        print("[Mesh] Estimating normals (k=30)… this may take minutes")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        if cam_center is None:
            cam_center = np.mean(P, axis=0)
        print(f"[Mesh] Orienting normals towards camera center {cam_center.tolist()}")
        pcd.orient_normals_towards_camera_location(camera_location=cam_center)
        print(f"[Mesh] Normals ready | took {(time.time()-t_norm0):.2f}s")
    except Exception as e:
        print("[Mesh][Error] Normal estimation/orientation failed:\n" + traceback.format_exc())

    # Poisson reconstruction
    try:
        print(f"[Mesh] Running Poisson (depth={int(depth)}, scale={float(scale)})… this may take minutes")
        t_poi0 = time.time()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(depth), scale=float(scale), linear_fit=True
        )
        print(f"[Mesh] Poisson done | vertices={len(mesh.vertices)}, faces={len(mesh.triangles)}, took {(time.time()-t_poi0):.2f}s")
    except Exception as e:
        _toast_error("Poisson reconstruction failed.")
        return None, f"Poisson reconstruction failed: {e}"

    densities = np.asarray(densities)
    if densities.size == 0 or len(mesh.vertices) == 0:
        return None, "Poisson reconstruction produced an empty mesh."

    trim_q = float(np.clip(trim_q if trim_q is not None else 0.15, 0.0, 0.9))
    if trim_q > 0:
        thr_d = np.quantile(densities, trim_q)
        keep_mask = densities >= thr_d
        try:
            pre_v = len(mesh.vertices)
            res = mesh.remove_vertices_by_mask(~keep_mask)
            # Open3D versions differ: some return a new mesh, some modify in-place and return None
            mesh = res if (res is not None) else mesh
            print(f"[Mesh] Trim by density | quantile={trim_q:.2f}, thr={thr_d:.6f}, vertices {pre_v} -> {len(mesh.vertices)}")
        except Exception as e:
            print("[Mesh][Warn] Density trim failed:\n" + traceback.format_exc())

    # Color transfer (projection / kNN)
    try:
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("Empty mesh after trimming")
        print("[Mesh] Transferring vertex colors…")
        t_col0 = time.time()
        verts_np = np.asarray(mesh.vertices)
        if not mesh.has_vertex_normals() or len(np.asarray(mesh.vertex_normals)) != verts_np.shape[0]:
            mesh.compute_vertex_normals()
        vnormals = np.asarray(mesh.vertex_normals)
        if bool(proj_color) and vnormals is not None and cam_centers_all is not None and frames_full is not None:
            # Projection-based: weight by distance and angle to viewing direction
            from scipy.spatial import cKDTree as _CKD
            kk = max(1, int(proj_k or 4))
            apow = float(proj_angle_pow or 2.0)
            # Use full-resolution color points for projection fidelity
            tree = _CKD(P_color_full)
            out_cols = []
            batch = 100000
            for s in range(0, verts_np.shape[0], batch):
                e = min(verts_np.shape[0], s + batch)
                vbatch = verts_np[s:e]
                d, idx = tree.query(vbatch, k=kk)
                if kk == 1:
                    d = d[:, None]
                    idx = idx[:, None]
                w_dist = 1.0 / np.maximum(d.astype(np.float64), 1e-6)
                cam_idx = frames_full[idx]
                cam_pos = cam_centers_all[cam_idx]
                view_dir = vbatch[:, None, :] - cam_pos
                view_dir /= np.maximum(np.linalg.norm(view_dir, axis=2, keepdims=True), 1e-6)
                nrm = vnormals[s:e][:, None, :]
                cosang = np.clip(np.sum(nrm * view_dir, axis=2), 0.0, 1.0)
                # Add small epsilon to avoid zeros wiping colors
                w_angle = np.power(cosang + 1e-3, apow)
                w = w_dist * w_angle
                w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
                cols = np.sum(RGB_color_full[idx] * w[..., None], axis=1)
                out_cols.append(cols)
                if (e % 200000) == 0:
                    print(f"[Mesh] Projection-colored {e}/{verts_np.shape[0]} vertices…")
            mesh_colors = np.vstack(out_cols)
            src_tag = f'projection(k={kk},pow={apow})'
        elif color_from_full:
            # kNN from full-resolution colors
            from scipy.spatial import cKDTree as _CKD
            k = max(1, int(color_k or 3))
            color_tree = _CKD(P_color_full)
            mesh_colors = []
            batch = 200000
            for s in range(0, verts_np.shape[0], batch):
                e = min(verts_np.shape[0], s + batch)
                d, idx = color_tree.query(verts_np[s:e], k=k)
                if k == 1:
                    cols = RGB_color_full[idx]
                else:
                    d = np.atleast_2d(d)
                    idx = np.atleast_2d(idx)
                    w = 1.0 / np.maximum(d.astype(np.float64), 1e-12)
                    w = w / np.sum(w, axis=1, keepdims=True)
                    cols = np.sum(RGB_color_full[idx] * w[..., None], axis=1)
                mesh_colors.append(cols)
                if (e % 200000) == 0:
                    print(f"[Mesh] Colorized {e}/{verts_np.shape[0]} vertices…")
            mesh_colors = np.vstack(mesh_colors)
            src_tag = f'full(k={k})'
        else:
            # kNN from deduped geometry cloud
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            k = max(1, int(color_k or 3))
            cols_np = np.asarray(pcd.colors)
            mesh_colors = []
            for idx_v, v in enumerate(verts_np):
                try:
                    _, idx, dist2 = kdtree.search_knn_vector_3d(v, k)
                    if len(idx) == 0:
                        mesh_colors.append([0.7, 0.7, 0.7])
                        continue
                    d = np.sqrt(np.maximum(np.asarray(dist2, dtype=np.float64), 1e-12))
                    if k == 1:
                        c = cols_np[np.asarray(idx)][0]
                    else:
                        w = 1.0 / d
                        w = w / np.sum(w)
                        c = np.sum(cols_np[np.asarray(idx)] * w[:, None], axis=0)
                    mesh_colors.append(np.clip(c, 0, 1))
                except Exception:
                    mesh_colors.append([0.7, 0.7, 0.7])
                if (idx_v + 1) % 200000 == 0:
                    print(f"[Mesh] Colorized {idx_v+1}/{verts_np.shape[0]} vertices…")
            mesh_colors = np.asarray(mesh_colors)
            src_tag = f'dedup(k={k})'
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh_colors))
        print(f"[Mesh] Color transfer done | source={src_tag}, took {(time.time()-t_col0):.2f}s")
    except Exception as e:
        print("[Mesh][Error] Color transfer failed:\n" + traceback.format_exc())

    # Optional smoothing and cleanup
    try:
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("Empty mesh before post-process")
        # Ensure vertex normals and orient them roughly toward global camera center
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        try:
            if cam_center is not None:
                Vp = np.asarray(mesh.vertices)
                Vn = np.asarray(mesh.vertex_normals)
                vec = Vp - cam_center[None, :]
                dot = np.sum(Vn * vec, axis=1, keepdims=True)
                flip = (dot > 0).astype(np.float64)
                Vn = Vn * (1.0 - 2.0 * flip)
                mesh.vertex_normals = o3d.utility.Vector3dVector(Vn)
        except Exception:
            pass
        if smooth_iters is not None and int(smooth_iters) > 0:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        print(f"[Mesh] Post-process complete | vertices={len(mesh.vertices)}, faces={len(mesh.triangles)}")
    except Exception as e:
        print("[Mesh][Warn] Post-process issue:\n" + traceback.format_exc())

    # Export PLY and GLB
    tag = f"d{int(depth)}_t{trim_q:.2f}".replace('.', 'p')
    ply_path = os.path.join(save_dir, f"mesh_poisson_{tag}.ply")
    glb_path = os.path.join(save_dir, f"mesh_poisson_{tag}.glb")
    try:
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("Empty mesh at write stage")
        # Convert vertex colors to linear (sRGB -> linear) for export
        try:
            if mesh.has_vertex_colors():
                VC = np.asarray(mesh.vertex_colors)
                if VC.size > 0:
                    VC_lin = _srgb_to_linear(np.clip(VC, 0.0, 1.0))
                    mesh.vertex_colors = o3d.utility.Vector3dVector(VC_lin.astype(np.float64))
        except Exception:
            pass
        o3d.io.write_triangle_mesh(ply_path, mesh)
        print(f"[Mesh] Wrote PLY | {ply_path}")
    except Exception as e:
        print("[Mesh][Error] PLY write failed:\n" + traceback.format_exc())

    try:
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("Empty mesh at GLB export stage")
        # Convert to trimesh for GLB export and apply same alignment as point cloud viz
        V = np.asarray(mesh.vertices)
        F = np.asarray(mesh.triangles)
        VC = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
        if VC is not None and VC.size == V.shape[0] * 3:
            tm.visual.vertex_colors = (np.clip(VC, 0, 1) * 255).astype(np.uint8)
        align_rotation = np.eye(4)
        align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()
        align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
        tm.apply_transform(align_rotation)
        tm.export(file_obj=glb_path)
        print(f"[Mesh] Wrote GLB | {glb_path}")
    except Exception as e:
        _toast_warn("GLB export failed. PLY saved.")
        print("[Mesh][Error] GLB export failed:\n" + traceback.format_exc())
        return None, f"GLB export failed: {e}. PLY saved at {ply_path}"

    info = f"Mesh reconstructed: vertices={len(mesh.vertices)}, faces={len(mesh.triangles)} | depth={depth}, trim_q={trim_q}, smooth={smooth_iters}. Saved: {os.path.basename(ply_path)}, {os.path.basename(glb_path)}"
    _toast_info("Mesh reconstruction finished.")
    print("[Mesh] Finished")
    return glb_path, info


def bake_texture_single_view(save_dir: str,
                             frame_idx: int,
                             fov_deg: float,
                             tex_size: int):
    _toast_info("Baking texture…")
    try:
        frame_idx = int(frame_idx)
        tex_size = int(tex_size)
        fov_deg = float(fov_deg)
    except Exception:
        pass
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not save_dir or not os.path.isdir(save_dir) or not os.path.exists(pred_path):
        _toast_error("No reconstruction found to bake.")
        return None, "No reconstruction found."

    # Load predictions for per-frame points and images
    data = np.load(pred_path)
    if 'points' not in data or 'images' not in data:
        _toast_error("Missing points/images for baking.")
        return None, "Missing points/images for baking."
    points = np.array(data['points'])   # (N,H,W,3)
    images = np.array(data['images'])   # NHWC or NCHW
    N = points.shape[0]
    if frame_idx < 0 or frame_idx >= N:
        _toast_error("Frame index out of range.")
        return None, f"Frame index out of range (0..{N-1})."

    # Prefer original input image file for maximum sharpness
    img_path = None
    try:
        images_root = os.path.join(save_dir, 'images')
        files = sorted([os.path.join(images_root, f) for f in os.listdir(images_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if 0 <= frame_idx < len(files):
            img_path = files[frame_idx]
    except Exception:
        img_path = None
    if img_path and os.path.exists(img_path):
        img = np.array(Image.open(img_path).convert('RGB')) / 255.0
    else:
        # fallback to cached image tensor
        img = images[frame_idx]
        if img.ndim == 3 and img.shape[0] == 3:  # NCHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
    H, W = img.shape[0], img.shape[1]

    # Load mesh (latest Poisson GLB)
    # Find most recent mesh_poisson_*.glb
    cand = [fn for fn in os.listdir(save_dir) if fn.startswith('mesh_poisson_') and fn.endswith('.glb')]
    if not cand:
        _toast_error("No mesh to bake.")
        return None, "No mesh to bake. Please reconstruct a mesh first."
    cand.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)
    glb_path = os.path.join(save_dir, cand[0])
    try:
        tm = trimesh.load(glb_path, force='mesh')
    except Exception as e:
        _toast_error("Failed to load mesh.")
        return None, f"Failed to load mesh: {e}"

    V = tm.vertices.view(np.ndarray)
    F = tm.faces.view(np.ndarray)
    # Compute UVs by nearest-neighbor lookup from 3D frame points -> pixels (intrinsic-free)
    Pf = points[frame_idx]  # (Hm,Wm,3) world coords at model resolution
    Hm, Wm = Pf.shape[0], Pf.shape[1]
    # Align frame points with the same rotation used for mesh export
    align_R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    Pf_flat = Pf.reshape(-1, 3)
    Pf_aligned = (align_R @ Pf_flat.T).T  # (Hm*Wm,3)
    valid = np.isfinite(Pf_aligned).all(axis=1)
    Pf_aligned_valid = Pf_aligned[valid]
    if Pf_aligned_valid.shape[0] == 0:
        _toast_error("No valid frame points for baking.")
        return None, "No valid frame points for baking."
    # Build UV grid for valid points based on frame grid scaled to the chosen image size
    ii, jj = np.meshgrid(np.arange(Hm), np.arange(Wm), indexing='ij')
    jj = jj.reshape(-1)
    ii = ii.reshape(-1)
    scale_x = (W - 1) / max(1, Wm - 1)
    scale_y = (H - 1) / max(1, Hm - 1)
    u_pix = jj * scale_x
    v_pix = ii * scale_y
    uv_grid = np.stack([u_pix, v_pix], axis=1)[valid]
    # Normalize to [0,1]
    uv_norm = np.zeros_like(uv_grid, dtype=np.float64)
    uv_norm[:, 0] = np.clip(uv_grid[:, 0] / max(1, W - 1), 0.0, 1.0)
    uv_norm[:, 1] = 1.0 - np.clip(uv_grid[:, 1] / max(1, H - 1), 0.0, 1.0)
    # KD-Tree from mesh vertices to frame points
    from scipy.spatial import cKDTree
    tree = cKDTree(Pf_aligned_valid)
    d, idx = tree.query(V, k=3)
    if np.ndim(d) == 1:
        d = d[:, None]; idx = idx[:, None]
    w = 1.0 / np.maximum(d.astype(np.float64), 1e-6)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    UV = np.sum(uv_norm[idx] * w[..., None], axis=1)
    UV = np.clip(UV, 0.0, 1.0)

    # Create a simple texture image (resampled original)
    tex_img = (img * 255).astype(np.uint8)
    tex_pil = Image.fromarray(tex_img)
    tex_pil = tex_pil.resize((tex_size, tex_size), Image.BILINEAR)
    tex_rgba = np.array(tex_pil.convert('RGBA'))

    # Attach UVs to mesh and write GLB with embedded texture using trimesh
    try:
        # Drop any existing vertex colors by constructing a fresh mesh
        tm2 = trimesh.Trimesh(vertices=V.copy(), faces=F.copy(), process=False)
        # Build a PBR material with embedded baseColorTexture
        from trimesh.visual.texture import SimpleMaterial
        material = SimpleMaterial(image=tex_pil)
        tm2.visual = trimesh.visual.texture.TextureVisuals(uv=UV, material=material)
        baked_glb = os.path.join(save_dir, f"mesh_baked_{frame_idx}_fov{int(fov_deg)}_{tex_size}px.glb")
        # Export as a scene to ensure material + texture embed in GLB
        scene = trimesh.Scene(tm2)
        scene.export(baked_glb)
    except Exception as e:
        _toast_error("Texture bake export failed.")
        return None, f"Bake export failed: {e}"

    _toast_info("Texture baking finished.")
    return baked_glb, f"Texture baked from frame {frame_idx} (FOV={fov_deg}°, {tex_size}px)."


def bake_texture_multiview(save_dir: str,
                           ref_frame_idx: int,
                           tex_size: int,
                           knn_vertices: int,
                           angle_pow: float):
    _toast_info("Baking multi-view texture…")
    try:
        ref_frame_idx = int(ref_frame_idx)
        tex_size = int(tex_size)
        knn_vertices = int(knn_vertices)
        angle_pow = float(angle_pow)
    except Exception:
        pass
    pred_path = os.path.join(save_dir, 'predictions.npz')
    if not save_dir or not os.path.isdir(save_dir) or not os.path.exists(pred_path):
        _toast_error("No reconstruction found to bake.")
        return None, "No reconstruction found."

    data = np.load(pred_path)
    if 'points' not in data or 'images' not in data or 'camera_poses' not in data:
        _toast_error("Missing points/images/camera poses for multiview baking.")
        return None, "Missing points/images/camera poses for multiview baking."
    points = np.array(data['points'])    # (N,H,W,3)
    images = np.array(data['images'])    # (N,H,W,3) or NCHW
    cams = np.array(data['camera_poses'])  # (N,4,4) C2W
    N = points.shape[0]
    if N == 0:
        return None, "No frames to bake."
    if ref_frame_idx < 0 or ref_frame_idx >= N:
        ref_frame_idx = 0

    # Load mesh (latest Poisson GLB)
    cand = [fn for fn in os.listdir(save_dir) if fn.startswith('mesh_poisson_') and fn.endswith('.glb')]
    if not cand:
        _toast_error("No mesh to bake.")
        return None, "No mesh to bake. Please reconstruct a mesh first."
    cand.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)
    glb_path = os.path.join(save_dir, cand[0])
    tm = trimesh.load(glb_path, force='mesh')
    V = tm.vertices.view(np.ndarray)
    F = tm.faces.view(np.ndarray)
    print(f"[MV-Bake] Start | frames={N}, verts={V.shape[0]}, tex={tex_size}px, ref={ref_frame_idx}, knn={knn_vertices}, angle_pow={angle_pow}")

    # Build UVs from reference frame via nearest-neighbor mapping (intrinsic-free)
    Pref = points[ref_frame_idx]
    Hm, Wm = Pref.shape[0], Pref.shape[1]
    align_R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    Pref_flat = Pref.reshape(-1, 3)
    Pref_aligned = (align_R @ Pref_flat.T).T
    valid = np.isfinite(Pref_aligned).all(axis=1)
    Pref_valid = Pref_aligned[valid]
    if Pref_valid.shape[0] == 0:
        return None, "Reference frame has no valid points for UVs."
    print(f"[MV-Bake] Ref valid points: {Pref_valid.shape[0]} / {Hm*Wm}")
    ii, jj = np.meshgrid(np.arange(Hm), np.arange(Wm), indexing='ij')
    jj = jj.reshape(-1)[valid]
    ii = ii.reshape(-1)[valid]
    # UV normalized
    UV_ref = np.stack([jj / max(1, Wm - 1), 1.0 - (ii / max(1, Hm - 1))], axis=1).astype(np.float64)
    from scipy.spatial import cKDTree
    tree_ref = cKDTree(Pref_valid)
    dV, idxV = tree_ref.query(V, k=max(1, int(knn_vertices)))
    if np.ndim(dV) == 1:
        dV = dV[:, None]; idxV = idxV[:, None]
    wV = 1.0 / np.maximum(dV.astype(np.float64), 1e-6)
    wV = wV / np.maximum(wV.sum(axis=1, keepdims=True), 1e-6)
    UV = np.sum(UV_ref[idxV] * wV[..., None], axis=1)
    UV = np.clip(UV, 0.0, 1.0)
    try:
        print(f"[MV-Bake] UV range: u[{UV[:,0].min():.3f},{UV[:,0].max():.3f}] v[{UV[:,1].min():.3f},{UV[:,1].max():.3f}]")
    except Exception:
        pass

    # Create empty texture and accumulate colors from all frames with visibility and angle weighting
    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    acc = np.zeros((tex_size, tex_size, 1), dtype=np.float64)
    # Precompute per-vertex normals for angle weights
    if not tm.visual or not tm.vertex_normals.any():
        tm.rezero()
        tm.vertex_normals
    vnorm = tm.vertex_normals.view(np.ndarray)
    vnorm = vnorm / np.maximum(np.linalg.norm(vnorm, axis=1, keepdims=True), 1e-8)

    # For each frame, accumulate colors using 3D kNN from frame point grid (intrinsic-free)
    acc_total_before = float(acc.sum())
    for i in range(N):
        # Load frame image
        img = images[i]
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img_lin = _srgb_to_linear(img)
        Hi, Wi = img.shape[0], img.shape[1]
        print(f"[MV-Bake][{i}] image={Wi}x{Hi}")
        # Align camera pose to mesh space for angle weights
        T_c2w = cams[i]
        T_align = np.eye(4); T_align[:3, :3] = align_R
        T_c2w_al = T_align @ T_c2w
        cam_pos = T_c2w_al[:3, 3]
        view_dir = V - cam_pos[None, :]
        view_dir = view_dir / np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-8)
        cosang = np.clip(np.sum(vnorm * view_dir, axis=1), 0.0, 1.0)
        w_ang = np.power(cosang + 1e-3, float(angle_pow))  # (Nv,)
        try:
            print(f"[MV-Bake][{i}] angle weight: min={w_ang.min():.3e}, max={w_ang.max():.3e}, mean={w_ang.mean():.3e}")
        except Exception:
            pass

        # Build KD from this frame's aligned points and map to image pixels
        Pf = points[i]
        Hm, Wm = Pf.shape[0], Pf.shape[1]
        Pf_flat = Pf.reshape(-1, 3)
        Pf_al = (align_R @ Pf_flat.T).T
        valid = np.isfinite(Pf_al).all(axis=1)
        if not np.any(valid):
            continue
        Pf_val = Pf_al[valid]
        ii, jj = np.meshgrid(np.arange(Hm), np.arange(Wm), indexing='ij')
        jj = jj.reshape(-1)[valid]
        ii = ii.reshape(-1)[valid]
        print(f"[MV-Bake][{i}] valid frame points: {Pf_val.shape[0]} / {Hm*Wm}")
        sx = (Wi - 1) / max(1, Wm - 1)
        sy = (Hi - 1) / max(1, Hm - 1)
        u_pix = jj * sx
        v_pix = ii * sy
        from scipy.spatial import cKDTree
        tree_i = cKDTree(Pf_val)
        kmap = max(1, int(knn_vertices))
        d, idx = tree_i.query(V, k=kmap)
        if np.ndim(d) == 1:
            d = d[:, None]; idx = idx[:, None]
        w_dist = 1.0 / np.maximum(d.astype(np.float64), 1e-6)
        w_dist = w_dist / np.maximum(w_dist.sum(axis=1, keepdims=True), 1e-6)  # (Nv,k)
        try:
            print(f"[MV-Bake][{i}] dist weight: min={w_dist.min():.3e}, max={w_dist.max():.3e}, mean={w_dist.mean():.3e}")
        except Exception:
            pass
        u_sel = u_pix[idx]
        v_sel = v_pix[idx]
        u_nn = np.clip(np.round(u_sel).astype(int), 0, Wi - 1)
        v_nn = np.clip(np.round(v_sel).astype(int), 0, Hi - 1)
        cols_nn = img_lin[v_nn, u_nn]  # (Nv,k,3)
        cols_vert = np.sum(cols_nn * w_dist[..., None], axis=1)  # (Nv,3)
        try:
            cm = cols_vert.mean(axis=0)
            print(f"[MV-Bake][{i}] color mean (lin): {cm[0]:.3f},{cm[1]:.3f},{cm[2]:.3f}")
        except Exception:
            pass
        Ui = np.clip((UV[:, 0] * (tex_size - 1)).round().astype(int), 0, tex_size - 1)
        Vi = np.clip((UV[:, 1] * (tex_size - 1)).round().astype(int), 0, tex_size - 1)
        w_total = w_ang[:, None]
        acc_before = float(acc.sum())
        np.add.at(tex, (Vi, Ui), cols_vert * w_total)
        np.add.at(acc, (Vi, Ui, 0), w_total.squeeze(1))
        acc_after = float(acc.sum())
        print(f"[MV-Bake][{i}] contributed acc += {acc_after - acc_before:.2f}")

    # Normalize and export
    acc_safe = np.maximum(acc, 1e-6)
    try:
        nz = int((acc[...,0] > 1e-6).sum())
        print(f"[MV-Bake] acc stats: sum={float(acc.sum()):.2f}, nonzero_pixels={nz}")
    except Exception:
        pass
    if float(acc.sum()) <= 1e-6:
        _toast_error("No contributions accumulated. Check camera poses and frame points.")
        return None, "Multi-view bake produced no contributions."
    tex_lin = np.clip(tex / acc_safe, 0.0, 1.0)
    tex_srgb = _linear_to_srgb(tex_lin)
    tex_rgb8 = (np.clip(tex_srgb, 0.0, 1.0) * 255).astype(np.uint8)
    # Inpaint holes where there was no contribution
    try:
        import cv2 as _cv2
        hole_mask = (acc[..., 0] <= 1e-6).astype(np.uint8) * 255
        nonzero = int((hole_mask == 0).sum())
        print(f"[MV-Bake] inpaint: nonzero tex pixels before={nonzero}")
        tex_bgr = _cv2.cvtColor(tex_rgb8, _cv2.COLOR_RGB2BGR)
        tex_bgr = _cv2.inpaint(tex_bgr, hole_mask, inpaintRadius=3, flags=_cv2.INPAINT_TELEA)
        tex_rgb8 = _cv2.cvtColor(tex_bgr, _cv2.COLOR_BGR2RGB)
    except Exception as _e:
        print(f"[MV-Bake] inpaint skipped: {_e}")
    alpha8 = np.full((tex_size, tex_size, 1), 255, dtype=np.uint8)
    tex_rgba = np.concatenate([tex_rgb8, alpha8], axis=2)
    # Build textured mesh with existing UVs and baked texture
    tm2 = trimesh.Trimesh(vertices=V.copy(), faces=F.copy(), process=False)
    from trimesh.visual.texture import SimpleMaterial
    material = SimpleMaterial(image=Image.fromarray(tex_rgba, mode='RGBA'))
    tm2.visual = trimesh.visual.texture.TextureVisuals(uv=UV, material=material)
    # Ensure no competing vertex colors
    try:
        if hasattr(tm2.visual, 'vertex_colors'):
            tm2.visual.vertex_colors = None
    except Exception:
        pass
    baked_glb = os.path.join(save_dir, f"mesh_baked_mv_ref{ref_frame_idx}_{tex_size}px.glb")
    # Export as GLTF binary to embed texture
    trimesh.exchange.gltf.export_glb(trimesh.Scene(tm2), baked_glb)
    _toast_info("Multi-view texture baking finished.")
    return baked_glb, f"Multi-view texture baked (ref={ref_frame_idx}, {tex_size}px)."

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


def seed_then_gradio_demo(target_dir,
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
                          zero_depth_edges,
                          keep_models_loaded,
                          seed_value):
    set_global_seed(seed_value or 42)
    return gradio_demo(
        target_dir,
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
        zero_depth_edges,
        keep_models_loaded,
    )

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
                    with gr.Row():
                        append_video = gr.Video(label="Append Video (does not reset images)", interactive=True)
                        append_btn = gr.Button("Append to Images", variant="secondary", scale=1)

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
                reconstruction_output = gr.Model3D(height=1440, zoom_speed=0.5, pan_speed=0.5, label="3D Output")

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
                    with gr.Row():
                        seed_box = gr.Textbox(value='42', label='Seed (int)', info='Applied at start of Reconstruct')
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
                    with gr.Row():
                        quad_filename = gr.Textbox(value='export_quads.ply', label='Quads Filename (GLB/PLY)')
                        quad_size = gr.Number(0.001, label='Quad Size (m)')
                    with gr.Row():
                        nerfstudio_folder = gr.Textbox(value='nerfstudio', label='Nerfstudio Folder Name')
                        export_nerfstudio_btn = gr.Button("Export Nerfstudio Dataset", variant="secondary")
                    export_btn = gr.Button("Export PLY", variant="secondary")
                    export_quads_btn = gr.Button("Export Quads (Unreal/Nanite)", variant="secondary")

                with gr.Group():
                    gr.Markdown("### 8. Mesh Reconstruction (Poisson)")
                    with gr.Row():
                        mesh_depth = gr.Number(10, label='Poisson Depth (8–11 typical)')
                        mesh_scale = gr.Number(1.2, label='Poisson Scale (1.0–1.5)')
                        mesh_trim_q = gr.Number(0.15, label='Density Trim Quantile (0–0.9)')
                    with gr.Row():
                        mesh_smooth = gr.Number(5, label='Smoothing Iters (Taubin)')
                        mesh_color_k = gr.Number(3, label='kNN for Color Transfer (1–5)')
                        mesh_color_from_full = gr.Checkbox(value=True, label='Use Full-Resolution Color Source')
                    with gr.Row():
                        mesh_proj_color = gr.Checkbox(value=False, label='Use Projection-based Texturing')
                        mesh_proj_k = gr.Number(4, label='Projection k (1–8)')
                        mesh_proj_angle_pow = gr.Number(2.0, label='Angle Weight Power (>=0)')
                    mesh_btn = gr.Button("Reconstruct Mesh (Poisson)", variant="secondary")

                with gr.Group():
                    gr.Markdown("### 9. Texture Baking (Single View UV)")
                    with gr.Row():
                        bake_frame_idx = gr.Number(value=0, label='Frame Index (0-based)')
                        bake_fov = gr.Number(value=60.0, label='Assumed FOV (deg)')
                        bake_tex_size = gr.Number(value=1024, label='Texture Size (px)')
                    bake_btn = gr.Button("Bake Texture (UV from Frame)", variant="secondary")

                with gr.Group():
                    gr.Markdown("### 10. Texture Baking (Multi-view)")
                    with gr.Row():
                        mv_ref_frame = gr.Number(value=0, label='UV Reference Frame (0-based)')
                        mv_tex_size = gr.Number(value=2048, label='Texture Size (px)')
                        mv_knn = gr.Number(value=5, label='kNN (per-vertex)')
                        mv_angle_pow = gr.Number(value=2.0, label='Angle Weight Power')
                    mv_bake_btn = gr.Button("Bake Texture (Multi-view)", variant="secondary")

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

        # Append video handler (does not clear existing images)
        append_btn.click(
            fn=append_video_to_images,
            inputs=[target_dir_output, append_video, fps],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

        # Reconstruct button
        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=lambda: "Loading and Reconstructing...", inputs=[], outputs=[log_output]
        ).then(
            fn=seed_then_gradio_demo,
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
                seed_box,
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

        # Export Quads for Unreal/Nanite
        export_quads_btn.click(
            export_quads_unreal,
            inputs=[
                save_dir_output,
                quad_filename,
                conf_thres,
                dedup_enable,
                dedup_radius,
                normals_k,
                normals_chunk,
                quad_size,
            ],
            outputs=[reconstruction_output, log_output],
        )

        export_nerfstudio_btn.click(
            export_nerfstudio_dataset,
            inputs=[
                save_dir_output,
                nerfstudio_folder,
            ],
            outputs=[reconstruction_output, log_output],
        )

        # Mesh reconstruction
        mesh_btn.click(
            reconstruct_mesh_poisson,
            inputs=[
                save_dir_output,
                conf_thres,
                dedup_enable,
                dedup_radius,
                mesh_depth,
                mesh_scale,
                mesh_trim_q,
                mesh_smooth,
                mesh_color_k,
                mesh_color_from_full,
                mesh_proj_color,
                mesh_proj_k,
                mesh_proj_angle_pow,
            ],
            outputs=[reconstruction_output, log_output],
        )

        # Texture baking (single view UV)
        bake_btn.click(
            bake_texture_single_view,
            inputs=[
                save_dir_output,
                bake_frame_idx,
                bake_fov,
                bake_tex_size,
            ],
            outputs=[reconstruction_output, log_output],
        )

        # Texture baking (multi-view)
        mv_bake_btn.click(
            bake_texture_multiview,
            inputs=[
                save_dir_output,
                mv_ref_frame,
                mv_tex_size,
                mv_knn,
                mv_angle_pow,
            ],
            outputs=[reconstruction_output, log_output],
        )

    demo.queue(max_size=20).launch(show_error=True, share=True)
