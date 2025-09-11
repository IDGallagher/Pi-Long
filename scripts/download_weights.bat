@echo off
setlocal EnableExtensions

rem Minimal Windows script to fetch required weights
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "WEIGHTS_DIR=%ROOT_DIR%\weights"

if not exist "%WEIGHTS_DIR%" mkdir "%WEIGHTS_DIR%"
pushd "%WEIGHTS_DIR%"

echo ===== Downloading weights into: %CD% =====
echo.

if not exist "dino_salad.ckpt" (
	echo [1/4] SALAD (dino_salad.ckpt)
	powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt' -OutFile 'dino_salad.ckpt'"
) else (
	echo [1/4] SALAD already exists, skipping
)

@REM if not exist "dinov2_vitb14_pretrain.pth" (
@REM 	echo [2/4] DINO (dinov2_vitb14_pretrain.pth)
@REM 	powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth' -OutFile 'dinov2_vitb14_pretrain.pth'"
@REM ) else (
@REM 	echo [2/4] DINO already exists, skipping
@REM )

@REM if not exist "ORBvoc.txt" (
@REM 	echo [3/4] DBoW (ORBvoc.txt)
@REM 	powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz' -OutFile 'ORBvoc.txt.tar.gz'"
@REM 	if not exist "ORBvoc.txt.tar.gz" (
@REM 		echo Primary URL failed, trying fallback...
@REM 		powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz' -OutFile 'ORBvoc.txt.tar.gz'"
@REM 	)
@REM 	if exist "ORBvoc.txt.tar.gz" (
@REM 		where tar >nul 2>&1
@REM 		if %errorlevel%==0 (
@REM 			echo Extracting with tar...
@REM 			tar -xzf ORBvoc.txt.tar.gz
@REM 		) else (
@REM 			echo 'tar' not found; trying Python to extract...
@REM 			python -c "import tarfile; tarfile.open('ORBvoc.txt.tar.gz','r:gz').extractall('.')"
@REM 		)
@REM 		if exist "ORBvoc.txt.tar.gz" del /f /q ORBvoc.txt.tar.gz
@REM 	) else (
@REM 		echo Failed to download ORBvoc.txt.tar.gz
@REM 	)
@REM ) else (
@REM 	echo [3/4] DBoW already exists, skipping
@REM )

@REM if not exist "model.safetensors" (
@REM 	echo [4/4] Pi3 (model.safetensors)
@REM 	powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors' -OutFile 'model.safetensors'"
@REM ) else (
@REM 	echo [4/4] Pi3 already exists, skipping
@REM )

echo.
echo Done.
popd
