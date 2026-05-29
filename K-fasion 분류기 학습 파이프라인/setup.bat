@echo off
chcp 65001 >nul
echo ================================
echo K-Fashion 분류기 환경 설정 (Windows + RTX 4060)
echo ================================

:: Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 3.10 또는 3.11 설치 후 다시 실행하세요.
    pause
    exit /b 1
)

echo.
echo [1/6] 가상환경 생성 중...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] 가상환경 생성 실패
    pause
    exit /b 1
)

echo.
echo [2/6] 가상환경 활성화 중...
call venv\Scripts\activate.bat

echo.
echo [3/6] pip 업그레이드 중...
python -m pip install --upgrade pip

echo.
echo [4/6] PyTorch 설치 중 (CUDA 12.1 / RTX 4060)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] PyTorch 설치 실패
    pause
    exit /b 1
)

echo.
echo [5/6] 나머지 패키지 설치 중...
pip install fashionclip pillow requests scikit-learn

echo.
echo [6/6] GPU 인식 확인 중...
python -c "import torch; print('[GPU 확인]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '[WARN] GPU 인식 안됨 - CUDA 드라이버 확인 필요')"

echo.
echo ================================
echo 설치 완료!
echo.
echo 앞으로 작업할 때마다 가상환경 활성화:
echo   venv\Scripts\activate.bat
echo.
echo 학습 실행:
echo   python train_classifier.py
echo ================================
pause