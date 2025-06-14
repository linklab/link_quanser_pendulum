# 🚀 link_quanser_pendulum 설치 가이드

## 📦 1. Quanser SDK 설치

### 🔗 GitHub 저장소
[Quanser SDK (Windows 64-bit)](https://github.com/quanser/quanser_sdk_win64)

### 📥 설치 명령어

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade --find-links "%QSDK_DIR%\python" "%QSDK_DIR%\python\quanser_api-*.whl"
```

### ⚠️ 에러 발생 시

```bash
ERROR: Invalid wheel filename (wrong number of parts): 'quanser_api-*'
```

위 에러가 발생하면 `%QSDK_DIR%` 대신 **Quanser SDK를 설치한 실제 경로**를 입력하세요.
```bash
python -m pip install --upgrade --find-links "C:\Program Files\Quanser\Quanser SDK\python" quanser_api quanser_common quanser_communications quanser_devices quanser_hardware quanser_image_processing quanser_multimedia
```