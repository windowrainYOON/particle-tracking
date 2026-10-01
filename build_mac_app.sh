#!/bin/bash
# 더블클릭으로 실행되는 ParticleTracker.app 만들기 (PyInstaller) + /Applications에 설치.
# 사용: ./build_mac_app.sh              → dist/ParticleTracker.app 만들고 /Applications/ParticleTracker.app 으로 설치
#       ./build_mac_app.sh --no-install → 설치하지 않고 dist에만
# 아이콘: assets/ParticleTracker.icns (다시 만들려면 ./.venv/bin/python tools/make_icon.py)
cd "$(dirname "$0")"
[ -d .venv ] || { python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt; }
./.venv/bin/pip install -q pyinstaller
./.venv/bin/pyinstaller --noconfirm --windowed --name ParticleTracker --icon assets/ParticleTracker.icns \
  --osx-bundle-identifier com.windowrainyoon.particletracker \
  --collect-all skimage --collect-all imageio_ffmpeg --collect-data matplotlib \
  launcher.py || exit 1
echo "빌드 완료: dist/ParticleTracker.app"
if [ "$1" != "--no-install" ]; then
  if pgrep -f "/Applications/ParticleTracker.app/Contents/MacOS/ParticleTracker" >/dev/null; then
    echo "⚠ /Applications의 ParticleTracker가 실행 중이라 설치를 건너뜁니다. 앱을 종료한 뒤 ./build_mac_app.sh 를 다시 실행하세요."
  else
    rm -rf /Applications/ParticleTracker.app && ditto dist/ParticleTracker.app /Applications/ParticleTracker.app \
      && touch /Applications/ParticleTracker.app && echo "설치 완료: /Applications/ParticleTracker.app (Dock에 끌어다 놓아 쓰세요)"
  fi
fi
