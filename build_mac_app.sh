#!/bin/bash
# (선택) 더블클릭으로 실행되는 ParticleTracker.app 만들기 — PyInstaller 사용.
# 사용: ./build_mac_app.sh   →  dist/ParticleTracker.app
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt; }
./.venv/bin/pip install pyinstaller
./.venv/bin/pyinstaller --noconfirm --windowed --name ParticleTracker \
  --collect-all skimage --collect-all imageio_ffmpeg --collect-data matplotlib \
  launcher.py
echo "완료: dist/ParticleTracker.app  (처음 열 때 우클릭 → 열기)"
