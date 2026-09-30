#!/bin/bash
# ParticleTracker 실행 (macOS). Finder에서 더블클릭하세요.
# 처음 실행 시 가상환경(.venv)을 만들고 필요한 패키지를 설치합니다 (인터넷 필요, 수 분 소요).
cd "$(dirname "$0")"
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
  osascript -e 'display alert "Python 3가 없습니다" message "https://www.python.org/downloads/macos/ 에서 Python 3.10 이상을 설치한 뒤 다시 실행하세요."'
  exit 1
fi
if [ ! -d ".venv" ]; then
  echo "가상환경을 만드는 중..."
  $PY -m venv .venv || exit 1
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt || { echo "패키지 설치 실패"; read -n 1; exit 1; }
fi
./.venv/bin/python -m ptracker
