#!/bin/bash
# 배포용 DMG 만들기: dist/ParticleTracker.app + Applications 바로가기 + 문서 → ParticleTracker-<버전>.dmg
# 사용: ./build_dmg.sh                 → 앱을 새로 빌드(설치는 건너뜀)한 뒤 dist/ 에 DMG 생성
#       ./build_dmg.sh --skip-build    → 이미 있는 dist/ParticleTracker.app 으로 DMG만
#       ./build_dmg.sh --out ~/Desktop → DMG를 저장할 폴더 지정 (기본: dist)
# DMG는 .gitignore에 있어 커밋되지 않습니다.
cd "$(dirname "$0")"
OUT=dist; SKIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) SKIP=1 ;;
    --out) shift; OUT="$1" ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac; shift
done
if [ $SKIP = 0 ]; then ./build_mac_app.sh --no-install || exit 1; fi
[ -d dist/ParticleTracker.app ] || { echo "dist/ParticleTracker.app 이 없습니다. ./build_mac_app.sh 를 먼저 실행하세요."; exit 1; }

VER=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' ptracker/__init__.py)
DMG="$OUT/ParticleTracker-$VER.dmg"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ditto dist/ParticleTracker.app "$STAGE/ParticleTracker.app"
ln -s /Applications "$STAGE/Applications"
mkdir "$STAGE/문서"
cp docs/사용방법.md docs/설치방법.md docs/빌드방법.md README.md CHANGELOG.md "$STAGE/문서/"

mkdir -p "$OUT"; rm -f "$DMG"
hdiutil create -volname "ParticleTracker $VER" -srcfolder "$STAGE" -fs HFS+ -format UDZO -imagekey zlib-level=9 -ov "$DMG" || exit 1
echo "DMG 완료: $DMG ($(du -h "$DMG" | cut -f1))"
