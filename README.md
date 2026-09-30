# ParticleTracker

Cy5(내부 대조) · pHrodo(pH 센서) · HT(holotomography z-projection) 타임랩스에서
입자를 검출·추적하고, 입자별 **I(pHrodo)/I(Cy5)** 변화를 세포 안/밖으로 나눠 분석하는 macOS용 GUI 프로그램입니다.
그래프·영상 출력, 여러 데이터셋 통합, 특이 입자(비율이 한 번 증가했다가 감소·유지되는 입자) 색인과 크롭 영상까지 포함합니다.

---

## 1. 설치와 실행 (macOS)

1. **Python 3.10 이상** 설치: <https://www.python.org/downloads/macos/> (Apple Silicon / Intel 모두 가능)
2. 이 폴더(`ParticleTracker`)를 원하는 위치에 둡니다.
3. **`run_mac.command`를 더블클릭**합니다.
   - 처음 한 번은 `.venv` 가상환경을 만들고 패키지를 설치합니다(인터넷 필요, 수 분).
   - "확인되지 않은 개발자" 경고가 뜨면 **우클릭 → 열기**로 실행하세요.
   - 터미널에서 실행하려면:
     ```bash
     cd ParticleTracker
     python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
     ./.venv/bin/python -m ptracker
     ```
4. (선택) 독립 실행 앱: `./build_mac_app.sh` → `dist/ParticleTracker.app`

필요 메모리: 1464×1464×38 프레임 데이터셋 1개당 약 3–4 GB. 처리 시간은 데이터셋당 약 5–10분(영상 포함).

---

## 2. GUI 사용법

| 탭 | 하는 일 |
|---|---|
| **1. 데이터·실행** | `tif 파일 추가`로 여러 파일을 한꺼번에 고르면 파일명 키워드(`Cy5`, `pHrodo`, `HT`)로 3채널씩 자동으로 묶습니다 (예: `Cy5_1.tif / pHrodo_1.tif / HT_1.tif` → 데이터셋 `1`). 실행할 **단계**를 고르고 ▶ 실행. 진행률과 로그가 표시되고 ■ 중지로 멈출 수 있습니다. |
| **2. 파라미터** | 모든 알고리즘 파라미터. 항목에 마우스를 올리면 설명이 보입니다. YAML로 저장/불러오기. |
| **3. 결과 보기** | 결과 폴더의 그림(PNG)·영상(MP4)·표(CSV)를 바로 미리보기. |
| **4. 통합 분석** | 여러 데이터셋 결과를 체크해 하나로 합치고, 합산 그래프와 통합 CSV를 만듭니다. |
| **5. 특이 입자** | 통합(또는 단일) 결과에서 I(pHrodo)/I(Cy5)가 **한 번 증가 → 급격한/장기적 감소 → 낮게 유지**되는 입자를 찾습니다. 기준값을 바로 바꿔 다시 분류할 수 있고, 표에서 입자를 고르면 곡선이 보이며 **크롭 영상 + 그래프**를 만들 수 있습니다. |

### 파라미터를 바꾼 뒤 어디부터 다시 돌리나?
각 단계 결과는 `<출력 폴더>/_cache/`에 저장되므로 바뀐 단계부터만 다시 실행하면 됩니다.

| 바꾼 파라미터 섹션 | 다시 실행할 단계 |
|---|---|
| 입자 검출 | 전체 |
| 정합 | 정합 → 이후 전부 |
| 세포 영역 / 퍼진 세포막 / 분열 세포 | 세포 → 이후 전부 |
| 트래킹 | 트래킹 → 측정 → 그래프 → 영상 |
| 측정·분류 | 측정 → 그래프 → 영상 |
| 특이 입자 | [특이 입자] 탭에서 분류만 다시 |
| 출력 | 그래프 / 영상 |

---

## 3. 알고리즘 요약

1. **입자 검출 (Cy5)** — 가우시안 배경(σ=25) 제거 → 프레임별 적응형 threshold(중앙값+10×MAD와 triangle 중 큰 값) → 3 px 이하 제거 → 국소 최대점 씨앗 → watershed로 붙은 입자 분리.
2. **정합** — Cy5를 HT 해상도로 축소 후 프레임별 상관 최대 이동(±8 px, subpixel), Cy5↔pHrodo 정수 이동(±3 px). 상관이 낮은 프레임은 중앙값 이동 사용.
3. **세포 (HT)** — opening으로 입자 제거 + 평활화 → triangle(세포 영역)/Otsu(세포 중심) threshold → watershed → 겹침으로 프레임 간 ID 연결. TV-L1 optical flow로 세포 움직임.
   **퍼진 세포막**: 배지(가장 어두운 배경) 분포 + 3σ 이상 = 세포 발자국, 그중 세포질 밖. (세포 안/밖 분류에는 쓰지 않고 표시·기록만)
   **분열 세포**: 평균 RI가 직전 6프레임 대비 급증 + 면적 급감(또는 둥글고 매우 밝음) → 해당 세포에 주로 속한 트랙 제외.
4. **트래킹 (v2)** — 세포 흐름 + Cy5 패턴 매칭(입자와 주변 입자 배치를 템플릿으로 다음 프레임에서 NCC 탐색)으로 위치 예측 + 이웃 일관성(주변 입자 이동 중앙값과의 차이에 벌점, 2-pass). 비용 = 거리 + 세포 불일치 + intensity/크기 차이 → LAP(헝가리안) 1:1 연결 → 1프레임 gap closing.
5. **측정** — 같은 ROI로 Cy5/pHrodo(배경 고리 중앙값 제거)와 HT RI 측정, I(pHrodo)/I(Cy5). 트랙의 50% 초과가 세포질이면 세포 안.
6. **특이 입자** — 3점 중앙값 곡선 기준 증가(×2, +0.01) 후 급격한 감소(한 프레임에 35% 이하) 또는 장기적 감소(끝이 최고점 50% 이하), 감소 후 재증가·반복 상승·1프레임 반짝·최고점 전 오르내림 제외, 감소 후 3시점 이상 유지.

---

## 4. 출력 파일 (데이터셋 폴더)

| 파일 | 내용 |
|---|---|
| `tracks_points_A_B_HT.csv` | 트랙·시점별 좌표(형광/HT), 영역(cytoplasm/spread_membrane/background), I(Cy5), I(pHrodo), 배경, 비율, HT RI, 이동량 |
| `tracks_summary_A_B_HT.csv` | 트랙별 요약, 그룹(inside/outside/short/excluded_mitotic) |
| `particles_all_slices.csv`, `particle_labels.tif` | 프레임별 검출 입자와 ROI 라벨 stack(Fiji에서 확인 가능) |
| `registration.csv`, `cell_stats_mitosis.csv`, `footprint_per_frame.csv` | 정합·분열 판정·영역 QC |
| `ratio_*.png`, `channel_timecourses.png`, `tracks_overlay.png`, `regions_*.png`, `mitosis_detection.png`, `qc_*.png` | 그래프 |
| `movie_*.mp4` | Cy5/pHrodo/HT × 세포 안/밖 트랙 영상, 영역 영상 |
| `dataset.json`, `config_used.yaml` | 입력 파일 경로와 사용한 파라미터(재현용) |

---

## 5. 명령줄 사용 (배치 처리)

```bash
./.venv/bin/python -m ptracker.cli run-folder /data/exp1 --out /data/results --config configs/default.yaml
./.venv/bin/python -m ptracker.cli combine /data/results/1 /data/results/2 --out /data/results/Combined
./.venv/bin/python -m ptracker.cli risefall /data/results/Combined/combined_tracks_points_A_B_HT.csv --out /data/results/RiseFall
# 일부 단계만:  ... run --cy5 a.tif --phrodo b.tif --ht c.tif --out out --stages track measure figures
```

---

## 6. 코드 구조와 유지보수

```
ptracker/
  config.py        모든 파라미터(dataclass). 필드를 추가하면 GUI 폼이 자동으로 생깁니다.
  io_utils.py      tif 로드, 파일명 키워드 자동 묶음, DatasetSpec
  detection.py     1. 입자 검출
  registration.py  2. 정합 (좌표 변환 fl_to_ht, pHrodo 정렬 align_b)
  cells.py         3. 세포 분할/ID, optical flow, 퍼진 세포막, 분열 판정
  tracking.py      4. 예측(흐름·패턴), 비용 행렬, LAP, 이웃 일관성, gap closing
  measurement.py   5. ROI 측정, 트랙 요약·그룹 분류
  analysis.py      6. 데이터셋 통합, 특이 입자 분류
  plots.py         그래프 함수
  movies.py        채널 영상, 크롭 영상
  pipeline.py      단계 실행·캐시(DatasetRunner), 통합/특이 입자/크롭 작업
  cli.py           명령줄
  style.py         한글 글꼴, ffmpeg 경로
  gui/             PySide6 GUI (main_window, param_form, worker, tasks, table_model)
tests/             합성 데이터 end-to-end 테스트 (python tests/test_pipeline.py)
configs/default.yaml
```

- **파라미터 추가**: `config.py`의 해당 dataclass에 `이름: 타입 = p(기본값, "라벨", "설명", min=.., max=..)` 한 줄 → 알고리즘 함수에서 `prm.이름` 사용. GUI·YAML 자동 반영.
- **알고리즘 수정**: 각 단계는 독립 모듈의 순수 함수입니다(입력 배열/표 → 출력). 수정 후 `python tests/test_pipeline.py`로 전체 흐름을 확인하세요.
- **단계 추가**: `pipeline.py`의 `STAGES`, `STAGE_LABELS`, `STAGE_WEIGHT`에 이름을 넣고 `DatasetRunner.stage_<이름>()` 메서드를 만들면 GUI 체크박스가 자동 추가됩니다. 캐시는 `self._save(key, obj)` / `self._get(key)`.
- **그래프 추가**: `plots.py`에 함수를 만들고 `stage_figures`에서 호출.

## 7. 문제 해결
- **영상이 GIF로 저장됨**: ffmpeg를 찾지 못한 경우입니다. `pip install imageio-ffmpeg` 또는 `brew install ffmpeg`.
- **한글이 네모로 보임**: macOS는 AppleGothic을 자동 사용합니다. 다른 OS는 Noto Sans CJK/나눔고딕 설치.
- **메모리 부족으로 멈춤**: 다른 앱을 닫고, 데이터셋을 하나씩 실행하세요. 끊긴 경우 캐시가 남은 단계 다음부터 다시 실행하면 됩니다.
- **채널 자동 인식 실패**: [파라미터 > 채널·시간]에서 파일명 키워드를 바꾸거나 [수동 추가]를 쓰세요.
