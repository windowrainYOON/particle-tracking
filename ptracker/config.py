"""
모든 분석 파라미터를 한 곳에서 관리합니다.

파라미터를 추가하려면 해당 dataclass에 field를 추가하고 `p(기본값, "라벨", "설명")`
형식으로 메타데이터를 달아 주세요. GUI의 파라미터 탭은 이 정보로 자동 생성됩니다.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
import yaml


def p(default, label: str, help: str = "", **kw):
    """dataclass field + GUI metadata (label/help/min/max/step)."""
    return field(default=default, metadata=dict(label=label, help=help, **kw))


@dataclass
class ChannelParams:
    cy5_keyword: str = p("Cy5", "Cy5(A) 파일 키워드", "파일명에 이 문자열이 있으면 A 채널(ROI 검출용 internal control)로 인식")
    phrodo_keyword: str = p("pHrodo", "pHrodo(B) 파일 키워드", "파일명에 이 문자열이 있으면 B 채널(pH 센서)로 인식")
    ht_keyword: str = p("HT", "HT 파일 키워드", "파일명에 이 문자열이 있으면 holotomography(z-projection)로 인식")
    frame_interval_min: float = p(20.0, "프레임 간격 (분)", "타임랩스 촬영 간격", min=0.1, max=1440, step=1)


@dataclass
class DetectionParams:
    sigma_bg: float = p(25.0, "배경 가우시안 σ (px)", "입자 반경의 ~5배 이상. 느리게 변하는 배경을 추정", min=1, max=200, step=1)
    sigma_smooth: float = p(1.0, "평활화 σ (px)", "노이즈가 심하면 1.5~2", min=0, max=10, step=0.1)
    k_mad: float = p(10.0, "threshold 배수 k (×MAD)", "중앙값 + k×MAD 와 triangle 중 큰 값을 threshold로 사용", min=1, max=50, step=0.5)
    min_distance: int = p(3, "봉우리 최소 간격 (px)", "watershed 씨앗(국소 최대점) 간 최소 거리", min=1, max=30)
    max_small: int = p(3, "노이즈 조각 크기 (px 이하 제거)", "이 크기 이하의 조각은 제거", min=0, max=100)


@dataclass
class RegistrationParams:
    fl_ht_search: int = p(8, "형광→HT 탐색 범위 (HT px)", "프레임별 이동량 탐색 반경", min=1, max=40)
    ab_search: int = p(3, "Cy5↔pHrodo 탐색 범위 (px)", "", min=0, max=20)
    fl_ht_min_corr: float = p(0.5, "형광→HT 최소 상관", "이보다 낮은 프레임은 다른 프레임 이동량 중앙값 사용", min=0, max=1, step=0.05)
    ab_min_corr: float = p(0.3, "Cy5↔pHrodo 최소 상관", "", min=0, max=1, step=0.05)


@dataclass
class CellParams:
    open_radius: int = p(5, "입자 제거 opening 반경 (px)", "HT에서 세포 몸체 영상을 만들 때 입자를 지우는 크기", min=1, max=30)
    smooth_sigma: float = p(3.0, "세포 영상 평활화 σ", "", min=0, max=20, step=0.5)
    min_cell_area: int = p(3000, "최소 세포 면적 (HT px)", "", min=100, max=200000, step=100)
    id_overlap: float = p(0.3, "세포 ID 연결 최소 겹침 비율", "다음 프레임 세포가 이 비율 이상 겹치면 같은 ID", min=0.05, max=1, step=0.05)
    flow_attachment: float = p(10.0, "optical flow attachment", "TV-L1 데이터 항 가중치", min=1, max=100, step=1)


@dataclass
class FootprintParams:
    open_radius: int = p(3, "입자 제거 반경 (px)", "퍼진 세포막 검출 전 입자 제거", min=1, max=20)
    smooth_sigma: float = p(2.0, "평활화 σ", "", min=0, max=10, step=0.5)
    k_sigma: float = p(3.0, "배경 + kσ", "배지(완전히 어두운 배경) 분포보다 k 표준편차 이상 밝으면 세포 발자국", min=0.5, max=20, step=0.5)
    min_area: int = p(1500, "최소 조각 면적 (px)", "", min=0, max=100000, step=100)
    keep_large: int = p(5000, "세포와 떨어져도 유지할 면적 (px)", "", min=0, max=500000, step=500)


@dataclass
class MitosisParams:
    enabled: bool = p(True, "분열 세포 제외", "분열(둥글게 수축) 세포에 주로 속한 트랙을 분석에서 제외")
    d_ri: float = p(50.0, "RI 급증 기준 (×10⁻⁴)", "직전 6프레임 중앙값 대비 평균 RI 증가량", min=1, max=1000, step=5)
    area_ratio: float = p(0.75, "면적 급감 기준 (비율)", "직전 6프레임 중앙값 대비 면적 비율 미만", min=0.1, max=1, step=0.05)
    abs_ri: float = p(13580.0, "절대 RI 기준 (×10⁴)", "이보다 밝고 둥근 세포도 분열로 판정", min=13000, max=15000, step=10)
    max_ecc: float = p(0.8, "둥근 모양 기준 (eccentricity 미만)", "", min=0, max=1, step=0.05)


@dataclass
class TrackingParams:
    gate: float = p(8.0, "gate (HT px)", "예측 위치에서 이 거리 밖 후보는 연결 불가", min=1, max=50, step=0.5)
    w_cell: float = p(2.0, "세포 불일치 벌점", "", min=0, max=20, step=0.5)
    w_int: float = p(2.0, "intensity 차이 가중치", "× |ln(I1/I2)|", min=0, max=20, step=0.5)
    w_area: float = p(1.0, "크기 차이 가중치", "× |ln(A1/A2)|", min=0, max=20, step=0.5)
    gap_closing: bool = p(True, "gap closing (1프레임 누락 연결)", "")
    pattern_half: int = p(15, "패턴 템플릿 반폭 (형광 px)", "입자와 주변 입자 배치를 담는 창 크기 = 2×반폭+1", min=3, max=60)
    pattern_search: int = p(12, "패턴 탐색 반경 (형광 px)", "", min=2, max=60)
    pattern_score_hi: float = p(0.5, "패턴 신뢰 상관 (≥ 이면 패턴 위치 사용)", "", min=0, max=1, step=0.05)
    pattern_score_lo: float = p(0.3, "패턴 부분 신뢰 상관 (흐름과 반반)", "", min=0, max=1, step=0.05)
    nb_radius: float = p(25.0, "이웃 반경 (형광 px)", "이웃 이동 중앙값 계산 범위", min=1, max=200, step=1)
    nb_min: int = p(2, "최소 이웃 수", "", min=1, max=20)
    w_nb: float = p(0.5, "이웃 일관성 가중치", "× |이 연결의 이동 − 이웃 이동 중앙값|", min=0, max=10, step=0.1)


@dataclass
class MeasurementParams:
    ring_in: int = p(3, "배경 고리 안쪽 (px)", "ROI 바깥 이 거리부터", min=0, max=30)
    ring_out: int = p(8, "배경 고리 바깥쪽 (px)", "", min=1, max=60)
    min_ia: float = p(5.0, "비율 계산 최소 I(Cy5)", "배경 뺀 Cy5가 이 값 이하면 비율 제외", min=0, max=1000, step=1)
    min_track: int = p(10, "장기 트랙 최소 프레임 수", "그래프·분류에 쓰는 트랙 길이 기준", min=2, max=200)
    inside_fraction: float = p(0.5, "세포 안 판정 비율", "트랙 프레임 중 이 비율 초과가 세포질 안이면 inside", min=0, max=1, step=0.05)


@dataclass
class RiseFallParams:
    rise_x: float = p(2.0, "증가 배수", "최고점 ≥ 시작값 × 배수", min=1, max=20, step=0.1)
    rise_delta: float = p(0.01, "최소 증가/감소 폭 (비율)", "", min=0, max=1, step=0.001, decimals=4)
    floor: float = p(0.005, "시작값 하한", "", min=0, max=1, step=0.001, decimals=4)
    drop_frac: float = p(0.35, "급격한 감소: 다음 프레임 ≤ 이전 × ", "", min=0, max=1, step=0.05)
    fall_frac: float = p(0.5, "장기적 감소: 끝 ≤ 최고점 × ", "", min=0, max=1, step=0.05)
    rebound_frac: float = p(0.6, "재증가 기준 (최고점 ×)", "감소 후 이 수준을 넘으면 진동으로 제외", min=0, max=1, step=0.05)
    rebound_raw_max: float = p(0.15, "감소 후 원본값 상승 허용 비율", "", min=0, max=1, step=0.05)
    min_post: int = p(3, "감소 후 최소 관찰 시점", "", min=1, max=50)
    min_high_frames: int = p(2, "최고 구간 최소 연속 프레임", "1프레임 반짝 신호 제외", min=1, max=10)
    min_points: int = p(8, "판정 최소 비율 시점 수", "", min=4, max=100)


@dataclass
class OutputParams:
    make_movies: bool = p(True, "채널별 영상 생성", "")
    movie_fps: int = p(3, "영상 fps", "", min=1, max=30)
    make_figures: bool = p(True, "그래프 생성", "")
    keep_cache: bool = p(True, "중간 결과(캐시) 보관", "파라미터를 바꾼 뒤 특정 단계부터 다시 돌릴 때 필요")
    crop_half: int = p(32, "크롭 반폭 (형광 px)", "", min=8, max=200)


@dataclass
class Config:
    channel: ChannelParams = field(default_factory=ChannelParams)
    detection: DetectionParams = field(default_factory=DetectionParams)
    registration: RegistrationParams = field(default_factory=RegistrationParams)
    cells: CellParams = field(default_factory=CellParams)
    footprint: FootprintParams = field(default_factory=FootprintParams)
    mitosis: MitosisParams = field(default_factory=MitosisParams)
    tracking: TrackingParams = field(default_factory=TrackingParams)
    measurement: MeasurementParams = field(default_factory=MeasurementParams)
    risefall: RiseFallParams = field(default_factory=RiseFallParams)
    output: OutputParams = field(default_factory=OutputParams)

    SECTION_LABELS = {
        "channel": "채널·시간", "detection": "입자 검출 (Cy5)", "registration": "정합",
        "cells": "세포 영역 (HT)", "footprint": "퍼진 세포막 (HT)", "mitosis": "분열 세포",
        "tracking": "트래킹", "measurement": "측정·분류", "risefall": "특이 입자 (증가 후 감소)",
        "output": "출력",
    }

    # ---- (de)serialisation ----
    def to_dict(self) -> dict:
        return {f.name: asdict(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for sec, vals in (d or {}).items():
            obj = getattr(cfg, sec, None)
            if obj is None or not is_dataclass(obj):
                continue
            for k, v in (vals or {}).items():
                if hasattr(obj, k):
                    setattr(obj, k, type(getattr(obj, k))(v))
        return cfg

    def save(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def sections(cfg: Config):
    """GUI helper: (section_name, label, dataclass_obj) 목록."""
    for f in fields(cfg):
        yield f.name, Config.SECTION_LABELS.get(f.name, f.name), getattr(cfg, f.name)
