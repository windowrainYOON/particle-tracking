"""파일 입출력과 데이터셋(채널 3개 묶음) 정의."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import tifffile


@dataclass
class DatasetSpec:
    name: str
    cy5: str          # A 채널 (internal control, ROI 검출)
    phrodo: str       # B 채널 (pH 센서)
    ht: str           # HT z-projection
    out_dir: str

    def validate(self) -> list[str]:
        errs = [f"{k} 파일 없음: {v}" for k, v in (("Cy5", self.cy5), ("pHrodo", self.phrodo), ("HT", self.ht)) if not Path(v).is_file()]
        return errs

    def save(self, path=None):
        path = Path(path or Path(self.out_dir) / "dataset.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_stack(path) -> np.ndarray:
    """(T, Y, X) 스택을 읽습니다. 2D면 T=1로 확장."""
    a = tifffile.imread(str(path))
    a = np.squeeze(a)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"{path}: (T,Y,X) 3차원 스택이 아닙니다 (shape={a.shape})")
    return a


def load_dataset(spec: DatasetSpec):
    A, B, H = load_stack(spec.cy5), load_stack(spec.phrodo), load_stack(spec.ht)
    if A.shape != B.shape:
        raise ValueError(f"Cy5 {A.shape} 와 pHrodo {B.shape} 크기가 다릅니다")
    if H.shape[0] != A.shape[0]:
        raise ValueError(f"HT 프레임 수({H.shape[0]})가 형광({A.shape[0]})과 다릅니다")
    return A, B, H


def auto_group_files(files, cy5_kw="Cy5", ph_kw="pHrodo", ht_kw="HT", out_root=None) -> list[DatasetSpec]:
    """파일명 키워드로 채널을 인식하고, 키워드를 뺀 나머지 이름이 같은 파일끼리 묶습니다.
    예) Cy5_1.tif / pHrodo_1.tif / HT_1.tif → 데이터셋 '_1'"""
    groups: dict[str, dict] = {}
    for f in map(Path, files):
        stem = f.stem
        for ch, kw in (("phrodo", ph_kw), ("cy5", cy5_kw), ("ht", ht_kw)):   # pHrodo 먼저 (HT 문자열 포함 방지)
            if re.search(re.escape(kw), stem, re.IGNORECASE):
                key = re.sub(re.escape(kw), "", stem, flags=re.IGNORECASE).strip("_- ") or "dataset"
                groups.setdefault(key, {})[ch] = str(f)
                break
    specs = []
    for key, d in sorted(groups.items()):
        if {"cy5", "phrodo", "ht"} <= d.keys():
            base = Path(out_root) if out_root else Path(d["cy5"]).parent / "ParticleTracker_results"
            specs.append(DatasetSpec(name=key, cy5=d["cy5"], phrodo=d["phrodo"], ht=d["ht"], out_dir=str(base / key)))
    return specs
