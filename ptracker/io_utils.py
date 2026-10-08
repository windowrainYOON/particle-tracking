"""파일 입출력과 데이터셋(역할 3개 묶음) 정의.

역할
  reference (A) : ROI 검출용 internal control 형광 (예: Cy5)
  measure   (B) : 측정 대상 형광 (예: pHrodo)
  RI            : 굴절률 z-projection (holotomography)

각 역할의 입력(source)은 tif 경로이거나, Tomocube raw TCF 파일의 채널 하나입니다.
  "/path/Cy5_1.tif"            → tif 스택 (T, Y, X)
  "/path/sample.TCF::CH0"      → TCF 형광 채널 CH0 (2DFLMIP, 없으면 3DFL의 z 최대값)
  "/path/sample.TCF::RI"       → TCF 굴절률 z-projection (2DMIP, 없으면 3D의 z 최대값), RI × 10⁴ 정수
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import tifffile

ROLES = ("measure", "reference", "ri")                       # 화면에 보이는 순서
ROLE_LABELS = {"measure": "measure", "reference": "reference", "ri": "RI"}
ROLE_HELP = {"measure": "측정 대상 형광 (B, 예: pHrodo)", "reference": "기준 형광 (A, ROI 검출용 internal control, 예: Cy5)",
             "ri": "굴절률 z-projection (holotomography)"}
TCF_SEP = "::"
RI_CHANNEL = "RI"
TCF_EXT = {".tcf"}
_OLD_KEYS = {"cy5": "reference", "phrodo": "measure", "ht": "ri"}   # 1.8 이전 dataset.json / 프로젝트 파일


@dataclass
class DatasetSpec:
    name: str
    reference: str    # A 채널 (internal control, ROI 검출)
    measure: str      # B 채널 (측정 대상, 예: pH 센서)
    ri: str           # RI z-projection
    out_dir: str

    def sources(self):
        return {"measure": self.measure, "reference": self.reference, "ri": self.ri}

    def validate(self) -> list[str]:
        errs = []
        for role, src in self.sources().items():
            path, ch = split_source(src)
            if not Path(path).is_file(): errs.append(f"{ROLE_LABELS[role]} 파일 없음: {path}"); continue
            if ch is not None:
                try: names = [c["name"] for c in tcf_channels(path)]
                except Exception as e: errs.append(f"{ROLE_LABELS[role]} TCF 파일을 읽을 수 없습니다: {path} ({e})"); continue   # noqa: BLE001
                if ch not in names: errs.append(f"{ROLE_LABELS[role]} 채널 {ch}가 TCF에 없습니다 (있는 채널: {', '.join(names)}): {path}")
        return errs

    def save(self, path=None):
        path = Path(path or Path(self.out_dir) / "dataset.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        return cls(**{_OLD_KEYS.get(k, k): v for k, v in d.items()})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ================================================================ source 문자열
def split_source(src) -> tuple[str, str | None]:
    """"a.TCF::CH0" → ("a.TCF", "CH0"), "a.tif" → ("a.tif", None)."""
    src = str(src)
    if TCF_SEP in src:
        p, ch = src.rsplit(TCF_SEP, 1)
        if is_tcf(p): return p, ch
    return src, None


def make_source(path, channel=None) -> str:
    return f"{path}{TCF_SEP}{channel}" if channel else str(path)


def is_tcf(path) -> bool:
    return Path(str(path)).suffix.lower() in TCF_EXT


def source_label(src) -> str:
    """표에 보일 짧은 이름: 'sample.TCF › CH0' 또는 'Cy5_1.tif'."""
    p, ch = split_source(src)
    return f"{Path(p).name} › {ch}" if ch else Path(p).name


# ================================================================ TCF (Tomocube, HDF5)
def _h5():
    import h5py
    return h5py


def _attr(o, k, default=None):
    v = o.attrs.get(k)
    if v is None: return default
    v = np.ravel(v)[0] if np.ndim(v) else v
    return v.decode() if isinstance(v, bytes) else v


def _tcf_groups(f, ch):
    """채널 → (2D 그룹, 3D 그룹) (없으면 None)."""
    D = f["Data"]
    if ch == RI_CHANNEL: return D.get("2DMIP"), D.get("3D")
    g2, g3 = D.get("2DFLMIP"), D.get("3DFL")
    return (g2.get(ch) if g2 is not None else None), (g3.get(ch) if g3 is not None else None)


@lru_cache(maxsize=64)
def _tcf_index(path, ch, mtime):
    """[(데이터셋 이름, 시간(s), 2D 여부)], 프레임 크기 (Y, X)."""
    with _h5().File(path, "r") as f:
        g2, g3 = _tcf_groups(f, ch)
        g, is2d = (g2, True) if g2 is not None and len(g2) else (g3, False)
        if g is None or not len(g): raise ValueError(f"TCF에 {ch} 데이터가 없습니다: {path}")
        keys = sorted(k for k in g.keys() if k.isdigit())
        idx = [(k, float(_attr(g[k], "Time", i)), is2d) for i, k in enumerate(keys)]
        shp = g[keys[0]].shape[-2:]
    return idx, tuple(int(s) for s in shp)


def _index(path, ch):
    return _tcf_index(str(path), ch, Path(path).stat().st_mtime)


def tcf_channels(path) -> list[dict]:
    """TCF에 들어 있는 채널 목록. [{name, label, frames, size, interval_min}] — 형광(CH0, CH1, …) 다음 RI."""
    out = []
    with _h5().File(str(path), "r") as f:
        D = f["Data"]; fl = set()
        for gname in ("2DFLMIP", "3DFL"):
            if gname in D: fl |= {k for k in D[gname].keys() if k.upper().startswith("CH")}
        meta = {}
        for ch in sorted(fl):
            g2, g3 = _tcf_groups(f, ch); g = g2 if g2 is not None else g3
            em = _attr(g, "Emission"); ex = _attr(g, "Excitation")
            meta[ch] = "형광" + (f" (Ex {ex * 1000:.0f} / Em {em * 1000:.0f} nm)" if em and ex else "")
        has_ri = "2DMIP" in D or "3D" in D
    for ch in sorted(fl) + ([RI_CHANNEL] if has_ri else []):
        try: idx, shp = _index(path, ch)
        except ValueError: continue
        t = [x[1] for x in idx]; dt = float(np.median(np.diff(t))) / 60 if len(t) > 1 else None
        out.append(dict(name=ch, label=meta.get(ch, "굴절률 z-projection"), frames=len(idx), size=shp, interval_min=dt))
    return out


def _tcf_read(path, ch, i) -> np.ndarray:
    idx, _ = _index(path, ch); key, _, is2d = idx[i]
    with _h5().File(str(path), "r") as f:
        g2, g3 = _tcf_groups(f, ch); g = g2 if is2d else g3
        a = g[key][()]
    if a.ndim == 3: a = a.max(0)                     # 3D만 있으면 z 최대값 투영
    return a.astype(np.uint16, copy=False)           # 프레임마다 uint8/uint16이 섞여 있음 (값은 원래 단위)


def tcf_frame_times(src) -> list[float] | None:
    p, ch = split_source(src)
    return [x[1] for x in _index(p, ch)[0]] if ch else None


def _frame_map(src, ref) -> list[int] | None:
    """src의 프레임 중 ref(reference 채널)의 각 프레임 시간과 가장 가까운 것. 둘 다 TCF가 아니면 None (같은 번호)."""
    ts, tr = tcf_frame_times(src), tcf_frame_times(ref) if ref else None
    if ts is None or tr is None: return None
    ts = np.asarray(ts); return [int(np.argmin(np.abs(ts - t))) for t in tr]


# ================================================================ 공통 읽기
def source_shape(src, ref=None) -> tuple[int, int, int]:
    """(T, Y, X). ref를 주면 TCF 채널의 T는 ref 프레임 수에 맞춥니다."""
    p, ch = split_source(src)
    if ch is not None:
        idx, shp = _index(p, ch); m = _frame_map(src, ref)
        return (len(m) if m is not None else len(idx), *shp)
    with tifffile.TiffFile(p) as t: s = [d for d in t.series[0].shape if d != 1]
    return (1, *s) if len(s) == 2 else tuple(s)


def read_frame(src, z, ref=None) -> np.ndarray:
    """한 프레임 (z는 0부터)."""
    p, ch = split_source(src)
    if ch is not None:
        m = _frame_map(src, ref); return _tcf_read(p, ch, m[z] if m is not None else z)
    T = source_shape(src)[0]
    a = np.squeeze(tifffile.imread(p, key=z)) if T > 1 else np.squeeze(tifffile.imread(p))
    if a.ndim != 2: a = np.squeeze(tifffile.imread(p))[z]           # 페이지 = 프레임이 아닌 파일
    return a


def load_stack(src, ref=None) -> np.ndarray:
    """(T, Y, X) 스택을 읽습니다. 2D면 T=1로 확장."""
    p, ch = split_source(src)
    if ch is not None:
        n = source_shape(src, ref)[0]
        return np.stack([read_frame(src, z, ref) for z in range(n)])
    a = tifffile.imread(str(p))
    a = np.squeeze(a)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"{p}: (T,Y,X) 3차원 스택이 아닙니다 (shape={a.shape})")
    return a


def load_dataset(spec: DatasetSpec):
    """(A = reference, B = measure, RI). TCF 채널은 reference 채널의 촬영 시각에 맞춰 프레임을 고릅니다."""
    A = load_stack(spec.reference)
    B, H = load_stack(spec.measure, spec.reference), load_stack(spec.ri, spec.reference)
    if A.shape != B.shape:
        raise ValueError(f"reference {A.shape} 와 measure {B.shape} 크기가 다릅니다")
    if H.shape[0] != A.shape[0]:
        raise ValueError(f"RI 프레임 수({H.shape[0]})가 형광({A.shape[0]})과 다릅니다")
    return A, B, H


# ================================================================ 자동 묶기
def auto_group_files(files, ref_kw="Cy5", meas_kw="pHrodo", ri_kw="HT", out_root=None) -> list[DatasetSpec]:
    """tif: 파일명 키워드로 역할을 인식하고, 키워드를 뺀 나머지 이름이 같은 파일끼리 묶습니다.
    예) Cy5_1.tif / pHrodo_1.tif / HT_1.tif → 데이터셋 '_1'.  (TCF는 tcf_specs로 채널을 직접 할당)"""
    groups: dict[str, dict] = {}
    for f in map(Path, files):
        stem = f.stem
        for role, kw in (("measure", meas_kw), ("reference", ref_kw), ("ri", ri_kw)):   # measure 먼저 (pHrodo에 HT 문자열 포함 방지)
            if kw and re.search(re.escape(kw), stem, re.IGNORECASE):
                key = re.sub(re.escape(kw), "", stem, flags=re.IGNORECASE).strip("_- ") or "dataset"
                groups.setdefault(key, {})[role] = str(f)
                break
    specs = []
    for key, d in sorted(groups.items()):
        if set(ROLES) <= d.keys():
            base = Path(out_root) if out_root else Path(d["reference"]).parent / "ParticleTracker_results"
            specs.append(DatasetSpec(name=key, reference=d["reference"], measure=d["measure"], ri=d["ri"], out_dir=str(base / key)))
    return specs


def tcf_specs(files, assign: dict, out_root=None) -> tuple[list[DatasetSpec], list[str]]:
    """TCF 파일마다 데이터셋 하나. assign = {"measure": "CH0", "reference": "CH1", "ri": "RI"}.
    반환 (데이터셋들, 건너뛴 파일 설명들)."""
    specs, skipped = [], []
    for f in map(Path, files):
        try: names = {c["name"] for c in tcf_channels(f)}
        except Exception as e: skipped.append(f"{f.name}: 읽기 실패 ({e})"); continue   # noqa: BLE001
        miss = [ch for ch in assign.values() if ch not in names]
        if miss: skipped.append(f"{f.name}: 채널 {', '.join(miss)} 없음 (있는 채널: {', '.join(sorted(names))})"); continue
        base = Path(out_root) if out_root else f.parent / "ParticleTracker_results"
        specs.append(DatasetSpec(name=f.stem, out_dir=str(base / f.stem), **{r: make_source(f, assign[r]) for r in ROLES}))
    return specs, skipped
