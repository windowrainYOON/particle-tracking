"""영상 표시용 LUT: cytosol-analysis(Cytosol Viewer)의 LUT 프리셋 JSON을 읽어 역할(reference/measure/RI)에 맞춥니다.

프리셋 형식 (cytosol.lut.save_preset):
    {"channels": {"<채널 이름>": {"vmin", "vmax", "color": [r, g, b], "gamma", "visible"}}, "order": [...]}

LUT은 저장하는 영상의 표시에만 쓰고, 측정은 언제나 raw 값으로 합니다.
채널 이름 매칭
  TCF 형광 CHn : 프리셋 이름이 "CHn" 이거나 "CHn "으로 시작 (Cytosol Viewer는 "CH0 ex561/em600"처럼 씀)
  TCF RI       : 프리셋 이름 "RI" (또는 "HT")
  tif          : 파일 이름에 프리셋 채널 이름(또는 '-'/공백 앞부분)이 들어 있으면
Cytosol Viewer는 RI를 실제 굴절률(1.33…)로, 이 앱은 RI × 10⁴ 정수로 다루므로 RI LUT 범위는 데이터에 맞게 × 10⁴ 합니다.
"""
from __future__ import annotations
import json, re, shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .io_utils import split_source, RI_CHANNEL


@dataclass
class ChannelLUT:
    name: str
    vmin: float
    vmax: float
    color: tuple = (1.0, 1.0, 1.0)
    gamma: float = 1.0

    def norm(self, plane) -> np.ndarray:
        n = np.clip((np.asarray(plane, np.float32) - self.vmin) / max(self.vmax - self.vmin, 1e-12), 0, 1)
        return n ** (1.0 / self.gamma) if self.gamma != 1.0 else n

    def rgb(self, plane) -> np.ndarray:
        """(Y, X, 3) float 0..1."""
        return self.norm(plane)[..., None] * np.asarray(self.color, np.float32)

    def to_dict(self) -> dict:
        return dict(name=self.name, vmin=self.vmin, vmax=self.vmax, color=list(self.color), gamma=self.gamma)


def auto_rgb(plane, lo, hi) -> np.ndarray:
    """프리셋이 없을 때: 주어진 범위의 회색."""
    n = np.clip((np.asarray(plane, np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.dstack([n] * 3)


def load_preset(path) -> dict:
    """{채널 이름: dict} (order 순서)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ch = d.get("channels")
    if not isinstance(ch, dict) or not ch: raise ValueError(f"LUT 프리셋 형식이 아닙니다 (channels 없음): {path}")
    order = [n for n in d.get("order", []) if n in ch] + [n for n in ch if n not in d.get("order", [])]
    return {n: ch[n] for n in order}


def _key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match_channel(preset: dict, src, role: str) -> str | None:
    """역할의 입력(source)에 맞는 프리셋 채널 이름, 없으면 None."""
    path, ch = split_source(src)
    names = list(preset)
    if ch is not None:
        if ch == RI_CHANNEL:
            return next((n for n in names if n.strip().upper() in ("RI", "HT")), None)
        return next((n for n in names if n.strip().upper() == ch.upper() or n.upper().startswith(ch.upper() + " ")), None)
    stem = _key(Path(path).stem); best = None
    for n in names:
        for cand in {n, re.split(r"[-\s]", n.strip())[0]}:
            k = _key(cand)
            if len(k) >= 2 and k in stem and (best is None or len(k) > best[0]): best = (len(k), n)
    if best is None and role == "ri": return next((n for n in names if n.strip().upper() in ("RI", "HT")), None)
    return best[1] if best else None


def role_luts(path, sources: dict, data: dict | None = None, log=print) -> dict:
    """{역할: ChannelLUT}. sources = {역할: source}, data = {역할: 스택} (RI 단위 맞춤용). 맞는 채널이 없는 역할은 빠짐."""
    preset = load_preset(path); out = {}
    for role, src in sources.items():
        n = match_channel(preset, src, role)
        if n is None:
            log(f"LUT 프리셋에 {role} 채널({Path(split_source(src)[0]).name} {split_source(src)[1] or ''})과 맞는 이름이 없어 자동 범위로 표시합니다 "
                f"(프리셋 채널: {', '.join(preset)})"); continue
        d = preset[n]; lo, hi = float(d["vmin"]), float(d["vmax"])
        if role == "ri" and data is not None and role in data and hi < 100 and float(np.nanmax(data[role][len(data[role]) // 2])) > 1000:
            lo, hi = lo * 1e4, hi * 1e4                  # 굴절률 → RI × 10⁴
        out[role] = ChannelLUT(n, lo, hi, tuple(float(c) for c in d.get("color", (1, 1, 1))), float(d.get("gamma", 1.0) or 1.0))
        log(f"LUT {role} ← 프리셋 '{n}' ({lo:g}–{hi:g}, γ {out[role].gamma:g})")
    return out


def save_with_outputs(path, out_dir, luts: dict) -> Path:
    """프리셋 파일을 출력 폴더에 그대로 복사하고, 역할별로 쓴 값을 lut_roles.json에 남깁니다."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); dst = out_dir / Path(path).name
    if Path(path).resolve() != dst.resolve(): shutil.copyfile(path, dst)
    (out_dir / "lut_roles.json").write_text(json.dumps(dict(preset=str(path), roles={r: l.to_dict() for r, l in luts.items()}),
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    return dst
