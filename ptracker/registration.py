"""2단계: 형광(Cy5) → HT, Cy5 ↔ pHrodo 프레임별 정합."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from .config import RegistrationParams


def prep(x, s=1):
    x = x.astype(np.float32)
    return np.clip(ndi.gaussian_filter(x, s) - ndi.gaussian_filter(x, 15), 0, None)


def best_shift(ref, mov, R, m=40):
    """mov를 (dy,dx)만큼 옮겼을 때 ref와 상관이 최대인 정수 이동. 반환 (corr, dy, dx)."""
    m = min(m, ref.shape[0] // 4)
    r = ref[m:-m, m:-m]; best = (-2.0, 0, 0)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            c = np.corrcoef(r.ravel(), mov[m - dy:mov.shape[0] - m - dy, m - dx:mov.shape[1] - m - dx].ravel())[0, 1]
            if np.isfinite(c) and c > best[0]:
                best = (float(c), dy, dx)
    return best


def register_frame(a, b, ht, prm: RegistrationParams):
    """한 프레임의 정합. 반환 (corr_FL_HT, dy_ht, dx_ht, corr_A_B, dy_B, dx_B, corr_A_B_noshift)."""
    NH = ht.shape[0]
    F = prep(resize(a.astype(np.float32), ht.shape, anti_aliasing=True)); H = prep(ht)
    c, dy, dx = best_shift(H, F, prm.fl_ht_search)
    Fs = np.roll(F, (dy, dx), (0, 1)); e = min(20, NH // 10)
    try:
        r, _, _ = phase_cross_correlation(H[e:-e, e:-e], Fs[e:-e, e:-e], upsample_factor=20)
        if np.all(np.abs(r) <= 1.5): dy += r[0]; dx += r[1]
    except Exception:
        pass
    cb, by, bx = best_shift(prep(a), prep(b), prm.ab_search)
    m = min(40, a.shape[0] // 4)
    c0 = np.corrcoef(prep(a)[m:-m, m:-m].ravel(), prep(b)[m:-m, m:-m].ravel())[0, 1]
    return c, dy, dx, cb, by, bx, c0


def register(A, B, HT, prm: RegistrationParams, progress=None) -> pd.DataFrame:
    """프레임별 이동량 표.
    - dy_ht, dx_ht : 형광 좌표 × 배율(S) + 이동 = HT 좌표
    - dy_B, dx_B   : np.roll(B, (dy_B, dx_B)) 하면 A와 정렬됨"""
    T = A.shape[0]; rows = []
    for z in range(T):
        rows.append((z + 1, *register_frame(A[z], B[z], HT[z], prm)))
        if progress: progress((z + 1) / T, f"정합 frame {z+1}/{T}")
    reg = pd.DataFrame(rows, columns=["frame", "corr_FL_HT", "dy_ht", "dx_ht", "corr_A_B", "dy_B", "dx_B", "corr_A_B_noshift"])
    reg["FL_HT_reliable"] = reg.corr_FL_HT >= prm.fl_ht_min_corr
    reg["A_B_reliable"] = reg.corr_A_B >= prm.ab_min_corr
    if reg.FL_HT_reliable.any():
        for c in ["dy_ht", "dx_ht"]: reg.loc[~reg.FL_HT_reliable, c] = reg.loc[reg.FL_HT_reliable, c].median()
    if reg.A_B_reliable.any():
        for c in ["dy_B", "dx_B"]: reg.loc[~reg.A_B_reliable, c] = np.round(reg.loc[reg.A_B_reliable, c].median())
    return reg


def align_b(B, reg) -> np.ndarray:
    sh = reg[["dy_B", "dx_B"]].values.astype(int)
    return np.stack([np.roll(B[z], (sh[z, 0], sh[z, 1]), (0, 1)) for z in range(B.shape[0])])


def fl_to_ht(y, x, frame, reg, S):
    """형광 좌표(y,x) → HT 좌표. frame은 1-based (배열 가능)."""
    i = np.asarray(frame) - 1
    return (np.asarray(y) + 0.5) * S - 0.5 + reg.dy_ht.values[i], (np.asarray(x) + 0.5) * S - 0.5 + reg.dx_ht.values[i]
