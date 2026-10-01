"""5단계: 같은 ROI로 Cy5(A) / pHrodo(B) / HT 측정, 트랙 요약, 세포 안·밖·분열 분류."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from .config import MeasurementParams


def measure_rois(tr, A, Bal, HT, LAB, reg, S, prm: MeasurementParams, progress=None):
    """ROI 평균 − 배경 고리 중앙값. HT는 ROI를 HT 좌표로 옮겨 평균/최대 RI."""
    T = A.shape[0]; rec = {}
    for z in range(T):
        for k, v in measure_frame(LAB[z], A[z], Bal[z], HT[z], reg.dy_ht.values[z], reg.dx_ht.values[z], S, prm).items(): rec[(z + 1, k)] = v
        if progress: progress((z + 1) / T, f"ROI 측정 {z+1}/{T}")
    M = pd.DataFrame([(k[0], k[1], *v) for k, v in rec.items()],
                     columns=["frame", "label", "I_A_raw", "I_B_raw", "I_B_max", "bg_A", "bg_B", "HT_RI_mean", "HT_RI_max"])
    tr = tr.merge(M, on=["frame", "label"], how="left")
    tr["I_A"] = tr.I_A_raw - tr.bg_A; tr["I_B"] = tr.I_B_raw - tr.bg_B
    tr["ratio_BA"] = np.where(tr.I_A > prm.min_ia, tr.I_B / tr.I_A, np.nan)
    tr["ratio_BA_raw"] = tr.I_B_raw / tr.I_A_raw
    return tr


def measure_frame(lab, a_img, b_img, ht, dy_ht, dx_ht, S, prm: MeasurementParams, labels=None, return_masks=False):
    """한 프레임의 ROI별 (I_A_raw, I_B_raw, I_B_max, bg_A, bg_B, HT_RI_mean, HT_RI_max).
    labels: 계산할 ROI 번호(없으면 전부). return_masks면 {label: (y0, x0, roi, ring)}도 반환 (미리보기용)."""
    NF, NW = lab.shape; NH, NHW = ht.shape
    se_i = np.ones((2 * prm.ring_in + 1,) * 2, bool); se_o = np.ones((2 * prm.ring_out + 1,) * 2, bool); rec = {}; masks = {}
    pad = prm.ring_out + 2; occ = ndi.binary_dilation(lab > 0, iterations=2); want = None if labels is None else set(labels)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None or (want is not None and k not in want): continue
        y0 = max(sl[0].start - pad, 0); y1 = min(sl[0].stop + pad, NF); x0 = max(sl[1].start - pad, 0); x1 = min(sl[1].stop + pad, NW)
        m = lab[y0:y1, x0:x1] == k
        ring = ndi.binary_dilation(m, se_o) & ~ndi.binary_dilation(m, se_i) & ~occ[y0:y1, x0:x1]
        a = a_img[y0:y1, x0:x1].astype(float); b = b_img[y0:y1, x0:x1].astype(float); ok = ring.sum() > 10
        yy, xx = np.nonzero(m)
        hy = np.clip(np.round((yy + y0 + 0.5) * S - 0.5 + dy_ht).astype(int), 0, NH - 1)
        hx = np.clip(np.round((xx + x0 + 0.5) * S - 0.5 + dx_ht).astype(int), 0, NHW - 1)
        hv = ht[hy, hx].astype(float) / 10000
        rec[k] = (a[m].mean(), b[m].mean(), b[m].max(), np.median(a[ring]) if ok else np.nan,
                  np.median(b[ring]) if ok else np.nan, hv.mean(), hv.max())
        if return_masks: masks[k] = (y0, x0, m, ring)
    return (rec, masks) if return_masks else rec


def flag_merged(tr, prm: MeasurementParams):
    """다른 입자와 합쳐지거나 가려진 시점 표시: 면적과 Cy5 총 밝기가 그 트랙의 직전 3시점 중앙값보다 함께 급증.
    exclude_merged면 그 시점의 비율을 비웁니다 (원래 값은 ratio_BA_unfiltered)."""
    tr = tr.sort_values(["track_id", "frame"]).copy(); g = tr.groupby("track_id")
    integ = tr.area_px * tr.mean_intensity
    ref_a = g.area_px.transform(lambda x: x.shift(1).rolling(3, min_periods=1).median())
    ref_i = integ.groupby(tr.track_id).transform(lambda x: x.shift(1).rolling(3, min_periods=1).median())
    raw = ((tr.area_px >= prm.merge_area_ratio * ref_a) & (integ >= prm.merge_int_ratio * ref_i)).fillna(False).values
    # 급증 뒤에도 면적이 (급증 전 기준 × 배수) 이상이면 최대 merge_max_frames까지 이어서 표시
    m = raw.copy(); tid = tr.track_id.values; area = tr.area_px.values; base = ref_a.values; i = 0; n = len(tr)
    while i < n:
        if raw[i]:
            b = base[i]; j = i + 1; k = 1
            while j < n and tid[j] == tid[i] and k < prm.merge_max_frames and area[j] >= prm.merge_area_ratio * b: m[j] = True; j += 1; k += 1
            i = j
        else: i += 1
    tr["merged"] = m
    tr["ratio_BA_unfiltered"] = tr.ratio_BA
    if prm.exclude_merged: tr.loc[tr.merged, "ratio_BA"] = np.nan
    return tr


def track_summary(tr, prm: MeasurementParams, CS, dt_min, exclude_mitotic=True):
    mit_cells = set(CS[CS.mitotic].cell_id)

    def one(t):
        t = t.sort_values("frame"); dy = t.y_ht.iloc[-1] - t.y_ht.iloc[0]; dx = t.x_ht.iloc[-1] - t.x_ht.iloc[0]
        path = t.step_ht.iloc[1:].sum(); dur = (t.frame.iloc[-1] - t.frame.iloc[0]) * dt_min; cells = t.cell_id[t.cell_id > 0]
        r = t.dropna(subset=["ratio_BA"]); sl = np.polyfit(r.time_min / 60, r.ratio_BA, 1)[0] if len(r) >= 3 else np.nan
        return pd.Series(dict(start_frame=t.frame.iloc[0], end_frame=t.frame.iloc[-1], n_points=len(t), gaps=int((t.frame.diff() > 1).sum()),
            duration_min=dur, main_cell_id=int(cells.mode().iloc[0]) if len(cells) else 0, frac_in_cell=t.inside_cell.mean(),
            frac_cytoplasm=(t.region == "cytoplasm").mean(), frac_spread_membrane=(t.region == "spread_membrane").mean(),
            frac_background=(t.region == "background").mean(), path_length_ht=path, net_disp_ht=np.hypot(dx, dy),
            straightness=np.hypot(dx, dy) / path if path > 0 else np.nan, mean_speed_htpx_per_h=path / dur * 60 if dur > 0 else np.nan,
            mean_cell_carried_ht=t.cell_carried_ht.mean(), mean_residual_ht=t.residual_ht.mean(), mean_radius_px=t.radius_equiv_px.mean(),
            mean_I_A=t.I_A.mean(), mean_I_B=t.I_B.mean(), mean_ratio_BA=r.ratio_BA.mean(),
            ratio_first3=r.ratio_BA.iloc[:3].mean(), ratio_last3=r.ratio_BA.iloc[-3:].mean(), ratio_slope_per_h=sl,
            mean_HT_RI=t.HT_RI_mean.mean(), ht_confirmed_frac=t.ht_confirmed.mean()))
    TS = tr.groupby("track_id").apply(one).reset_index()
    TS["group"] = np.where(TS.n_points < prm.min_track, "short", np.where(TS.frac_in_cell > prm.inside_fraction, "inside", "outside"))
    TS["mitotic_cell"] = TS.main_cell_id.isin(mit_cells) & (TS.group == "inside")
    if exclude_mitotic:
        TS.loc[TS.mitotic_cell, "group"] = "excluded_mitotic"
    return TS


def per_cell(tr, CID, dt_min):
    T = CID.shape[0]
    area = pd.DataFrame([(c, z + 1, int((CID[z] == c).sum())) for z in range(T) for c in np.unique(CID[z]) if c > 0],
                        columns=["cell_id", "frame", "cell_area_htpx"])
    pc = tr[tr.cell_id > 0].groupby(["cell_id", "frame"]).agg(n_particles=("track_id", "size"), median_ratio_BA=("ratio_BA", "median"),
                                                              sum_I_A=("I_A", "sum"), sum_I_B=("I_B", "sum")).reset_index()
    pc = area.merge(pc, how="left"); pc["time_min"] = (pc.frame - 1) * dt_min
    return pc


POINT_COLUMNS = ["track_id", "group", "frame", "time_min", "label", "x", "y", "x_ht", "y_ht", "cell_id", "inside_cell", "region",
                 "on_footprint", "in_mitotic_cell", "ht_confirmed", "pat_score", "pat2_score", "pat_refined", "gate_used", "area_px", "radius_equiv_px", "eccentricity",
                 "I_A_raw", "bg_A", "I_A", "I_B_raw", "bg_B", "I_B", "I_B_max", "ratio_BA", "ratio_BA_raw", "merged", "ratio_BA_unfiltered", "HT_RI_mean", "HT_RI_max",
                 "step_ht", "cell_carried_ht", "residual_ht", "link_margin", "link_cost"]


def points_table(tr):
    cols = [c for c in POINT_COLUMNS if c in tr.columns]
    return tr[cols].rename(columns={"x": "x_fl", "y": "y_fl", "label": "roi_label"})
