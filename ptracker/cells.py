"""3단계: HT 기반 세포 영역(세포질), 세포 ID 연결, 세포 움직임(optical flow),
퍼진 세포막 영역(footprint), 분열 세포 판정."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation, morphology
from skimage.transform import resize
from skimage.registration import optical_flow_tvl1
from .config import CellParams, FootprintParams, MitosisParams


def cell_body_images(HT, prm: CellParams):
    """입자를 opening으로 지우고 평활화한 세포 몸체 영상."""
    return np.stack([ndi.gaussian_filter(morphology.opening(h.astype(np.float32), morphology.disk(prm.open_radius)), prm.smooth_sigma)
                     for h in HT]).astype(np.float32)


def cell_thresholds(sample):
    """세포 영역(triangle)·세포 중심(Otsu) threshold. sample = 세포 몸체 영상의 4프레임 간격 표본."""
    return float(filters.threshold_triangle(sample.ravel())), float(filters.threshold_otsu(sample.ravel()))


def segment_frame(img, low, core_thr, prm: CellParams):
    """한 프레임 세포 몸체 영상 → 세포질 라벨 (ID 연결 전)."""
    m = morphology.remove_small_objects(img > low, max_size=prm.min_cell_area)
    m = ndi.binary_fill_holes(morphology.closing(m, morphology.disk(4)))
    core = morphology.remove_small_objects(ndi.binary_erosion(img > core_thr, morphology.disk(3)), max_size=800)
    lab = segmentation.watershed(-img, measure.label(core), mask=m)
    rest = measure.label(m & (lab == 0)); rest[rest > 0] += lab.max()
    rest = morphology.remove_small_objects(rest, max_size=prm.min_cell_area)
    return lab + rest


def segment_cells(cellimg, prm: CellParams, progress=None):
    """세포질 라벨 스택(프레임 간 ID 연결 포함)과 자동 threshold."""
    T = len(cellimg)
    low, core_thr = cell_thresholds(cellimg[::4])
    labs = []
    for z in range(T):
        labs.append(segment_frame(cellimg[z], low, core_thr, prm))
        if progress: progress((z + 1) / T * 0.5, f"세포 분할 frame {z+1}/{T}")
    CID = np.zeros(cellimg.shape, np.int32); nxt = 1
    for z in range(T):
        cur = labs[z]
        for r in measure.regionprops(cur):
            if z > 0:
                ov = CID[z - 1][cur == r.label]; ov = ov[ov > 0]
                if ov.size > prm.id_overlap * r.area:
                    v, cn = np.unique(ov, return_counts=True); CID[z][cur == r.label] = v[cn.argmax()]; continue
            CID[z][cur == r.label] = nxt; nxt += 1
    return CID, low, core_thr


def cell_flow(cellimg, prm: CellParams, progress=None):
    """연속 프레임 사이 세포 몸체의 이동 벡터장 (T-1, 2, H, W), 단위 HT px/프레임."""
    T, NH, NW = cellimg.shape; out = np.zeros((max(T - 1, 0), 2, NH, NW), np.float32)
    for z in range(T - 1):
        a = resize(cellimg[z], (NH // 2, NW // 2)); b = resize(cellimg[z + 1], (NH // 2, NW // 2))
        lo, hi = np.percentile(cellimg[z], [1, 99.5]); d = max(hi - lo, 1e-6)
        v, u = optical_flow_tvl1((a - lo) / d, (b - lo) / d, attachment=prm.flow_attachment)
        out[z, 0] = resize(v, (NH, NW)) * 2; out[z, 1] = resize(u, (NH, NW)) * 2
        if progress: progress(0.5 + (z + 1) / (T - 1) * 0.5, f"세포 흐름 {z+1}/{T-1}")
    return out


def footprint(HT, CID, prm: FootprintParams):
    """배경(완전히 어두운 배지) 밖 = 세포 발자국. 세포질 밖 부분이 '퍼진 세포막'."""
    T = len(HT); FP = np.zeros(HT.shape, bool); stats = []
    for z in range(T):
        FP[z], st = footprint_frame(HT[z], CID[z], prm); stats.append(dict(frame=z + 1, **st))
    return FP, pd.DataFrame(stats)


def footprint_frame(ht, cid, prm: FootprintParams):
    """한 프레임의 세포 발자국 마스크와 배경 통계."""
    o = ndi.gaussian_filter(morphology.opening(ht.astype(np.float32), morphology.disk(prm.open_radius)), prm.smooth_sigma)
    v = o.ravel(); mode = np.median(v[v < np.percentile(v, 60)]); low = v[v < mode]
    sig = float(np.sqrt(np.mean((low - mode) ** 2))) if low.size else float(np.std(v))
    fp = o > mode + prm.k_sigma * sig
    fp = morphology.closing(fp, morphology.disk(3)); fp = morphology.remove_small_objects(fp, max_size=prm.min_area)
    fp = ndi.binary_fill_holes(fp); fp = ~morphology.remove_small_objects(~fp, max_size=500)
    lab = measure.label(fp); cellm = ndi.binary_dilation(cid > 0, iterations=3); keep = np.zeros(lab.max() + 1, bool)
    for r in measure.regionprops(lab):
        if r.area >= prm.keep_large or cellm[tuple(r.coords.T)].any(): keep[r.label] = True
    FP = keep[lab] | (cid > 0)
    return FP, dict(bg_mode=float(mode), bg_sigma=sig, threshold=float(mode + prm.k_sigma * sig),
                    footprint_frac=FP.mean(), cytoplasm_frac=(cid > 0).mean(), spread_membrane_frac=(FP & (cid == 0)).mean())


def mitosis_table(HT, CID, prm: MitosisParams) -> pd.DataFrame:
    """세포·프레임별 면적/평균 RI와 분열 판정."""
    rows = []
    for z in range(len(HT)):
        for r in measure.regionprops(CID[z], intensity_image=HT[z].astype(float)):
            rows.append(dict(frame=z + 1, cell_id=r.label, area=r.area, meanRI=r.intensity_mean, ecc=r.eccentricity, solidity=r.solidity))
    CS = pd.DataFrame(rows, columns=["frame", "cell_id", "area", "meanRI", "ecc", "solidity"]).sort_values(["cell_id", "frame"])
    return classify_mitosis(CS, prm)


def classify_mitosis(CS, prm: MitosisParams) -> pd.DataFrame:
    """세포·프레임별 면적/평균 RI 표에 분열 판정 열을 붙입니다 (파라미터만 바꿔 다시 판정할 때도 사용)."""
    CS = CS[["frame", "cell_id", "area", "meanRI", "ecc", "solidity"]].copy(); g = CS.groupby("cell_id")
    CS["ref_RI"] = g.meanRI.transform(lambda x: x.shift(1).rolling(6, min_periods=2).median())
    CS["ref_area"] = g.area.transform(lambda x: x.shift(1).rolling(6, min_periods=2).median())
    CS["dRI"] = CS.meanRI - CS.ref_RI; CS["area_ratio"] = CS.area / CS.ref_area
    CS["mitotic"] = ((CS.dRI > prm.d_ri) & (CS.area_ratio < prm.area_ratio)) | ((CS.meanRI > prm.abs_ri) & (CS.ecc < prm.max_ecc))
    if not prm.enabled:
        CS["mitotic"] = False
    return CS
