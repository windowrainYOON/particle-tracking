"""1단계: Cy5(A) 채널 입자 검출 – 배경 제거 → 적응형 threshold → 씨앗 → watershed."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation, feature, morphology
from .config import DetectionParams


def background_subtracted(raw: np.ndarray, prm: DetectionParams) -> np.ndarray:
    raw = raw.astype(np.float32)
    return ndi.gaussian_filter(raw, prm.sigma_smooth) - ndi.gaussian_filter(raw, prm.sigma_bg)


def detect_frame(raw: np.ndarray, prm: DetectionParams):
    """한 프레임 → (라벨 영상, threshold)."""
    sm = background_subtracted(raw, prm)
    med = np.median(sm); mad = np.median(np.abs(sm - med)) * 1.4826
    thr = max(med + prm.k_mad * mad, filters.threshold_triangle(sm))
    mask = morphology.remove_small_objects(sm > thr, max_size=prm.max_small)
    pk = feature.peak_local_max(sm, min_distance=prm.min_distance, threshold_abs=thr, labels=measure.label(mask))
    mk = np.zeros(raw.shape, np.int32); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(-sm, mk, mask=mask)
    lab, _, _ = segmentation.relabel_sequential(lab)
    return lab, float(thr)


def detect_stack(A: np.ndarray, prm: DetectionParams, progress=None):
    """전체 스택 → (라벨 스택, 입자 표, 프레임 요약)."""
    T = A.shape[0]
    LAB = np.zeros(A.shape, np.uint32)
    rows, summ = [], []
    for z in range(T):
        lab, thr = detect_frame(A[z], prm); LAB[z] = lab
        p = pd.DataFrame(measure.regionprops_table(lab, intensity_image=A[z].astype(float), properties=(
            "label", "centroid", "area", "equivalent_diameter_area", "perimeter", "eccentricity",
            "intensity_mean", "intensity_max", "intensity_min")))
        p = p.rename(columns={"centroid-0": "y", "centroid-1": "x", "area": "area_px", "intensity_mean": "mean_intensity",
                              "intensity_max": "max_intensity", "intensity_min": "min_intensity"})
        p["radius_equiv_px"] = p.pop("equivalent_diameter_area") / 2
        p["integrated_intensity"] = p.area_px * p.mean_intensity
        p.insert(0, "frame", z + 1); rows.append(p)
        summ.append(dict(frame=z + 1, threshold=thr, n_particles=len(p), median_area_px=p.area_px.median(),
                         median_radius_px=p.radius_equiv_px.median(), median_mean_intensity=p.mean_intensity.median()))
        if progress: progress((z + 1) / T, f"검출 frame {z+1}/{T}: {len(p)}개")
    if LAB.max() < 65535: LAB = LAB.astype(np.uint16)
    return LAB, pd.concat(rows, ignore_index=True), pd.DataFrame(summ)
