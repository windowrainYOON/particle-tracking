"""작은 합성 데이터(세포 2개 + 입자, 드리프트 포함)를 만들어 파이프라인 동작을 점검합니다."""
import numpy as np, tifffile
from pathlib import Path
from scipy import ndimage as ndi


def make(folder, T=8, N=360, seed=0):
    rng = np.random.default_rng(seed); folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    NH = int(N * 0.72); yy, xx = np.mgrid[:NH, :NH]
    cells = [(NH * .35, NH * .35, 45, 30), (NH * .65, NH * .6, 50, 35)]
    parts = np.column_stack([rng.uniform(20, N - 20, 70), rng.uniform(20, N - 20, 70)])
    A = np.zeros((T, N, N), np.uint16); B = np.zeros((T, N, N), np.uint8); H = np.zeros((T, NH, NH), np.uint16)
    for t in range(T):
        pos = parts + rng.normal(0, .8, parts.shape) + np.array([t * .5, 0]); parts = pos
        img = np.zeros((N, N)); bimg = np.zeros((N, N)); hht = np.full((NH, NH), 13380.0)
        for (cy, cx, ry, rx) in cells:
            hht += 60 * np.exp(-(((yy - cy - t) / ry) ** 2 + ((xx - cx) / rx) ** 2))
        for i, (y, x) in enumerate(pos):
            yi, xi = int(y), int(x)
            if 3 <= yi < N - 3 and 3 <= xi < N - 3:
                img[yi, xi] += 400 + 50 * (i % 5); inside = hht[min(int(y * .72), NH - 1), min(int(x * .72), NH - 1)] > 13400
                bimg[yi, xi] += (8 if inside else 1) * (1 + t / T); hht[min(int(y * .72), NH - 1), min(int(x * .72), NH - 1)] += 300
        A[t] = np.clip(ndi.gaussian_filter(img, 1.5) * 9 + rng.poisson(3, (N, N)), 0, 65535)
        B[t] = np.clip(ndi.gaussian_filter(bimg, 1.5) * 9 + rng.poisson(1, (N, N)), 0, 255)
        H[t] = np.clip(ndi.gaussian_filter(hht, 1) + rng.normal(0, 2, (NH, NH)), 0, 65535)
    tifffile.imwrite(folder / "Cy5_syn.tif", A, imagej=True); tifffile.imwrite(folder / "pHrodo_syn.tif", B, imagej=True)
    tifffile.imwrite(folder / "HT_syn.tif", H, imagej=True)
    return folder


if __name__ == "__main__":
    make("synthetic_data")
