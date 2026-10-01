"""앱 아이콘 만들기: assets/icon_1024.png, assets/ParticleTracker.icns (macOS iconutil 필요).
모티프: 어두운 남색 바탕의 둥근 사각형 안에 세포(반투명 막), 그 안을 지나는 입자 궤적
(Cy5 자홍 → pHrodo 주황으로 바뀌는 색 = 산성화), 끝의 입자는 빛나는 점."""
import subprocess, shutil, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Circle
from matplotlib.path import Path as MPath
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[1] / "assets"
N = 1024


def draw(path_png):
    fig = plt.figure(figsize=(N / 100, N / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_alpha(0)
    # macOS 아이콘 격자: 1024 캔버스에 824 둥근 사각형 (여백 100)
    m = 100 / N; box = FancyBboxPatch((m, m), 1 - 2 * m, 1 - 2 * m, boxstyle="round,pad=0,rounding_size=0.18", lw=0, fc="none")
    ax.add_patch(box)
    yy, xx = np.mgrid[0:1:512j, 0:1:512j]
    bg = LinearSegmentedColormap.from_list("bg", ["#0b1030", "#14245e", "#0c5b6e"])((0.65 * yy + 0.35 * (1 - xx)))
    im = ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", interpolation="bicubic"); im.set_clip_path(box)
    # 세포: 부드러운 막 (여러 겹의 반투명 윤곽)
    t = np.linspace(0, 2 * np.pi, 400); r = 0.30 + 0.025 * np.sin(3 * t + .6) + 0.018 * np.cos(5 * t)
    cx, cy = 0.47, 0.50; xs, ys = cx + 1.08 * r * np.cos(t), cy + 0.92 * r * np.sin(t)
    cell = PathPatch(MPath(np.c_[xs, ys]), fc="#5ec8e6", alpha=0.10, lw=0); ax.add_patch(cell)
    for k, a in [(6, .06), (3.5, .12), (1.6, .45)]:
        ax.plot(xs, ys, color="#8fe3f5", lw=k, alpha=a, solid_capstyle="round")
    # 핵
    ax.add_patch(Circle((0.40, 0.44), 0.085, fc="#7fd3ea", alpha=0.10, lw=0)); ax.plot(0.40 + .085 * np.cos(t), 0.44 + .085 * np.sin(t), color="#9be6f7", lw=1.2, alpha=.35)
    # 주변의 흐린 입자들
    rng = np.random.default_rng(3)
    for _ in range(18):
        px, py = rng.uniform(.18, .82), rng.uniform(.18, .82); s = rng.uniform(.004, .009)
        ax.add_patch(Circle((px, py), s, fc="#c9b8ff", alpha=.35, lw=0))
    # 입자 궤적: 부드러운 곡선, 색이 자홍(Cy5) → 주황(pHrodo)
    u = np.linspace(0, 1, 300)                                       # 3차 베지어: 부드러운 S자
    P0, P1, P2, P3 = np.array([.22, .27]), np.array([.30, .62]), np.array([.62, .30]), np.array([.76, .70])
    pts = ((1 - u) ** 3)[:, None] * P0 + (3 * (1 - u) ** 2 * u)[:, None] * P1 + (3 * (1 - u) * u ** 2)[:, None] * P2 + (u ** 3)[:, None] * P3
    segs = np.stack([pts[:-1], pts[1:]], 1)
    cm = LinearSegmentedColormap.from_list("tr", ["#e040fb", "#ff4f8b", "#ff9a3c"])
    for lw, a in [(26, .07), (16, .12), (7, 1.0)]:
        lc = LineCollection(segs, colors=cm(u[:-1]), linewidths=lw, alpha=a, capstyle="round"); ax.add_collection(lc)
    for k in np.linspace(0, len(u) - 40, 6).astype(int):          # 지나온 시점 표시 점
        ax.add_patch(Circle(pts[k], 0.012, fc="white", alpha=.9, lw=0)); ax.add_patch(Circle(pts[k], 0.012, fc=cm(u[k]), alpha=.6, lw=0))
    # 끝의 빛나는 입자
    ex, ey = pts[-1]
    for rr, a in [(.075, .08), (.05, .15), (.033, .35), (.022, 1.0)]:
        ax.add_patch(Circle((ex, ey), rr, fc="#ffb15c" if rr > .022 else "#fff3e0", alpha=a, lw=0))
    # 윗부분 은은한 광택
    gl = np.clip(1.6 * (yy - .55), 0, 1) ** 2 * .10; sheen = np.zeros(yy.shape + (4,)); sheen[..., :3] = 1; sheen[..., 3] = gl
    im2 = ax.imshow(sheen, extent=(0, 1, 0, 1), origin="lower"); im2.set_clip_path(box)
    for art in ax.patches[1:] + ax.lines + ax.collections: art.set_clip_path(box)     # 모두 둥근 사각형 안으로
    fig.savefig(path_png, dpi=100, transparent=True); plt.close(fig)


def icns(png, out):
    from PIL import Image
    tmp = Path(tempfile.mkdtemp()) / "icon.iconset"; tmp.mkdir(); im = Image.open(png).convert("RGBA")
    for s in (16, 32, 128, 256, 512):
        im.resize((s, s), Image.LANCZOS).save(tmp / f"icon_{s}x{s}.png"); im.resize((2 * s, 2 * s), Image.LANCZOS).save(tmp / f"icon_{s}x{s}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(tmp), "-o", str(out)], check=True); shutil.rmtree(tmp.parent)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); png = OUT / "icon_1024.png"; draw(png); icns(png, OUT / "ParticleTracker.icns"); print("saved", png, OUT / "ParticleTracker.icns")
