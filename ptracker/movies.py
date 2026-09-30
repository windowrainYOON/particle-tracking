"""영상 생성: 채널별 트랙 영상, 영역(세포질/퍼진 세포막/분열) 영상, 단일 입자 크롭 영상."""
from __future__ import annotations
import numpy as np, pandas as pd
from . import style
style.setup()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from .plots import track_trace, CNAME, GNAME, _disp

GCOL = {"inside": "lime", "outside": "red", "excluded_mitotic": "yellow"}


def _stamp(ax, text):
    ax.text(.01, .985, text, transform=ax.transAxes, va="top", color="yellow", fontsize=11, bbox=dict(fc="black", alpha=.6, ec="none"))


def channel_movie(stack, pts, path_noext, fl_coords, S, CID, reg, dt_min, title, fps=3, colors=None, cellmask_fp=None, CS=None, progress=None):
    """stack 위에 트랙(최근 5프레임 꼬리 + ROI 원) 표시. fl_coords=False면 HT 좌표."""
    writer, ext = style.movie_writer(fps)
    T, N = stack.shape[0], stack.shape[1]; NH = CID.shape[1]; X, Y = ("x", "y") if fl_coords else ("x_ht", "y_ht"); sc = 1 if fl_coords else S
    lo, hi = np.percentile(stack[T // 2], [.5, 99.8]); colors = colors or {}
    fig = plt.figure(figsize=(9, 9), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    alive = pts.groupby("frame").track_id.apply(set).to_dict()
    with writer.saving(fig, path_noext + ext, dpi=100):
        for f in range(1, T + 1):
            ax.clear(); ax.axis("off")
            if cellmask_fp is not None and not fl_coords:
                ov = np.dstack([np.clip((stack[f - 1].astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)] * 3)
                sm = cellmask_fp[f - 1] & (CID[f - 1] == 0); ov[sm] = ov[sm] * .5 + np.array([.1, .55, 1]) * .5; ax.imshow(ov)
            else:
                ax.imshow(stack[f - 1], cmap="gray", vmin=lo, vmax=hi)
            if fl_coords:
                xs = (np.arange(NH) - reg.dx_ht.values[f - 1] + .5) / S - .5; ys = (np.arange(NH) - reg.dy_ht.values[f - 1] + .5) / S - .5
                ax.contour(xs, ys, CID[f - 1] > 0, [.5], colors="w", linewidths=.5)
            else:
                ax.contour(CID[f - 1] > 0, [.5], colors="r" if cellmask_fp is not None else "w", linewidths=.6)
                if CS is not None:
                    for c_ in CS[(CS.frame == f) & CS.mitotic].cell_id: ax.contour(CID[f - 1] == c_, [.5], colors="yellow", linewidths=2)
            cur = pts[pts.frame == f]; past = pts[(pts.frame <= f) & (pts.frame >= f - 5) & pts.track_id.isin(alive.get(f, set()))]
            segs, cs = [], []
            for t_, t in past.groupby("track_id"):
                if len(t) > 1: segs.append(t.sort_values("frame")[[X, Y]].values); cs.append(colors.get(t_, GCOL.get(t.group.iloc[0], "cyan")))
            ax.add_collection(LineCollection(segs, colors=cs, linewidths=1.1))
            ax.add_collection(PatchCollection([Circle((r[0], r[1]), max(r[2] * sc, 1.5)) for r in cur[[X, Y, "radius_equiv_px"]].values],
                                              facecolor="none", edgecolor=[colors.get(t, GCOL.get(g, "cyan")) for t, g in zip(cur.track_id, cur.group)], linewidths=.9))
            ax.set_xlim(0, N - 1); ax.set_ylim(N - 1, 0)
            tm = (f - 1) * dt_min; _stamp(ax, f"{title} | frame {f} | t = {int(tm//60)} h {int(tm%60):02d} min | n = {len(cur)}")
            writer.grab_frame()
            if progress: progress(f / T, f"영상 {title} {f}/{T}")
    plt.close(fig)
    return path_noext + ext


def crop_movie(A, Bal, HT, t, S, path_noext, info="", r=None, half=32, fps=3):
    """입자를 따라가는 Cy5 / pHrodo / HT 크롭 + 비율 곡선(현재 시점 표시)."""
    writer, ext = style.movie_writer(fps, 1800)
    t = t.sort_values("frame").set_index("frame"); fr = np.arange(t.index.min(), t.index.max() + 1)
    pos = t[["x_fl", "y_fl", "x_ht", "y_ht"]].reindex(fr).interpolate(); hh = max(int(half * S), 4)
    hlo, hhi = np.percentile(HT[len(HT) // 2], [.5, 99.8])

    def crop(im, x, y, h):
        xi, yi = int(round(x)), int(round(y)); p = np.pad(im, h); return p[yi:yi + 2 * h, xi:xi + 2 * h]
    cA = [crop(A[f - 1], *pos.loc[f, ["x_fl", "y_fl"]], half) for f in fr]; cB = [crop(Bal[f - 1], *pos.loc[f, ["x_fl", "y_fl"]], half) for f in fr]
    va = np.percentile(np.stack(cA), 99.8); vb = max(np.percentile(np.stack(cB), 99.8), 3)
    fig = plt.figure(figsize=(10, 7.4), dpi=80); axs = [fig.add_axes([.02 + i * .33, .42, .3, .5]) for i in range(3)]; ax = fig.add_axes([.08, .07, .86, .28])
    tt = t.reset_index(); dt_h = (tt.time_min.diff() / tt.frame.diff()).median() / 60 if len(tt) > 1 else 1 / 3
    track_trace(ax, tt, r); vl = ax.axvline((fr[0] - 1) * dt_h, color="k", lw=1.5)
    with writer.saving(fig, path_noext + ext, 80):
        for i, f in enumerate(fr):
            x, y, xh, yh = pos.loc[f]; det = f in t.index
            for a, im, vmax, nm, rad, s_ in [(axs[0], cA[i], va, "Cy5", half, 1), (axs[1], cB[i], vb, "pHrodo", half, 1), (axs[2], crop(HT[f - 1], xh, yh, hh), None, "HT", hh, S)]:
                a.clear(); a.axis("off"); a.imshow(im, cmap="gray", vmin=hlo if vmax is None else 0, vmax=hhi if vmax is None else vmax)
                rr = (t.loc[f, "radius_equiv_px"] if det else 4) * s_
                a.add_patch(Circle((rad - .5, rad - .5), rr + 2 * s_, fill=False, ec="lime" if det else "yellow", lw=1.3, ls="-" if det else "--")); a.set_title(nm, fontsize=11)
            vl.set_xdata([(f - 1) * dt_h] * 2); reg_ = t.loc[f, "region"] if (det and "region" in t) else "-"
            fig.suptitle(f"{info} | frame {f} | t = {(f-1)*dt_h:.2f} h | 영역: {reg_}" + ("" if det else " (미검출, 위치 보간)"), fontsize=11)
            writer.grab_frame()
    plt.close(fig)
    return path_noext + ext
