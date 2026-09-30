"""그래프 생성 함수 모음. 모든 함수는 (데이터, 저장 경로)를 받아 PNG를 저장합니다."""
from __future__ import annotations
import numpy as np, pandas as pd
from . import style
style.setup()
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

GNAME = {"inside": "세포 안", "outside": "세포 밖"}
CNAME = {"abrupt_drop": "급격한 감소", "gradual_decline": "장기적 감소"}
DS_COLORS = plt.get_cmap("tab10")


def _save(fig, path, dpi=110):
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def _disp(im, p=(0.5, 99.8)):
    lo, hi = np.percentile(im, p); return np.clip((im.astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)


def _track_colors(ids, seed=0):
    rng = np.random.default_rng(seed); cm = plt.get_cmap("hsv"); return {t: cm(rng.random()) for t in ids}


def _spaghetti(ax, g, var, ylab, title, col, ylim=None, idcol="track_id"):
    for t_, t in g.groupby(idcol):
        t = t.sort_values("frame"); ax.plot(t.t_h, t[var], "-", lw=.6, alpha=.35, color=col.get(t_, "gray"))
    q = g.groupby("t_h")[var].quantile([.25, .5, .75]).unstack()
    if len(q):
        ax.fill_between(q.index, q[.25], q[.75], color="k", alpha=.15, label="IQR"); ax.plot(q.index, q[.5], "k-", lw=2.5, label="중앙값")
    ax.set_xlabel("시간 (h)"); ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.3); ax.legend(loc="upper left")
    if ylim is not None and np.all(np.isfinite(ylim)): ax.set_ylim(*ylim)


def ratio_figures(pts, out_dir, prefix="", idcol="track_id"):
    """세포 안·밖 비율 비교 + 그룹별 입자 곡선. pts는 group∈{inside,outside} 장기 트랙."""
    d = pts[pts.group.isin(["inside", "outside"])].copy()
    if d.empty: return
    d["t_h"] = d.time_min / 60; col = _track_colors(d[idcol].unique())
    yR = (np.nanpercentile(d.ratio_BA, .5), np.nanpercentile(d.ratio_BA, 99.5))
    for g in ["inside", "outside"]:
        s = d[d.group == g]
        if s.empty: continue
        fig, ax = plt.subplots(figsize=(11, 6.5))
        _spaghetti(ax, s, "ratio_BA", "I(pHrodo)/I(Cy5)", f"{GNAME[g]} 입자: I(pHrodo)/I(Cy5) ({s[idcol].nunique()} 트랙)", col, yR, idcol)
        _save(fig, f"{out_dir}/{prefix}ratio_timecourse_{g}.png")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for g, c in [("inside", "tab:red"), ("outside", "tab:blue")]:
        q = d[d.group == g].groupby("t_h").ratio_BA.quantile([.25, .5, .75]).unstack()
        if len(q):
            ax.fill_between(q.index, q[.25], q[.75], color=c, alpha=.2)
            ax.plot(q.index, q[.5], "-", color=c, lw=2.5, label=f"{GNAME[g]} (n={d[d.group==g][idcol].nunique()}, 중앙값 ± IQR)")
    ax.set_xlabel("시간 (h)"); ax.set_ylabel("I(pHrodo)/I(Cy5)"); ax.set_title("I(pHrodo)/I(Cy5): 세포 안 vs 세포 밖"); ax.grid(alpha=.3); ax.legend()
    _save(fig, f"{out_dir}/{prefix}ratio_inside_vs_outside.png")
    fig, axs = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
    for j, g in enumerate(["inside", "outside"]):
        s = d[d.group == g]
        if s.empty: continue
        _spaghetti(axs[0, j], s, "I_A", "I(Cy5) (배경 제거)", f"{GNAME[g]}: I(Cy5)", col, (0, np.nanpercentile(d.I_A, 99.7)), idcol)
        _spaghetti(axs[1, j], s, "I_B", "I(pHrodo) (배경 제거)", f"{GNAME[g]}: I(pHrodo)", col, (np.nanpercentile(d.I_B, .3), np.nanpercentile(d.I_B, 99.7)), idcol)
        _spaghetti(axs[2, j], s, "HT_RI_mean", "ROI 평균 RI", f"{GNAME[g]}: HT 굴절률", col, (np.nanpercentile(d.HT_RI_mean, .3), np.nanpercentile(d.HT_RI_mean, 99.7)), idcol)
    _save(fig, f"{out_dir}/{prefix}channel_timecourses.png", 90)


def per_dataset_medians(pts, path):
    d = pts[pts.group.isin(["inside", "outside"])].copy(); d["t_h"] = d.time_min / 60
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, ds in enumerate(d.dataset.unique()):
        for g, ls in [("inside", "-"), ("outside", "--")]:
            m = d[(d.dataset == ds) & (d.group == g)].groupby("t_h").ratio_BA.median(); ax.plot(m.index, m.values, ls, color=DS_COLORS(i), lw=1.8, label=f"{ds} {GNAME[g]}")
    ax.set_xlabel("시간 (h)"); ax.set_ylabel("I(pHrodo)/I(Cy5) 중앙값"); ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8); ax.set_title("데이터셋별 중앙값 (실선 = 세포 안, 점선 = 세포 밖)")
    _save(fig, path)


def qc_detection_cells(A, P, HT, CID, path):
    T = A.shape[0]; frs = sorted(set([1, max(1, T // 4), max(1, T // 2), T]))
    fig, ax = plt.subplots(2, len(frs), figsize=(5 * len(frs), 10.5), squeeze=False)
    for i, z in enumerate(frs):
        ax[0, i].imshow(_disp(A[z - 1], (1, 99.8)), cmap="gray"); q = P[P.frame == z]
        ax[0, i].plot(q.x, q.y, "r+", ms=3, mew=.5); ax[0, i].set_title(f"Cy5 frame {z}: {len(q)} 입자"); ax[0, i].axis("off")
        ax[1, i].imshow(_disp(HT[z - 1]), cmap="gray"); ax[1, i].imshow(np.ma.masked_equal(CID[z - 1], 0) % 20, cmap="tab20", alpha=.3, vmin=0, vmax=19)
        ax[1, i].contour(CID[z - 1] > 0, [.5], colors="r", linewidths=.6); ax[1, i].set_title(f"HT frame {z}: 세포"); ax[1, i].axis("off")
    _save(fig, path, 70)


def tracks_overlay(HT, CID, pts, path, idcol="track_id"):
    fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(_disp(HT[-1]), cmap="gray"); ax.contour(CID[-1] > 0, [.5], colors="w", linewidths=.5)
    segs, cs = [], []
    for t_, t in pts.groupby(idcol):
        t = t.sort_values("frame"); xy = t[["x_ht", "y_ht"]].values; segs += [xy[i:i + 2] for i in range(len(xy) - 1)]; cs += list(t.frame.values[1:])
    if segs:
        lc = LineCollection(segs, cmap="plasma", linewidths=1); lc.set_array(np.array(cs)); lc.set_clim(1, HT.shape[0]); ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=.03).set_label("frame")
    ax.axis("off"); ax.set_title(f"장기 트랙 {pts[idcol].nunique()}개 (색 = 프레임)"); _save(fig, path, 90)


def regions_figure(HT, CID, FP, CS, path):
    T = HT.shape[0]; frs = sorted(set([1, max(1, T // 4), max(1, T // 2), T])); fig, axs = plt.subplots(1, len(frs), figsize=(6 * len(frs), 6.5), squeeze=False)
    for a, f in zip(axs[0], frs):
        ov = np.dstack([_disp(HT[f - 1], (.5, 99.7))] * 3); sm = FP[f - 1] & (CID[f - 1] == 0); ov[sm] = ov[sm] * .45 + np.array([.1, .55, 1]) * .55
        a.imshow(ov); a.contour(CID[f - 1] > 0, [.5], colors="r", linewidths=.8)
        for c_ in CS[(CS.frame == f) & CS.mitotic].cell_id: a.contour(CID[f - 1] == c_, [.5], colors="yellow", linewidths=2)
        a.set_title(f"frame {f}: 세포질(빨강) / 퍼진 세포막(파랑) / 분열(노랑)"); a.axis("off")
    _save(fig, path, 75)


def mitosis_figure(CS, path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5)); mc = set(CS[CS.mitotic].cell_id)
    for c_ in CS.cell_id.unique():
        s = CS[CS.cell_id == c_]
        if len(s) < 5: continue
        lw, al = (2.5, 1) if c_ in mc else (1, .4)
        axs[0].plot(s.frame, s.meanRI / 1e4, lw=lw, alpha=al, label=f"cell {c_}" + (" (분열)" if c_ in mc else "")); axs[1].plot(s.frame, s.area, lw=lw, alpha=al)
        m = s[s.mitotic]; axs[0].plot(m.frame, m.meanRI / 1e4, "r*", ms=12); axs[1].plot(m.frame, m.area, "r*", ms=12)
    axs[0].set_ylabel("세포 평균 RI"); axs[1].set_ylabel("세포 면적 (HT px)"); [a.set_xlabel("frame") for a in axs]; axs[0].legend(fontsize=7)
    axs[0].set_title("세포별 평균 굴절률 (빨간 별 = 분열 판정)"); axs[1].set_title("세포별 면적"); _save(fig, path, 85)


# ---------------------------------------------------------------- 특이 입자
def risefall_fraction(F, path):
    ds = list(F.dataset.unique()); x = np.arange(len(ds)); w = .2; fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(ds)), 5.8))
    for i, (g, col) in enumerate([("inside", "tab:purple"), ("outside", "tab:blue")]):
        t = F[F.group == g].set_index("dataset").reindex(ds)
        ax.bar(x + (2 * i - 1.5) * w, t.candidate_pct, w, color=col, alpha=.3, label=f"{GNAME[g]} — 필터 전 후보")
        ax.bar(x + (2 * i - .5) * w, t.kept_pct, w, color=col, label=f"{GNAME[g]} — 최종 선별")
        for xi, (p, k, n) in enumerate(zip(t.kept_pct, t.kept, t.n_tracks)):
            if np.isfinite(p): ax.text(xi + (2 * i - .5) * w, p + .1, f"{p:.1f}%\n({int(k)}/{int(n)})", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(ds); ax.set_ylabel("트랙 비율 (%)"); ax.legend(ncol=2, fontsize=9)
    ax.set_title("I(pHrodo)/I(Cy5)가 한 번 증가 후 감소·유지된 입자 비율"); _save(fig, path, 100)


def track_trace(ax, t, r=None, show_channels=False):
    """한 트랙의 비율 곡선 (+선택: I(A), I(B))."""
    tv = t.dropna(subset=["ratio_BA"]).sort_values("frame"); th = tv.time_min / 60
    ax.plot(th, tv.ratio_BA, "o-", color="tab:purple", ms=3, lw=.8, alpha=.6, label="I(pHrodo)/I(Cy5)")
    ax.plot(th, tv.ratio_BA.rolling(3, center=True, min_periods=1).median(), "k-", lw=1.8, label="3점 중앙값")
    if r is not None and np.isfinite(getattr(r, "peak_time_h", np.nan)): ax.axvline(r.peak_time_h, color="green", ls=":", lw=1.2)
    if r is not None and np.isfinite(getattr(r, "decline_time_h", np.nan)): ax.axvline(r.decline_time_h, color="red", ls=":", lw=1.2)
    ax.axhline(0, color="gray", lw=.6); ax.set_xlabel("시간 (h)"); ax.set_ylabel("I(pHrodo)/I(Cy5)"); ax.grid(alpha=.3)
    if show_channels:
        a2 = ax.twinx(); a2.plot(t.time_min / 60, t.I_A, "-", color="tab:red", alpha=.4, lw=1, label="I(Cy5)"); a2.plot(t.time_min / 60, t.I_B * 10, "-", color="tab:orange", alpha=.4, lw=1, label="I(pHrodo)×10")
        a2.set_ylabel("intensity"); a2.legend(loc="upper right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)


def risefall_multiples(R, pts, path, title):
    n = len(R)
    if n == 0: return
    nc = 6; nr = int(np.ceil(n / nc)); fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 2.4), sharex=True, squeeze=False)
    for a, r in zip(axs.flat, R.itertuples()):
        t = pts[pts.track_uid == r.track_uid].sort_values("frame").dropna(subset=["ratio_BA"])
        a.plot(t.time_min / 60, t.ratio_BA, "o-", color="tab:purple" if r.category == "abrupt_drop" else "tab:olive", ms=2, lw=.7, alpha=.6)
        a.plot(t.time_min / 60, t.ratio_BA.rolling(3, center=True, min_periods=1).median(), "k-", lw=1.4)
        a.axvline(r.peak_time_h, color="green", ls=":", lw=1)
        if np.isfinite(r.decline_time_h): a.axvline(r.decline_time_h, color="red", ls=":", lw=1)
        a.set_title(f"{r.track_uid} {CNAME.get(r.category,'')}" + ("" if r.keep else f"\n[제외] {r.filter_result}"), fontsize=7, color="k" if r.keep else "firebrick")
        a.tick_params(labelsize=7)
    for a in axs.flat[n:]: a.axis("off")
    fig.suptitle(title, fontsize=12); plt.tight_layout(rect=[0, 0, 1, .97]); _save(fig, path, 80)


def risefall_aligned(R, pts, path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for a, col, lab in [(axs[0], "peak_frame", "최고점"), (axs[1], "decline_frame", "감소 시점")]:
        for g, cl in [("inside", "tab:purple"), ("outside", "tab:blue")]:
            s = R[(R.group == g) & R.keep]; rows = []
            for r in s.itertuples():
                t = pts[pts.track_uid == r.track_uid].dropna(subset=["ratio_BA"]); dt_h = (t.time_min / 60).diff().median() if len(t) > 1 else 1
                rows += list(zip((t.frame - getattr(r, col)) * (dt_h if np.isfinite(dt_h) else 1), t.ratio_BA))
            if not rows: continue
            d = pd.DataFrame(rows, columns=["rel", "r"]); d = d[(d.rel >= -5) & (d.rel <= 5)]; gg = d.groupby("rel"); ok = gg.size() >= max(3, len(s) // 4)
            q = gg.r.quantile([.25, .5, .75]).unstack()[ok]
            if len(q): a.fill_between(q.index, q[.25], q[.75], color=cl, alpha=.15); a.plot(q.index, q[.5], "o-", color=cl, ms=3, label=f"{GNAME[g]} ({len(s)}개)")
        a.axvline(0, color="k", ls=":"); a.set_xlabel(f"{lab} 기준 시간 (h)"); a.set_ylabel("I(pHrodo)/I(Cy5)"); a.set_title(f"선별 입자 — {lab} 정렬 (중앙값 ± IQR)"); a.grid(alpha=.3)
        if a.get_legend_handles_labels()[0]: a.legend()
    _save(fig, path, 100)
