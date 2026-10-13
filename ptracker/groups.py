"""그룹 비교: 여러 이미지(데이터셋) 결과를 그룹으로 묶어 시간 추이·분포를 비교합니다.

그룹 = 이름 + 결과 출처(source) 목록. 출처로 쓸 수 있는 것
  · 데이터셋 결과 폴더            (tracks_points_A_B_HT.csv, 있으면 cell_intensity.csv)
  · 통합 결과 폴더 또는 그 CSV     (combined_tracks_points_A_B_HT.csv, 있으면 combined_cell_intensity.csv / datasets_map.json)
  · 여러 데이터셋 결과 폴더를 담은 상위 폴더 (한 단계 아래 폴더들을 모두 읽음)

단위(unit)
  · 입자: 트랙 하나 = 선 하나. 값은 point 표의 열 (I(measure)/I(reference) 등)
  · 세포 (세포 내 입자): 세포마다 그 세포 안 입자들의 값을 모은 것 (트래킹 결과만 있으면 됨)
  · 세포 (세포 전체): 세포 영역 전체의 평균 형광 (cell_intensity.csv — 1.10 이후 측정 단계에서 만듦)

계산 순서: 필터 → 단위·시간 칸(bin)별 평균 → (정규화) → 그룹(·위치)·시간별 중심값과 범위.
집계를 '이미지'로 하면 이미지마다 중심값을 먼저 구한 뒤 이미지들 사이의 평균 ± 범위를 냅니다 (이미지 = 반복 실험).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np, pandas as pd
from . import analysis, style

POINTS_FILE = "tracks_points_A_B_HT.csv"; CELLS_FILE = "cell_intensity.csv"
COMB_POINTS = "combined_tracks_points_A_B_HT.csv"; COMB_CELLS = "combined_cell_intensity.csv"

# 값(metric): key → (화면 이름, 단위 종류, 열/계산)
METRICS = {
    "ratio_BA": ("I(measure)/I(reference)", "particle"),
    "I_B": ("I(measure) (배경 제거)", "particle"),
    "I_A": ("I(reference) (배경 제거)", "particle"),
    "ratio_BA_raw": ("I(measure)/I(reference) (배경 제거 전)", "particle"),
    "HT_RI_mean": ("ROI 평균 RI", "particle"),
    "area_px": ("입자 면적 (px)", "particle"),
    "cp_median_ratio": ("세포 내 입자 I(measure)/I(reference) 중앙값", "cell_particles"),
    "cp_sum_ratio": ("세포 내 입자 ΣI(measure)/ΣI(reference)", "cell_particles"),
    "cp_mean_I_B": ("세포 내 입자 평균 I(measure)", "cell_particles"),
    "cp_mean_I_A": ("세포 내 입자 평균 I(reference)", "cell_particles"),
    "cp_n": ("세포 내 입자 수", "cell_particles"),
    "cell_ratio_BA": ("세포 전체 I(measure)/I(reference)", "cell_whole"),
    "cell_I_B": ("세포 전체 I(measure) (배경 제거)", "cell_whole"),
    "cell_I_A": ("세포 전체 I(reference) (배경 제거)", "cell_whole"),
    "cell_area_flpx": ("세포 면적 (형광 px)", "cell_whole"),
}
LEVELS = {"particle": "입자 (트랙)", "cell_particles": "세포 (세포 내 입자)", "cell_whole": "세포 (세포 전체)"}
LOCATIONS = {"inside": "세포 안", "outside": "세포 밖", "both": "세포 안+밖 합침", "split": "세포 안·밖 따로"}
CENTERS = {"median": "중앙값", "mean": "평균"}
SPREADS = {"iqr": "IQR (25–75%)", "sem": "SEM", "sd": "SD", "ci95": "95% CI", "none": "없음"}
NORMS = {"none": "없음", "fold": "처음 대비 배수 (÷ 처음 n점 평균)", "delta": "처음 대비 변화 (− 처음 n점 평균)"}
AGGS = {"unit": "모든 입자·세포를 합쳐서", "image": "이미지별 값을 먼저 구한 뒤 이미지들 사이에서"}
GROUP_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


@dataclass
class Group:
    name: str
    sources: list = field(default_factory=list)
    color: str = ""
    enabled: bool = True


@dataclass
class CompareSettings:
    metric: str = "ratio_BA"
    location: str = "inside"
    min_points: int = 10             # 단위(트랙·세포)마다 값이 있는 시점 수 최소
    bin_h: float = 0.0               # 시간 칸 (h). 0이면 촬영 시점 그대로
    t_min_h: float = 0.0
    t_max_h: float = 0.0             # 0이면 끝까지
    center: str = "median"
    spread: str = "iqr"
    norm: str = "none"
    norm_n: int = 3
    agg: str = "unit"
    min_n: int = 3                   # 시점마다 단위(트랙·세포)가 이만큼 미만이면 그 점은 그리지 않음
    unit_filter: str = ""            # 쉼표로 구분한 글자. 이름에 하나라도 들어 있는 단위만 (예: "P01 · cell 3, P02 · cell 1")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


# ================================================================ 불러오기
_cache: dict = {}


def _mtime(p: Path):
    try: return p.stat().st_mtime
    except OSError: return 0


def _dataset_dirs_in(folder: Path) -> dict:
    """폴더 하나 또는 그 바로 아래 폴더들 중 데이터셋 결과 폴더 {이름: 경로}."""
    if (folder / POINTS_FILE).exists(): return {folder.name: str(folder)}
    return {d.name: str(d) for d in sorted(folder.iterdir()) if d.is_dir() and (d / POINTS_FILE).exists()}


def load_source(src) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """출처 하나 → (points, 세포 전체 밝기 표, 설명). points/세포 표에는 dataset 열이 있습니다."""
    p = Path(src); key = (str(p), _mtime(p), _mtime(p / POINTS_FILE), _mtime(p / COMB_POINTS))
    if key in _cache: return _cache[key]
    csv = None
    if p.is_file(): csv = p
    elif (p / COMB_POINTS).exists(): csv = p / COMB_POINTS
    if csv is not None:
        pts = pd.read_csv(csv); folder = csv.parent
        if "dataset" not in pts:            # 단일 데이터셋 point CSV
            pts.insert(0, "dataset", folder.name if csv.name == POINTS_FILE else csv.stem)
        if "track_uid" not in pts: pts.insert(1, "track_uid", pts.dataset.astype(str) + "::" + pts.track_id.astype(str))
        cells = pd.DataFrame()
        if (folder / COMB_CELLS).exists(): cells = pd.read_csv(folder / COMB_CELLS)
        elif csv.name == POINTS_FILE and (folder / CELLS_FILE).exists(): cells = pd.read_csv(folder / CELLS_FILE).assign(dataset=pts.dataset.iloc[0])
        elif (folder / "datasets_map.json").exists():
            try: cells = analysis.combine_cells(json.loads((folder / "datasets_map.json").read_text(encoding="utf-8")))
            except Exception: cells = pd.DataFrame()   # noqa: BLE001
        desc = f"통합 결과: 이미지 {pts.dataset.nunique()}개"
    elif p.is_dir():
        dirs = _dataset_dirs_in(p)
        if not dirs: raise ValueError(f"결과 파일({POINTS_FILE} 또는 {COMB_POINTS})이 없습니다: {p}")
        pts = analysis.combine(dirs); cells = analysis.combine_cells(dirs)
        desc = f"결과 폴더: 이미지 {len(dirs)}개" if len(dirs) > 1 else "데이터셋 결과"
    else:
        raise ValueError(f"없는 경로: {p}")
    if not cells.empty: desc += f", 세포 전체 밝기 있음"
    out = (pts, cells, desc); _cache[key] = out
    return out


def load_groups(groups: list[Group], log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """사용할 그룹의 출처를 모두 읽어 (points, cells) 반환. 열 group·image 추가 (image = 그룹 안에서 고유한 이미지 이름)."""
    P, C = [], []
    for g in groups:
        if not g.enabled: continue
        seen = {}
        for src in g.sources:
            try: pts, cells, _ = load_source(src)
            except Exception as e:   # noqa: BLE001
                if log: log(f"[{g.name}] 읽기 실패: {src} ({e})")
                continue
            ren = {}
            for ds in pts.dataset.astype(str).unique():   # 같은 그룹 안에서 이름이 겹치면 상위 폴더 이름을 붙임
                nm = ds if ds not in seen else f"{Path(src).name}/{ds}"
                k = 2
                while nm in seen: nm = f"{ds} ({k})"; k += 1
                seen[nm] = src; ren[ds] = nm
            pts = pts.rename(columns={"group": "track_group"}).assign(group=g.name, image=pts.dataset.astype(str).map(ren)); P.append(pts)
            if not cells.empty: C.append(cells.assign(group=g.name, image=cells.dataset.astype(str).map(ren)))
    P = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    C = pd.concat(C, ignore_index=True) if C else pd.DataFrame()
    return P, C


# ================================================================ 계산
def _series_name(group, loc):
    return group if loc is None else f"{group} · {LOCATIONS[loc]}"


def unit_values(P: pd.DataFrame, C: pd.DataFrame, s: CompareSettings) -> pd.DataFrame:
    """단위·시간 칸별 값. 열: series, group, location, image, unit, t_h, value."""
    level = METRICS[s.metric][1]; cols = ["series", "group", "location", "image", "unit", "t_h", "value"]
    if level == "cell_whole":
        if C.empty: return pd.DataFrame(columns=cols)
        d = C.assign(location=None, unit=C.image.astype(str) + " · cell " + C.cell_id.astype(str), v=C[s.metric])
    else:
        if P.empty or (level == "particle" and s.metric not in P): return pd.DataFrame(columns=cols)
        d = P; tg = P.track_group          # track_group = 트랙 분류(inside/outside/short…), group = 사용자 그룹
        if level == "particle":
            keep = tg.isin(["inside", "outside"]) if s.location in ("both", "split") else tg == s.location
            d = d[keep].assign(location=tg[keep] if s.location == "split" else None)
            d = d.assign(unit=d.image.astype(str) + "::" + d.track_id.astype(str), v=d[s.metric])
        else:   # 세포 내 입자: 세포 안에 있는 시점의 입자 (트랙 분류가 세포 안/밖이든) 중 장기 트랙만
            d = d[(d.cell_id > 0) & tg.isin(["inside", "outside"])].copy()
            g = d.groupby(["group", "image", "cell_id", "frame"])
            d = g.agg(time_min=("time_min", "first"), cp_median_ratio=("ratio_BA", "median"), sI_B=("I_B", "sum"), sI_A=("I_A", "sum"),
                      cp_mean_I_B=("I_B", "mean"), cp_mean_I_A=("I_A", "mean"), cp_n=("track_id", "size")).reset_index()
            d["cp_sum_ratio"] = np.where(d.sI_A > 0, d.sI_B / d.sI_A, np.nan)
            d = d.assign(location=None, unit=d.image.astype(str) + " · cell " + d.cell_id.astype(str), v=d[s.metric])
    d = d[["group", "location", "image", "unit", "time_min", "v"]].dropna(subset=["v"])
    d = d[np.isfinite(d.v)]
    d["t_h"] = d.time_min / 60
    if s.t_min_h > 0: d = d[d.t_h >= s.t_min_h]
    if s.t_max_h > 0: d = d[d.t_h <= s.t_max_h]
    if s.bin_h > 0: d["t_h"] = (np.floor(d.t_h / s.bin_h) * s.bin_h + s.bin_h / 2).round(6)
    else: d["t_h"] = d.t_h.round(6)
    d["location"] = d.location.astype(object)
    key = ["group", "location", "image", "unit"]
    d = d.groupby(key + ["t_h"], dropna=False).v.mean().reset_index()
    n = d.groupby(key, dropna=False).v.transform("size"); d = d[n >= s.min_points]
    if s.norm != "none" and len(d):
        d = d.sort_values(key + ["t_h"])
        base = d.groupby(key, dropna=False).v.transform(lambda x: x.iloc[:max(s.norm_n, 1)].mean())
        d["v"] = d.v / base.where(base != 0) if s.norm == "fold" else d.v - base
        d = d.dropna(subset=["v"])
    toks = [t.strip() for t in s.unit_filter.split(",") if t.strip()]
    if toks:      # 단위 이름과 똑같은 글자는 그 단위만, 아니면 이름에 들어 있는 단위 모두 (예: "P01" → P01의 모든 단위)
        names = set(d.unit.astype(str)); ex = {t for t in toks if t in names}; sub = [t for t in toks if t not in names]
        d = d[d.unit.astype(str).apply(lambda u: u in ex or any(t in u for t in sub))]
    d["series"] = [_series_name(g, l if isinstance(l, str) else None) for g, l in zip(d.group, d.location)]
    return d.rename(columns={"v": "value"})[cols].reset_index(drop=True)


def _center_spread(x: np.ndarray, s: CompareSettings):
    x = x[np.isfinite(x)]; n = len(x)
    if n == 0: return np.nan, np.nan, np.nan
    c = np.median(x) if s.center == "median" else np.mean(x)
    if s.spread == "iqr": lo, hi = np.percentile(x, [25, 75])
    elif s.spread == "none" or n < 2: lo = hi = np.nan
    else:
        sd = np.std(x, ddof=1); e = {"sd": sd, "sem": sd / np.sqrt(n), "ci95": 1.96 * sd / np.sqrt(n)}[s.spread]; lo, hi = c - e, c + e
    return c, lo, hi


def per_image(U: pd.DataFrame, s: CompareSettings) -> pd.DataFrame:
    """이미지·시간별 중심값. 열: series, group, location, image, t_h, n_units, value"""
    if U.empty: return pd.DataFrame(columns=["series", "group", "location", "image", "t_h", "n_units", "value"])
    f = np.median if s.center == "median" else np.mean
    return U.groupby(["series", "group", "location", "image", "t_h"], dropna=False).value.agg(n_units="size", value=f).reset_index()


def summarize(U: pd.DataFrame, s: CompareSettings) -> pd.DataFrame:
    """그룹(series)·시간별 중심값과 범위. 열: series, group, location, t_h, n_units, n_images, center, low, high"""
    cols = ["series", "group", "location", "t_h", "n_units", "n_images", "center", "low", "high"]
    if U.empty: return pd.DataFrame(columns=cols)
    src = per_image(U, s) if s.agg == "image" else U; rows = []
    for (ser, g, loc, t), x in src.groupby(["series", "group", "location", "t_h"], dropna=False):
        sub = U[(U.series == ser) & (U.t_h == t)]
        c, lo, hi = _center_spread(x.value.values, s) if sub.unit.nunique() >= max(s.min_n, 1) else (np.nan, np.nan, np.nan)
        rows.append((ser, g, loc, t, sub.unit.nunique(), sub.image.nunique(), c, lo, hi))
    return pd.DataFrame(rows, columns=cols).sort_values(["series", "t_h"]).reset_index(drop=True)


def unit_means(U: pd.DataFrame, s: CompareSettings) -> pd.DataFrame:
    """분포 비교용: 단위(또는 이미지)마다 시간 범위 안 평균. 열: series, group, location, image, unit, value"""
    if U.empty: return pd.DataFrame(columns=["series", "group", "location", "image", "unit", "value"])
    M = U.groupby(["series", "group", "location", "image", "unit"], dropna=False).value.mean().reset_index()
    if s.agg == "image":
        f = np.median if s.center == "median" else np.mean
        M = M.groupby(["series", "group", "location", "image"], dropna=False).value.agg(f).reset_index().assign(unit=lambda x: x.image)
    return M


def stats_table(M: pd.DataFrame) -> pd.DataFrame:
    """series 쌍별 Mann-Whitney U (Holm 보정) + 3개 이상이면 Kruskal-Wallis."""
    from scipy import stats
    ser = list(dict.fromkeys(M.series)); vals = {k: M[M.series == k].value.dropna().values for k in ser}; rows = []
    if len(ser) >= 3 and all(len(v) >= 2 for v in vals.values()):
        h, p = stats.kruskal(*vals.values()); rows.append(dict(test="Kruskal-Wallis", a="전체", b="", n_a=sum(map(len, vals.values())), n_b=np.nan, statistic=h, p=p, p_holm=p))
    pairs = []
    for i in range(len(ser)):
        for j in range(i + 1, len(ser)):
            a, b = vals[ser[i]], vals[ser[j]]
            if len(a) >= 2 and len(b) >= 2:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                pairs.append(dict(test="Mann-Whitney U", a=ser[i], b=ser[j], n_a=len(a), n_b=len(b), statistic=u, p=p))
    if pairs:
        ps = np.array([r["p"] for r in pairs]); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
        for k, idx in enumerate(o): run = max(run, min(1, (m - k) * ps[idx])); adj[idx] = run
        for r, a in zip(pairs, adj): r["p_holm"] = a
        rows += pairs
    return pd.DataFrame(rows, columns=["test", "a", "b", "n_a", "n_b", "statistic", "p", "p_holm"])


def metric_label(s: CompareSettings):
    lab = METRICS[s.metric][0]
    if s.norm == "fold": lab += " (처음 대비 배수)"
    elif s.norm == "delta": lab += " (처음 대비 변화)"
    return lab


# ================================================================ 그리기
def series_colors(groups: list[Group]) -> dict:
    return {g.name: (g.color or GROUP_COLORS[i % len(GROUP_COLORS)]) for i, g in enumerate(groups)}


def plot_timecourse(ax, U, S, s: CompareSettings, colors: dict, individual="none", band=True):
    """individual: none | image | unit"""
    for ser, q in S.groupby("series", sort=False):
        g = q.group.iloc[0]; loc = q.location.iloc[0]; col = colors.get(g, "k"); ls = "--" if loc == "outside" else "-"
        if individual != "none":
            src = per_image(U[U.series == ser], s) if individual == "image" else U[U.series == ser]
            key = "image" if individual == "image" else "unit"
            for _, t in src.groupby(key): ax.plot(t.t_h, t.value, ls, color=col, lw=.7, alpha=.35 if individual == "unit" else .55)
        q = q.dropna(subset=["center"])
        if band and s.spread != "none" and q.low.notna().any(): ax.fill_between(q.t_h, q.low, q.high, color=col, alpha=.18, lw=0)
        nu = U[U.series == ser].unit.nunique(); ni = U[U.series == ser].image.nunique()
        ax.plot(q.t_h, q.center, ls, color=col, lw=2.4, marker="o" if len(q) <= 40 else None, ms=3,
                label=f"{ser} (이미지 {ni}, {'세포' if METRICS[s.metric][1] != 'particle' else '트랙'} {nu})")
    ax.set_xlabel("시간 (h)"); ax.set_ylabel(metric_label(s)); ax.grid(alpha=.3)
    if s.norm == "fold": ax.axhline(1, color="gray", lw=.8, ls=":")
    elif s.norm == "delta": ax.axhline(0, color="gray", lw=.8, ls=":")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=9)
    ax.set_title(f"{metric_label(s)} — {CENTERS[s.center]}" + ("" if s.spread == "none" else f" ± {SPREADS[s.spread]}")
                 + (" (이미지 단위)" if s.agg == "image" else ""))


def plot_distribution(ax, M, s: CompareSettings, colors: dict, T: pd.DataFrame | None = None):
    ser = list(dict.fromkeys(M.series))
    if not ser: return
    data = [M[M.series == k].value.dropna().values for k in ser]
    bp = ax.boxplot(data, widths=.55, showfliers=False, patch_artist=True)
    rng = np.random.default_rng(0)
    for i, (k, v) in enumerate(zip(ser, data), 1):
        g = M[M.series == k].group.iloc[0]; col = colors.get(g, "k")
        bp["boxes"][i - 1].set(facecolor=col, alpha=.25, edgecolor=col)
        ax.plot(i + rng.uniform(-.15, .15, len(v)), v, "o", color=col, ms=3 if len(v) > 50 else 5, alpha=.55)
    for m in bp["medians"]: m.set(color="k", lw=2)
    ax.set_xticks(range(1, len(ser) + 1)); ax.set_xticklabels([k.replace(" · ", "\n") + f"\n(n={len(v)})" for k, v in zip(ser, data)], fontsize=8)
    unit = "이미지" if s.agg == "image" else ("세포" if METRICS[s.metric][1] != "particle" else "트랙")
    ax.set_ylabel(metric_label(s)); ax.grid(alpha=.3, axis="y")
    rng_txt = f"{s.t_min_h:g}–{s.t_max_h:g} h" if s.t_max_h > 0 else (f"{s.t_min_h:g} h 이후" if s.t_min_h > 0 else "전체 시간")
    ax.set_title(f"{unit}별 평균 {metric_label(s)} ({rng_txt})", fontsize=10)
    if T is not None and len(T) and len(ser) <= 4:     # 유의 표시 (쌍별 Holm 보정 p). 비교가 많으면 [통계 검정] 표에서
        top = max((np.nanmax(v) for v in data if len(v)), default=0); lo_ = min((np.nanmin(v) for v in data if len(v)), default=0)
        h = (top - lo_) * .07 or 1; y = top + h; idx = {k: i + 1 for i, k in enumerate(ser)}
        for r in T[T.test == "Mann-Whitney U"].itertuples():
            a, b = idx[r.a], idx[r.b]; p = r.p_holm
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            ax.plot([a, a, b, b], [y, y + h * .3, y + h * .3, y], "k-", lw=.8); ax.text((a + b) / 2, y + h * .35, f"{star} (p={p:.2g})", ha="center", fontsize=8)
            y += h * 1.1


# ================================================================ 저장
def save_setup(path, groups: list[Group], s: CompareSettings, extra: dict | None = None):
    d = dict(groups=[asdict(g) for g in groups], settings=asdict(s), **(extra or {}))
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def load_setup(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Group(**g) for g in d.get("groups", [])], CompareSettings.from_dict(d.get("settings")), d


def export(base, fig, U, S, M, T, groups, s: CompareSettings, extra=None) -> list[str]:
    """base: 확장자가 있으면 그 형식으로 그림 저장(png/pdf/svg/tif), 같은 이름으로 CSV·설정 저장. 저장한 파일 목록 반환."""
    style.setup(); base = Path(base); ext = base.suffix.lower() if base.suffix.lower() in (".png", ".pdf", ".svg", ".tif", ".tiff", ".jpg") else ".png"
    stem = base.with_suffix(""); stem.parent.mkdir(parents=True, exist_ok=True); out = []
    f = stem.with_suffix(ext); fig.savefig(f, dpi=300 if ext in (".png", ".tif", ".tiff", ".jpg") else None, bbox_inches="tight"); out.append(str(f))
    for nm, df in [("summary", S), ("per_unit", U), ("per_image", per_image(U, s)), ("unit_means", M), ("stats", T)]:
        if df is None: continue
        p = Path(f"{stem}_{nm}.csv"); df.to_csv(p, index=False, encoding="utf-8-sig"); out.append(str(p))
    p = Path(f"{stem}_settings.json"); save_setup(p, groups, s, extra); out.append(str(p))
    return out
