"""trials.csv 분석: 모드별 기술통계, 착륙 성공률, 대응표본 검정, 그래프 저장.

사용 예
    python3 analyze.py                      # trials.csv -> results/
    python3 analyze.py --input trials.csv --outdir results --test auto

분석 단위
- 기술통계(평균 ± 표준편차)는 시도(trial) 단위.
- 대응표본 검정은 참가자 단위: 참가자 x 모드별 평균을 먼저 낸 뒤, 두 모드를 모두
  수행한 참가자끼리 짝지어 비교한다 (같은 참가자의 반복 시도를 독립 표본으로 세지 않기 위함).
- 경로 오차는 시나리오마다 단위·크기가 달라 시나리오별로 따로 분석한다.
- target_time_s는 목표에 도달한 시도만 값이 있다 (실패·시간초과는 제외됨).
- outcome = aborted 인 시도는 분석에서 뺀다.
"""

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRICS = {
    "reaction_time_s": "Reaction time (s)",
    "target_time_s": "Time to target (s)",
    "path_error_m": "Path error (m)",
}
SCENARIO_ORDER = ["takeoff", "straight", "turn", "landing", "circuit"]
ALPHA = 0.05

# 색: 모드마다 고정 순서로 배정 (필터링돼도 같은 모드는 항상 같은 색)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                          "#898781", "#e1e0d9", "#c3c2b7")


# ---------------------------------------------------------------- 데이터

def load(path: Path, env: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "environment" not in df.columns:   # 환경 열이 생기기 전의 기록은 단순 모델(off)
        df["environment"] = "off"
    df["environment"] = df["environment"].fillna("off")
    envs = sorted(df["environment"].unique())
    if env is not None:
        df = df[df["environment"] == env]
        if df.empty:
            raise SystemExit(f"environment = {env} 인 기록이 없습니다 (있는 환경: {envs}).")
    elif len(envs) > 1:
        print(f"주의: 서로 다른 환경 {envs}의 기록이 섞여 있습니다. 환경마다 난이도가 달라 함께 "
              f"분석하면 결과가 섞입니다. --env 로 하나만 골라 분석하세요.\n")
    n_aborted = (df["outcome"] == "aborted").sum()
    df = df[df["outcome"] != "aborted"].copy()
    df["landing_success"] = df["landing_success"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    print(f"{path}: {len(df)}개 시도, 참가자 {df['participant_id'].nunique()}명, "
          f"모드 {sorted(df['mode'].unique())} (aborted {n_aborted}개 제외)\n")
    return df


def mode_order(df):
    modes = sorted(df["mode"].unique(), key=lambda m: (m != "gesture", m))
    return modes, {m: SERIES[i % len(SERIES)] for i, m in enumerate(modes)}


def scenarios_in(df):
    return [s for s in SCENARIO_ORDER if s in set(df["scenario"])]


# ---------------------------------------------------------------- 기술통계

def descriptives(df) -> pd.DataFrame:
    rows = []
    for (scenario, mode), g in df.groupby(["scenario", "mode"]):
        for metric in METRICS:
            v = g[metric].dropna()
            rows.append({"scenario": scenario, "mode": mode, "metric": metric, "n": len(v),
                         "mean": v.mean() if len(v) else np.nan,
                         "sd": v.std(ddof=1) if len(v) > 1 else np.nan})
    out = pd.DataFrame(rows)
    out["scenario"] = pd.Categorical(out["scenario"], SCENARIO_ORDER, ordered=True)
    return out.sort_values(["scenario", "metric", "mode"]).reset_index(drop=True)


def landing_rates(df) -> pd.DataFrame:
    land = df[df["scenario"] == "landing"]
    out = land.groupby("mode")["landing_success"].agg(successes="sum", n="count")
    out["successes"] = out["successes"].astype(int)
    out["rate_pct"] = 100 * out["successes"] / out["n"]
    lo, hi = wilson_ci(out["successes"].to_numpy(), out["n"].to_numpy())
    out["ci95_low_pct"], out["ci95_high_pct"] = 100 * lo, 100 * hi
    return out.reset_index()


def wilson_ci(k, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간 (표본이 작을 때도 0~1 범위를 벗어나지 않음)."""
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


# ---------------------------------------------------------------- 검정

def holm(pvals):
    """Holm-Bonferroni 보정 p값."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_test(a, b, method="auto"):
    """a, b: 같은 참가자 순서로 짝지은 값. auto면 차이값의 Shapiro-Wilk로 t / Wilcoxon 선택."""
    d = a - b
    n = len(d)
    res = {"n_pairs": n, "mean_diff": d.mean(), "shapiro_p": np.nan}
    if n < 3:
        return {**res, "test": "skipped", "note": "짝지은 참가자 3명 미만"}
    if np.allclose(d, 0):
        return {**res, "test": "skipped", "note": "모든 차이가 0"}

    if method == "auto":
        res["shapiro_p"] = stats.shapiro(d).pvalue
        method = "t" if res["shapiro_p"] >= ALPHA else "wilcoxon"

    if method == "t":
        r = stats.ttest_rel(a, b)
        sd = d.std(ddof=1)
        return {**res, "test": "paired t", "statistic": r.statistic, "df": n - 1,
                "p": r.pvalue, "effect": "Cohen's dz",
                "effect_size": d.mean() / sd if sd > 0 else np.nan, "note": ""}

    r = stats.wilcoxon(a, b)
    # 짝지은 순위이연상관 (matched-pairs rank-biserial r): -1~1
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    r_rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return {**res, "test": "Wilcoxon", "statistic": r.statistic, "df": np.nan,
            "p": r.pvalue, "effect": "rank-biserial r", "effect_size": r_rb, "note": ""}


def run_tests(df, modes, method) -> pd.DataFrame:
    targets = [(s, m) for s in scenarios_in(df) for m in METRICS]
    if "landing" in set(df["scenario"]):
        targets.append(("landing", "landing_success"))

    rows = []
    for scenario, metric in targets:
        per_person = (df[df["scenario"] == scenario]
                      .groupby(["participant_id", "mode"])[metric].mean().unstack())
        for m1, m2 in itertools.combinations(modes, 2):
            if m1 not in per_person or m2 not in per_person:
                continue
            pair = per_person[[m1, m2]].dropna()
            res = paired_test(pair[m1].to_numpy(), pair[m2].to_numpy(), method)
            rows.append({"scenario": scenario, "metric": metric,
                         "comparison": f"{m1} - {m2}", **res})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    tested = out["p"].notna() if "p" in out else pd.Series(False, index=out.index)
    out["p_holm"] = np.nan
    if tested.any():
        out.loc[tested, "p_holm"] = holm(out.loc[tested, "p"])
    return out


# ---------------------------------------------------------------- 출력

def print_report(desc, rates, tests, modes):
    print("=" * 72)
    print("모드별 평균 ± 표준편차 (시도 단위)")
    print("=" * 72)
    for (scenario, metric), g in desc.groupby(["scenario", "metric"], observed=True, sort=False):
        cells = [f"{r.mode}: {r.mean:.2f} ± {'-' if np.isnan(r.sd) else f'{r.sd:.2f}'} (n={r.n})"
                 if r.n else f"{r.mode}: -" for r in g.itertuples()]
        print(f"{scenario:9s} {metric:16s} " + "   ".join(cells))

    print("\n" + "=" * 72)
    print("착륙 성공률")
    print("=" * 72)
    if rates.empty:
        print("착륙 시도가 없습니다.")
    for r in rates.itertuples():
        print(f"{r.mode:10s} {r.rate_pct:5.1f}%  ({r.successes}/{r.n}, "
              f"95% CI {r.ci95_low_pct:.1f}–{r.ci95_high_pct:.1f}%)")

    print("\n" + "=" * 72)
    print("대응표본 검정 (참가자별 평균, Holm 보정)")
    print("=" * 72)
    if len(modes) < 2:
        print(f"모드가 {len(modes)}개({', '.join(modes)})뿐이라 비교할 수 없습니다. "
              "두 가지 이상의 모드로 수집된 데이터가 필요합니다.")
        return
    for r in tests.itertuples():
        head = f"{r.scenario:9s} {r.metric:16s} {r.comparison:22s}"
        if r.test == "skipped":
            print(f"{head} 검정 생략: {r.note}")
            continue
        stat = (f"t({int(r.df)}) = {r.statistic:.3f}" if r.test == "paired t"
                else f"W = {r.statistic:.1f}")
        sig = "*" if r.p_holm < ALPHA else ""
        normal = "" if np.isnan(r.shapiro_p) else f", Shapiro p={r.shapiro_p:.3f}"
        print(f"{head} {r.test:9s} {stat}, p = {r.p:.4f}, p_holm = {r.p_holm:.4f}{sig}, "
              f"{r.effect} = {r.effect_size:.2f}  (n={r.n_pairs}, 평균 차이 {r.mean_diff:+.3f}{normal})")
    print("\n* Holm 보정 후 p < 0.05")


# ---------------------------------------------------------------- 그래프

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def box_plot(df, metric, modes, colors, path):
    scenarios = [s for s in scenarios_in(df) if df.loc[df["scenario"] == s, metric].notna().any()]
    if not scenarios:
        return False
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios) + 0.6, 4.2),
                             squeeze=False, facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for ax, scenario in zip(axes[0], scenarios):
        style_axes(ax)
        sub = df[df["scenario"] == scenario]
        data = [sub.loc[sub["mode"] == m, metric].dropna().to_numpy() for m in modes]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 2},
                        whiskerprops={"color": MUTED, "linewidth": 1.2},
                        capprops={"color": MUTED, "linewidth": 1.2})
        for patch, m in zip(bp["boxes"], modes):
            patch.set_facecolor(colors[m] + "40")
            patch.set_edgecolor(colors[m])
            patch.set_linewidth(1.5)
        for i, (vals, m) in enumerate(zip(data, modes), start=1):
            x = i + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=22, color=colors[m], edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_xticks(range(1, len(modes) + 1), [f"{m}\n(n={len(v)})" for m, v in zip(modes, data)])
        ax.set_title(scenario, color=INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel(METRICS[metric], color=INK_2)
    fig.suptitle(f"{METRICS[metric]} by mode (each dot = one trial)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def landing_bar(rates, colors, path):
    if rates.empty:
        return False
    fig, ax = plt.subplots(figsize=(1.6 * len(rates) + 2.4, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(rates))
    ax.bar(x, rates["rate_pct"], width=0.55,
           color=[colors[m] for m in rates["mode"]], edgecolor=SURFACE, linewidth=2)
    yerr = [rates["rate_pct"] - rates["ci95_low_pct"], rates["ci95_high_pct"] - rates["rate_pct"]]
    ax.errorbar(x, rates["rate_pct"], yerr=yerr, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4)
    for xi, r in zip(x, rates.itertuples()):
        ax.text(xi + 0.32, r.rate_pct, f"{r.rate_pct:.0f}%\n({r.successes}/{r.n})",
                va="center", ha="left", color=INK, fontsize=9)
    ax.set_xticks(x, rates["mode"])
    ax.set_xlim(-0.6, len(rates) - 0.1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Landing success (%)", color=INK_2)
    ax.set_title("Landing success rate by mode (error bars = Wilson 95% CI)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="trials.csv 분석")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "trials.csv")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--env", default=None,
                        help="이 환경(off / calm / normal / rough)의 기록만 분석. 생략하면 전부")
    parser.add_argument("--test", choices=["auto", "t", "wilcoxon"], default="auto",
                        help="auto = 차이값 Shapiro-Wilk p >= 0.05면 t검정, 아니면 Wilcoxon")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input}이 없습니다. experiment.py로 먼저 데이터를 수집하세요.")
    df = load(args.input, args.env)
    if df.empty:
        raise SystemExit("분석할 시도가 없습니다.")
    modes, colors = mode_order(df)

    desc = descriptives(df)
    rates = landing_rates(df)
    tests = run_tests(df, modes, args.test)
    print_report(desc, rates, tests, modes)

    args.outdir.mkdir(parents=True, exist_ok=True)
    desc.to_csv(args.outdir / "descriptives.csv", index=False)
    rates.to_csv(args.outdir / "landing_success.csv", index=False)
    tests.to_csv(args.outdir / "paired_tests.csv", index=False)
    saved = ["descriptives.csv", "landing_success.csv", "paired_tests.csv"]
    for metric, fname in [("reaction_time_s", "box_reaction_time.png"),
                          ("path_error_m", "box_path_error.png")]:
        if box_plot(df, metric, modes, colors, args.outdir / fname):
            saved.append(fname)
    if landing_bar(rates, colors, args.outdir / "bar_landing_success.png"):
        saved.append("bar_landing_success.png")
    print(f"\n저장: {args.outdir}/ -> {', '.join(saved)}")


if __name__ == "__main__":
    main()
