"""
Report figure + LaTeX table for the midway report (report/fig_main.pdf, report/table_main.tex).

    python src/fig_report.py

Panels: (a,b) superiority at rollout step 50 vs unroll length N for each solver; (c) one-step gain of the mode-1
wave vs its input amplitude (the mechanism); (d) knife-edge test: superiority at step 100 vs test amplitude.
Everything is regenerated from results/runs/ and results/eval_amplitude.csv, so numbers in the report stay in
sync with the cached runs.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
import emusup  # noqa: E402

REPORT = os.path.join(emusup.PROJ, "report")
os.makedirs(REPORT, exist_ok=True)
RANDAMP = "indamp;1;0.5;1.5"
C_MODE = analyze.C_MODE
C_RAND = "#4a3aa7"
C_N = analyze.C_N
INK, MUTED, GREY = analyze.INK, analyze.MUTED, analyze.C_AUTH
DIVERGED = 10.0  # a run with superiority > 10 at step 50 blew up; it is annotated, not averaged

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.2,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 1.2, "lines.markersize": 2.6})


def agg_vs_N(rows, metric="sup@50"):
    """median / min / max / n / n_diverged per N, diverged seeds excluded from the statistics."""
    out = []
    for N, g in rows.groupby("N"):
        v = g[metric].values
        ok = v[v < DIVERGED]
        out.append({"N": N, "med": np.median(ok) if len(ok) else np.nan, "lo": ok.min() if len(ok) else np.nan,
                    "hi": ok.max() if len(ok) else np.nan, "n": len(ok), "div": int((v >= DIVERGED).sum())})
    return pd.DataFrame(out)


def panel_vs_N(ax, df, g, title):
    fno = df[(df["arch"] == "FNO") & (df["gamma_1"] == g)]
    conv = df[(df["arch"] == "Conv") & (df["gamma_1"] == g)]
    series = [
        (fno[fno["std"] & (fno["mode"] == "sup")], C_MODE["sup"], "-", "o", "FNO, sup;N"),
        (fno[fno["std"] & (fno["mode"] == "div_nog")], C_MODE["div_nog"], "-", "s", "FNO, div;N NOG"),
        (fno[fno["std"] & (fno["mode"] == "div_wig")], C_MODE["div_wig"], "-", "^", "FNO, div;N WIG"),
        (conv[conv["std"] & (conv["mode"] == "sup")], C_MODE["sup"], "--", "o", "ConvNet, sup;N"),
        (fno[(fno["ic_train"] == RANDAMP) & (fno["mode"] == "sup")], C_RAND, "-", "D",
         "FNO, sup;N, random-amp. data"),
    ]
    for rows, color, ls, marker, label in series:
        if rows.empty:
            continue
        a = agg_vs_N(rows)
        a = a[a["n"] > 0]
        ax.plot(a["N"], a["med"], ls=ls, marker=marker, color=color, label=label, mfc="white" if ls == "--" else color)
        if (a["n"] > 1).any():
            ax.fill_between(a["N"], a["lo"], a["hi"], color=color, alpha=0.13, lw=0)
        for _, r in a[a["div"] > 0].iterrows():
            ax.text(0.04, 0.97, f"N={int(r['N'])}: {int(r['div'])} seed diverged", transform=ax.transAxes, fontsize=5,
                    color=color, ha="left", va="top")
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_xlim(0.85, 12)
    ax.set_ylim(0, 1.85)
    ax.set_xlabel("unroll length N")
    ax.set_title(title)
    ax.grid(color=analyze.GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.minorticks_off()


def panel_gain(ax, df, g=-0.9):
    d = df[(df["arch"] == "FNO") & (df["gamma_1"] == g) & (df["cfg_seed_init"] == 0)]
    picks = [(d[d["std"] & (d["mode"] == "sup") & (d["N"] == N)], C_N[N], f"N={N} (c,d)") for N in (1, 2, 5)]
    picks.append((d[(d["ic_train"] == RANDAMP) & (d["mode"] == "sup") & (d["N"] == 1)], C_RAND, "random-amp. N=1 (c,d)"))
    solver = None
    for rows, color, label in picks:
        if rows.empty:
            continue
        name = rows.iloc[0]["name"]
        p = os.path.join(emusup.RUNS_DIR, name, "symbol_amp.csv")
        sa = pd.read_csv(p).query("k == 1 and amplitude >= 0.9")
        ax.plot(sa["amplitude"], sa["gain"], "o-", color=color, label=label)
        if solver is None:
            s1 = pd.read_csv(os.path.join(emusup.RUNS_DIR, name, "symbol.csv")).query("k == 1").iloc[0]
            solver = (s1["gain_solver"], s1["gain_exact"])
    if solver is not None:
        ax.axhline(solver[0], color=GREY, ls=":", lw=1.1, label="solver")
        ax.axhline(solver[1], color=INK, ls="--", lw=0.7, label="exact")
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("input amplitude")
    ax.set_ylabel("gain of mode 1", labelpad=1)
    ax.set_title("(c) gain vs. amplitude")
    ax.grid(color=analyze.GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_knife(ax, g=-0.9):
    df = pd.read_csv(os.path.join(emusup.PROJ, "results", "eval_amplitude.csv"))
    d = df[(df["arch"] == "FNO") & (df["gamma_1"] == g)]
    series = [
        (d[(d["ic_train"] == "ind;1;True") & (d["H"] == d["N"]) & (d["mode"] == "sup") & (d["N"] == 1)], C_N[1], "N=1 (c,d)"),
        (d[(d["ic_train"] == "ind;1;True") & (d["H"] == d["N"]) & (d["mode"] == "sup") & (d["N"] == 5)], C_N[5], "N=5 (c,d)"),
        (d[(d["ic_train"] == RANDAMP) & (d["N"] == 1)], C_RAND, "random-amp. N=1 (c,d)"),
    ]
    for rows, color, label in series:
        if rows.empty:
            continue
        a = rows.groupby("amplitude")["sup@100"].agg(["median", "min", "max", "count"]).reset_index()
        ax.plot(a["amplitude"], a["median"], "o-", color=color, label=label)
        ax.fill_between(a["amplitude"], a["min"], a["max"], color=color, alpha=0.13, lw=0)
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.axvline(1.0, color=GREY, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_xlabel("test amplitude")
    ax.set_ylabel("superiority, step 100", labelpad=1)
    ax.set_title("(d) rescaled test wave")
    ax.grid(color=analyze.GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_main(df):
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 0.88))
    panel_vs_N(axes[0], df, -0.9, "(a) explicit, CFL 0.9")
    panel_vs_N(axes[1], df, -3.0, "(b) implicit, CFL 3.0")
    axes[0].set_ylabel("superiority, step 50")
    axes[1].tick_params(labelleft=False)
    panel_gain(axes[2], df)
    panel_knife(axes[3])
    analyze.fig_legend(fig, axes, ncol=6, y=-0.22, fontsize=5.6)
    fig.subplots_adjust(wspace=0.74)
    fig.savefig(os.path.join(REPORT, "fig_main.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(REPORT, "fig_main.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def cell(rows, metric):
    v = rows[metric].values
    ok = v[v < DIVERGED]
    if len(v) == 0:
        return "--"
    s = f"{np.median(ok):.2f}" if len(ok) else "--"
    if len(ok) > 1:
        s += f" [{ok.min():.2f}, {ok.max():.2f}]"
    if (v >= DIVERGED).sum():
        s += r"$^\dagger$"
    return s


def table_main(df):
    """Rows: training recipe; columns: step 50 and step 100 for each solver. Median [min, max] over seeds."""
    rows = [
        ("FNO", "one-step (paper's protocol)", lambda d: d["std"] & (d["mode"] == "sup") & (d["N"] == 1)),
        ("FNO", "data-unrolled, N=2", lambda d: d["std"] & (d["mode"] == "sup") & (d["N"] == 2)),
        ("FNO", "data-unrolled, N=5", lambda d: d["std"] & (d["mode"] == "sup") & (d["N"] == 5)),
        ("FNO", "data-unrolled, N=10", lambda d: d["std"] & (d["mode"] == "sup") & (d["N"] == 10)),
        ("FNO", "random-amplitude data, one-step", lambda d: (d["ic_train"] == RANDAMP) & (d["mode"] == "sup") & (d["N"] == 1)),
        ("FNO", "random-amplitude data, N=5", lambda d: (d["ic_train"] == RANDAMP) & (d["mode"] == "sup") & (d["N"] == 5)),
    ]
    lines = []
    last_arch = None
    for arch, label, filt in rows:
        da = df[df["arch"] == arch]
        r = da[filt(da)]
        if r.empty:
            continue
        ex, im = r[r["gamma_1"] == -0.9], r[r["gamma_1"] == -3.0]
        n = max(len(ex), len(im))
        if arch != last_arch and last_arch is not None:
            lines.append(r"\midrule")
        last_arch = arch
        name = label
        lines.append(f"{name} & {n} & {cell(ex, 'sup@50')} & {cell(ex, 'sup@100')} & {cell(im, 'sup@50')} & {cell(im, 'sup@100')} \\\\")
    body = "\n".join(lines)
    tex = (r"\begin{tabular}{@{}lcllll@{}}" "\n" r"\toprule" "\n"
           r" & & \multicolumn{2}{c}{explicit upwind, CFL 0.9} & \multicolumn{2}{c}{implicit upwind, CFL 3.0} \\" "\n"
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
           r"training recipe & seeds & step 50 & step 100 & step 50 & step 100 \\" "\n" r"\midrule" "\n"
           + body + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    with open(os.path.join(REPORT, "table_main.tex"), "w") as f:
        f.write(tex)
    print(tex)


def main():
    df = analyze.load()
    fig_main(df)
    table_main(df)
    # numbers quoted in the text
    for arch in ("FNO", "Conv"):
        for g in (-0.9, -3.0):
            d = df[(df["arch"] == arch) & (df["gamma_1"] == g) & (df["ic_train"] == RANDAMP)]
            if d.empty:
                continue
            print(arch, g, "random-amplitude runs:")
            print(d[["mode", "N", "cfg_seed_init", "sup@10", "sup@50", "sup@100", "loss_end"]].sort_values(["mode", "N", "cfg_seed_init"]).to_string(index=False))


if __name__ == "__main__":
    main()
