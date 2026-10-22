"""
Three-panel figure for the story version of the midway report (report/fig_story.pdf).

    python src/fig_story.py

(a) superiority at step 50 vs unroll length N, FNO, data-unrolled training: paper's data (solid) vs random-amplitude
    data (dashed), one colour per solver;  (b) one-step gain vs input amplitude, explicit solver, seed 0;
(c) knife-edge test: superiority at step 100 vs test-wave amplitude, explicit solver, 3 seeds.
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
from fig_report import agg_vs_N, RANDAMP, C_N, C_RAND, INK, GREY  # noqa: E402

REPORT = os.path.join(emusup.PROJ, "report")
C_EXP, C_IMP = "#2a78d6", "#eb6834"  # explicit / implicit solver

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
                     "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.3, "lines.markersize": 3})


def _style(ax, tag):
    ax.grid(color=analyze.GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_unroll(ax, df):
    fno = df[df["arch"] == "FNO"]
    for g, color, name in ((-0.9, C_EXP, "explicit solver"), (-3.0, C_IMP, "implicit solver")):
        for data, ls, mk, mfc, lab in ((fno["std"], "-", "o", None, "paper data"),
                                       (fno["ic_train"] == RANDAMP, "--", "D", "white", "random amp.")):
            rows = fno[data & (fno["gamma_1"] == g) & (fno["mode"] == "sup")]
            if rows.empty:
                continue
            a = agg_vs_N(rows)
            a = a[a["n"] > 0]
            ax.plot(a["N"], a["med"], ls=ls, marker=mk, color=color, mfc=mfc or color, label=f"{name.split()[0]}, {lab}")
            if (a["n"] > 1).any():
                ax.fill_between(a["N"], a["lo"], a["hi"], color=color, alpha=0.12, lw=0)
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 12)
    ax.set_ylim(0, 2.25)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("unroll length N in training")
    ax.set_ylabel("superiority ratio, step 50")
    _style(ax, "(a)")


def panel_gain(ax, df):
    d = df[(df["arch"] == "FNO") & (df["gamma_1"] == -0.9) & (df["cfg_seed_init"] == 0)]
    picks = [(d[d["std"] & (d["mode"] == "sup") & (d["N"] == 1)], C_N[1], "paper data, N=1"),
             (d[d["std"] & (d["mode"] == "sup") & (d["N"] == 5)], C_N[5], "paper data, N=5"),
             (d[(d["ic_train"] == RANDAMP) & (d["mode"] == "sup") & (d["N"] == 1)], C_RAND, "random amp.")]
    solver = None
    for rows, color, label in picks:
        if rows.empty:
            continue
        name = rows.iloc[0]["name"]
        sa = pd.read_csv(os.path.join(emusup.RUNS_DIR, name, "symbol_amp.csv")).query("k == 1 and amplitude >= 0.9")
        ax.plot(sa["amplitude"], sa["gain"], "o-", color=color, label=label)
        if solver is None:
            s1 = pd.read_csv(os.path.join(emusup.RUNS_DIR, name, "symbol.csv")).query("k == 1").iloc[0]
            solver = (s1["gain_solver"], s1["gain_exact"])
    if solver is not None:
        ax.axhline(solver[0], color=GREY, ls=":", lw=1.2)
        ax.axhline(solver[1], color=INK, ls="--", lw=0.7)
    ax.axvline(1.0, color=GREY, lw=0.6, ls="-", alpha=0.6)
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("input amplitude (training: 1)")
    ax.set_ylabel("one-step gain")
    _style(ax, "(b)")


def panel_knife(ax):
    df = pd.read_csv(os.path.join(emusup.PROJ, "results", "eval_amplitude.csv"))
    d = df[(df["arch"] == "FNO") & (df["gamma_1"] == -0.9)]
    series = [(d[(d["ic_train"] == "ind;1;True") & (d["H"] == d["N"]) & (d["mode"] == "sup") & (d["N"] == 1)], C_N[1], "paper data, N=1"),
              (d[(d["ic_train"] == RANDAMP) & (d["N"] == 1)], C_RAND, "random amp., N=1")]
    for rows, color, label in series:
        if rows.empty:
            continue
        a = rows.groupby("amplitude")["sup@100"].agg(["median", "min", "max"]).reset_index()
        ax.plot(a["amplitude"], a["median"], "o-", color=color, label=label)
        ax.fill_between(a["amplitude"], a["min"], a["max"], color=color, alpha=0.12, lw=0)
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.axvline(1.0, color=GREY, lw=0.6, ls="-", alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("test amplitude (training: 1)")
    ax.set_ylabel("superiority ratio, step 100")
    _style(ax, "(c)")


def main():
    df = analyze.load()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 0.9))
    panel_unroll(axes[0], df)
    panel_gain(axes[1], df)
    panel_knife(axes[2])
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(os.path.join(REPORT, "fig_story.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(REPORT, "fig_story.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
