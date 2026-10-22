"""
Aggregate finished runs in results/runs/ into figures/ and results/summary.md.

    python src/analyze.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emusup  # noqa: E402

PROJ = emusup.PROJ
FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)
AUTHORS = os.path.join(emusup.AUTHORS_EXP, "advection_data")

# palette (validated reference palette of the dataviz guide)
C_MODE = {"sup": "#2a78d6", "div_nog": "#eb6834", "div_wig": "#1baf7a"}
C_N = {1: "#86b6ef", 2: "#3987e5", 5: "#1c5cab", 10: "#0d366b", 20: "#06203f"}
C_ARCH = {"Conv": "#2a78d6", "FNO": "#eb6834"}
C_AUTH = "#898781"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
MODE_LABEL = {"sup": "data-unrolled (sup)", "div_nog": "solver-in-loop, no grad (div-NOG)",
              "div_wig": "solver-in-loop, with grad (div-WIG)"}
MODE_TITLE = {"sup": "data-unrolled\n(sup;N)", "div_nog": "solver-in-the-loop, no grad\n(div;N, NOG)",
              "div_wig": "solver-in-the-loop, with grad\n(div;N, WIG)"}
PDE_LABEL = {-0.9: "explicit upwind, CFL 0.9", -3.0: "implicit upwind, CFL 3.0"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


# ------------------------------------------------------------------------------------------
def load():
    df = emusup.load_all_summaries()
    if df.empty:
        return df
    df["arch"] = df["cfg_network"].str.split(";").str[0]
    df["gamma_1"] = df["cfg_linear_difficulties"].apply(lambda t: t[1])
    df["pde"] = df["gamma_1"].map(PDE_LABEL)
    df["mode"] = df["cfg_train_mode"]
    df["N"] = df["cfg_num_unrolled_steps"]
    df["H"] = df["cfg_temporal_horizon_train"].where(df["cfg_temporal_horizon_train"] >= 0, df["N"])
    df["seed"] = df["cfg_seed_init"].astype(str) + "-" + df["cfg_seed_train"].astype(str) + "-" + \
        df["cfg_seed_test"].astype(str) + "-" + df["cfg_seed_shuffle"].astype(str)
    df["ic_train"] = df["cfg_ic_train"]
    # the paper's protocol: max-normalised single-mode ICs, one window per trajectory (H = N)
    df["std"] = (df["ic_train"] == "ind;1;True") & (df["H"] == df["N"]) & (df["cfg_ic_test"] == "ind;1;True")
    df = df[(df["cfg_convection_difficulty"] == 0.0) & (df["cfg_num_training_steps"] == 10000)]
    # an explicit temporal_horizon_train equal to N duplicates the default (-1) config; keep one
    df = df.drop_duplicates(subset=["cfg_network", "gamma_1", "mode", "N", "H", "ic_train", "seed", "cfg_ic_test"])
    return df


def load_burgers():
    df = emusup.load_all_summaries()
    if df.empty:
        return df
    df = df[df["cfg_convection_difficulty"] != 0.0].copy()
    if df.empty:
        return df
    df["arch"] = df["cfg_network"].str.split(";").str[0]
    df["mode"] = df["cfg_train_mode"]
    df["N"] = df["cfg_num_unrolled_steps"]
    df["seed"] = df["cfg_seed_init"].astype(str)
    return df


def curves(df_rows, tmax=100):
    """Stack superiority curves (time_step <= tmax) of the given summary rows -> (n_runs, tmax)."""
    out = []
    for name in df_rows["name"]:
        m = emusup.load_metrics(name)
        out.append(m["superiority"].values[:tmax])
    return np.array(out)


def authors_curves(network, gamma_1, tmax=100):
    out = []
    for d in glob.glob(os.path.join(AUTHORS, "*")):
        try:
            s = json.loads(os.path.basename(d))
        except Exception:
            continue
        if s["network"] == network and s["gamma_1"] == gamma_1 and s["test_ic_distribution"] == "ind;1;True":
            m = pd.read_csv(os.path.join(d, "metrics.csv.gz"))
            out.append(m["superiority"].values[:tmax])
    return np.array(out)


def band(ax, x, ys, color, label=None, ls="-", lw=1.6, alpha=0.18):
    med = np.median(ys, axis=0)
    ax.plot(x, med, color=color, label=label, ls=ls, lw=lw)
    if len(ys) > 1:
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=alpha, lw=0)


def style_sup_axis(ax, tmax=100, ylim=(0.0, 1.6)):
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_xlim(1, tmax)
    ax.set_ylim(*ylim)
    ax.set_xlabel("rollout step")


def fig_legend(fig, axes, ncol=2, y=-0.02, fontsize=7):
    """One legend for the whole figure, built from the unique labels of all panels."""
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=fontsize,
                   frameon=False)


# ------------------------------------------------------------------------------------------
def fig_repro(df):
    d = df[df["std"] & (df["mode"] == "sup") & (df["N"] == 1)]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), sharey=True)
    x = np.arange(1, 101)
    for ax, g in zip(axes, [-0.9, -3.0]):
        for arch, net in [("Conv", "Conv;34;10;relu"), ("FNO", "FNO;12;18;4;gelu")]:
            a = authors_curves(net, g)
            if len(a):
                ax.plot(x, np.median(a, axis=0), color=C_AUTH, ls=":", lw=1.3,
                        label="authors' runs (median)" if arch == "Conv" else None)
            r = d[(d["arch"] == arch) & (d["gamma_1"] == g)]
            if len(r):
                band(ax, x, curves(r), C_ARCH[arch], f"{arch}Net, ours ({len(r)} seeds)" if arch == "Conv" else f"{arch}, ours ({len(r)} seeds)")
        style_sup_axis(ax)
        ax.set_title(PDE_LABEL[g])
    axes[0].set_ylabel("superiority = err(emulator)/err(solver)")
    fig_legend(fig, axes, ncol=3)
    fig.savefig(os.path.join(FIG, "fig_repro.pdf"))
    fig.savefig(os.path.join(FIG, "fig_repro.png"))
    plt.close(fig)


def fig_unroll_curves(df, arch):
    d = df[(df["arch"] == arch) & df["std"]]
    if d.empty:
        return
    modes = ["sup", "div_nog", "div_wig"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True, sharey=True)
    x = np.arange(1, 101)
    for i, g in enumerate([-0.9, -3.0]):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            base = d[(d["gamma_1"] == g) & (d["mode"] == "sup") & (d["N"] == 1)]
            if len(base):
                band(ax, x, curves(base), C_N[1], f"N=1 (one-step, {len(base)} seeds)")
            for N in sorted(d["N"].unique()):
                if N == 1:
                    continue
                r = d[(d["gamma_1"] == g) & (d["mode"] == mode) & (d["N"] == N)]
                if len(r):
                    band(ax, x, curves(r), C_N.get(N, INK), f"N={N} ({len(r)})")
            style_sup_axis(ax)
            if i == 0:
                ax.set_title(MODE_TITLE[mode])
            if j == 0:
                ax.set_ylabel(f"{PDE_LABEL[g]}\nsuperiority")
            if i == 1:
                ax.set_xlabel("rollout step")
            ax.legend(loc="lower left", ncol=1)
    fig.suptitle(f"{arch}: superiority along the rollout for each unroll length N", y=1.01)
    fig.savefig(os.path.join(FIG, f"fig_unroll_curves_{arch}.pdf"))
    fig.savefig(os.path.join(FIG, f"fig_unroll_curves_{arch}.png"))
    plt.close(fig)


def fig_unroll_summary(df, metric="sup@50", label="superiority at rollout step 50"):
    d = df[df["std"]]
    if d.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), sharex=True, sharey="row")
    for i, g in enumerate([-0.9, -3.0]):
        for j, arch in enumerate(["Conv", "FNO"]):
            ax = axes[i, j]
            for mode in ["sup", "div_nog", "div_wig"]:
                r = d[(d["gamma_1"] == g) & (d["arch"] == arch) & ((d["mode"] == mode) | (d["N"] == 1))]
                if r.empty:
                    continue
                diverged = r[r[metric] > 10]  # autoregressive blow-ups are reported, not plotted
                r = r[r[metric] <= 10]
                agg = r.groupby("N")[metric].agg(["median", "min", "max", "count"]).reset_index()
                ax.plot(agg["N"], agg["median"], "o-", color=C_MODE[mode], label=MODE_LABEL[mode], ms=4)
                ax.fill_between(agg["N"], agg["min"], agg["max"], color=C_MODE[mode], alpha=0.15, lw=0)
                for N_d, n_d in diverged.groupby("N").size().items():
                    ax.annotate(f"{n_d} seed{'s' if n_d > 1 else ''} diverged", (N_d, 1.9), ha="center", va="top",
                                fontsize=6, color=C_MODE[mode])
            ax.axhline(1.0, color=INK, lw=0.8, ls="--")
            ax.set_ylim(0.0, 2.0)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10])
            ax.set_xticklabels(["1", "2", "5", "10"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(f"{arch}Net" if arch == "Conv" else arch)
            if j == 0:
                ax.set_ylabel(f"{PDE_LABEL[g]}\n{label}")
            if i == 1:
                ax.set_xlabel("unroll length N (training)")
    fig_legend(fig, axes, ncol=3)
    fig.savefig(os.path.join(FIG, f"fig_unroll_summary_{metric.replace('@', '')}.pdf"))
    fig.savefig(os.path.join(FIG, f"fig_unroll_summary_{metric.replace('@', '')}.png"))
    plt.close(fig)


def load_symbols(df):
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(emusup.RUNS_DIR, r["name"], "symbol.csv")
        if os.path.exists(p):
            s = pd.read_csv(p).query("k == 1").iloc[0]
            rows.append({**r[["name", "arch", "gamma_1", "pde", "mode", "N", "H", "seed"]].to_dict(),
                         "gain_dev_emu": s["gain_emulator"] - s["gain_exact"],
                         "gain_dev_solver": s["gain_solver"] - s["gain_exact"],
                         "phase_dev_emu": np.angle(np.exp(1j * (s["phase_emulator"] - s["phase_exact"]))),
                         "phase_dev_solver": np.angle(np.exp(1j * (s["phase_solver"] - s["phase_exact"]))),
                         "leak": s["leak_emulator"]})
    return pd.DataFrame(rows)


def fig_symbol(df):
    d = df[df["std"]]
    sym = load_symbols(d)
    if sym.empty:
        return sym
    for arch in ["Conv", "FNO"]:
        s = sym[sym["arch"] == arch]
        if s.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8), sharex=True)
        for j, g in enumerate([-0.9, -3.0]):
            for i, (col, lab) in enumerate([("gain_dev", "per-step amplitude error of mode 1"),
                                            ("phase_dev", "per-step phase error of mode 1 [rad]")]):
                ax = axes[i, j]
                ss = s[s["gamma_1"] == g]
                if ss.empty:
                    continue
                ax.axhline(ss[f"{col}_solver"].iloc[0], color=C_AUTH, ls=":", lw=1.3, label="low-fidelity solver")
                ax.axhline(0.0, color=INK, lw=0.8, ls="--", label="exact")
                for mode in ["sup", "div_nog", "div_wig"]:
                    r = ss[(ss["mode"] == mode) | (ss["N"] == 1)]
                    agg = r.groupby("N")[f"{col}_emu"].agg(["median", "min", "max"]).reset_index()
                    ax.plot(agg["N"], agg["median"], "o-", color=C_MODE[mode], ms=4, label=MODE_LABEL[mode])
                    ax.fill_between(agg["N"], agg["min"], agg["max"], color=C_MODE[mode], alpha=0.15, lw=0)
                ax.set_xscale("log")
                ax.set_xticks([1, 2, 5, 10])
                ax.set_xticklabels(["1", "2", "5", "10"])
                ax.minorticks_off()
                if i == 0:
                    ax.set_title(PDE_LABEL[g])
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 1:
                    ax.set_xlabel("unroll length N (training)")
        fig_legend(fig, axes, ncol=3)
        fig.suptitle(f"{arch}: learned one-step Fourier symbol vs. exact and vs. training solver", y=1.01)
        fig.savefig(os.path.join(FIG, f"fig_symbol_{arch}.pdf"))
        fig.savefig(os.path.join(FIG, f"fig_symbol_{arch}.png"))
        plt.close(fig)
    return sym


def predicted_superiority(name, T=100):
    """Linear-response prediction: for a single-mode initial condition and a *linear* one-step map
    with mode-1 symbol r (emulator), g (solver) and e (exact), the RMSE ratio after t steps is
    |r^t - e^t| / |g^t - e^t|.  Deviations from the measured curve quantify nonlinearity / leakage."""
    p = os.path.join(emusup.RUNS_DIR, name, "symbol.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p).query("k == 1").iloc[0]
    r = s["gain_emulator"] * np.exp(1j * s["phase_emulator"])
    g = s["gain_solver"] * np.exp(1j * s["phase_solver"])
    e = s["gain_exact"] * np.exp(1j * s["phase_exact"])
    t = np.arange(1, T + 1)
    return np.abs(r ** t - e ** t) / np.abs(g ** t - e ** t)


def fig_symbol_prediction(df):
    """Measured superiority curve vs. the prediction from the learned one-step symbol (seed 0 runs)."""
    d = df[df["std"] & (df["cfg_seed_init"] == 0)]
    if d.empty:
        return
    x = np.arange(1, 101)
    for arch in ["Conv", "FNO"]:
        da = d[d["arch"] == arch]
        if da.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True, sharey=True)
        for i, g in enumerate([-0.9, -3.0]):
            for j, mode in enumerate(["sup", "div_nog", "div_wig"]):
                ax = axes[i, j]
                for N in sorted(da["N"].unique()):
                    r = da[(da["gamma_1"] == g) & (da["N"] == N) & ((da["mode"] == mode) | (N == 1))]
                    if r.empty:
                        continue
                    name = r.iloc[0]["name"]
                    ax.plot(x, curves(r.iloc[:1])[0], color=C_N.get(N, INK), label=f"N={N} measured")
                    pred = predicted_superiority(name)
                    if pred is not None:
                        ax.plot(x, pred, color=INK, ls=(0, (1, 2)), lw=1.1, zorder=5,
                                label="symbol prediction" if N == 1 else None)
                style_sup_axis(ax)
                if i == 0:
                    ax.set_title(MODE_TITLE[mode])
                if j == 0:
                    ax.set_ylabel(f"{PDE_LABEL[g]}\nsuperiority")
                if i == 1:
                    ax.set_xlabel("rollout step")
        fig_legend(fig, axes, ncol=4)
        fig.suptitle(f"{arch}: measured superiority (colour, seed 0) vs. prediction from the learned mode-1 symbol (black dotted)", y=1.03)
        fig.savefig(os.path.join(FIG, f"fig_symbol_prediction_{arch}.pdf"))
        fig.savefig(os.path.join(FIG, f"fig_symbol_prediction_{arch}.png"))
        plt.close(fig)


def fig_errors(df):
    """Absolute rollout errors (emulator per N, and the low-fidelity solver) for the sup mode, seed 0."""
    d = df[df["std"] & (df["cfg_seed_init"] == 0) & (df["mode"] == "sup")]
    if d.empty:
        return
    x = np.arange(1, 101)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), sharex=True)
    for i, g in enumerate([-0.9, -3.0]):
        for j, arch in enumerate(["Conv", "FNO"]):
            ax = axes[i, j]
            r = d[(d["gamma_1"] == g) & (d["arch"] == arch)]
            if r.empty:
                continue
            m0 = emusup.load_metrics(r.iloc[0]["name"])
            ax.plot(x, m0["error_solver"].values[:100], color=C_AUTH, ls=":", lw=1.4, label="low-fidelity solver")
            for N in sorted(r["N"].unique()):
                m = emusup.load_metrics(r[r["N"] == N].iloc[0]["name"])
                ax.plot(x, m["error_emulator"].values[:100], color=C_N.get(N, INK), label=f"emulator, N={N}")
            ax.set_yscale("log")
            ax.set_xlim(1, 100)
            if i == 0:
                ax.set_title(f"{arch}Net" if arch == "Conv" else arch)
            if j == 0:
                ax.set_ylabel(f"{PDE_LABEL[g]}\nRMSE vs. reference")
            if i == 1:
                ax.set_xlabel("rollout step")
    fig_legend(fig, axes, ncol=3)
    fig.savefig(os.path.join(FIG, "fig_errors.pdf"))
    fig.savefig(os.path.join(FIG, "fig_errors.png"))
    plt.close(fig)


CONTROL_SERIES = [  # label, row filter, colour
    ("one-step, paper's data (N=1, H=1)", lambda d: d["std"] & (d["N"] == 1), "#2a78d6"),
    ("one-step on 10-step trajectories (N=1, H=10)", lambda d: (d["ic_train"] == "ind;1;True") & (d["N"] == 1) & (d["H"] == 10), "#eda100"),
    ("one-step, random-amplitude ICs (N=1, H=1)", lambda d: (d["ic_train"] == "indamp;1;0.5;1.5") & (d["N"] == 1) & (d["H"] == 1), "#1baf7a"),
    ("data-unrolled N=2 (sup)", lambda d: d["std"] & (d["N"] == 2) & (d["mode"] == "sup"), "#4a3aa7"),
]


def fig_controls(df):
    """Does training-data coverage of amplitudes below 1 remove superiority without any unrolling?"""
    x = np.arange(1, 101)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), sharex=True, sharey=True)
    any_data = False
    for i, g in enumerate([-0.9, -3.0]):
        for j, arch in enumerate(["Conv", "FNO"]):
            ax = axes[i, j]
            da = df[(df["arch"] == arch) & (df["gamma_1"] == g)]
            for label, filt, color in CONTROL_SERIES:
                r = da[filt(da)]
                if len(r):
                    any_data = True
                    band(ax, x, curves(r), color, f"{label} [{len(r)}]")
            style_sup_axis(ax)
            if i == 0:
                ax.set_title(f"{arch}Net" if arch == "Conv" else arch)
            if j == 0:
                ax.set_ylabel(f"{PDE_LABEL[g]}\nsuperiority")
            if i == 1:
                ax.set_xlabel("rollout step")
    if not any_data:
        plt.close(fig)
        return
    fig_legend(fig, axes, ncol=2)
    fig.savefig(os.path.join(FIG, "fig_controls.pdf"))
    fig.savefig(os.path.join(FIG, "fig_controls.png"))
    plt.close(fig)


def _series_of(row):
    """(label, colour) for the seed-0 diagnostic figures, or None to skip the run."""
    if row["std"] and row["mode"] == "sup":
        return f"sup, N={row['N']}", C_N.get(row["N"], INK)
    if row["std"] and row["N"] == 2:
        return {"div_nog": "div-NOG, N=2", "div_wig": "div-WIG, N=2"}[row["mode"]], C_MODE[row["mode"]]
    if row["ic_train"] == "indamp;1;0.5;1.5" and row["N"] == 1:
        return "random-amplitude ICs, N=1", "#1baf7a"
    if row["ic_train"] == "ind;1;True" and row["N"] == 1 and row["H"] == 10:
        return "10-step data, N=1", "#eda100"
    return None


def fig_amplitude_response(df):
    """Mode-1 gain of the trained emulator vs. input amplitude (seed 0). A linear map has a flat line."""
    for arch in ["Conv", "FNO"]:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
        any_data = False
        for ax, g in zip(axes, [-0.9, -3.0]):
            rows = df[(df["arch"] == arch) & (df["gamma_1"] == g) & (df["cfg_seed_init"] == 0)]
            for _, r in rows.iterrows():
                ser = _series_of(r)
                p = os.path.join(emusup.RUNS_DIR, r["name"], "symbol_amp.csv")
                if ser is None or not os.path.exists(p):
                    continue
                sa = pd.read_csv(p).query("k == 1 and amplitude >= 0.9")
                ax.plot(sa["amplitude"], sa["gain"], "o-", ms=3, color=ser[1], label=ser[0])
                any_data = True
            sy = os.path.join(emusup.RUNS_DIR, rows.iloc[0]["name"], "symbol.csv") if len(rows) else None
            if sy and os.path.exists(sy):
                s1 = pd.read_csv(sy).query("k == 1").iloc[0]
                ax.axhline(s1["gain_solver"], color=C_AUTH, ls=":", lw=1.3, label="low-fidelity solver")
                ax.axhline(s1["gain_exact"], color=INK, ls="--", lw=0.8, label="exact")
            ax.set_title(PDE_LABEL[g])
            ax.set_xlabel("input amplitude of the mode-1 wave")
            ax.set_xlim(0.9, 1.0)
        if not any_data:
            plt.close(fig)
            continue
        axes[0].set_ylabel("one-step gain of mode 1")
        fig_legend(fig, axes, ncol=3, y=-0.08)
        fig.suptitle(f"{arch}: amplitude dependence of the learned one-step map (training amplitude = 1)", y=1.03)
        fig.savefig(os.path.join(FIG, f"fig_amplitude_response_{arch}.pdf"))
        fig.savefig(os.path.join(FIG, f"fig_amplitude_response_{arch}.png"))
        plt.close(fig)


def fig_mode_rollout(df):
    """Amplitude and phase error of a unit mode-1 wave rolled out 100 steps (seed 0)."""
    for arch in ["Conv", "FNO"]:
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8), sharex=True)
        any_data = False
        for j, g in enumerate([-0.9, -3.0]):
            rows = df[(df["arch"] == arch) & (df["gamma_1"] == g) & (df["cfg_seed_init"] == 0)]
            ref_done = False
            for _, r in rows.iterrows():
                ser = _series_of(r)
                p = os.path.join(emusup.RUNS_DIR, r["name"], "mode_rollout.csv")
                if ser is None or not os.path.exists(p):
                    continue
                mr = pd.read_csv(p)
                t = mr["time_step"]
                if not ref_done:
                    axes[0, j].plot(t, mr["amp_solver"], color=C_AUTH, ls=":", lw=1.3, label="low-fidelity solver")
                    axes[0, j].plot(t, mr["amp_exact"], color=INK, ls="--", lw=0.8, label="exact")
                    dphi_s = np.angle(np.exp(1j * (mr["phase_solver"] - mr["phase_exact"])))
                    axes[1, j].plot(t, dphi_s, color=C_AUTH, ls=":", lw=1.3)
                    axes[1, j].axhline(0, color=INK, ls="--", lw=0.8)
                    ref_done = True
                axes[0, j].plot(t, mr["amp_emulator"], color=ser[1], label=ser[0])
                dphi = np.angle(np.exp(1j * (mr["phase_emulator"] - mr["phase_exact"])))
                axes[1, j].plot(t, dphi, color=ser[1])
                any_data = True
            axes[0, j].set_title(PDE_LABEL[g])
            axes[1, j].set_xlabel("rollout step")
            axes[1, j].set_xlim(1, 100)
        if not any_data:
            plt.close(fig)
            continue
        axes[0, 0].set_ylabel("amplitude of mode 1")
        axes[1, 0].set_ylabel("phase error of mode 1 [rad]")
        fig_legend(fig, axes, ncol=3)
        fig.suptitle(f"{arch}: rollout of a unit mode-1 wave under emulator, solver and exact dynamics", y=1.02)
        fig.savefig(os.path.join(FIG, f"fig_mode_rollout_{arch}.pdf"))
        fig.savefig(os.path.join(FIG, f"fig_mode_rollout_{arch}.png"))
        plt.close(fig)


def fig_burgers(db):
    """Burgers (paper Sec. 4.2 setting): superiority along the rollout per training mode and N."""
    if db is None or db.empty:
        return
    x = np.arange(1, 21)  # the one-Picard-step solver's error saturates (~0.85 nRMSE) by step 10
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for ax, mode in zip(axes, ["sup", "div_nog", "div_wig"]):
        for N in sorted(db["N"].unique()):
            r = db[((db["mode"] == mode) | (N == 1)) & (db["N"] == N)]
            if r.empty:
                continue
            band(ax, x, curves(r, tmax=20), C_N.get(N, INK), f"N={N} [{len(r)}]")
        ax.axhline(1.0, color=INK, lw=0.8, ls="--")
        ax.set_xlim(1, 20)
        ax.set_ylim(0.0, 2.0)
        ax.set_xlabel("rollout step")
        ax.set_title(MODE_TITLE[mode])
    axes[0].set_ylabel("superiority")
    fig_legend(fig, axes, ncol=4, y=-0.08)
    fig.suptitle("Burgers, FNO: implicit upwind with one Picard iteration vs. converged reference", y=1.22)
    fig.savefig(os.path.join(FIG, "fig_burgers.pdf"))
    fig.savefig(os.path.join(FIG, "fig_burgers.png"))
    plt.close(fig)


def burgers_md(db):
    if db is None or db.empty:
        return []
    lines = ["## Burgers (FNO, 100k steps): superiority (median [min, max] over seeds)", "",
             "| mode | N | n | step 1 | step 5 | step 10 | step 20 | step 50 | final loss |", "|---|---|---|---|---|---|---|---|---|"]
    for (mode, N), r in db.groupby(["mode", "N"]):
        cells = [fmt(r[m].agg(["median", "min", "max", "count"])) for m in ("sup@1", "sup@5", "sup@10", "sup@20", "sup@50")]
        lines.append(f"| {MODE_LABEL[mode]} | {N} | {len(r)} | " + " | ".join(cells) + f" | {r['loss_end'].median():.2e} |")
    lines.append("")
    return lines


def mechanism_table(df):
    """Per run: amplitude-only prediction of the step-100 superiority from the unit-mode rollout,
    |a_emu(100) - a_exact(100)| / |a_solver(100) - a_exact(100)|, next to the measured value."""
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(emusup.RUNS_DIR, r["name"], "mode_rollout.csv")
        if not os.path.exists(p):
            continue
        mr = pd.read_csv(p).iloc[99]
        pred = abs(mr["amp_emulator"] - mr["amp_exact"]) / abs(mr["amp_solver"] - mr["amp_exact"])
        dphi_e = np.angle(np.exp(1j * (mr["phase_emulator"] - mr["phase_exact"])))
        dphi_s = np.angle(np.exp(1j * (mr["phase_solver"] - mr["phase_exact"])))
        rows.append({"name": r["name"], "arch": r["arch"], "gamma_1": r["gamma_1"], "mode": r["mode"], "N": r["N"],
                     "H": r["H"], "ic_train": r["ic_train"], "std": r["std"], "measured": r["sup@100"],
                     "pred_amplitude_only": pred, "amp_emu_100": mr["amp_emulator"], "amp_solver_100": mr["amp_solver"],
                     "phase_err_emu_100": dphi_e, "phase_err_solver_100": dphi_s})
    return pd.DataFrame(rows)


def fig_mechanism_scatter(df):
    mt = mechanism_table(df)
    if mt.empty:
        return mt
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for g, color in [(-0.9, "#2a78d6"), (-3.0, "#eb6834")]:
        for arch, marker in [("Conv", "s"), ("FNO", "o")]:
            r = mt[(mt["gamma_1"] == g) & (mt["arch"] == arch)]
            if r.empty:
                continue
            ax.scatter(r["pred_amplitude_only"], r["measured"], s=16, marker=marker, color=color, alpha=0.8, lw=0,
                       label=f"{PDE_LABEL[g]}, {arch}")
    lo, hi = 0.03, 3.0
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted from mode-1 amplitude alone")
    ax.set_ylabel("measured superiority at step 100")
    ax.legend(loc="upper left", fontsize=6)
    fig.savefig(os.path.join(FIG, "fig_mechanism_scatter.pdf"))
    fig.savefig(os.path.join(FIG, "fig_mechanism_scatter.png"))
    plt.close(fig)
    mt.to_csv(os.path.join(PROJ, "results", "mechanism_table.csv"), index=False)
    return mt


# ------------------------------------------------------------------------------------------
def fmt(agg_row):
    if agg_row["count"] == 1:
        return f"{agg_row['median']:.3f}"
    return f"{agg_row['median']:.3f} [{agg_row['min']:.3f}, {agg_row['max']:.3f}]"


def summary_md(df, sym):
    lines = ["# Results summary (auto-generated by src/analyze.py)", ""]
    lines.append(f"Finished runs: {len(df)}  |  total training wall time: {df['time_train_s'].sum() / 3600:.1f} h (CPU)")
    lines.append("")
    # reproduction table
    d = df[df["std"] & (df["mode"] == "sup") & (df["N"] == 1)]
    if len(d):
        lines += ["## Reproduction of the one-step protocol (superiority, median over seeds; authors' median in brackets)", "",
                  "| setting | arch | seeds | t=1 | t=10 | t=50 | t=100 |", "|---|---|---|---|---|---|---|"]
        for g in [-0.9, -3.0]:
            for arch, net in [("Conv", "Conv;34;10;relu"), ("FNO", "FNO;12;18;4;gelu")]:
                r = d[(d["arch"] == arch) & (d["gamma_1"] == g)]
                if r.empty:
                    continue
                a = authors_curves(net, g)
                am = np.median(a, axis=0) if len(a) else None
                cells = []
                for t in [1, 10, 50, 100]:
                    ours = r[f"sup@{t}"].median()
                    cells.append(f"{ours:.3f}" + (f" ({am[t - 1]:.3f})" if am is not None else ""))
                lines.append(f"| {PDE_LABEL[g]} | {arch} | {len(r)} | " + " | ".join(cells) + " |")
        lines.append("")
    # main table
    d = df[df["std"]]
    for metric, lab in [("sup@50", "superiority at step 50"), ("sup@100", "superiority at step 100"),
                        ("sup_mean_1_100", "mean superiority over steps 1-100"), ("sup@1", "one-step superiority"),
                        ("loss_end", "final training loss")]:
        lines += [f"## {lab} by training mode and unroll length (median [min, max] over seeds)", ""]
        for g in [-0.9, -3.0]:
            for arch in ["Conv", "FNO"]:
                r = d[(d["gamma_1"] == g) & (d["arch"] == arch)]
                if r.empty:
                    continue
                Ns = sorted(r["N"].unique())
                lines += [f"**{PDE_LABEL[g]}, {arch}**", "", "| mode | " + " | ".join(f"N={N}" for N in Ns) + " |",
                          "|---|" + "---|" * len(Ns)]
                for mode in ["sup", "div_nog", "div_wig"]:
                    cells = []
                    for N in Ns:
                        rr = r[((r["mode"] == mode) | (r["N"] == 1)) & (r["N"] == N)]
                        if rr.empty:
                            cells.append("–")
                        else:
                            agg = rr[metric].agg(["median", "min", "max", "count"])
                            cells.append((f"{agg['median']:.2e}" if metric == "loss_end" else fmt(agg)) + f" (n={int(agg['count'])})")
                    lines.append(f"| {MODE_LABEL[mode]} | " + " | ".join(cells) + " |")
                lines.append("")
    # controls table
    lines += ["## Mechanism controls (median [min, max] over seeds)", "",
              "| setting | arch | series | n | sup@10 | sup@50 | sup@100 |", "|---|---|---|---|---|---|---|"]
    for g in [-0.9, -3.0]:
        for arch in ["Conv", "FNO"]:
            da = df[(df["arch"] == arch) & (df["gamma_1"] == g)]
            for label, filt, _ in CONTROL_SERIES:
                r = da[filt(da)]
                if r.empty:
                    continue
                cells = [fmt(r[m].agg(["median", "min", "max", "count"])) for m in ("sup@10", "sup@50", "sup@100")]
                lines.append(f"| {PDE_LABEL[g]} | {arch} | {label} | {len(r)} | " + " | ".join(cells) + " |")
    lines.append("")
    # random-amplitude training data vs unroll length (added 2026-10-19: does unrolling create superiority
    # once the amplitude artefact is removed from the training distribution?)
    d = df[df["ic_train"] == "indamp;1;0.5;1.5"]
    if len(d):
        lines += ["## Random-amplitude training ICs (amplitude ~ U(0.5, 1.5), test ICs unchanged) vs unroll length "
                  "(median [min, max] over seeds)", "",
                  "| setting | arch | mode | N | n | sup@10 | sup@50 | sup@100 | final loss |", "|---|---|---|---|---|---|---|---|---|"]
        for g in [-0.9, -3.0]:
            for arch in ["Conv", "FNO"]:
                for (mode, N), r in d[(d["gamma_1"] == g) & (d["arch"] == arch)].groupby(["mode", "N"]):
                    cells = [fmt(r[m].agg(["median", "min", "max", "count"])) for m in ("sup@10", "sup@50", "sup@100")]
                    lines.append(f"| {PDE_LABEL[g]} | {arch} | {MODE_LABEL[mode]} | {N} | {len(r)} | " + " | ".join(cells)
                                 + f" | {r['loss_end'].median():.2e} |")
        lines.append("")
    # symbol table
    if sym is not None and not sym.empty:
        lines += ["## Learned one-step symbol at mode k=1 (median over seeds): amplitude and phase error per step", "",
                  "| setting | arch | mode | N | gain err emulator | gain err solver | phase err emulator | phase err solver | leakage |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for g in [-0.9, -3.0]:
            for arch in ["Conv", "FNO"]:
                s = sym[(sym["gamma_1"] == g) & (sym["arch"] == arch)]
                for (mode, N), r in s.groupby(["mode", "N"]):
                    lines.append(f"| {PDE_LABEL[g]} | {arch} | {mode} | {N} | {r['gain_dev_emu'].median():+.2e} | "
                                 f"{r['gain_dev_solver'].median():+.2e} | {r['phase_dev_emu'].median():+.2e} | "
                                 f"{r['phase_dev_solver'].median():+.2e} | {r['leak'].median():.3f} |")
        lines.append("")
    lines += burgers_md(load_burgers())
    # timing
    lines += ["## Wall time per run on the laptop CPU (median seconds, training only)", ""]
    t = df.groupby(["arch", "mode", "N"])["time_train_s"].median().reset_index()
    lines += ["| arch | mode | N | train s |", "|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['arch']} | {r['mode']} | {r['N']} | {r['time_train_s']:.0f} |")
    with open(os.path.join(PROJ, "results", "summary.md"), "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def main():
    df = load()
    if df.empty:
        print("no finished runs")
        return
    fig_repro(df)
    fig_unroll_curves(df, "Conv")
    fig_unroll_curves(df, "FNO")
    fig_unroll_summary(df, "sup@50", "superiority at rollout step 50")
    fig_unroll_summary(df, "sup_mean_1_100", "mean superiority, steps 1-100")
    sym = fig_symbol(df)
    fig_symbol_prediction(df)
    fig_errors(df)
    fig_controls(df)
    fig_amplitude_response(df)
    fig_mode_rollout(df)
    fig_burgers(load_burgers())
    fig_mechanism_scatter(df)
    print(summary_md(df, sym))


if __name__ == "__main__":
    main()
