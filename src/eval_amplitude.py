"""
Post-hoc test of the amplitude mechanism: every saved emulator (linear advection runs) is evaluated on the
standard test initial conditions rescaled to amplitude a in AMPS.  The training protocol only ever shows
amplitude-1 inputs, so if superiority comes from the emulator's response below amplitude 1, superiority at
amplitude a < 1 should look very different from a = 1 (and the random-amplitude control should not care).

    python src/eval_amplitude.py        -> results/eval_amplitude.csv, figures/fig_test_amplitude_*.pdf|png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
import exponax as ex  # noqa: E402
import apebench  # noqa: E402
import emusup  # noqa: E402

AMPS = (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5)
OUT = os.path.join(emusup.PROJ, "results", "eval_amplitude.csv")
FIG = os.path.join(emusup.PROJ, "figures")


def evaluate(cfg: emusup.Config, model, T=100):
    train_sim = emusup.build_train_simulator(cfg)
    test_sim = emusup.build_test_simulator(cfg)
    metric = apebench.components.metric_dict[cfg.metric_test](cfg.metric_test)
    ic = emusup.make_ic_set(cfg.ic_test, cfg.num_points, cfg.num_samples_test, jax.random.key(cfg.seed_test))
    rows = []
    for a in AMPS:
        ic_a = ic * a
        ref = jax.vmap(ex.rollout(test_sim, T, include_init=False))(ic_a)
        e_s = np.asarray(emusup.rollout_error(train_sim, ic_a, ref, metric))
        e_m = np.asarray(emusup.rollout_error(model, ic_a, ref, metric))
        s = e_m / e_s
        rows.append({"amplitude": a, "sup@1": s[0], "sup@10": s[9], "sup@50": s[49], "sup@100": s[T - 1],
                     "err_emu@100": e_m[T - 1], "err_solver@100": e_s[T - 1]})
    return rows


def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(prev["name"])
    else:
        prev = pd.DataFrame()
    new = []
    for name in sorted(os.listdir(emusup.RUNS_DIR)):
        d = os.path.join(emusup.RUNS_DIR, name)
        if name in done or not os.path.exists(os.path.join(d, "model.eqx")):
            continue
        cfg = emusup.config_from_summary(os.path.join(d, "summary.json"))
        if not cfg.is_linear():
            continue
        model = emusup.load_model(cfg)
        for r in evaluate(cfg, model):
            r.update({"name": name, "arch": cfg.network.split(";")[0], "gamma_1": cfg.linear_difficulties[1],
                      "mode": cfg.train_mode, "N": cfg.num_unrolled_steps, "H": cfg.horizon_train(),
                      "ic_train": cfg.ic_train, "seed_init": cfg.seed_init})
            new.append(r)
        print("evaluated", name, flush=True)
    df = pd.concat([prev, pd.DataFrame(new)], ignore_index=True) if new else prev
    if df.empty:
        print("nothing to evaluate")
        return
    df.to_csv(OUT, index=False)
    plot(df)


def plot(df):
    C = {"paper N=1": "#2a78d6", "sup N=2": "#eb6834", "sup N=5": "#1baf7a", "random-amp N=1": "#4a3aa7",
         "10-step data N=1": "#eda100"}

    def series(r):
        if r["ic_train"] == "indamp;1;0.5;1.5":
            return "random-amp N=1"
        if r["ic_train"] == "ind;1;True" and r["N"] == 1 and r["H"] == 10:
            return "10-step data N=1"
        if r["ic_train"] == "ind;1;True" and r["H"] == r["N"] and r["mode"] == "sup":
            return {1: "paper N=1", 2: "sup N=2", 5: "sup N=5"}.get(r["N"])
        return None

    df = df.copy()
    df["series"] = df.apply(series, axis=1)
    df = df[df["series"].notna()]
    for arch in ["Conv", "FNO"]:
        da = df[df["arch"] == arch]
        if da.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
        for ax, g in zip(axes, [-0.9, -3.0]):
            d = da[da["gamma_1"] == g]
            for s_name, color in C.items():
                r = d[d["series"] == s_name]
                if r.empty:
                    continue
                agg = r.groupby("amplitude")["sup@100"].agg(["median", "min", "max", "count"]).reset_index()
                ax.plot(agg["amplitude"], agg["median"], "o-", ms=3, color=color, label=f"{s_name} [{int(agg['count'].max())}]")
                ax.fill_between(agg["amplitude"], agg["min"], agg["max"], color=color, alpha=0.15, lw=0)
            ax.axhline(1.0, color="#0b0b0b", lw=0.8, ls="--")
            ax.axvline(1.0, color="#898781", lw=0.8, ls=":")
            ax.set_title({-0.9: "explicit upwind, CFL 0.9", -3.0: "implicit upwind, CFL 3.0"}[g], fontsize=8.5)
            ax.set_xlabel("test amplitude (training: 1.0)", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylim(0.1, 100)
            ax.tick_params(labelsize=7)
            ax.grid(color="#e1e0d9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("superiority at step 100", fontsize=8)
        h, l = axes[1].get_legend_handles_labels()
        if not h:
            h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=7, frameon=False)
        fig.suptitle(f"{arch}: superiority when the test wave has a different amplitude than the training waves", y=1.04, fontsize=8.5)
        fig.savefig(os.path.join(FIG, f"fig_test_amplitude_{arch}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIG, f"fig_test_amplitude_{arch}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
