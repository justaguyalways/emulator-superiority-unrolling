"""
Sweep driver.  Usage:

    python src/run_sweep.py --grid repro            # list + run all configs of a grid
    python src/run_sweep.py --grid unroll --shard 0 4   # run every 4th config starting at 0
    python src/run_sweep.py --grid unroll --dry-run     # only list

Each config is cached in results/runs/<name>/ ; finished configs are skipped.
Launch several shards as separate processes for CPU parallelism, e.g.

    for i in 0 1 2 3; do XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        nohup .venv/bin/python src/run_sweep.py --grid unroll --shard $i 4 > results/logs/unroll_$i.log 2>&1 & done
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emusup  # noqa: E402
from emusup import Config  # noqa: E402

SEEDS = [(0, 10, 20, 30), (1, 10, 20, 30), (2, 10, 20, 30)]  # (init, train, test, shuffle)
PDE_SETTINGS = {  # name -> (linear_difficulties, train_simulator)
    "explicit_cfl0.9": ((0.0, -0.9, 0.0, 0.0, 0.0), "fd;0.0;1"),  # explicit first-order upwind, CFL 0.9
    "implicit_cfl3.0": ((0.0, -3.0, 0.0, 0.0, 0.0), "fd;1.0;1"),  # implicit first-order upwind, CFL 3.0
}
ARCHS = {"Conv": "Conv;34;10;relu", "FNO": "FNO;12;18;4;gelu"}


def seeds_cfg(s):
    return dict(seed_init=s[0], seed_train=s[1], seed_test=s[2], seed_shuffle=s[3])


def grid_repro():
    """The paper's one-step protocol; compare against the authors' CSVs."""
    cfgs = []
    for s, (pde, (ld, sim)), (an, net) in itertools.product(SEEDS, PDE_SETTINGS.items(), ARCHS.items()):
        cfgs.append(Config(network=net, linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                           num_unrolled_steps=1, tag="repro", **seeds_cfg(s)))
    return cfgs


def grid_unroll(Ns=(1, 2, 5, 10), modes=("sup", "div_nog", "div_wig"), seeds=SEEDS, archs=ARCHS,
                pdes=PDE_SETTINGS, tag="unroll"):
    """Main study: training mode x unroll length x architecture x PDE setting x seed."""
    cfgs = []
    for N, s, (an, net), (pde, (ld, sim)), mode in itertools.product(Ns, seeds, archs.items(), pdes.items(), modes):
        if mode != "sup" and N == 1:
            continue  # diverted chain with N = 1 is identical to one-step supervised
        cfgs.append(Config(network=net, linear_difficulties=ld, train_simulator=sim, train_mode=mode,
                           num_unrolled_steps=N, tag=tag, **seeds_cfg(s)))
    return cfgs


def grid_unroll_fixedH(Ns=(1, 2, 5, 10), H=10, seeds=SEEDS[:1]):
    """Control: same training horizon H for every N (windows from all time levels)."""
    cfgs = []
    for N, s, (an, net), (pde, (ld, sim)) in itertools.product(Ns, seeds, ARCHS.items(), PDE_SETTINGS.items()):
        cfgs.append(Config(network=net, linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                           num_unrolled_steps=N, temporal_horizon_train=H, tag="fixedH", **seeds_cfg(s)))
    return cfgs


def _cfg(net, pde, mode, N, s, tag):
    ld, sim = PDE_SETTINGS[pde]
    return Config(network=ARCHS[net], linear_difficulties=ld, train_simulator=sim, train_mode=mode,
                  num_unrolled_steps=N, tag=tag, **seeds_cfg(s))


def _ctrl(net, pde, s, kind, N=1):
    """Controls that separate 'unrolling' from 'training-data coverage of the radial direction':
    fixedH  : one window per time level of a 10-step low-fidelity trajectory (amplitudes below 1 seen in training)
    randamp : single-mode ICs with random amplitude instead of max-normalised ones (same test ICs)."""
    ld, sim = PDE_SETTINGS[pde]
    if kind == "fixedH":
        return Config(network=ARCHS[net], linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                      num_unrolled_steps=N, temporal_horizon_train=10, tag="fixedH", **seeds_cfg(s))
    if kind == "randamp":  # amplitude ~ U(0.5, 1.5) instead of exactly 1; test ICs unchanged (amplitude 1)
        return Config(network=ARCHS[net], linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                      num_unrolled_steps=N, ic_train="indamp;1;0.5;1.5", tag="randamp", **seeds_cfg(s))
    raise ValueError(kind)


def grid_midterm(net):
    """Priority-ordered list for one architecture: cheapest, most important results first.
    Measured CPU cost (M1): Conv ~10 min x N per run, FNO ~1.1 min x N per run."""
    P = list(PDE_SETTINGS)
    cfgs = []
    # 1. reproduction baseline, 3 seeds
    cfgs += [_cfg(net, pde, "sup", 1, s, "midterm") for s in SEEDS for pde in P]
    # 2. first unroll-length curve, seed 0
    cfgs += [_cfg(net, pde, "sup", N, SEEDS[0], "midterm") for N in (2, 5) for pde in P]
    # 3. mechanism controls at N=1, seed 0: 10-step training data; random-amplitude ICs
    cfgs += [_ctrl(net, pde, SEEDS[0], "fixedH") for pde in P]
    cfgs += [_ctrl(net, pde, SEEDS[0], "randamp") for pde in P]
    # 4. solver-in-the-loop at N=2, seed 0
    cfgs += [_cfg(net, pde, m, 2, SEEDS[0], "midterm") for m in ("div_nog", "div_wig") for pde in P]
    # 5. more seeds for the unroll curve and the controls
    cfgs += [_cfg(net, pde, "sup", N, s, "midterm") for N in (2, 5) for s in SEEDS[1:] for pde in P]
    cfgs += [_ctrl(net, pde, s, k) for s in SEEDS[1:] for k in ("fixedH", "randamp") for pde in P]
    # 6. solver-in-the-loop at N=5, seed 0; N=10 supervised, seed 0
    cfgs += [_cfg(net, pde, m, 5, SEEDS[0], "midterm") for m in ("div_nog", "div_wig") for pde in P]
    cfgs += [_cfg(net, pde, "sup", 10, SEEDS[0], "midterm") for pde in P]
    # 7. remaining seeds for div modes and N=10
    cfgs += [_cfg(net, pde, m, N, s, "midterm") for N in (2, 5) for s in SEEDS[1:] for m in ("div_nog", "div_wig") for pde in P]
    cfgs += [_cfg(net, pde, m, 10, SEEDS[0], "midterm") for m in ("div_nog", "div_wig") for pde in P]
    cfgs += [_cfg(net, pde, "sup", 10, s, "midterm") for s in SEEDS[1:] for pde in P]
    cfgs += [_cfg(net, pde, m, 10, s, "midterm") for s in SEEDS[1:] for m in ("div_nog", "div_wig") for pde in P]
    seen, out = set(), []
    for c in cfgs:
        if c.name() not in seen:
            seen.add(c.name())
            out.append(c)
    return out


def burgers_cfg(net, mode, N, s, tag="burgers"):
    """The paper's Sec. 4.2 Burgers setting (gamma_2 = 0.2, delta_1 = -8, 60 points, implicit FOU with one
    Picard iteration as the low-fidelity solver, converged Picard (100 iterations) as the reference)."""
    return Config(network=ARCHS[net], linear_difficulties=(0.0, 0.0, 0.2, 0.0, 0.0), convection_difficulty=-8.0,
                  num_points=60, train_simulator="fd;1.0;1", test_simulator="fd;1.0;100",
                  ic_train="fourier;1;False;True", ic_test="fourier;1;False;True",
                  num_samples_train=500, num_samples_test=20, num_training_steps=100_000, lr=3e-4,
                  warmup_steps=5_000, metric_test="mean_nRMSE", temporal_horizon_test=100,
                  train_mode=mode, num_unrolled_steps=N, tag=tag, **seeds_cfg(s))


def grid_extra_fno():
    """Cheap FNO additions, run after the main FNO grid: (a) fixed-horizon control, (b) Burgers."""
    P = list(PDE_SETTINGS)
    cfgs = []
    for N in (1, 2, 5, 10):
        for pde in P:
            ld, sim = PDE_SETTINGS[pde]
            cfgs.append(Config(network=ARCHS["FNO"], linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                               num_unrolled_steps=N, temporal_horizon_train=10, tag="fixedH", **seeds_cfg(SEEDS[0])))
    # the paper's third setting: implicit solver, multi-mode test ICs ("state-space generalisation")
    ld, sim = PDE_SETTINGS["implicit_cfl3.0"]
    for N in (1, 2):
        cfgs.append(Config(network=ARCHS["FNO"], linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                           num_unrolled_steps=N, ic_test="fourier;5;True;True", tag="multimode", **seeds_cfg(SEEDS[0])))
    for mode, N in [("sup", 1), ("sup", 2), ("sup", 5), ("div_nog", 2), ("div_wig", 2), ("div_nog", 5), ("div_wig", 5)]:
        cfgs.append(burgers_cfg("FNO", mode, N, SEEDS[0]))
    for mode, N in [("sup", 1), ("sup", 2), ("sup", 5)]:
        cfgs.append(burgers_cfg("FNO", mode, N, SEEDS[1]))
    for N in (2, 5, 10):
        for pde in P:
            ld, sim = PDE_SETTINGS[pde]
            for s in SEEDS[1:]:
                cfgs.append(Config(network=ARCHS["FNO"], linear_difficulties=ld, train_simulator=sim, train_mode="sup",
                                   num_unrolled_steps=N, temporal_horizon_train=10, tag="fixedH", **seeds_cfg(s)))
    return cfgs

def _randamp(net, pde, mode, N, s):
    """Random-amplitude single-mode training ICs (amplitude ~ U(0.5, 1.5)) with any training mode / unroll length."""
    ld, sim = PDE_SETTINGS[pde]
    return Config(network=ARCHS[net], linear_difficulties=ld, train_simulator=sim, train_mode=mode,
                  num_unrolled_steps=N, ic_train="indamp;1;0.5;1.5", tag="randamp", **seeds_cfg(s))


def grid_tonight():
    """Queue added 2026-10-19 evening (user: 'continue the experiments'), cheapest / most decisive first.
    1. The project's question asked on a training distribution WITHOUT the amplitude artefact: random-amplitude
       ICs + unrolled training (FNO, N=2 and 5, 3 seeds, both solvers). If the artefact explanation is right,
       every N should give ratio ~ 1 (plain imitation of the solver); superiority re-appearing here would be a
       genuine effect of unrolling and would change the report's conclusion.
    2. Solver-in-the-loop with random amplitudes (seed 0) for the same reason.
    3. The ConvNet random-amplitude control currently has one seed per solver; add seeds 1 and 2.
    4. Burgers: second seed for one-step and N=2 (the stability contrast), then div-WIG N=5 seed 0."""
    P = list(PDE_SETTINGS)
    cfgs = []
    cfgs += [_randamp("FNO", pde, "sup", 2, s) for s in SEEDS for pde in P]
    cfgs += [_randamp("FNO", pde, "sup", 5, s) for s in SEEDS for pde in P]
    cfgs += [_randamp("FNO", pde, "div_wig", N, SEEDS[0]) for N in (2, 5) for pde in P]
    cfgs += [_randamp("Conv", pde, "sup", 1, s) for s in SEEDS[1:] for pde in P]
    cfgs += [burgers_cfg("FNO", "sup", N, SEEDS[1]) for N in (1, 2)]
    cfgs += [burgers_cfg("FNO", "div_wig", 5, SEEDS[0])]
    return cfgs


GRIDS = {
    "tonight": grid_tonight,
    "extra_fno": grid_extra_fno,
    "repro": grid_repro,
    "unroll": grid_unroll,
    "unroll_seed0": lambda: grid_unroll(seeds=SEEDS[:1]),
    "unroll_sup": lambda: grid_unroll(modes=("sup",)),
    "unroll_div": lambda: grid_unroll(modes=("div_nog", "div_wig")),
    "fixedH": grid_unroll_fixedH,
    "midterm_conv": lambda: grid_midterm("Conv"),
    "midterm_fno": lambda: grid_midterm("FNO"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=sorted(GRIDS))
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "K"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="exit with code 3 after this many runs (process recycling frees JAX's compile cache)")
    args = ap.parse_args()
    cfgs = GRIDS[args.grid]()
    i, k = args.shard
    mine = cfgs[i::k]
    todo = [c for c in mine if args.force or not emusup.is_done(c)]
    print(f"grid={args.grid}: {len(cfgs)} configs, shard {i}/{k}: {len(mine)} ({len(todo)} to do)", flush=True)
    if args.dry_run:
        for c in mine:
            print(("  done  " if emusup.is_done(c) else "  todo  ") + c.name())
        return
    t0 = time.time()
    for j, c in enumerate(todo):
        try:
            emusup.run_and_save(c, force=args.force)
        except Exception as e:  # keep the sweep alive
            print(f"FAILED {c.name()}: {e!r}", flush=True)
        print(f"  progress {j + 1}/{len(todo)}  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if args.max_runs and j + 1 >= args.max_runs and j + 1 < len(todo):
            print(f"  recycling process after {args.max_runs} runs", flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main()
