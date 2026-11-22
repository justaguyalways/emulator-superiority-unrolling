"""
Emulator-superiority runner with extended training modes.

Re-implements the protocol of Koehler & Thuerey (NeurIPS 2025, "Neural Emulator
Superiority") on the MIT-licensed apebench / exponax / trainax stack and adds
the training modes needed for the unrolled-training study:

    sup      : data-unrolled supervised training over N steps ("sup;N" in APEBench;
               N = 1 is the paper's one-step training)
    div_nog  : solver-in-the-loop ("diverted chain", "div;N"), the low-fidelity solver
               is applied to the emulator's own prediction at every step; gradients are
               NOT propagated through the solver (NOG)
    div_wig  : same, but gradients flow through the solver (WIG)

The authors' unlicensed repository is used only for two small pieces that
guarantee bit-compatible numerics with their published CSVs: the single-mode
initial-condition generator (`individual_modes_ic.py`) and the finite-difference
solver parser (`superape/_parser.py`, wrapping their `picardax` package).

Superiority at rollout step t is   err_emulator(t) / err_low_fidelity_solver(t),
both measured against the ETDRK reference (exact for linear advection).
Values < 1 mean the emulator beats the solver it was trained on.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
AUTHORS_EXP = os.path.join(PROJ, "repos", "emulator-superiority", "code", "experiments")
RUNS_DIR = os.path.join(PROJ, "results", "runs")
sys.path.insert(0, AUTHORS_EXP)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import equinox as eqx  # noqa: E402
import optax  # noqa: E402
import exponax as ex  # noqa: E402
import trainax as tx  # noqa: E402
import apebench  # noqa: E402
import pandas as pd  # noqa: E402

from individual_modes_ic import individual_modes_gen  # noqa: E402  (authors' IC generator)
from superape._parser import parse_simulator  # noqa: E402  (authors' FD / ETDRK parser)

apebench.components.ic_dict["ind"] = individual_modes_gen


def individual_modes_random_amplitude_gen(ic_config: str, num_spatial_dims: int):
    """'indamp;<modes>;<lo>;<hi>': the authors' single-mode IC (random phase, max |u| = 1) rescaled by a
    random amplitude ~ U(lo, hi). The paper's 'ind;k;True' has amplitude exactly 1 for every sample."""
    parts = ic_config.split(";")
    active_modes = [int(m) for m in parts[1].split(",")]
    lo, hi = float(parts[2]), float(parts[3])

    def generate(num_points: int, key):
        state_hat = jnp.zeros((1, num_points // 2 + 1), dtype=jnp.complex64)
        for mode in active_modes:
            v = jax.random.normal(jax.random.fold_in(key, mode), (), dtype=jnp.complex64)
            state_hat = state_hat.at[0, mode].set(v)
        state = jnp.fft.irfft(state_hat, n=num_points)
        state = state / jnp.max(jnp.abs(state))
        amp = jax.random.uniform(jax.random.fold_in(key, 7919), (), minval=lo, maxval=hi)
        return state * amp

    return generate


apebench.components.ic_dict["indamp"] = individual_modes_random_amplitude_gen


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # emulator
    network: str = "Conv;34;10;relu"
    # PDE in APEBench "difficulty" units: linear_difficulties = (gamma_0 drag, gamma_1 advection,
    # gamma_2 diffusion, gamma_3 dispersion, gamma_4 hyper-diffusion); gamma_1 = -CFL
    linear_difficulties: tuple = (0.0, -0.9, 0.0, 0.0, 0.0)
    convection_difficulty: float = 0.0  # Burgers nonlinearity (0 = linear advection)
    num_points: int = 100
    # low-fidelity training solver and high-fidelity reference
    train_simulator: str = "fd;0.0;1"  # fd;theta;picard_iters  (theta 0 = explicit, 1 = implicit)
    test_simulator: str = "etdrk;2"
    ic_train: str = "ind;1;True"
    ic_test: str = "ind;1;True"
    # training protocol
    train_mode: str = "sup"  # sup | div_nog | div_wig
    num_unrolled_steps: int = 1
    temporal_horizon_train: int = -1  # -1 -> equals num_unrolled_steps (one window per trajectory)
    num_samples_train: int = 300
    num_training_steps: int = 10_000
    lr: float = 1e-3
    warmup_steps: int = 1_000
    batch_size: int = 32
    metric_train: str = "mean_MSE"
    # evaluation
    temporal_horizon_test: int = 200
    num_samples_test: int = 10
    metric_test: str = "mean_RMSE"
    symbol_modes: int = 20  # Fourier-symbol diagnostic for modes 1..symbol_modes (linear PDEs)
    # seeds
    seed_init: int = 0
    seed_train: int = 10
    seed_test: int = 20
    seed_shuffle: int = 30
    # implementation switches
    linearize_fd: bool = True  # replace the FD solver by its exact matrix if the PDE is linear
    tag: str = ""  # free-form label for grouping runs

    def horizon_train(self) -> int:
        return self.num_unrolled_steps if self.temporal_horizon_train < 0 else self.temporal_horizon_train

    def is_linear(self) -> bool:
        return self.convection_difficulty == 0.0

    def key(self) -> str:
        d = asdict(self)
        d["linear_difficulties"] = list(d["linear_difficulties"])
        d.pop("tag", None)  # labels do not change the experiment
        s = json.dumps(d, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:10]

    def name(self) -> str:
        g = self.linear_difficulties
        pde = f"adv{g[1]:+g}" if self.is_linear() else f"burg_g2{g[2]:g}_d{self.convection_difficulty:+g}"
        net = self.network.split(";")[0]
        return (f"{pde}_{self.train_simulator.replace(';', '-')}_{net}_{self.train_mode}"
                f"_N{self.num_unrolled_steps}_H{self.horizon_train()}_s{self.seed_init}-{self.seed_train}"
                f"-{self.seed_test}-{self.seed_shuffle}_{self.key()}")

    def to_json(self) -> dict:
        d = asdict(self)
        d["linear_difficulties"] = list(d["linear_difficulties"])
        return d


# --------------------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------------------
class LinearStepper(eqx.Module):
    """u -> M u  for a channel-1 state of shape (1, P). Exact replacement of a linear FD step."""

    M: jax.Array

    def __call__(self, u):
        return (self.M @ u[0])[None]


class OnePicardStepper(eqx.Module):
    """Wraps the authors' picardax stepper restricted to exactly one Picard step so that
    reverse-mode AD differentiates through the actual computation (one linear solve at the
    previous state) instead of jaxopt's implicit differentiation of a converged fixed point."""

    stepper: eqx.Module

    def __call__(self, u):
        return self.stepper.p1_step(u)


def build_train_simulator(cfg: Config):
    sim = parse_simulator(cfg.train_simulator, cfg.num_points, tuple(cfg.linear_difficulties),
                          cfg.convection_difficulty)
    if cfg.train_simulator.startswith("fd") and cfg.is_linear() and cfg.linearize_fd:
        # column j of M is the solver applied to the j-th unit vector (no autodiff needed;
        # the picardax solver uses jaxopt implicit diff, which has no forward-mode rule)
        M = jax.vmap(lambda e: sim(e[None])[0])(jnp.eye(cfg.num_points)).T
        # verify linearity on a random state
        u = jax.random.normal(jax.random.key(0), (1, cfg.num_points))
        err = float(jnp.max(jnp.abs(sim(u) - (M @ u[0])[None])))
        assert err < 1e-4, f"FD stepper is not linear (max dev {err})"
        return LinearStepper(M)
    if cfg.train_simulator.startswith("fd") and int(cfg.train_simulator.split(";")[2]) == 1:
        return OnePicardStepper(sim)
    return sim


def build_test_simulator(cfg: Config):
    return parse_simulator(cfg.test_simulator, cfg.num_points, tuple(cfg.linear_difficulties),
                           cfg.convection_difficulty)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def make_ic_set(ic_config: str, num_points: int, num_samples: int, key):
    gen = apebench.components.ic_dict[ic_config.split(";")[0].lower()](ic_config, 1)
    return ex.build_ic_set(gen, num_points=num_points, num_samples=num_samples, key=key)


def make_windows(trj_set, window: int):
    """(S, T+1, C, P) -> (S*(T+2-window), window, C, P) consecutive sub-trajectories."""
    sub = jax.vmap(ex.stack_sub_trajectories, in_axes=(0, None))(trj_set, window)
    return jnp.concatenate(sub)


def make_batch_indices(num_windows: int, num_steps: int, batch_size: int, key):
    """Same sampling scheme as the authors: concatenated random permutations (epochs), sliced."""
    num_epochs = math.ceil(num_steps * batch_size / num_windows)
    perms = jax.vmap(jax.random.permutation, in_axes=(0, None))(
        jax.random.split(key, num_epochs), jnp.arange(num_windows))
    return jnp.concatenate(perms)[: num_steps * batch_size].reshape(num_steps, batch_size)


# --------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------
def build_network(cfg: Config, key):
    name = cfg.network.split(";")[0].lower()
    act = apebench.components.activation_fn_dict[cfg.network.split(";")[-1].lower()]("")
    return apebench.components.architecture_dict[name](cfg.network, 1, cfg.num_points, 1, act, key)


def build_loss_config(cfg: Config):
    metric = apebench.components.metric_dict[cfg.metric_train](cfg.metric_train)
    N = cfg.num_unrolled_steps
    if cfg.train_mode == "sup":
        return tx.configuration.Supervised(N, time_level_loss=metric)
    if cfg.train_mode == "div_nog":
        return tx.configuration.DivertedChainBranchOne(N, time_level_loss=metric, cut_div_chain=True)
    if cfg.train_mode == "div_wig":
        return tx.configuration.DivertedChainBranchOne(N, time_level_loss=metric, cut_div_chain=False)
    raise ValueError(cfg.train_mode)


def build_optimizer(cfg: Config):
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, cfg.warmup_steps, cfg.num_training_steps)
    return optax.adam(sched)


def train(model, windows, loss_cfg, ref_stepper, optimizer, batch_idx):
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    def loss_fn(p, batch):
        return loss_cfg(eqx.combine(p, static), batch, ref_stepper=ref_stepper)

    def step(carry, idx):
        p, s = carry
        loss, grads = jax.value_and_grad(loss_fn)(p, windows[idx])
        updates, s = optimizer.update(grads, s)
        p = optax.apply_updates(p, updates)
        return (p, s), loss

    (params, _), losses = jax.lax.scan(step, (params, opt_state), batch_idx)
    return eqx.combine(params, static), losses


def rollout_error(stepper, ic_set, ref_trj, metric):
    """Autoregressive rollout from ic_set, error against ref_trj (S, T, C, P) at every step."""

    def scan_fn(state, ref_next):
        nxt = jax.vmap(stepper)(state)
        return nxt, metric(nxt, ref_next)

    _, err = jax.lax.scan(scan_fn, ic_set, ref_trj.transpose(1, 0, 2, 3))
    return err


def fourier_symbol(stepper, num_points: int, modes: int, amplitude: float = 1.0):
    """Apply a stepper to cosines of wavenumber k = 1..modes and amplitude `amplitude` and return the
    complex response at mode k (gain and phase) plus the relative energy leaked into other modes.
    For a linear stepper the response is independent of the amplitude."""
    x = jnp.arange(num_points) / num_points
    ks = jnp.arange(1, modes + 1)
    u = amplitude * jnp.cos(2 * jnp.pi * ks[:, None] * x[None, :])[:, None, :]  # (K, 1, P)
    out = jax.vmap(stepper)(u)[:, 0, :]
    U = jnp.fft.rfft(u[:, 0, :], axis=-1)
    O = jnp.fft.rfft(out, axis=-1)
    resp = O[jnp.arange(modes), ks] / U[jnp.arange(modes), ks]
    energy = jnp.sum(jnp.abs(O) ** 2, axis=-1)
    own = jnp.abs(O[jnp.arange(modes), ks]) ** 2
    leak = jnp.sqrt(jnp.clip(energy - own, 0.0) / jnp.maximum(own, 1e-30))
    return resp, leak


AMPLITUDES = (1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5)


def amplitude_response(stepper, num_points: int, modes: int = 3) -> pd.DataFrame:
    """Gain/phase of the mode-k response as a function of the input amplitude (nonlinearity probe:
    training data in the paper's protocol has max-amplitude exactly 1, rollouts drift below it)."""
    rows = []
    for a in AMPLITUDES:
        r, leak = fourier_symbol(stepper, num_points, modes, amplitude=a)
        r, leak = np.asarray(r), np.asarray(leak)
        for k in range(modes):
            rows.append({"amplitude": a, "k": k + 1, "gain": float(np.abs(r[k])),
                         "phase": float(np.angle(r[k])), "leak": float(leak[k])})
    return pd.DataFrame(rows)


def mode_rollout(stepper, num_points: int, steps: int = 100, k: int = 1, amplitude: float = 1.0):
    """Roll a single cosine mode through a stepper; return amplitude |mode k| and phase per step."""
    x = jnp.arange(num_points) / num_points
    u = amplitude * jnp.cos(2 * jnp.pi * k * x)[None, :]

    def scan_fn(state, _):
        nxt = stepper(state)
        c = jnp.fft.rfft(nxt[0])[k] / (num_points / 2)
        return nxt, (jnp.abs(c), jnp.angle(c))

    _, (amp, ph) = jax.lax.scan(scan_fn, u, None, length=steps)
    return np.asarray(amp), np.asarray(ph)


def run(cfg: Config, verbose: bool = True) -> dict:
    t0 = time.time()
    train_sim = build_train_simulator(cfg)
    test_sim = build_test_simulator(cfg)
    metric_test = apebench.components.metric_dict[cfg.metric_test](cfg.metric_test)

    key_init, key_train, key_test, key_shuffle = (jax.random.key(s) for s in
                                                  (cfg.seed_init, cfg.seed_train, cfg.seed_test, cfg.seed_shuffle))
    network = build_network(cfg, key_init)
    n_params = sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(eqx.filter(network, eqx.is_array)))

    # training data from the low-fidelity solver
    H = cfg.horizon_train()
    train_ic = make_ic_set(cfg.ic_train, cfg.num_points, cfg.num_samples_train, key_train)
    train_trj = jax.vmap(ex.rollout(train_sim, H, include_init=True))(train_ic)  # (S, H+1, 1, P)
    windows = make_windows(train_trj, cfg.num_unrolled_steps + 1)
    batch_idx = make_batch_indices(windows.shape[0], cfg.num_training_steps, cfg.batch_size, key_shuffle)

    # test data from the high-fidelity reference, plus the low-fidelity solver's own rollout
    test_ic = make_ic_set(cfg.ic_test, cfg.num_points, cfg.num_samples_test, key_test)
    test_ref = jax.vmap(ex.rollout(test_sim, cfg.temporal_horizon_test, include_init=False))(test_ic)
    err_solver = rollout_error(train_sim, test_ic, test_ref, metric_test)

    loss_cfg = build_loss_config(cfg)
    optimizer = build_optimizer(cfg)
    t1 = time.time()
    trained, losses = eqx.filter_jit(train)(network, windows, loss_cfg, train_sim, optimizer, batch_idx)
    losses = jax.block_until_ready(losses)
    t_train = time.time() - t1

    err_emu = rollout_error(trained, test_ic, test_ref, metric_test)
    sup = err_emu / err_solver
    neural_trj = jax.vmap(ex.rollout(trained, min(cfg.temporal_horizon_test, 100), include_init=False))(test_ic[:2])
    solver_trj = jax.vmap(ex.rollout(train_sim, min(cfg.temporal_horizon_test, 100), include_init=False))(test_ic[:2])

    out = {
        "config": cfg.to_json(),
        "name": cfg.name(),
        "n_params": n_params,
        "num_windows": int(windows.shape[0]),
        "losses": np.asarray(losses),
        "error_emulator": np.asarray(err_emu),
        "error_solver": np.asarray(err_solver),
        "superiority": np.asarray(sup),
        "trj_ref": np.asarray(test_ref[:2, :100]),
        "trj_neural": np.asarray(neural_trj),
        "trj_solver": np.asarray(solver_trj),
        "time_train_s": t_train,
        "time_total_s": time.time() - t0,
    }
    out["model"] = trained
    if cfg.is_linear():
        K = cfg.symbol_modes
        r_emu, leak_emu = fourier_symbol(trained, cfg.num_points, K)
        r_fd, _ = fourier_symbol(train_sim, cfg.num_points, K)
        r_ex, _ = fourier_symbol(test_sim, cfg.num_points, K)
        out["symbol"] = pd.DataFrame({
            "k": np.arange(1, K + 1),
            "gain_emulator": np.abs(np.asarray(r_emu)), "phase_emulator": np.angle(np.asarray(r_emu)),
            "gain_solver": np.abs(np.asarray(r_fd)), "phase_solver": np.angle(np.asarray(r_fd)),
            "gain_exact": np.abs(np.asarray(r_ex)), "phase_exact": np.angle(np.asarray(r_ex)),
            "leak_emulator": np.asarray(leak_emu),
        })
        out["symbol_amp"] = amplitude_response(trained, cfg.num_points)
        amp_e, ph_e = mode_rollout(trained, cfg.num_points)
        amp_s, ph_s = mode_rollout(train_sim, cfg.num_points)
        amp_x, ph_x = mode_rollout(test_sim, cfg.num_points)
        out["mode_rollout"] = pd.DataFrame({
            "time_step": np.arange(1, 101), "amp_emulator": amp_e, "phase_emulator": ph_e,
            "amp_solver": amp_s, "phase_solver": ph_s, "amp_exact": amp_x, "phase_exact": ph_x})
    if verbose:
        s = out["superiority"]
        sup_str = " ".join(f"@{t}={s[t - 1]:.3f}" for t in (1, 10, 50, 100) if t <= len(s))
        print(f"[{cfg.name()}] params={n_params} windows={windows.shape[0]} train={t_train:.0f}s "
              f"total={out['time_total_s']:.0f}s loss_end={float(losses[-1]):.2e} sup {sup_str}", flush=True)
    return out


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def run_dir(cfg: Config) -> str:
    return os.path.join(RUNS_DIR, cfg.name())


def is_done(cfg: Config) -> bool:
    return os.path.exists(os.path.join(run_dir(cfg), "summary.json"))


def save(out: dict, cfg: Config):
    d = run_dir(cfg)
    os.makedirs(d, exist_ok=True)
    T = len(out["superiority"])
    pd.DataFrame({"time_step": np.arange(1, T + 1), "error_emulator": out["error_emulator"],
                  "error_solver": out["error_solver"], "superiority": out["superiority"]}
                 ).to_csv(os.path.join(d, "metrics.csv"), index=False)
    pd.DataFrame({"update_step": np.arange(1, len(out["losses"]) + 1), "loss": out["losses"]}
                 ).to_csv(os.path.join(d, "loss.csv.gz"), index=False, compression="gzip")
    if "symbol" in out:
        out["symbol"].to_csv(os.path.join(d, "symbol.csv"), index=False)
    if "symbol_amp" in out:
        out["symbol_amp"].to_csv(os.path.join(d, "symbol_amp.csv"), index=False)
    if "mode_rollout" in out:
        out["mode_rollout"].to_csv(os.path.join(d, "mode_rollout.csv"), index=False)
    if "model" in out:
        eqx.tree_serialise_leaves(os.path.join(d, "model.eqx"), out["model"])
    np.savez_compressed(os.path.join(d, "trj.npz"), ref=out["trj_ref"], neural=out["trj_neural"],
                        solver=out["trj_solver"])
    s = out["superiority"]
    summary = {
        "name": out["name"], "config": out["config"], "n_params": out["n_params"],
        "num_windows": out["num_windows"], "time_train_s": out["time_train_s"],
        "time_total_s": out["time_total_s"], "loss_end": float(np.median(out["losses"][-100:])),
        **{f"sup@{t}": float(s[t - 1]) for t in (1, 2, 5, 10, 20, 50, 100, 200) if t <= T},
        **{f"err@{t}": float(out["error_emulator"][t - 1]) for t in (1, 10, 50, 100) if t <= T},
        **{f"err_solver@{t}": float(out["error_solver"][t - 1]) for t in (1, 10, 50, 100) if t <= T},
        "sup_mean_1_50": float(np.mean(s[:50])), "sup_mean_1_100": float(np.mean(s[:100])),
    }
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def run_and_save(cfg: Config, force: bool = False, verbose: bool = True):
    if is_done(cfg) and not force:
        with open(os.path.join(run_dir(cfg), "summary.json")) as f:
            return json.load(f)
    return save(run(cfg, verbose=verbose), cfg)


def load_all_summaries() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(RUNS_DIR):
        return pd.DataFrame()
    for name in sorted(os.listdir(RUNS_DIR)):
        p = os.path.join(RUNS_DIR, name, "summary.json")
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            row = {k: v for k, v in s.items() if k != "config"}
            row.update({f"cfg_{k}": (tuple(v) if isinstance(v, list) else v) for k, v in s["config"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def load_metrics(cfg_or_name) -> pd.DataFrame:
    name = cfg_or_name if isinstance(cfg_or_name, str) else cfg_or_name.name()
    return pd.read_csv(os.path.join(RUNS_DIR, name, "metrics.csv"))


def load_model(cfg: Config):
    """Rebuild the architecture and load the trained weights of a finished run."""
    skeleton = build_network(cfg, jax.random.key(cfg.seed_init))
    return eqx.tree_deserialise_leaves(os.path.join(run_dir(cfg), "model.eqx"), skeleton)


def config_from_summary(path: str) -> Config:
    with open(path) as f:
        d = json.load(f)["config"]
    d["linear_difficulties"] = tuple(d["linear_difficulties"])
    return Config(**d)
