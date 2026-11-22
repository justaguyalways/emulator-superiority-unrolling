"""Resumable, auditable extension of the midway study. See EXPERIMENT_PLAN.md."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emusup-mpl")
import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import emusup
from emusup import Config
from fast_models import accelerate
from run_sweep import ARCHS, PDE_SETTINGS, SEEDS, seeds_cfg

ROOT = Path(emusup.PROJ)
OUT = ROOT / "results" / "extension_2026_11_22"
HIST = ROOT / "results" / "runs"


def grid(group):
    cases = []
    if group == "conv":
        for pde, (ld, sim) in PDE_SETTINGS.items():
            for amp in ["ind;1;True", "indamp;1;0.5;1.5"]:
                for n in [1, 2, 5]:
                    for s in SEEDS:
                        cases.append(Config(network=ARCHS["Conv"], linear_difficulties=ld,
                            train_simulator=sim, ic_train=amp, num_unrolled_steps=n,
                            tag="extension_conv", **seeds_cfg(s)))
    elif group == "fixedH":
        for ld, sim in PDE_SETTINGS.values():
            for n in [1, 2, 5]:
                for s in SEEDS:
                    cases.append(Config(network=ARCHS["FNO"], linear_difficulties=ld,
                        train_simulator=sim, num_unrolled_steps=n,
                        temporal_horizon_train=5 if n != 5 else -1,
                        tag="extension_fixedH5", **seeds_cfg(s)))
    elif group == "burgers":
        for mode, n in [("sup", 1), ("sup", 2), ("sup", 5), ("div_wig", 2)]:
            for s in SEEDS:
                cases.append(Config(network="UNet;12;2;relu",
                    linear_difficulties=(0., 0., .2, 0., 0.), convection_difficulty=-8.,
                    num_points=60, train_simulator="fd;1.0;1", test_simulator="fd;1.0;100",
                    ic_train="fourier;1;False;True", ic_test="fourier;1;False;True",
                    num_samples_train=500, num_samples_test=100,
                    temporal_horizon_test=100, metric_test="mean_nRMSE",
                    num_training_steps=100_000, warmup_steps=5_000, lr=3e-4,
                    train_mode=mode, num_unrolled_steps=n, tag="extension_burgers",
                    **seeds_cfg(s)))
    elif group == "calibration":
        cases = [Config(network=ARCHS["FNO"], tag="extension_calibration")]
    else:
        raise ValueError(group)
    return cases


def historical(cfg):
    p = HIST / cfg.name() / "summary.json"
    return p if p.exists() else None


def provenance(backend):
    packages = ["jax", "jaxlib", "equinox", "optax", "apebench", "trainax",
                "exponax", "pdequinox", "numpy", "lineax", "jaxopt"]
    return {"backend": backend, "platform": platform.platform(),
        "python": platform.python_version(), "devices": [str(d) for d in jax.devices()],
        "dependencies": {p: importlib.metadata.version(p) for p in packages},
        "source_sha256": {p: hashlib.sha256((ROOT / "src" / p).read_bytes()).hexdigest()
            for p in ["extension_study.py", "emusup.py", "fast_models.py"]},
        "authors_commit": subprocess.check_output(["git", "-C", str(ROOT / "repos" /
            "emulator-superiority"), "rev-parse", "HEAD"], text=True).strip()}


@eqx.filter_jit
def train_chunk(params, state, static, windows, indices, loss_cfg, teacher, optimizer):
    def step(carry, idx):
        p, s = carry
        def objective(p):
            return loss_cfg(eqx.combine(p, static), windows[idx], ref_stepper=teacher)
        loss, grad = jax.value_and_grad(objective)(p)
        update, s = optimizer.update(grad, s, p)
        return (optax.apply_updates(p, update), s), loss
    return jax.lax.scan(step, (params, state), indices)


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def finite_json(x):
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def evaluate(cfg, model, teacher, directory):
    ic = emusup.make_ic_set(cfg.ic_test, cfg.num_points, cfg.num_samples_test,
                           jax.random.key(cfg.seed_test))
    reference = emusup.build_test_simulator(cfg)
    ref = jax.vmap(ex.rollout(reference, cfg.temporal_horizon_test))(ic)
    metric = ex.metrics.RMSE if cfg.metric_test == "mean_RMSE" else ex.metrics.nRMSE

    @eqx.filter_jit
    def errors(stepper, ic, ref):
        def step(state, target):
            nxt = jax.vmap(stepper)(state)
            return nxt, jax.vmap(metric)(nxt, target)
        return jax.lax.scan(step, ic, ref.transpose(1, 0, 2, 3))[1]

    em = np.asarray(errors(model, ic, ref))
    so = np.asarray(errors(teacher, ic, ref))
    mean_em, mean_so = em.mean(axis=1), so.mean(axis=1)
    ratio = mean_em / mean_so
    pd.DataFrame({"time_step": np.arange(1, len(ratio)+1),
        "error_emulator": mean_em, "error_solver": mean_so,
        "superiority": ratio}).to_csv(directory / "metrics.csv", index=False)
    np.savez_compressed(directory / "per_sample_errors.npz", emulator=em, solver=so)
    out = {}
    for h in [20, 50, 100, 200]:
        if h > len(ratio):
            continue
        out[f"integrated_ratio_1_{h}"] = float(mean_em[:h].sum() / mean_so[:h].sum())
        out[f"mean_ratio_1_{h}"] = float(np.mean(ratio[:h]))
        out[f"fraction_superior_1_{h}"] = float(np.mean(ratio[:h] < 1))
    for t in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
        if t <= len(ratio):
            out[f"sup@{t}"] = float(ratio[t-1])
            out[f"err@{t}"] = float(mean_em[t-1])
            out[f"err_solver@{t}"] = float(mean_so[t-1])
    bad = np.flatnonzero(~np.isfinite(em).all(axis=1))
    out["first_nonfinite_step"] = int(bad[0]+1) if len(bad) else None
    bad = np.flatnonzero(np.any(em > 10, axis=1))
    out["first_any_sample_error_gt10"] = int(bad[0]+1) if len(bad) else None

    # Deterministic paired bootstrap over test examples, conditional on this model.
    h = 100 if cfg.is_linear() else 20
    rng = np.random.default_rng(20261122)
    idx = rng.integers(0, cfg.num_samples_test, (2000, cfg.num_samples_test))
    a, b = em[:h].sum(axis=0), so[:h].sum(axis=0)
    draws = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    out["primary_test_bootstrap95"] = [float(x) for x in np.quantile(draws, [.025, .975])]
    out["bootstrap_scope"] = "test examples only; conditional on trained model"

    if cfg.is_linear():
        emusup.amplitude_response(model, cfg.num_points).to_csv(directory / "symbol_amp.csv", index=False)
        rows = []
        for amp in [.95, 1., 1.05]:
            r = jax.vmap(ex.rollout(reference, 100))(ic * amp)
            a = np.asarray(errors(model, ic * amp, r)).mean(axis=1)
            b = np.asarray(errors(teacher, ic * amp, r)).mean(axis=1)
            rows.append({"amplitude": amp, "sup@100": float(a[-1]/b[-1]),
                         "integrated_ratio_1_100": float(a.sum()/b.sum())})
        pd.DataFrame(rows).to_csv(directory / "test_amplitude.csv", index=False)
    else:
        tighter = eqx.tree_at(lambda s:s.picard_tol, reference, 1e-7)
        tighter = eqx.tree_at(lambda s:s.picard_maxiter, tighter, 200)
        r = jax.vmap(ex.rollout(tighter, 20))(ic)
        a = np.asarray(errors(model, ic, r)).mean(axis=1)
        b = np.asarray(errors(teacher, ic, r)).mean(axis=1)
        out['integrated_ratio_1_20_tighter_reference'] = float(a.sum()/b.sum())
        out['tighter_reference_primary_difference'] = float(a.sum()/b.sum()-out['integrated_ratio_1_20'])
        pd.DataFrame({'time_step':np.arange(1,21),'error_emulator':a,
                      'error_solver':b,'superiority':a/b}).to_csv(directory/'metrics_tighter_reference.csv',index=False)
    # A small selection for inspecting propagation; not selected by performance.
    np.savez_compressed(directory / "trj.npz", ref=np.asarray(ref[:3, :100]),
        neural=np.asarray(jax.vmap(ex.rollout(model, min(100, cfg.temporal_horizon_test)))(ic[:3])),
        solver=np.asarray(jax.vmap(ex.rollout(teacher, min(100, cfg.temporal_horizon_test)))(ic[:3])))
    return out


def run(cfg, backend="fast", chunk=500, reuse=True):
    d = OUT / "runs" / cfg.name()
    done = d / "summary.json"
    if done.exists():
        return json.loads(done.read_text())
    if reuse and historical(cfg):
        print("REUSED", cfg.name(), flush=True)
        return None
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = emusup.build_network(cfg, jax.random.key(cfg.seed_init))
    if backend == "fast":
        model = accelerate(model, cfg.network)
    params, static = eqx.partition(model, eqx.is_array)
    optimizer = emusup.build_optimizer(cfg)
    state = optimizer.init(params)
    teacher = emusup.build_train_simulator(cfg)
    ic = emusup.make_ic_set(cfg.ic_train, cfg.num_points, cfg.num_samples_train,
                           jax.random.key(cfg.seed_train))
    trj = jax.vmap(ex.rollout(teacher, cfg.horizon_train(), include_init=True))(ic)
    windows = emusup.make_windows(trj, cfg.num_unrolled_steps+1)
    indices = emusup.make_batch_indices(len(windows), cfg.num_training_steps,
                                       cfg.batch_size, jax.random.key(cfg.seed_shuffle))
    loss_cfg = emusup.build_loss_config(cfg)
    meta = {"name": cfg.name(), "config": cfg.to_json(), "provenance": provenance(backend),
            "num_windows": len(windows), "n_params": sum(a.size for a in jax.tree_util.tree_leaves(params))}
    cursor, elapsed, loss_history = 0, 0., []
    progress = d / "checkpoint.json"
    if progress.exists():
        prior = json.loads(progress.read_text())
        if prior["config"] != cfg.to_json() or prior["provenance"] != meta["provenance"]:
            raise ValueError("Checkpoint configuration or implementation changed")
        params, state = eqx.tree_deserialise_leaves(d / "checkpoint.eqx", (params, state))
        cursor, elapsed = prior["completed_updates"], prior["training_seconds"]
        loss_history = list(np.load(d / "loss.npy"))
    print("START", cfg.name(), "from", cursor, flush=True)
    while cursor < cfg.num_training_steps:
        end = min(cfg.num_training_steps, cursor + chunk)
        ts = time.time()
        (params, state), losses = train_chunk(params, state, static, windows,
            indices[cursor:end], loss_cfg, teacher, optimizer)
        losses = np.asarray(jax.block_until_ready(losses))
        elapsed += time.time()-ts
        if not np.isfinite(losses).all():
            atomic_json(d / "failure.json", {**meta, "completed_updates": cursor,
                "reason": "nonfinite training loss", "training_seconds": elapsed})
            raise FloatingPointError(f"Nonfinite training loss: {cfg.name()}")
        loss_history.extend(losses.tolist())
        cursor = end
        checkpoint_tmp = d / "checkpoint.eqx.tmp"
        eqx.tree_serialise_leaves(checkpoint_tmp, (params, state))
        checkpoint_tmp.replace(d / "checkpoint.eqx")
        np.save(d / "loss.npy", np.array(loss_history))
        atomic_json(progress, {**meta, "completed_updates": cursor, "training_seconds": elapsed})
        print(json.dumps({"run": cfg.name(), "updates": cursor,
            "loss": float(np.mean(losses[-100:])), "train_s": round(elapsed, 2)}), flush=True)
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(d / "model.eqx", model)
    # Independent teacher-generated validation states, never used for selection.
    vi = emusup.make_ic_set(cfg.ic_train, cfg.num_points, 100, jax.random.key(110))
    vt = jax.vmap(ex.rollout(teacher, cfg.horizon_train(), include_init=True))(vi)
    vw = emusup.make_windows(vt, cfg.num_unrolled_steps+1)
    val = float(eqx.filter_jit(lambda m,w: loss_cfg(m,w,ref_stepper=teacher))(model,vw))
    summary = {**meta, "time_train_s": elapsed, "time_final_process_s": time.time()-t0,
        "completed_updates": cursor, "loss_end": float(np.median(loss_history[-100:])),
        "teacher_validation_loss": val, **evaluate(cfg, model, teacher, d)}
    atomic_json(done, finite_json(summary))
    print("FINISHED", cfg.name(), "primary", summary.get("integrated_ratio_1_20" if not cfg.is_linear()
                                                           else "integrated_ratio_1_100"), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--group", choices=["conv", "fixedH", "burgers", "calibration"], required=True)
    p.add_argument("--backend", choices=["fast", "original"], default="fast")
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--shard", nargs=2, type=int, default=[0, 1])
    p.add_argument("--no-reuse", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    cases = grid(args.group)[args.shard[0]::args.shard[1]]
    if args.action == "plan":
        for c in cases:
            status = "new"
            if (OUT/"runs"/c.name()/"summary.json").exists(): status = "completed extension"
            elif historical(c) and not args.no_reuse: status = "historical"
            print(status, c.name())
        return
    count = 0
    for c in cases:
        if args.limit and count >= args.limit: break
        if not (OUT/"runs"/c.name()/"summary.json").exists() and (args.no_reuse or not historical(c)):
            run(c, args.backend, args.chunk, reuse=not args.no_reuse)
            count += 1
            jax.clear_caches()
    print("GROUP FINISHED", args.group, flush=True)


if __name__ == "__main__":
    main()
