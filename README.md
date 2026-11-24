# Does neural emulator superiority survive unrolled training?

CMU 10-749 "AI for Scientific Computing" (Fall 2026), course project.
Poornash Anandan Sangeetha (`aanandan`) and Abhinav Raghuvanshi (`araghuva`).

Reproduction-with-extension of Koehler & Thuerey, *Neural Emulator Superiority: When
Machine Learning for PDEs Surpasses its Training Data* (NeurIPS 2025, arXiv:2510.23111).

**Question.** The paper reports that an emulator trained only on trajectories of a cheap,
inaccurate solver can be *more* accurate than that solver (superiority ratio < 1), using
one-step training. Does the effect survive unrolled and solver-in-the-loop training, and
what produces it?

**Midway finding.** In the paper's linear-advection protocol, superiority is an
amplitude-extrapolation artefact of training only on waves of amplitude exactly 1.
Unrolling does not remove it for the explicit solver and flips it for the implicit one;
a 5 % change of the test amplitude or training on diverse amplitudes removes it entirely.

## Layout

| path | what |
|---|---|
| `src/emusup.py` | runner: config, solvers, training modes `sup` / `div_nog` / `div_wig`, rollout metrics, gain diagnostics, caching |
| `src/run_sweep.py` | experiment grids, sharded CPU execution |
| `src/analyze.py` | aggregates `results/runs/` into `figures/` and `results/summary.md` |
| `src/eval_amplitude.py` | knife-edge test: re-evaluates saved emulators on rescaled test waves |
| `src/fig_report.py`, `src/fig_story.py` | figures and table used in the midway report |
| `results/summary.md` | auto-generated result tables for all 137 runs |
| `results/eval_amplitude.csv`, `results/mechanism_table.csv` | knife-edge and gain/fixed-point data |
| `results/runs/<run>/` | per run: `summary.json`, `metrics.csv` (error and superiority per step), `symbol.csv`, `symbol_amp.csv` (gain vs amplitude), `mode_rollout.csv` |
| `figures/` | all diagnostic figures |

Large per-run artefacts (trajectories, model weights, loss curves) are not included; the
summaries above are sufficient to regenerate every figure and table.

## Setup

```bash
brew install uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python jax "apebench==0.1.1" jaxtyping jax-tqdm lineax exponax sympy pandas matplotlib seaborn jaxopt tqdm tabulate
mkdir -p repos && git clone https://github.com/tum-pbs/emulator-superiority repos/emulator-superiority
```

The authors' repository has no licence and is not redistributed here; it is used only for
their initial-condition generator and solver parser, for bit-compatibility with the paper.
Our runner is built on the MIT-licensed APEBench / Exponax / Trainax stack.

## Running

```bash
.venv/bin/python src/run_sweep.py --grid midterm_fno --dry-run   # list configs
./run_grid.sh midterm_fno                                         # run (cached by name)
.venv/bin/python src/analyze.py                                   # figures + results/summary.md
.venv/bin/python src/eval_amplitude.py                            # knife-edge test
.venv/bin/python src/fig_report.py && .venv/bin/python src/fig_story.py   # writes to ../report/ if present
```

## Protocol (paper's Sec. 4.1 unless stated)

* 1D linear advection on 100 periodic points; explicit first-order upwind at CFL 0.9 or
  implicit first-order upwind at CFL 3.0; reference: exact ETDRK solution.
* Training data: 300 single-mode initial conditions of amplitude exactly 1, rolled out N
  steps by the low-fidelity solver. Adam, 10 000 steps, warm-up-cosine schedule, batch 32.
* Emulators: ConvNet (31.8k parameters) and FNO (32.5k), as in the paper.
* Recipes: one-step (N = 1); data-unrolled `sup;N`; solver-in-the-loop `div;N` without (NOG)
  and with (WIG) gradients through the solver; N in {2, 5, 10}; 3 seeds per cell.
* Metric: superiority(t) = RMSE(emulator rollout)(t) / RMSE(solver rollout)(t), both
  against the exact solution, 10 test initial conditions, 200 steps.
* Controls: knife-edge test (test amplitude rescaled), random-amplitude training
  (U(0.5, 1.5)), one-step training on 10-step trajectories, multi-mode test, Burgers.

## November 22 extension (completed)

The focused extension is specified in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
All 66 primary configurations completed their full budgets with finite rollouts;
the separate calibration brings the fresh total to 67. The completion,
source hashes, unchanged historical summaries, and common teacher baselines are
verified in `results/extension_2026_11_22/audit.json`.
It runs 36 ConvNet amplitude-control cells, 18 FNO cells using the same five-step
teacher trajectories, and 12 UNet Burgers cells. Each comparison uses three
initialization seeds. One additional FNO calibration is retained separately.
All primary baselines are freshly trained: historical and new environments are
not numerically interchangeable and their results are not pooled.

Use Python 3.12 and the pinned dependencies in `requirements-extension.txt`.
The authors' dependency is pinned to commit
`461eb07d2714c958d43d4821abc78f2d31efff97` under the existing ignored `repos/`
directory. The current execution uses CPU JAX; workers are independent processes.
Run one pool at a time. Completed cells are skipped; unfinished cells resume their
saved weights **and Adam state** if configuration and source hashes match.

```bash
.venv/bin/python -m pip install -r requirements-extension.txt
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/validate_extension.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/run_extension_pool.py --workers 6
.venv/bin/python src/study_status.py

# Frozen-model diagnostics after training; these never select checkpoints.
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/homogeneity_control.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/common_teacher_fit.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/burgers_metric_sensitivity.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/diagnose_teacher_coverage.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/analyze_extension.py
.venv/bin/python src/audit_extension.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/export_extension_report.py
```

[Extension results](results/extension_2026_11_22/RESULTS.md) contain the complete
tables, figures, interpretation, and report-ready paragraphs.
The [extension PDF](results/extension_2026_11_22/Experiment_Results.pdf) and
[original midway report](reports/midway_report_story.pdf) are included.
The final PDF exporter also requires a reviewed `DISCUSSION.json` and all
prescribed outcomes. Seed ranges are descriptive, and conditional test-example
bootstrap intervals do not replace variability across training seeds.

Configurations, metrics, binned training curves, frozen-model diagnostics,
per-example errors, and numerical-validation records are retained with the
extension results. Full weights, optimizer checkpoints, trajectories, loss arrays,
and execution logs for all 67 fresh runs are included in the repository. The
cached numerical results suffice to regenerate the tables and figures; the saved
weights support reevaluation on new inputs. See the plan for the distinctions between total-error
ratios, individual rollout-step ratios, and the converged discrete Burgers
reference.

The exact numerical sources executed for the fresh runs are preserved in
`results/extension_2026_11_22/source_snapshot/`. After training finished, the live
runner received a serialization-only fix for nested nonfinite JSON values;
it does not change training or evaluation. Existing results retain their original
source hashes. Interrupted workers resumed saved Adam state; the resume check
in `validation.json` agrees exactly with uninterrupted updates.

## 2D forced-flow extension (completed November 24)

The fixed protocol is in [TURBULENCE_EXPERIMENT_PLAN.md](TURBULENCE_EXPERIMENT_PLAN.md).
All 12 prescribed FNO models completed 12,000 updates each with finite 100-step
forecasts: two independent 128-flow training datasets, three initialization seeds,
and supervised one-step versus four-step training on shared eight-step trajectories.
Training, warmup, and validation use only the rough solver. Testing uses 64 held-out
flows and an independently refined numerical reference.

The physical problem is periodic 2D incompressible Navier-Stokes with Kolmogorov
forcing, viscosity 0.01, scalar drag -0.1, and forcing-based Re=100. The rough
solver uses 32x32 points; references use 96x96 and 128x128. The initial fields have
multiple modes and naturally varying heights. This is a modest-Re forced-flow
study, without a claim to a developed inertial range or high-Re turbulence.

**Finding:** no model beats its teacher on the predefined 20-step score. Median
error ratios are 1.127 (one-step training) and 1.160 (four-step). The 48-grid
numerical control gives 0.121, while time refinement on the same 32-grid gives
0.999, indicating a dominant spatial-resolution effect in this comparison.
A zero-field control scores 0.885 in the later window despite zero energy,
demonstrating a limitation of long-horizon pointwise superiority metrics.

The [2D report](results/turbulence_2d_2026_11_23/Turbulence_2D_Results.pdf) and
[complete results](results/turbulence_2d_2026_11_23/RESULTS.md) include seed and
dataset ranges, conditional trajectory bootstraps, field and velocity errors,
energy/enstrophy/spectrum checks, fixed-example plots, and error decomposition.
The 2D implementation is independent of the authors' unlicensed repository.

```bash
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/turbulence_data.py coarse
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/turbulence_data.py reference
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/validate_turbulence.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/run_turbulence_pool.py --workers 3
.venv/bin/python src/turbulence_status.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/evaluate_turbulence.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/turbulence_long_reference.py
# Benchmark after training workers have finished.
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/benchmark_turbulence.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/audit_turbulence.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/analyze_turbulence.py
MPLCONFIGDIR=/tmp/emusup-mpl .venv/bin/python src/export_turbulence_report.py
```

Use the pinned dependencies in `requirements-extension.txt`. Numerical validation
caught a doubled-drag convention in Exponax 0.1.0; the 2D subclass applies the
declared scalar drag once and passes analytic decay and forcing checks. All main
data were regenerated after that preflight correction. Earlier preflight artifacts
are quarantined separately and are excluded from main results.

`results/turbulence_2d_2026_11_23/audit.json` verifies the full grid, budgets,
source/data/model hashes, score reconstruction, full-field projection identity,
and spectral energy consistency. Final weights, optimizer checkpoints, data,
losses, per-example metrics, and the executed numerical sources are retained.
Checkpoint resumption was validated against uninterrupted Adam updates.
