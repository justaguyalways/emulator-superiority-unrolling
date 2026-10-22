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
