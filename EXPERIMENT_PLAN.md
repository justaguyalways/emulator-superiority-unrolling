# Extension of the midway report: experiments 1 and 2

Specified on 2026-11-22 before collecting extension outcomes. The 137 existing
runs are historical evidence; their identities and hashes are recorded in
`results/extension_2026_11_22/historical_manifest.json`. Historical runs must not
be described as newly trained. All failed and unstable runs remain in the results.

## Questions and comparisons

1. **Architecture replication of the amplitude control.** Complete the ConvNet
   comparison between amplitude exactly 1 and amplitude uniform on [0.5, 1.5],
   for supervised unroll lengths N=1,2,5 and both advection solvers. Use the
   original 300 training samples, 10,000 Adam updates, batch 32, peak rate 0.001,
   1,000-step warmup, training/test/shuffle seeds 10/20/30, and initialization
   seeds 0/1/2. Training horizon H=N follows the midway study. Reuse identified
   historical cells where available; record every new cell separately.
   Hypothesis: amplitude diversity reduces the large, time-growing advantage;
   this does not require every individual ratio to equal one.

2. **Separate training horizon from loss unrolling.** With FNO and unit-amplitude
   waves, hold the underlying training trajectories fixed at H=5, and compare
   N=1,2,5 on both solvers and all three initialization seeds. N=1 and N=2
   use sliding windows of those same trajectories. Compare with the historical
   H=N study. The number of windows differs, but trajectories, update count,
   batch size, schedule, and data seeds are held fixed. Report the distinction
   between more starting-state coverage and the effect of the loss itself.

3. **Nonlinear Burgers experiment with the paper's UNet.** Use UNet;12;2;relu
   (27,193 parameters), 60 grid points, diffusion difficulty 0.2, convection
   difficulty -8, the authors' `fourier;1;False;True` initial conditions,
   500 training samples, one-Picard-step teacher and converged-Picard reference.
   Compare supervised N=1,2,5 and solver-in-loop WIG N=2, each with initialization
   seeds 0/1/2 and common train/test/shuffle seeds 10/20/30. Budget: 100,000 Adam
   updates, batch 32, peak rate 0.0003 and warmup 5,000. The main test uses
   100 fixed held-out examples over steps 1-20; steps 21-100 are a separate
   long-rollout stability extension. These data seeds differ from the authors'
   released notebook and are explicitly retained for continuity with this project.
   H=N changes the available states as well as the loss, so this comparison alone
   must not be called an isolated causal effect of unrolling.

## Outcomes chosen in advance

- Advection primary: ratio of summed mean RMSE over steps 1-100. Also report
  step 1, 10, 50, 100, the mean pointwise ratio, and seed median/min/max.
- Burgers primary: ratio of summed mean nRMSE over steps 1-20. Also report
  steps 3,5,10,20,50,100, and the fraction of steps and seeds with ratio <1.
  A transient improvement is distinct from an improvement over the full window.
- Report absolute emulator and solver errors alongside ratios. A small solver
  error can amplify a tiny imitation mismatch into a large relative change.
- Three seeds describe initialization variability, not uncertainty over all
  possible datasets. Per-test-sample paired bootstrap intervals, when available,
  are conditional on a trained model and are not a replacement for seed variation.
- Any nonfinite error is an unstable result, never silently removed. Record the
  first nonfinite step and the first nRMSE >10 as descriptive failure diagnostics.
- Record training loss and held-out teacher-imitation loss to identify a worse
  fit as an alternative explanation for worse physical predictions.
- No checkpoint or hyperparameter is selected using superiority on the test set.
  Report the final prescribed checkpoint, including unfavorable outcomes.

## Numerical validation and reproducibility

Before full runs, check network parameter counts, accelerated/original forward
and gradient agreement, chunked/resumed versus uninterrupted Adam updates,
linear solver matrix equivalence, the WIG derivative against a directional
finite difference, and reference sensitivity to a tighter Picard iteration cap.

Calibration finding, before extension comparisons: a fresh original-backend FNO
one-step run (initialization 0, explicit solver) gave superiority at step 100
0.108 versus the historical 0.233, despite the same nominal configuration.
The qualitative effect reproduces, but old and new environments are not
numerically interchangeable. Therefore, primary extension comparisons will use
fresh runs for all their cells (`--no-reuse`). Historical runs remain context
and are analyzed as a separate cohort. This supersedes reuse in the first
paragraphs of the plan. It is not appropriate to pair old and new outcomes as
though they differed only in the experimental treatment.

The optional channel-last backend evaluates the *same* periodic convolutions
as matrix products; it retains all original parameters and initialization.
Floating-point summation order can change, so backend and dependency versions
must accompany every newly trained result. Historical-versus-new comparisons
must acknowledge this implementation difference rather than assert bitwise
reproduction. Checkpoints retain both weights and optimizer state.

## Reporting boundary

An additional post-training diagnostic was specified while the main sweeps were
running: on linear advection only, compare each frozen model with
`f_hom(u) = max(abs(u)) * f(u / max(abs(u)))` (zero input maps to zero).
This enforces positive amplitude homogeneity without retraining and leaves the
first step on unit-maximum-amplitude test inputs unchanged. Compare the same
predefined R100 outcome before and after this intervention, with identical
weights/seeds/test inputs. Test homogeneity and first-step invariance numerically.
This is a mechanism diagnostic, not an extra primary benchmark, and must not be
applied to Burgers, whose nonlinear dynamics are not amplitude homogeneous.

Additional fit diagnostic, specified after the first completed baselines:
recipe-specific validation losses have different objectives and must not be
treated as a common measure of teacher imitation. Evaluate every frozen model
on identical held-out teacher states from 100 profiles (validation seed 110),
teacher time levels 0-4, using a single-step MSE at every time level. Report both
initial-state and all-five-level MSE. Linear models are evaluated on unit-height
and random-height validation profiles; Burgers uses the original Fourier family.
This diagnostic is not used to select models or hyperparameters.

The original report already contains the main FNO amplitude result and some
ConvNet one-step controls. New contributions must be the newly completed cells,
the fixed-data-horizon comparison, the UNet Burgers study, and new analyses that
are labeled as analyses of historical data. Neither a small seed study nor a
single nonlinear PDE establishes a universal claim about neural superiority.

## Completion record (after collecting outcomes)

All 66 primary cells and one separate runtime calibration completed on 2026-11-22.
Every primary cell received its prescribed update budget and produced finite
rollouts. Interrupted workers resumed weights and Adam states without changing
the recipe. `audit.json` verifies the full unique grid, common numerical provenance,
recomputed scores, exactly matching teacher baseline errors within each physical
test condition, and the unchanged 137 historical summary hashes.
The executed numerical sources are archived under `source_snapshot/` before a
post-run serialization-only fix to handle nested nonfinite JSON values. No model,
seed, hyperparameter, or checkpoint was selected using favorable test outcomes.
