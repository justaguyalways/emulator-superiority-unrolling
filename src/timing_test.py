"""Timing + regression test: reproduce one of the authors' advection runs on CPU and
import os
compare against their stored CSV for the identical seeds."""
import sys, time, json
EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repos", "emulator-superiority", "code", "experiments")
sys.path.insert(0, EXP)
t0 = time.time()
import jax, jax.numpy as jnp, pandas as pd
import apebench, exponax, trainax, equinox
print("jax", jax.__version__, jax.devices(), "| apebench", apebench.__version__, "| exponax", exponax.__version__,
      "| trainax", __import__("importlib.metadata").metadata.version("trainax"), "| equinox", equinox.__version__, flush=True)
import superape
from individual_modes_ic import individual_modes_gen
apebench.components.ic_dict["ind"] = individual_modes_gen
print(f"imports {time.time()-t0:.1f}s", flush=True)

setting = {"network": "Conv;34;10;relu", "gamma_1": -0.9, "train_simulator": "fd;0.0;1",
           "train_ic_distribution": "ind;1;True", "test_ic_distribution": "ind;1;True",
           "seed_init": 0, "seed_train": 10, "seed_test": 20, "seed_shuffle": 30}
config = dict(
    network=setting["network"],
    linear_difficulties=(0.0, setting["gamma_1"], 0.0, 0.0, 0.0),
    convection_difficulty=0.0, num_points=100,
    simulator_train=setting["train_simulator"], simulator_test="etdrk;2",
    ic_distribution_train=setting["train_ic_distribution"], ic_distribution_test=setting["test_ic_distribution"],
    temporal_horizon_train=1, temporal_horizon_test=5000, num_warmup_steps=0,
    metric_fn_train="mean_MSE", metric_fn_test="mean_RMSE",
    num_samples_train=300, num_samples_test=10, num_unrolled_steps=1,
    optimizer="adam;10_000;warmup_cosine;0.0;1e-3;1_000", batch_size=32,
    seed_init=setting["seed_init"], seed_train=setting["seed_train"],
    seed_test=setting["seed_test"], seed_shuffle=setting["seed_shuffle"],
)
t1 = time.time()
net, losses, metrics, trj = superape.run_no_recording(**config)
sup = jax.block_until_ready(metrics["superiority_trj"])
print(f"full run (10k steps, 5000-step test rollout): {time.time()-t1:.1f}s", flush=True)
ref = pd.read_csv(f"{EXP}/advection_data/{json.dumps(setting)}/metrics.csv.gz", compression="gzip")
for t in [1, 2, 5, 10, 20, 50, 100]:
    print(f"t={t:4d}  ours={float(sup[t-1]):.4f}  authors={float(ref.query('time_step==@t')['superiority'].iloc[0]):.4f}")
print("final loss ours", float(losses[-1]))
