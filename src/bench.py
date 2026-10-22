"""Throughput benchmark: seconds per training step for the real configs on this CPU.
Run with different XLA thread settings / in parallel to pick the sweep strategy."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
import equinox as eqx  # noqa: E402
import emusup  # noqa: E402
from emusup import Config  # noqa: E402

STEPS = int(os.environ.get("BENCH_STEPS", "300"))


def bench(cfg: Config):
    train_sim = emusup.build_train_simulator(cfg)
    network = emusup.build_network(cfg, jax.random.key(0))
    train_ic = emusup.make_ic_set(cfg.ic_train, cfg.num_points, cfg.num_samples_train, jax.random.key(1))
    import exponax as ex
    trj = jax.vmap(ex.rollout(train_sim, cfg.horizon_train(), include_init=True))(train_ic)
    windows = emusup.make_windows(trj, cfg.num_unrolled_steps + 1)
    loss_cfg = emusup.build_loss_config(cfg)
    opt = emusup.build_optimizer(cfg)
    f = eqx.filter_jit(emusup.train)
    idx = emusup.make_batch_indices(windows.shape[0], 20, cfg.batch_size, jax.random.key(2))
    t = time.time(); jax.block_until_ready(f(network, windows, loss_cfg, train_sim, opt, idx)[1]); t_c = time.time() - t
    idx = emusup.make_batch_indices(windows.shape[0], STEPS, cfg.batch_size, jax.random.key(2))
    t = time.time(); jax.block_until_ready(f(network, windows, loss_cfg, train_sim, opt, idx)[1]); t_r = time.time() - t
    ms = 1000 * t_r / STEPS
    print(f"{cfg.network:18s} {cfg.train_mode:8s} N={cfg.num_unrolled_steps:2d}  compile {t_c:5.1f}s  "
          f"{ms:7.1f} ms/step  -> {ms * cfg.num_training_steps / 60000:5.1f} min per 10k-step run", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = []
    for net in ["Conv;34;10;relu", "FNO;12;18;4;gelu"]:
        cfgs.append(Config(network=net, train_mode="sup", num_unrolled_steps=1))
        if which == "all":
            cfgs.append(Config(network=net, train_mode="sup", num_unrolled_steps=5))
            cfgs.append(Config(network=net, train_mode="div_wig", num_unrolled_steps=5))
    print("XLA_FLAGS =", os.environ.get("XLA_FLAGS", "<default>"), "| devices", jax.devices())
    for c in cfgs:
        bench(c)
