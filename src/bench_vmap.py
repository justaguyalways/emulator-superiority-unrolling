"""Does vmapping several seeds into one computation raise CPU throughput?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, equinox as eqx, exponax as ex
import emusup
from emusup import Config
STEPS = int(os.environ.get("BENCH_STEPS", "100"))
def bench(cfg, S):
    train_sim = emusup.build_train_simulator(cfg)
    keys = jax.random.split(jax.random.key(0), S)
    nets = eqx.filter_vmap(lambda k: emusup.build_network(cfg, k))(keys)
    ic = emusup.make_ic_set(cfg.ic_train, cfg.num_points, cfg.num_samples_train, jax.random.key(1))
    trj = jax.vmap(ex.rollout(train_sim, cfg.horizon_train(), include_init=True))(ic)
    windows = emusup.make_windows(trj, cfg.num_unrolled_steps + 1)
    loss_cfg = emusup.build_loss_config(cfg); opt = emusup.build_optimizer(cfg)
    f = eqx.filter_jit(eqx.filter_vmap(emusup.train, in_axes=(eqx.if_array(0), None, None, None, None, 0)))
    def idx(n): return jnp.stack([emusup.make_batch_indices(windows.shape[0], n, cfg.batch_size, jax.random.key(10+s)) for s in range(S)])
    t=time.time(); jax.block_until_ready(f(nets, windows, loss_cfg, train_sim, opt, idx(10))[1]); tc=time.time()-t
    t=time.time(); jax.block_until_ready(f(nets, windows, loss_cfg, train_sim, opt, idx(STEPS))[1]); tr=time.time()-t
    ms = 1000*tr/STEPS
    print(f"{cfg.network:18s} {cfg.train_mode:8s} N={cfg.num_unrolled_steps:2d} vmap S={S}: compile {tc:5.1f}s {ms:7.1f} ms/step "
          f"-> {ms/S:6.1f} ms/step/seed -> {ms*cfg.num_training_steps/60000/S:5.1f} min per run-equivalent", flush=True)
if __name__ == "__main__":
    for S in [1, 3, 6]:
        bench(Config(network="Conv;34;10;relu"), S)
    for S in [1, 3, 6]:
        bench(Config(network="FNO;12;18;4;gelu"), S)
    bench(Config(network="Conv;34;10;relu", num_unrolled_steps=5), 3)
