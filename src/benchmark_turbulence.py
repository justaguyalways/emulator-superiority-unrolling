"""Synchronized CPU inference timing after all training workers finish."""
import json
from pathlib import Path
import platform
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from turbulence_core import FlowConfig,OUT,ResidualFNO,atomic_json,build_solver,resize,warmup


def main():
    assert len(list((OUT/'runs').glob('*/training_summary.json')))==12
    folder=OUT/'runs'/'data1101_N1_seed0';s=json.loads((folder/'training_summary.json').read_text())
    c=s['config'];cfg=FlowConfig(**c['flow'])
    model=eqx.tree_deserialise_leaves(folder/'model.eqx',ResidualFNO(cfg,c['scale'],0))
    x=jnp.asarray(np.load(OUT/'data'/'test.npz')['initial'][:8]);rows=[]
    for name,stepper,initial in [('FNO',model,x),('teacher',build_solver(cfg),x),
            ('refined48',build_solver(cfg,n=48,substeps=20,order=4),resize(x,48)),
            ('reference96',build_solver(cfg,n=96),resize(x,96))]:
        warmup(stepper,initial,20).block_until_ready()
        times=[]
        for i in range(5):
            start=time.perf_counter();warmup(stepper,initial,20).block_until_ready();times.append(time.perf_counter()-start)
        rows.append({'method':name,'batch_size':8,'forecast_steps':20,'median_seconds':float(np.median(times)),
            'milliseconds_per_field_step':float(np.median(times)/160*1000),'all_repeats_seconds':times})
    atomic_json(OUT/'inference_timing.json',{'platform':platform.platform(),'devices':[str(d) for d in jax.devices()],
        'rows':rows,'notes':'Compiled, synchronized, same batch size and forecast interval. Compilation excluded; CPU otherwise idle. One representative frozen FNO; both recipes share inference architecture.'})
    print(rows)


if __name__=='__main__':main()
