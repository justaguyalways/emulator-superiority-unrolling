"""Secondary checks: absolute RMSE and per-example wins, not new primary tests."""
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/emusup-mpl')
import equinox as eqx
import exponax as ex
import jax
import numpy as np
import pandas as pd

import emusup
from extension_study import OUT,grid


def main():
    paths=[]
    for p in (OUT/'runs').glob('*/summary.json'):
        summary=json.loads(p.read_text())
        if summary['config']['convection_difficulty']!=0:paths.append((p,summary))
    if not paths:
        print('No completed Burgers models yet.');return
    c=grid('burgers')[0]
    cache=OUT/'burgers_reference_rms.npz'
    if cache.exists():
        rms=np.load(cache)['reference_rms']
    else:
        ic=emusup.make_ic_set(c.ic_test,c.num_points,c.num_samples_test,jax.random.key(c.seed_test))
        ref=emusup.build_test_simulator(c)
        states=jax.vmap(ex.rollout(ref,c.temporal_horizon_test))(ic)
        # One channel, unit domain: nRMSE(u,v)*RMS(v) = absolute RMSE(u,v).
        rms=np.sqrt(np.mean(np.asarray(states)**2,axis=(-2,-1))).T
        probe=states[0,0]
        lhs=float(ex.metrics.nRMSE(probe*.9,probe))*float(np.sqrt(np.mean(np.asarray(probe)**2)))
        rhs=float(ex.metrics.RMSE(probe*.9,probe))
        assert abs(lhs-rhs)<1e-6
        np.savez_compressed(cache,reference_rms=rms)
    records=[]
    for p,s in sorted(paths):
        cfg=s['config']
        for key in ['num_points','num_samples_test','seed_test','ic_test','temporal_horizon_test','test_simulator']:
            assert cfg[key]==getattr(c,key),(key,cfg[key],getattr(c,key))
        a=np.load(p.parent/'per_sample_errors.npz')
        e,te=a['emulator'],a['solver']
        per=e[:20].sum(axis=0)/te[:20].sum(axis=0)
        absolute=(e[:20]*rms[:20]).sum()/(te[:20]*rms[:20]).sum()
        records.append({'name':s['name'],'mode':cfg['train_mode'],'N':cfg['num_unrolled_steps'],
            'seed':cfg['seed_init'],'nRMSE_R20':s['integrated_ratio_1_20'],
            'absolute_RMSE_R20':float(absolute),
            'test_examples_with_lower_total_nRMSE_fraction':float(np.mean(per<1)),
            'test_example_ratio_median':float(np.median(per)),
            'test_example_ratio_p05':float(np.quantile(per,.05)),
            'test_example_ratio_p95':float(np.quantile(per,.95)),
            'conditional_bootstrap95_low':s['primary_test_bootstrap95'][0],
            'conditional_bootstrap95_high':s['primary_test_bootstrap95'][1]})
    pd.DataFrame(records).to_csv(OUT/'burgers_secondary_metrics.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':main()
