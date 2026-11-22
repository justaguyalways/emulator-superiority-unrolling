"""Post-training amplitude intervention on linear-advection emulators only.

No weights or test-selected parameters change. Enforce f(a*u)=a*f(u), a>0,
by evaluating the original network on unit-maximum-amplitude states. The first
step from the unit-height test inputs is unchanged. This is a diagnostic
intervention, not a claim that this wrapper is a learned better simulator.
"""
import json
from pathlib import Path

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import emusup
from extension_study import OUT, finite_json, atomic_json
from fast_models import accelerate


class Homogeneous(eqx.Module):
    model: eqx.Module

    def __call__(self,u):
        amplitude=jnp.max(jnp.abs(u))
        return amplitude*self.model(u/jnp.maximum(amplitude,1e-12))


@eqx.filter_jit
def rollout_error(model,ic,ref):
    def step(u,r):
        u=jax.vmap(model)(u)
        return u,jax.vmap(ex.metrics.RMSE)(u,r)
    return jax.lax.scan(step,ic,ref.transpose(1,0,2,3))[1]


def main():
    rows=[];data_cache={}
    for p in sorted((OUT/'runs').glob('*/summary.json')):
        summary=json.loads(p.read_text());cfg=emusup.Config(**summary['config'])
        if not cfg.is_linear():continue
        dst=p.parent/'homogeneity_control.json'
        if dst.exists():
            rows.append(json.loads(dst.read_text()));continue
        model=emusup.build_network(cfg,jax.random.key(cfg.seed_init))
        if summary['provenance']['backend']=='fast':model=accelerate(model,cfg.network)
        model=eqx.tree_deserialise_leaves(p.parent/'model.eqx',model)
        wrapped=Homogeneous(model)
        key=(cfg.ic_test,cfg.num_points,cfg.num_samples_test,cfg.seed_test,
             tuple(cfg.linear_difficulties),cfg.train_simulator,cfg.test_simulator)
        if key not in data_cache:
            ic=emusup.make_ic_set(cfg.ic_test,cfg.num_points,cfg.num_samples_test,jax.random.key(cfg.seed_test))
            ref=jax.vmap(ex.rollout(emusup.build_test_simulator(cfg),100))(ic)
            teacher=emusup.build_train_simulator(cfg)
            te=np.asarray(rollout_error(teacher,ic,ref))
            data_cache[key]=(ic,ref,te)
        ic,ref,te=data_cache[key]
        raw=np.asarray(rollout_error(model,ic,ref));hom=np.asarray(rollout_error(wrapped,ic,ref))
        first_step_difference=float(np.max(np.abs(raw[0]-hom[0])))
        assert first_step_difference<2e-6,first_step_difference
        u=ic[0]
        scale_error=float(jnp.max(jnp.abs(wrapped(.7*u)-.7*wrapped(u))))
        assert scale_error<2e-6,scale_error
        result={'name':cfg.name(),'arch':cfg.network.split(';')[0],
            'group':cfg.tag,'solver':'explicit' if cfg.linear_difficulties[1]==-.9 else 'implicit',
            'N':cfg.num_unrolled_steps,'H':cfg.horizon_train(),'amplitudes':cfg.ic_train,
            'seed':cfg.seed_init,'original_R100':float(raw.sum()/te.sum()),
            'homogeneous_R100':float(hom.sum()/te.sum()),
            'original_sup100':float(raw[-1].mean()/te[-1].mean()),
            'homogeneous_sup100':float(hom[-1].mean()/te[-1].mean()),
            'max_first_step_error_difference':first_step_difference,
            'homogeneity_max_difference_at_scale07':scale_error}
        assert np.isclose(result['original_R100'],summary['integrated_ratio_1_100'],rtol=1e-5),result
        atomic_json(dst,finite_json(result))
        pd.DataFrame({'time_step':np.arange(1,101),'original_error':raw.mean(axis=1),
            'homogeneous_error':hom.mean(axis=1),'teacher_error':te.mean(axis=1),
            'original_ratio':raw.mean(axis=1)/te.mean(axis=1),
            'homogeneous_ratio':hom.mean(axis=1)/te.mean(axis=1)}).to_csv(p.parent/'homogeneity_metrics.csv',index=False)
        print(json.dumps(result),flush=True)
        rows.append(result)
    pd.DataFrame(rows).to_csv(OUT/'homogeneity_results.csv',index=False)


if __name__=='__main__':main()
