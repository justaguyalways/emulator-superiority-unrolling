"""Compare frozen models on identical held-out one-step teacher targets.

Recipe-specific validation losses use different objectives and cannot be
compared as though they measure the same fit. This diagnostic fixes the
validation states, targets, and one-step MSE across recipes. No test-reference
solutions or superiority scores are used to build this validation set.
"""
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/emusup-mpl')
from pathlib import Path

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import emusup
from extension_study import OUT,atomic_json,finite_json
from fast_models import accelerate,FastModel


@eqx.filter_jit
def mse(model,x,y):
    pred=jax.vmap(model)(x)
    return jnp.mean((pred-y)**2,axis=(-2,-1))


@eqx.filter_jit
def trained_backend_difference(model,x):
    return jnp.max(jnp.abs(jax.vmap(model)(x)-jax.vmap(model.model)(x)))


def main():
    caches={};records=[]
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p in sorted((OUT/'runs').glob('*/summary.json')):
        s=json.loads(p.read_text());c=emusup.Config(**s['config'])
        dest=p.parent/'common_teacher_fit.json'
        if dest.exists():
            cached=json.loads(dest.read_text())
            if cached.get('source_sha256')==source_hash:
                records.extend(cached['rows']);continue
        model=emusup.build_network(c,jax.random.key(c.seed_init))
        if s['provenance']['backend']=='fast':model=accelerate(model,c.network)
        model=eqx.tree_deserialise_leaves(p.parent/'model.eqx',model)
        sets=['ind;1;True','indamp;1;0.5;1.5'] if c.is_linear() else [c.ic_train]
        model_rows=[]
        for ic_config in sets:
            key=(tuple(c.linear_difficulties),c.convection_difficulty,c.num_points,c.train_simulator,ic_config)
            if key not in caches:
                teacher=emusup.build_train_simulator(c)
                ic=emusup.make_ic_set(ic_config,c.num_points,100,jax.random.key(110))
                trj=jax.vmap(ex.rollout(teacher,5,include_init=True))(ic)
                caches[key]=(trj[:,:-1].reshape(-1,1,c.num_points),
                             trj[:,1:].reshape(-1,1,c.num_points))
            x,y=caches[key]
            # Keep inference batches modest while training workers occupy RAM.
            errors=np.concatenate([np.asarray(mse(model,x[i:i+50],y[i:i+50]))
                                   for i in range(0,len(x),50)]).reshape(100,5)
            row={'name':c.name(),'group':c.tag,'arch':c.network.split(';')[0],
                'pde':'advection' if c.is_linear() else 'Burgers',
                'solver':'Burgers P1' if not c.is_linear() else ('explicit' if c.linear_difficulties[1]==-.9 else 'implicit'),
                'N':c.num_unrolled_steps,'mode':c.train_mode,'seed':c.seed_init,
                'training_heights':c.ic_train,'validation_distribution':ic_config,
                'common_initial_MSE':float(errors[:,0].mean()),
                'common_H5_MSE':float(errors.mean()),
                **{f'MSE_at_input_time_{t}':float(errors[:,t].mean()) for t in range(5)}}
            if isinstance(model,FastModel):
                delta=float(trained_backend_difference(model,x[:50]))
                assert delta<2e-5,(c.name(),ic_config,delta)
                row['trained_native_backend_max_difference']=delta
            if c.train_mode=='sup' and c.num_unrolled_steps==1 and ic_config==c.ic_train:
                if c.horizon_train() in [1,5]:
                    value=row['common_initial_MSE' if c.horizon_train()==1 else 'common_H5_MSE']
                    original=s['teacher_validation_loss']
                    assert np.isclose(value,original,rtol=.02,atol=1e-11),(c.name(),value,original)
                    row['one_step_validation_relative_difference']=abs(value-original)/max(original,1e-30)
            model_rows.append(row);records.append(row)
            print(json.dumps(row),flush=True)
        atomic_json(dest,finite_json({'source_sha256':source_hash,'validation_seed':110,
            'validation_profiles':100,'teacher_horizon':5,'metric':'one-step absolute mean MSE',
            'note':'Common states/targets across all recipes; post-training diagnostic, no model selection.',
            'rows':model_rows}))
    pd.DataFrame(records).to_csv(OUT/'common_teacher_fit.csv',index=False)


if __name__=='__main__':main()
