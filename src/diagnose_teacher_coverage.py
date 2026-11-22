"""Measure how the Burgers training states change as the horizon is extended."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/emusup-mpl')
import equinox as eqx
import exponax as ex
import jax
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emusup
from extension_study import grid,OUT,atomic_json


def main():
    c=grid('burgers')[0]
    ic=emusup.make_ic_set(c.ic_train,c.num_points,c.num_samples_train,jax.random.key(c.seed_train))
    teacher=emusup.build_train_simulator(c);ref=emusup.build_test_simulator(c)
    states=jax.vmap(ex.rollout(teacher,5,include_init=True))(ic)

    @eqx.filter_jit
    def probe(batch):
        coarse=jax.vmap(teacher)(batch)
        accurate=jax.vmap(ref)(batch)
        gap=jax.vmap(ex.metrics.nRMSE)(coarse,accurate)
        its=jax.vmap(lambda u:ref._run_iterator(u).state.iter_num)(batch)
        return gap,its

    records=[]
    for t in range(6):
        u=np.asarray(states[:,t,0,:]);spec=np.abs(np.fft.rfft(u,axis=-1))**2
        fraction=spec[:,2:].sum(axis=1)/np.maximum(spec[:,1:].sum(axis=1),1e-30)
        gap,its=map(np.asarray,probe(states[:,t]))
        records.append({'teacher_time':t,'samples':len(ic),
            'mean_higher_mode_energy_fraction':float(fraction.mean()),
            'median_higher_mode_energy_fraction':float(np.median(fraction)),
            'mean_one_step_teacher_reference_nRMSE':float(gap.mean()),
            'median_picard_iterations_to_reference_tolerance':float(np.median(its)),
            'max_picard_iterations':int(its.max())})
        print(json.dumps(records[-1]),flush=True)
    pd.DataFrame(records).to_csv(OUT/'burgers_teacher_coverage.csv',index=False)
    atomic_json(OUT/'burgers_teacher_coverage.json',{'config':c.to_json(),
        'definition':'Energy in Fourier modes k>=2 divided by non-DC energy (k>=1). '
        'Local teacher/reference gap is measured from the same teacher state, not from separate trajectories.',
        'rows':records})
    fig,axs=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    df=pd.DataFrame(records)
    axs[0].plot(df.teacher_time,df.mean_higher_mode_energy_fraction,'o-',color='#2563eb')
    axs[0].set_ylabel('Mean energy fraction in higher modes');axs[0].set_title('More complex shapes enter training')
    axs[1].plot(df.teacher_time,df.mean_one_step_teacher_reference_nRMSE,'o-',color='#d97706')
    axs[1].set_ylabel('Mean one-step nRMSE');axs[1].set_title('Local error of the training teacher')
    for ax in axs:
        ax.set_xlabel('Teacher time level used as input');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'burgers_teacher_coverage.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
