"""Extended-reference check on the predetermined first-eight test subset."""
import json
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from turbulence_core import FlowConfig,OUT,ResidualFNO,atomic_json,rollout


def rmse(z):return np.sqrt(np.mean(z*z,axis=(-3,-2,-1)))


def score(p,t,r,start,end):
    scale=np.maximum(rmse(r),1e-12)
    a=rmse(p-r)/scale;b=rmse(t-r)/scale
    return float(a[:,start:end].sum()/b[:,start:end].sum())


def main():
    ref=np.load(OUT/'data'/'reference.npz');test=np.load(OUT/'data'/'test.npz')
    r=ref['resolved'][:8];tight=ref['tighter_long_first8'];teacher=test['trajectory'][:8,1:]
    difference=rmse(tight-r);gap=rmse(teacher-r)
    rows=[]
    for p in sorted((OUT/'runs').glob('*/training_summary.json')):
        s=json.loads(p.read_text());c=s['config'];cfg=FlowConfig(**c['flow'])
        model=eqx.tree_deserialise_leaves(p.parent/'model.eqx',ResidualFNO(cfg,c['scale'],c['initialization_seed']))
        pred=np.asarray(rollout(model,jnp.asarray(test['initial'][:8]),100))
        row={'name':p.parent.name,'N':c['unroll'],'dataset':c['training_dataset'],'seed':c['initialization_seed']}
        for label,start,end in [('R20',0,20),('R100',0,100),('R21_100',20,100)]:
            row[label+'_reference96']=score(pred,teacher,r,start,end)
            row[label+'_reference128']=score(pred,teacher,tight,start,end)
            row[label+'_change']=row[label+'_reference128']-row[label+'_reference96']
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'long_reference_sensitivity.csv',index=False)
    zero=np.zeros_like(teacher)
    atomic_json(OUT/'long_reference_sensitivity.json',{
        'subset':'Predetermined first eight of the 64 test flows; same subset for all models.',
        'reference_refinement_fraction_of_teacher_gap_primary':float(difference[:,:20].mean()/gap[:,:20].mean()),
        'reference_refinement_fraction_of_teacher_gap_later':float(difference[:,20:].mean()/gap[:,20:].mean()),
        'maximum_model_R100_change_on_subset':max(abs(r['R100_change']) for r in rows),
        'maximum_model_later_ratio_change_on_subset':max(abs(r['R21_100_change']) for r in rows),
        'zero_later_ratio_subset_reference96':score(zero,teacher,r,20,100),
        'zero_later_ratio_subset_reference128':score(zero,teacher,tight,20,100)})
    print(json.loads((OUT/'long_reference_sensitivity.json').read_text()))


if __name__=='__main__':main()
