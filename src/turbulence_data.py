"""Generate rough-solver-only data and separate, never-trained-on references."""
import argparse
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import jax.numpy as jnp
from turbulence_core import (FlowConfig,OUT,atomic_json,build_solver,initial_fields,
    warmup,rollout,resize,resolved_reference,velocity,provenance,cutoff)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def coarse_data(pilot=False):
    cfg=FlowConfig();teacher=build_solver(cfg)
    folder=OUT/'data';folder.mkdir(parents=True,exist_ok=True)
    cases=([('train_9101',9101,32,8),('pilot_validation',9102,16,8)] if pilot else
        [('train_1101',1101,128,8),('train_1102',1102,128,8),
         ('validation',2101,32,8),('test',3101,64,100)])
    for label,seed,count,horizon in cases:
        dest=folder/f'{label}.npz'
        if dest.exists():
            previous=json.loads(dest.with_suffix('.json').read_text())
            assert previous['provenance']['source_sha256']==provenance()['source_sha256'],'Changed data source'
            print('cached',label,flush=True);continue
        start=time.time();initial=initial_fields(cfg,seed,count)
        fields=[];trajectories=[]
        for i in range(0,count,8):
            x=warmup(teacher,initial[i:i+8],cfg.warmup_steps)
            trj=jnp.concatenate([x[:,None],rollout(teacher,x,horizon)],axis=1)
            a=np.asarray(trj)
            assert np.isfinite(a).all(),label
            fields.append(np.asarray(x));trajectories.append(a)
            print(label,'flows',i+8,'/',count,'elapsed',round(time.time()-start,1),flush=True)
        fields=np.concatenate(fields);trajectories=np.concatenate(trajectories)
        np.savez_compressed(dest,initial=fields,trajectory=trajectories)
        flat=fields.reshape(count,-1)
        corr=flat@flat.T/np.sqrt(np.sum(flat**2,axis=1)[:,None]*np.sum(flat**2,axis=1)[None,:])
        rms=np.sqrt(np.mean(fields**2,axis=(1,2,3)))
        atomic_json(dest.with_suffix('.json'),{'label':label,'seed':seed,'flows':count,
            'horizon':horizon,'config':asdict(cfg),'provenance':provenance(),
            'initial_RMS_min_median_max':[rms.min(),np.median(rms),rms.max()],
            'initial_maxabs_min_median_max':np.quantile(np.max(abs(fields),axis=(1,2,3)),[0,.5,1]),
            'pairwise_initial_correlation_median':np.median(corr[np.triu_indices(count,k=1)]),
            'global_training_scale':float(np.sqrt(np.mean(trajectories**2))),
            'seconds':time.time()-start,'npz_sha256':digest(dest),
            'source':'rough solver only; no fine-reference states or targets'})


def radial_spectra(v):
    n=v.shape[-1];k=np.fft.fftfreq(n)*n
    radius=np.rint(np.sqrt(k[:,None]**2+k[None,:]**2)).astype(int)
    power=.5*np.sum(abs(np.fft.fft2(v,norm='forward'))**2,axis=-3)
    return np.stack([power[...,radius==i].sum(axis=-1) for i in range(radius.max()+1)],axis=-1)


def references():
    cfg=FlowConfig();folder=OUT/'data'
    test=np.load(folder/'test.npz');initial=test['initial'];teacher=test['trajectory'][:,1:]
    ref_folder=folder/'reference_batches';ref_folder.mkdir(exist_ok=True)
    manifest={'config':asdict(cfg),'provenance':provenance(),'test_npz_sha256':digest(folder/'test.npz')}
    manifest_path=folder/'reference_manifest.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest,'Changed reference protocol'
    else:atomic_json(manifest_path,manifest)
    for i in range(0,len(initial),8):
        dest=ref_folder/f'batch_{i//8:02}.npz'
        if dest.exists():print('cached reference',i,flush=True);continue
        start=time.time();x=jnp.asarray(initial[i:i+8])
        r=rollout(build_solver(cfg,n=96),resize(x,96),100)
        low=resolved_reference(r,32)
        rv=velocity(r,cfg.domain);lv=velocity(low,cfg.domain)
        rr=np.asarray(r);ll=np.asarray(low);vv=np.asarray(rv);vl=np.asarray(lv)
        assert np.isfinite(rr).all()
        fields={'resolved':ll,
            'full_vorticity_variance':np.mean(rr**2,axis=(2,3,4)),
            'unresolved_vorticity_variance':np.maximum(0,np.mean(rr**2,axis=(2,3,4))-np.mean(ll**2,axis=(2,3,4))),
            'full_velocity_variance':np.mean(vv**2,axis=(2,3,4)),
            'unresolved_velocity_variance':np.maximum(0,np.mean(vv**2,axis=(2,3,4))-np.mean(vl**2,axis=(2,3,4))),
            'kinetic_energy':.5*np.mean(np.sum(vv**2,axis=2),axis=(2,3)),
            'enstrophy':.5*np.mean(rr**2,axis=(2,3,4)),
            'energy_spectrum':radial_spectra(vv)}
        if i==0:
            fields['snapshot_example_indices']=np.arange(3)
            fields['snapshot_steps']=np.array([0,1,5,20,100])
            fields['fine_snapshots']=np.concatenate([np.asarray(resize(x[:3],96))[:,None],rr[:3,[0,4,19,99]]],axis=1)
        # Finer reference on every primary test trajectory; long horizon on first 8.
        h=100 if i==0 else 20
        tight=rollout(build_solver(cfg,n=128,substeps=40,order=4),resize(x,128),h)
        fields['tighter_resolved']=np.asarray(resolved_reference(tight,32))
        fields['tighter_full_vorticity_variance']=np.mean(np.asarray(tight)**2,axis=(2,3,4))
        np.savez_compressed(dest,**fields)
        print('reference batch',i//8,'seconds',round(time.time()-start,1),flush=True)
    batches=[np.load(ref_folder/f'batch_{i:02}.npz') for i in range(8)]
    shared=[k for k in batches[0].files if not k.startswith(('snapshot','fine_snapshots','tighter_'))]
    merged={k:np.concatenate([b[k] for b in batches]) for k in shared}
    merged['tighter_resolved']=np.concatenate([b['tighter_resolved'][:,:20] for b in batches])
    merged['tighter_long_first8']=batches[0]['tighter_resolved']
    for key in ['snapshot_example_indices','snapshot_steps','fine_snapshots']:
        merged[key]=batches[0][key]
    np.savez_compressed(folder/'reference.npz',**merged)
    ref=merged['resolved'];tight=merged['tighter_resolved']
    rms=lambda z:np.sqrt(np.mean(z*z,axis=(2,3,4)))
    gap=rms(teacher[:,:20]-ref[:,:20]);difference=rms(tight-ref[:,:20])
    atomic_json(folder/'reference_validation.json',{'config':asdict(cfg),
        'test_flows':len(initial),'primary_steps':20,
        'teacher_mean_resolved_RMSE_1_20':float(gap.mean()),
        'refinement_mean_resolved_RMSE_1_20':float(difference.mean()),
        'refinement_fraction_of_teacher_gap':float(difference.mean()/gap.mean()),
        'refinement_max_RMSE_fraction_of_mean_teacher_gap':float(difference.max()/gap.mean()),
        'reference_npz_sha256':digest(folder/'reference.npz'),
        'unresolved_enstrophy_fraction_mean_1_20':float((merged['unresolved_vorticity_variance'][:,:20]/merged['full_vorticity_variance'][:,:20]).mean()),
        'unresolved_kinetic_energy_fraction_mean_1_20':float((merged['unresolved_velocity_variance'][:,:20]/merged['full_velocity_variance'][:,:20]).mean()),
        'note':'All references are evaluation-only. Every fine solve starts with the same spectrally interpolated coarse field.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['coarse','reference'])
    p.add_argument('--pilot',action='store_true');args=p.parse_args()
    coarse_data(args.pilot) if args.action=='coarse' else references()
