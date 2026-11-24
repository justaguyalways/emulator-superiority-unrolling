"""Train solely on coarse trajectories; this module never loads fine references."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from turbulence_core import FlowConfig,OUT,ROOT,ResidualFNO,atomic_json,provenance

BUDGET=12000
BATCH=8


def make_windows(trj,n):
    return jnp.stack([trj[:,i:i+n+1] for i in range(trj.shape[1]-n)],axis=1).reshape(
        -1,n+1,*trj.shape[2:])


def optimizer(budget=BUDGET):
    schedule=optax.warmup_cosine_decay_schedule(1e-5,1e-3,min(500,budget//5),budget,end_value=1e-5)
    return optax.chain(optax.clip_by_global_norm(1.),optax.adam(schedule))


@eqx.filter_jit
def train_chunk(params,state,static,windows,indices,opt,n,scale):
    def loss(p,batch):
        model=eqx.combine(p,static);z=batch[:,0];total=0.
        for t in range(n):
            z=jax.vmap(model)(z)
            total+=jnp.mean(((z-batch[:,t+1])/scale)**2)
        return total/n
    def step(carry,idx):
        p,s=carry
        value,grad=eqx.filter_value_and_grad(loss)(p,windows[idx])
        updates,s=opt.update(grad,s,p)
        return (eqx.apply_updates(p,updates),s),value
    return jax.lax.scan(step,(params,state),indices)


@eqx.filter_jit
def predict(model,x):return jax.vmap(model)(x)


def model_provenance():
    p=provenance()
    p['source_sha256']['train_turbulence.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return p


def train(dataset,seed,n,budget=BUDGET,chunk=300,pilot=False,width=16,modes=10,blocks=4):
    cfg=replace(FlowConfig(),hidden_channels=width,fno_modes=modes,fno_blocks=blocks)
    suffix=f'_w{width}_m{modes}_b{blocks}' if pilot else ''
    folder=OUT/('pilot_runs' if pilot else 'runs')/f'data{dataset}_N{n}_seed{seed}{suffix}'
    folder.mkdir(parents=True,exist_ok=True)
    summary_path=folder/'training_summary.json'
    if summary_path.exists():print('cached',folder.name,flush=True);return
    data_path=OUT/'data'/f'train_{dataset}.npz'
    val_path=OUT/'data'/('pilot_validation.npz' if pilot else 'validation.npz')
    raw=np.load(data_path)['trajectory']
    # Pilot uses a declared subset and disjoint model initialization, never fine data.
    if pilot:raw=raw[:32]
    scale=float(np.sqrt(np.mean(raw**2)))
    trj=jnp.asarray(raw);windows=make_windows(trj,n)
    model=ResidualFNO(cfg,scale,seed)
    params,static=eqx.partition(model,eqx.is_inexact_array)
    opt=optimizer(budget);state=opt.init(params)
    indices=jax.random.randint(jax.random.key(dataset+12345),(budget,BATCH),0,len(windows))
    config={'flow':asdict(cfg),'training_dataset':dataset,'initialization_seed':seed,
        'unroll':n,'training_horizon':8,'budget':budget,'batch':BATCH,
        'learning_rate':.001,'warmup':min(500,budget//5),'clip_norm':1.,
        'final_learning_rate':1e-5,'scale':scale,'pilot':pilot,
        'training_flows':len(raw),'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'validation_sha256':hashlib.sha256(val_path.read_bytes()).hexdigest()}
    meta={'config':config,'provenance':model_provenance(),
        'n_params':sum(a.size for a in jax.tree_util.tree_leaves(params)),
        'training_information':'Coarse training and validation trajectories only. No fine reference loaded.'}
    cursor=0;elapsed=0.;loss_history=[]
    progress=folder/'checkpoint.json'
    if progress.exists():
        prior=json.loads(progress.read_text())
        assert prior['config']==config and prior['provenance']==meta['provenance'],'Changed checkpoint protocol/source'
        params,state=eqx.tree_deserialise_leaves(folder/prior['checkpoint_file'],(params,state))
        cursor=prior['completed_updates'];elapsed=prior['training_seconds']
        loss_history=np.load(folder/'loss.npy')[:cursor].tolist()
        assert len(loss_history)==cursor
    print('START',folder.name,'from',cursor,flush=True)
    for start in range(cursor,budget,chunk):
        end=min(start+chunk,budget);t0=time.time()
        (params,state),loss=train_chunk(params,state,static,windows,indices[start:end],opt,n,scale)
        loss=np.asarray(jax.block_until_ready(loss));elapsed+=time.time()-t0
        if not np.isfinite(loss).all():
            atomic_json(folder/'failure.json',{**meta,'completed_updates':start,'reason':'nonfinite training loss'})
            raise FloatingPointError(folder.name)
        loss_history.extend(loss.tolist())
        checkpoint_file=f'checkpoint_{end:06}.eqx'
        tmp=folder/(checkpoint_file+'.tmp');eqx.tree_serialise_leaves(tmp,(params,state));tmp.replace(folder/checkpoint_file)
        tmp=folder/'loss.npy.tmp'
        with tmp.open('wb') as f:np.save(f,np.array(loss_history))
        tmp.replace(folder/'loss.npy')
        atomic_json(progress,{**meta,'completed_updates':end,'training_seconds':elapsed,'checkpoint_file':checkpoint_file})
        for old in sorted(folder.glob('checkpoint_*.eqx'))[:-2]:old.unlink()
        print(json.dumps({'run':folder.name,'updates':end,'loss':float(loss[-100:].mean()),'train_s':round(elapsed,2)}),flush=True)
    model=eqx.combine(params,static);eqx.tree_serialise_leaves(folder/'model.eqx',model)
    validation=np.load(val_path)['trajectory']
    x=validation[:,:-1].reshape(-1,1,cfg.coarse_points,cfg.coarse_points)
    y=validation[:,1:].reshape(x.shape)
    error=[]
    for i in range(0,len(x),8):error.append(np.mean((np.asarray(predict(model,jnp.asarray(x[i:i+8])))-y[i:i+8])**2,axis=(1,2,3)))
    error=np.concatenate(error).reshape(len(validation),8)
    # Temporal coverage check: all models use identical validation states and targets.
    atomic_json(summary_path,{**meta,'completed_updates':budget,'training_seconds':elapsed,
        'loss_start':float(np.mean(loss_history[:100])),'loss_end':float(np.mean(loss_history[-300:])),
        'common_validation_one_step_MSE':float(error.mean()),
        'common_validation_normalized_one_step_MSE':float(error.mean()/scale**2),
        'validation_MSE_by_input_time':error.mean(axis=0),
        'validation_initial_MSE':float(error[:,0].mean()),
        'model_sha256':hashlib.sha256((folder/'model.eqx').read_bytes()).hexdigest()})
    final_checkpoint=json.loads(progress.read_text())['checkpoint_file']
    for old in folder.glob('checkpoint_*.eqx'):
        if old.name!=final_checkpoint:old.unlink()
    print('FINISHED',folder.name,'validation normalized MSE',float(error.mean()/scale**2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=int,default=1101)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--unroll',type=int,choices=[1,4],default=1)
    p.add_argument('--budget',type=int,default=BUDGET);p.add_argument('--pilot',action='store_true')
    p.add_argument('--width',type=int,default=16);p.add_argument('--modes',type=int,default=10)
    p.add_argument('--blocks',type=int,default=4)
    a=p.parse_args();train(a.dataset,a.seed,a.unroll,a.budget,pilot=a.pilot,width=a.width,modes=a.modes,blocks=a.blocks)
