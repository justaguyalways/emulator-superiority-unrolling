"""Numerical design checks on pilot-only seeds; not main superiority results."""
from dataclasses import asdict
import time
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from turbulence_core import (FlowConfig,OUT,atomic_json,build_solver,initial_fields,
    warmup,rollout,resize,resolved_reference,velocity,ResidualFNO,project,provenance)


def main():
    cfg=FlowConfig();t0=time.time()
    teacher=build_solver(cfg)
    x=initial_fields(cfg,9001,4)
    states=[np.asarray(x)]
    for i in range(6):
        x=warmup(teacher,x,50);x.block_until_ready();states.append(np.asarray(x))
        print('warmup physical time',(i+1)*10,'RMS',np.sqrt(np.mean(np.asarray(x)**2,axis=(1,2,3))),flush=True)
    assert np.isfinite(x).all()
    np.savez_compressed(OUT/'pilot_fields.npz',warmup=np.array(states))
    base=rollout(teacher,x,20)
    refs={}
    for n,sub in [(64,20),(96,20),(128,40)]:
        r=rollout(build_solver(cfg,n=n,substeps=sub,order=4),resize(x,n),20)
        refs[n]=np.asarray(resolved_reference(r,cfg.coarse_points))
        print('reference',n,'finite',np.isfinite(refs[n]).all(),flush=True)
    def rmse(a,b):return np.sqrt(np.mean((a-b)**2,axis=(2,3,4)))
    gap=rmse(np.asarray(base),refs[96])
    validation={'config':asdict(cfg),'provenance':provenance(),
        'warmup_rms_by_time':np.sqrt(np.mean(np.array(states)**2,axis=(2,3,4))),
        'teacher_reference_mean_RMSE':gap.mean(),
        'teacher_reference_RMSE_at_20':gap[:,-1].mean(),
        '64_to_96_reference_mean_RMSE':rmse(refs[64],refs[96]).mean(),
        '96_to_128_reference_mean_RMSE':rmse(refs[96],refs[128]).mean(),
        '96_to_128_fraction_of_teacher_gap':rmse(refs[96],refs[128]).mean()/gap.mean(),
        'resize_roundtrip_max_difference':float(jnp.max(jnp.abs(x-resize(resize(x,96),32)))),
        'nonforcing_energy_fraction':float(1-np.sum(abs(np.fft.rfft2(np.asarray(x))[:,:,0,4])**2)*2/np.sum(abs(np.fft.fft2(np.asarray(x)))**2)),
        'elapsed_physics_s':time.time()-t0}
    print(validation,flush=True)
    trj=jnp.concatenate([x[:,None],rollout(teacher,x,8)],axis=1)
    model=ResidualFNO(cfg,float(jnp.sqrt(jnp.mean(x*x))),9001)
    params,static=eqx.partition(model,eqx.is_inexact_array)
    opt=optax.adam(1e-3);state=opt.init(params)
    batch=trj[:,:5]
    @eqx.filter_jit
    def chunk(p,s,updates,n):
        def loss(pp):
            m=eqx.combine(pp,static);z=batch[:,0];total=0.
            for i in range(n):
                z=jax.vmap(m)(z);total+=jnp.mean(((z-batch[:,i+1])/m.scale)**2)
            return total/n
        def step(c,_):
            p,s=c;v,g=eqx.filter_value_and_grad(loss)(p)
            update,s=opt.update(g,s,p)
            return (eqx.apply_updates(p,update),s),v
        return jax.lax.scan(step,(p,s),None,length=updates)
    for n in [1,4]:
        start=time.time();(_, _),loss=chunk(params,state,10,n);loss.block_until_ready();compile_s=time.time()-start
        (_, _),loss=chunk(params,state,100,n);loss.block_until_ready()
        start=time.time();(_, _),loss=chunk(params,state,100,n);loss.block_until_ready();elapsed=time.time()-start
        validation[f'training_N{n}_compile_s']=compile_s
        validation[f'training_N{n}_100_updates_s']=elapsed
        validation[f'training_N{n}_pilot_loss_start_end']=[float(loss[0]),float(loss[-1])]
        print('training benchmark',n,compile_s,elapsed,flush=True)
    validation['n_params']=sum(a.size for a in jax.tree_util.tree_leaves(params))
    atomic_json(OUT/'pilot_validation.json',validation)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    main()
