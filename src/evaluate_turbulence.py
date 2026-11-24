"""Frozen-model evaluation against fine solutions; never used for training."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from turbulence_core import (FlowConfig,OUT,ResidualFNO,atomic_json,rollout,velocity,
    build_solver,resize,resolved_reference)
from turbulence_data import radial_spectra

EVAL_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def evaluation_identity():
    return {'evaluator_source_sha256':EVAL_HASH,
        'reference_sha256':hashlib.sha256((OUT/'data'/'reference.npz').read_bytes()).hexdigest(),
        'test_data_sha256':hashlib.sha256((OUT/'data'/'test.npz').read_bytes()).hexdigest()}


def cached(folder,identity):
    p=folder/'evaluation.json'
    return p.exists() and json.loads(p.read_text()).get('evaluation_provenance')==identity


def rmse(z):return np.sqrt(np.mean(z*z,axis=(-3,-2,-1)))


def normalized(error,denominator):
    return error/np.maximum(denominator,1e-12)


def ratio(a,b,h=20,start=0):
    a=a[:,start:h];b=b[:,start:h]
    return float(a.mean(axis=0).sum()/b.mean(axis=0).sum()) if np.isfinite(a).all() else float('inf')


def bootstrap(a,b,h=20):
    aa=a[:,:h].sum(axis=1);bb=b[:,:h].sum(axis=1)
    if not np.isfinite(aa).all():return [float('inf'),float('inf')]
    idx=np.random.default_rng(20261123).integers(0,len(aa),(4000,len(aa)))
    draws=aa[idx].sum(axis=1)/bb[idx].sum(axis=1)
    return np.quantile(draws,[.025,.975]).tolist()


def metrics(pred,teacher,ref,cfg,folder,identity):
    folder.mkdir(parents=True,exist_ok=True)
    target=ref['resolved']
    r_rms=rmse(target);p_rmse=rmse(pred-target);t_rmse=rmse(teacher-target)
    p_n=normalized(p_rmse,r_rms);t_n=normalized(t_rmse,r_rms)
    p_vel=np.asarray(velocity(jnp.asarray(pred),cfg.domain))
    t_vel=np.asarray(velocity(jnp.asarray(teacher),cfg.domain))
    r_vel=np.asarray(velocity(jnp.asarray(target),cfg.domain))
    pv=rmse(p_vel-r_vel);tv=rmse(t_vel-r_vel);vrms=rmse(r_vel)
    un=ref['unresolved_vorticity_variance'];uv=ref['unresolved_velocity_variance']
    full_p=np.sqrt(p_rmse*p_rmse+un);full_t=np.sqrt(t_rmse*t_rmse+un)
    full_vp=np.sqrt(pv*pv+uv);full_vt=np.sqrt(tv*tv+uv)
    full_rms=np.sqrt(ref['full_vorticity_variance'])
    full_vrms=np.sqrt(ref['full_velocity_variance'])
    energy=.5*np.mean(np.sum(p_vel*p_vel,axis=2),axis=(2,3))
    enstrophy=.5*np.mean(pred*pred,axis=(2,3,4))
    corr=np.mean(pred*target,axis=(2,3,4))/np.maximum(rmse(pred)*r_rms,1e-12)
    tighter=ref['tighter_resolved'];tight_rms=rmse(tighter)
    tight_p=normalized(rmse(pred[:,:20]-tighter),tight_rms)
    tight_t=normalized(rmse(teacher[:,:20]-tighter),tight_rms)
    per_step=pd.DataFrame({'step':np.arange(1,101),'physical_time':np.arange(1,101)*cfg.output_dt,
        'emulator_nRMSE':p_n.mean(axis=0),'teacher_nRMSE':t_n.mean(axis=0),
        'ratio':p_n.mean(axis=0)/t_n.mean(axis=0),
        'emulator_RMSE':p_rmse.mean(axis=0),'teacher_RMSE':t_rmse.mean(axis=0),
        'velocity_nRMSE':normalized(pv,vrms).mean(axis=0),
        'teacher_velocity_nRMSE':normalized(tv,vrms).mean(axis=0),
        'correlation':corr.mean(axis=0),'kinetic_energy':energy.mean(axis=0),
        'reference_kinetic_energy':ref['kinetic_energy'].mean(axis=0),
        'enstrophy':enstrophy.mean(axis=0),'reference_enstrophy':ref['enstrophy'].mean(axis=0)})
    per_step.to_csv(folder/'metrics.csv',index=False)
    # Squared error identity: e_model=e_teacher+(model-teacher).
    delta=pred-teacher;bias=teacher-target
    teacher_mse=np.mean(bias*bias,axis=(0,2,3,4))
    deviation=np.mean(delta*delta,axis=(0,2,3,4))
    cross=2*np.mean(bias*delta,axis=(0,2,3,4))
    model_mse=np.mean((pred-target)**2,axis=(0,2,3,4))
    pd.DataFrame({'step':np.arange(1,101),'teacher_MSE':teacher_mse,
        'deviation_MSE':deviation,'cross_term':cross,'emulator_MSE':model_mse,
        'identity_residual':model_mse-teacher_mse-deviation-cross}).to_csv(folder/'error_decomposition.csv',index=False)
    spectra=radial_spectra(p_vel)
    np.savez_compressed(folder/'diagnostics.npz',
        emulator_nRMSE=p_n,teacher_nRMSE=t_n,emulator_RMSE=p_rmse,teacher_RMSE=t_rmse,
        velocity_nRMSE=normalized(pv,vrms),teacher_velocity_nRMSE=normalized(tv,vrms),
        full_nRMSE=normalized(full_p,full_rms),teacher_full_nRMSE=normalized(full_t,full_rms),
        kinetic_energy=energy,enstrophy=enstrophy,correlation=corr,
        mean_energy_spectrum_1_20=spectra[:,:20].mean(axis=(0,1)),
        mean_energy_spectrum_21_100=spectra[:,20:].mean(axis=(0,1)),
        snapshots=pred[:3,[0,4,19,99]])
    bad=np.flatnonzero(~np.isfinite(p_n).all(axis=0))
    extreme=np.flatnonzero(np.any(p_n>10,axis=0))
    primary=ratio(p_n,t_n)
    summary={**identity,'evaluation_provenance':evaluation_identity(),
        'status':'completed' if not len(bad) else 'nonfinite_rollout',
        'R20':primary,'R100':ratio(p_n,t_n,100),'R21_100':ratio(p_n,t_n,100,20),
        'absolute_RMSE_R20':ratio(p_rmse,t_rmse),
        'velocity_R20':ratio(normalized(pv,vrms),normalized(tv,vrms)),
        'full_field_R20':ratio(normalized(full_p,full_rms),normalized(full_t,full_rms)),
        'full_velocity_R20':ratio(normalized(full_vp,full_vrms),normalized(full_vt,full_vrms)),
        'primary_bootstrap95':bootstrap(p_n,t_n),
        'bootstrap_scope':'64 test trajectories, conditional on the trained model; not independent across models',
        'test_example_win_fraction':float(np.mean(p_n[:,:20].sum(axis=1)<t_n[:,:20].sum(axis=1))),
        'emulator_mean_nRMSE_1_20':float(p_n[:,:20].mean()),
        'teacher_mean_nRMSE_1_20':float(t_n[:,:20].mean()),
        'mean_correlation_1_20':float(corr[:,:20].mean()),
        'mean_correlation_21_100':float(corr[:,20:].mean()),
        'kinetic_energy_relative_bias_1_20':float(energy[:,:20].mean()/ref['kinetic_energy'][:,:20].mean()-1),
        'kinetic_energy_relative_bias_21_100':float(energy[:,20:].mean()/ref['kinetic_energy'][:,20:].mean()-1),
        'enstrophy_relative_bias_1_20':float(enstrophy[:,:20].mean()/ref['enstrophy'][:,:20].mean()-1),
        'enstrophy_relative_bias_21_100':float(enstrophy[:,20:].mean()/ref['enstrophy'][:,20:].mean()-1),
        'first_nonfinite_step':int(bad[0]+1) if len(bad) else None,
        'first_any_example_nRMSE_gt10':int(extreme[0]+1) if len(extreme) else None,
        'R20_tighter_reference':ratio(tight_p,tight_t),
        'reference_primary_change':ratio(tight_p,tight_t)-primary,
        'deviation_MSE_fraction_1_20':float(deviation[:20].mean()/teacher_mse[:20].mean()),
        'cross_term_fraction_1_20':float(cross[:20].mean()/teacher_mse[:20].mean()),
        'decomposition_max_abs_residual':float(np.nanmax(abs(model_mse-teacher_mse-deviation-cross)))}
    for h in [1,5,10,20,50,100]:summary[f'ratio_at_{h}']=float(p_n[:,h-1].mean()/t_n[:,h-1].mean())
    atomic_json(folder/'evaluation.json',summary)
    print(identity.get('name'),primary,flush=True)
    return summary


def main():
    base=FlowConfig();data=OUT/'data'
    test=np.load(data/'test.npz');initial=test['initial'];teacher=test['trajectory'][:,1:]
    ref=np.load(data/'reference.npz')
    baseline_folder=OUT/'baselines'
    eval_identity=evaluation_identity()
    if not cached(baseline_folder/'teacher',eval_identity):
        metrics(teacher,teacher,ref,base,baseline_folder/'teacher',{'name':'teacher','method':'teacher'})
    if not cached(baseline_folder/'persistence',eval_identity):
        metrics(np.repeat(initial[:,None],100,axis=1),teacher,ref,base,baseline_folder/'persistence',{'name':'persistence','method':'persistence'})
    if not cached(baseline_folder/'zero',eval_identity):
        metrics(np.zeros_like(teacher),teacher,ref,base,baseline_folder/'zero',{'name':'zero','method':'zero'})
    if not cached(baseline_folder/'linear_flow',eval_identity):
        n=base.coarse_points;k=np.fft.fftfreq(n)*n
        symbol=-base.viscosity*(k[:,None]**2+k[None,:]**2)+base.drag
        y=np.arange(n)*base.domain/n
        f=np.broadcast_to(-4*np.cos(4*y)[None,None,:],initial.shape[1:])
        h0=np.fft.fft2(initial);fh=np.fft.fft2(f)
        frames=[]
        for t in np.arange(1,101)*base.output_dt:
            e=np.exp(symbol*t);frames.append(np.fft.ifft2(e*h0+fh*np.expm1(symbol*t)/symbol).real.astype(np.float32))
        metrics(np.stack(frames,axis=1),teacher,ref,base,baseline_folder/'linear_flow',{'name':'linear_flow','method':'linear_flow'})
    # Secondary numerical controls, added without changing any trained model.
    for name,grid in [('time_refined32',32),('space_refined48',48)]:
        folder=baseline_folder/name
        if cached(folder,eval_identity):continue
        stepper=build_solver(base,n=grid,substeps=20,order=4)
        fields=[]
        for i in range(0,len(initial),8):
            z=rollout(stepper,resize(jnp.asarray(initial[i:i+8]),grid),100)
            fields.append(np.asarray(resolved_reference(z,32)))
        metrics(np.concatenate(fields),teacher,ref,base,folder,{'name':name,'method':name,
            'note':'Secondary numerical control, projected to the common band; not a trained model.'})
    for summary in sorted((OUT/'runs').glob('*/training_summary.json')):
        folder=summary.parent
        if cached(folder,eval_identity):continue
        s=json.loads(summary.read_text());c=s['config'];cfg=FlowConfig(**c['flow'])
        model=ResidualFNO(cfg,c['scale'],c['initialization_seed'])
        assert hashlib.sha256((folder/'model.eqx').read_bytes()).hexdigest()==s['model_sha256']
        model=eqx.tree_deserialise_leaves(folder/'model.eqx',model)
        predictions=[]
        for i in range(0,len(initial),8):
            predictions.append(np.asarray(rollout(model,jnp.asarray(initial[i:i+8]),100)))
        pred=np.concatenate(predictions)
        metrics(pred,teacher,ref,cfg,folder,{'name':folder.name,'method':'FNO',
            'N':c['unroll'],'dataset':c['training_dataset'],'seed':c['initialization_seed'],
            'completed_updates':s['completed_updates'],
            'common_validation_MSE':s['common_validation_one_step_MSE'],
            'common_validation_normalized_MSE':s['common_validation_normalized_one_step_MSE']})


if __name__=='__main__':main()
