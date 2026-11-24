"""Numerical checks before main training; failures stop the experiment."""
from dataclasses import replace
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from turbulence_core import (FlowConfig,OUT,atomic_json,build_solver,initial_fields,
    project,resize,resolved_reference,rollout,ResidualFNO,provenance)
from train_turbulence import make_windows,optimizer,train_chunk


def main():
    cfg=replace(FlowConfig(),hidden_channels=16,fno_modes=10,fno_blocks=4);n=cfg.coarse_points
    t=jnp.arange(n)*2*jnp.pi/n
    wave=(jnp.cos(2*t[:,None]+3*t[None,:]))[None]
    transfer=float(jnp.max(jnp.abs(wave-resize(resize(wave,96),32))))
    assert transfer<1e-5,transfer
    tg=(2*jnp.cos(t[:,None])*jnp.cos(t[None,:]))[None]
    unforced=replace(cfg,forcing_scale=0.)
    trajectory=rollout(build_solver(unforced),tg[None],10)[0]
    expected=tg*jnp.exp((-2*cfg.viscosity+cfg.drag)*2.)
    analytic=float(jnp.sqrt(jnp.mean((trajectory[-1]-expected)**2)/jnp.mean(expected**2)))
    assert analytic<2e-5,analytic
    teacher=build_solver(cfg)
    zero=jnp.zeros_like(wave);pred=teacher(zero)
    forcing=-4*jnp.cos(4*t)[None,None,:]
    exact=jnp.broadcast_to(forcing,pred.shape)*jnp.expm1((-16*cfg.viscosity+cfg.drag)*cfg.output_dt)/(-16*cfg.viscosity+cfg.drag)
    forced=float(jnp.max(jnp.abs(pred-exact)))
    assert forced<1e-5,forced
    x=initial_fields(cfg,9901,4)
    mask=teacher.stepper._integrator._nonlinear_fun
    # Remove the known injection from the nonlinear RHS to check transport alone.
    h=jnp.fft.rfft2(x[0])
    adv=mask(h)-mask.injection
    adv=adv*__import__('turbulence_core').fourier_mask(n)
    a=jnp.fft.irfft2(adv,s=(n,n))
    k=jnp.fft.fftfreq(n)*n;kr=jnp.fft.rfftfreq(n)*n
    k2=k[:,None]**2+kr[None,:]**2
    psi=jnp.fft.irfft2(h/jnp.where(k2>0,-k2,1),s=(n,n))
    enstrophy_defect=float(abs(jnp.mean(x[0]*a)))
    energy_defect=float(abs(jnp.mean(psi*a)))
    assert max(enstrophy_defect,energy_defect)<1e-5
    trj=jnp.concatenate([x[:,None],rollout(teacher,x,8)],axis=1)
    model=ResidualFNO(cfg,float(jnp.sqrt(jnp.mean(x*x))),9901)
    params,static=eqx.partition(model,eqx.is_inexact_array)
    opt=optimizer(30);state=opt.init(params)
    windows=make_windows(trj,4);idx=jax.random.randint(jax.random.key(99),(8,8),0,len(windows))
    (whole,ws),_=train_chunk(params,state,static,windows,idx,opt,4,model.scale)
    (half,hs),_=train_chunk(params,state,static,windows,idx[:4],opt,4,model.scale)
    (half,hs),_=train_chunk(half,hs,static,windows,idx[4:],opt,4,model.scale)
    resume=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree_util.tree_leaves(whole),jax.tree_util.tree_leaves(half)))
    assert resume==0.,resume
    def loss(p):
        z=jax.vmap(eqx.combine(p,static))(x)
        return jnp.mean(((z-trj[:,1])/model.scale)**2)
    grad=eqx.filter_grad(loss)(params)
    # An all-ones direction nearly cancels this derivative in float32. Use a
    # unit gradient direction so subtraction is well-conditioned; validate the
    # actual finite difference, not a near-zero difference of similar losses.
    norm=jnp.sqrt(sum(jnp.sum(a*a) for a in jax.tree_util.tree_leaves(grad)))
    direction=jax.tree_util.tree_map(lambda a:a/norm,grad)
    dot=sum(jnp.sum(a*b) for a,b in zip(jax.tree_util.tree_leaves(grad),jax.tree_util.tree_leaves(direction)))
    eps=.001
    plus=jax.tree_util.tree_map(lambda a,d:a+eps*d,params,direction)
    minus=jax.tree_util.tree_map(lambda a,d:a-eps*d,params,direction)
    fd=(loss(plus)-loss(minus))/(2*eps)
    relative=float(abs(dot-fd)/jnp.maximum(abs(dot),1e-6))
    assert relative<.02,(dot,fd,relative)
    pilot=np.load(OUT/'pilot_fields.npz')['warmup'][-1]
    r1=rollout(build_solver(cfg,n=96,substeps=20,order=4),resize(jnp.asarray(pilot),96),20)
    r2=rollout(build_solver(cfg,n=96,substeps=40,order=4),resize(jnp.asarray(pilot),96),20)
    time_difference=float(jnp.sqrt(jnp.mean((resolved_reference(r1,32)-resolved_reference(r2,32))**2)))
    primary=__import__('json').loads((OUT/'pilot_validation.json').read_text())
    assert primary['96_to_128_fraction_of_teacher_gap']<.05
    assert time_difference/primary['teacher_reference_mean_RMSE']<.01
    result={'passed':True,'provenance':provenance(),
        'fourier_interpolation_roundtrip_max_error':transfer,
        'Taylor_Green_relative_error_after_time_2':analytic,
        'forced_Stokes_max_error':forced,
        'nonlinear_advection_enstrophy_defect':enstrophy_defect,
        'nonlinear_advection_kinetic_energy_defect':energy_defect,
        'resumed_Adam_parameter_difference':resume,
        'FNO_directional_derivative_relative_difference':relative,
        'reference_time_refinement_RMS_difference':time_difference,
        'reference_time_refinement_fraction_of_teacher_gap':time_difference/primary['teacher_reference_mean_RMSE'],
        'n_trainable_parameters':sum(a.size for a in jax.tree_util.tree_leaves(params)),
        'fixed_forcing_is_not_trainable':not hasattr(params,'forcing')}
    atomic_json(OUT/'validation.json',result);print(result)


if __name__=='__main__':main()
