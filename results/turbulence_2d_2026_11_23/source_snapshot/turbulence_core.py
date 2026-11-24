"""Periodic, dealiased 2D Kolmogorov flow and a coarse-only residual FNO.

Independent extension: no authors' unlicensed code is imported. Exponax and
PDEquinox are used under their own licenses. Arrays are (..., channels, x, y).
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/emusup-mpl')

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pdequinox as pdeq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'turbulence_2d_2026_11_23'


@dataclass(frozen=True)
class FlowConfig:
    coarse_points: int = 32
    reference_points: int = 96
    domain: float = 2 * np.pi
    viscosity: float = .01
    drag: float = -.1
    forcing_mode: int = 4
    forcing_scale: float = 1.
    output_dt: float = .2
    teacher_substeps: int = 10
    reference_substeps: int = 20
    warmup_steps: int = 300
    training_horizon: int = 8
    test_horizon: int = 100
    primary_horizon: int = 20
    hidden_channels: int = 12
    fno_modes: int = 8
    fno_blocks: int = 3


def atomic_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (tuple, list)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer): return int(x)
        return x
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def cutoff(n):
    return n//3-1


def fourier_mask(n, kmax=None):
    kmax=cutoff(n) if kmax is None else kmax
    kx=jnp.fft.fftfreq(n)*n
    ky=jnp.fft.rfftfreq(n)*n
    return ((jnp.abs(kx[:,None])<=kmax)&(ky[None,:]<=kmax))[None]


def project(x, kmax=None):
    n=x.shape[-1]
    h=jnp.fft.rfft2(x)*fourier_mask(n,kmax)
    h=h.at[...,0,0].set(0.)
    return jnp.fft.irfft2(h,s=(n,n))


def resize(x, n):
    """Fourier interpolation/restriction; input has zero Nyquist modes.

    Forward-normalized coefficients preserve amplitudes across grid sizes.
    Fine-to-coarse callers must first filter to the common coarse band.
    """
    old=x.shape[-1]
    h=jnp.fft.fftshift(jnp.fft.fft2(x,norm='forward'),axes=(-2,-1))
    if n>=old:
        pad=(n-old)//2
        h=jnp.pad(h,[(0,0)]*(h.ndim-2)+[(pad,pad),(pad,pad)])
    else:
        start=(old-n)//2
        h=h[...,start:start+n,start:start+n]
    return jnp.fft.ifft2(jnp.fft.ifftshift(h,axes=(-2,-1)),norm='forward').real


def resolved_reference(x, coarse_points):
    return resize(project(x,cutoff(coarse_points)),coarse_points)


class ProjectedStep(eqx.Module):
    stepper: eqx.Module
    substeps: int = eqx.field(static=True)

    def __call__(self,x):
        n=x.shape[-1];mask=fourier_mask(n)
        h=jnp.fft.rfft2(x)
        def body(i,z):
            z=self.stepper.step_fourier(z)*mask
            return z.at[...,0,0].set(0.)
        h=jax.lax.fori_loop(0,self.substeps,body,h)
        return jnp.fft.irfft2(h,s=(n,n))


class PhysicalKolmogorovFlow(ex.stepper.KolmogorovFlowVorticity):
    """Use scalar drag exactly once.

    Exponax 0.1.0 constructs the drag using sum(derivative_operator**0),
    which multiplies the documented drag by the number of spatial axes.
    An analytic Taylor-Green check detects this. Override the linear symbol
    rather than silently change the requested physical coefficient.
    """
    def _build_linear_operator(self,derivative_operator):
        return self.diffusivity*jnp.sum(derivative_operator**2,axis=0,keepdims=True)+self.drag


def build_solver(cfg, n=None, substeps=None, order=None):
    n=cfg.coarse_points if n is None else n
    coarse=n==cfg.coarse_points
    substeps=(cfg.teacher_substeps if coarse else cfg.reference_substeps) if substeps is None else substeps
    order=(2 if coarse else 4) if order is None else order
    base=PhysicalKolmogorovFlow(2,cfg.domain,n,cfg.output_dt/substeps,
        diffusivity=cfg.viscosity,drag=cfg.drag,injection_mode=cfg.forcing_mode,
        injection_scale=cfg.forcing_scale,order=order,dealiasing_fraction=2/3)
    return ProjectedStep(base,substeps)


def initial_fields(cfg, seed, count):
    n=cfg.coarse_points
    key=jax.random.key(seed)
    noise=jax.random.normal(key,(count,1,n,n))
    kx=jnp.fft.fftfreq(n)*n;ky=jnp.fft.rfftfreq(n)*n
    envelope=jnp.exp(-(kx[:,None]**2+ky[None,:]**2)/(2*3.**2))
    x=jnp.fft.irfft2(jnp.fft.rfft2(noise)*envelope,s=(n,n))
    x=project(x)
    amp=jax.random.uniform(jax.random.fold_in(key,1),(count,1,1,1),minval=.5,maxval=1.5)
    return x/jnp.sqrt(jnp.mean(x*x,axis=(-2,-1),keepdims=True))*amp


@eqx.filter_jit
def warmup(stepper, x, steps):
    def body(i,z): return jax.vmap(stepper)(z)
    return jax.lax.fori_loop(0,steps,body,x)


@eqx.filter_jit
def rollout(stepper, x, steps):
    def step(z,_):
        z=jax.vmap(stepper)(z)
        return z,z
    return jax.lax.scan(step,x,None,length=steps)[1].swapaxes(0,1)


class ResidualFNO(eqx.Module):
    net: eqx.Module
    forcing_mode: int = eqx.field(static=True)
    forcing_coefficient: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)

    def __init__(self,cfg,scale,seed):
        self.net=pdeq.arch.ClassicFNO(2,2,1,hidden_channels=cfg.hidden_channels,
            num_modes=cfg.fno_modes,num_blocks=cfg.fno_blocks,
            activation=jax.nn.gelu,key=jax.random.key(seed))
        self.forcing_mode=cfg.forcing_mode
        self.forcing_coefficient=-cfg.forcing_mode*(2*np.pi/cfg.domain)*cfg.forcing_scale
        self.scale=float(scale)
        self.residual_scale=cfg.output_dt

    def __call__(self,x):
        n=x.shape[-1]
        f=self.forcing_coefficient*jnp.cos(self.forcing_mode*2*np.pi*jnp.arange(n)/n)
        forcing=jnp.broadcast_to(f[None,None,:],(1,n,n))
        features=jnp.concatenate([x/self.scale,forcing/self.scale],axis=0)
        return project(x+self.residual_scale*self.scale*self.net(features))


def velocity(x, domain=2*np.pi):
    n=x.shape[-1];kx=jnp.fft.fftfreq(n)*n*2*np.pi/domain
    ky=jnp.fft.rfftfreq(n)*n*2*np.pi/domain
    k2=kx[:,None]**2+ky[None,:]**2
    psi=jnp.fft.rfft2(x)/jnp.where(k2>0,-k2,1.)
    u=jnp.fft.irfft2(1j*ky[None,:]*psi,s=(n,n))
    v=jnp.fft.irfft2(-1j*kx[:,None]*psi,s=(n,n))
    return jnp.concatenate([u,v],axis=-3)


def provenance():
    return {'dependencies':{p:importlib.metadata.version(p) for p in
        ['jax','jaxlib','exponax','pdequinox','equinox','optax','numpy']},
        'devices':[str(x) for x in jax.devices()],
        'source_sha256':{name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                         for name in ['turbulence_core.py']}}
