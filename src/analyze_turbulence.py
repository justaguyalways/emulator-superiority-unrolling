"""Regenerate transparent tables and scientific figures from frozen results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from turbulence_core import OUT

FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLORS={1:'#2563eb',4:'#d97706'}


def interval(v):
    a=np.asarray(v,float)
    if not np.isfinite(a).all():return 'unstable (retained)'
    return f'{np.median(a):.3f} [{a.min():.3f}, {a.max():.3f}]'


def save(fig,name,layout=True):
    if layout:fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(FIG/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def collect():
    records=[]
    for p in sorted((OUT/'runs').glob('*/evaluation.json')):
        d=json.loads(p.read_text());d['directory']=str(p.parent)
        d['CI_low'],d['CI_high']=d['primary_bootstrap95']
        if d['R20'] is None:d['R20']=np.inf
        records.append(d)
    for p in sorted((OUT/'runs').glob('*/failure.json')):
        if (p.parent/'evaluation.json').exists():continue
        s=json.loads(p.read_text());c=s['config']
        records.append({'name':p.parent.name,'directory':str(p.parent),'method':'FNO',
            'N':c['unroll'],'dataset':c['training_dataset'],'seed':c['initialization_seed'],
            'status':'training_failed','R20':np.inf,'R100':np.inf,'R21_100':np.inf,
            'velocity_R20':np.inf,'full_field_R20':np.inf,'CI_low':np.inf,'CI_high':np.inf})
    df=pd.DataFrame(records)
    if len(df):df.to_csv(OUT/'per_model_results.csv',index=False)
    baselines=pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'baselines').glob('*/evaluation.json')])
    if len(baselines):baselines.to_csv(OUT/'baseline_results.csv',index=False)
    return df,baselines


def main():
    df,base=collect()
    if len(df)==0:return
    tables=[]
    for (n,dataset),g in df.groupby(['N','dataset']):
        tables.append({'N':n,'training_data_seed':dataset,'models':len(g),
            **{m:interval(g[m]) for m in ['R20','R100','velocity_R20','full_field_R20']},
            'models_better':f'{sum(g.R20<1)}/{len(g)}'})
    pd.DataFrame(tables).to_csv(OUT/'group_results.csv',index=False)
    contrasts=[]
    for (dataset,seed),g in df.groupby(['dataset','seed']):
        if set(g.N)=={1,4}:
            a=g[g.N==1].iloc[0];b=g[g.N==4].iloc[0]
            contrasts.append({'dataset':dataset,'seed':seed,'N1_R20':a.R20,'N4_R20':b.R20,
                'N4_minus_N1_R20':b.R20-a.R20,'N1_R100':a.R100,'N4_R100':b.R100,
                'N4_minus_N1_R100':b.R100-a.R100})
    pd.DataFrame(contrasts).to_csv(OUT/'paired_training_contrasts.csv',index=False)
    all_df=df
    df=df[df.status=='completed']
    if len(df)==0:return
    ref=np.load(OUT/'data'/'reference.npz')
    teacher=pd.read_csv(OUT/'baselines'/'teacher'/'metrics.csv')
    zero=pd.read_csv(OUT/'baselines'/'zero'/'metrics.csv')
    curves={r['name']:pd.read_csv(Path(r['directory'])/'metrics.csv') for r in records_as_dict(df)}

    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for n,g in df.groupby('N'):
        array=np.array([curves[name].ratio.to_numpy() for name in g.name])
        for ax,h in zip(axs,[20,100]):
            t=np.arange(1,h+1)*.2
            ax.plot(t,np.median(array[:,:h],axis=0),color=COLORS[n],label=f'N={n}')
            ax.fill_between(t,np.min(array[:,:h],axis=0),np.max(array[:,:h],axis=0),color=COLORS[n],alpha=.16)
    for ax,h in zip(axs,[20,100]):
        ax.axhline(1,color='#475569',ls='--',lw=1,label='Equal to teacher')
        ax.set_xlabel('Physical forecast time');ax.grid(alpha=.2)
        ax.set_title('Primary window: 20 steps' if h==20 else 'Extended forecast: 100 steps')
    zero_late=float(base[base.name=='zero'].iloc[0].R21_100)
    axs[1].text(.03,.03,f'Zero-field later-window score: {zero_late:.3f} (energy = 0)',
        transform=axs[1].transAxes,fontsize=7.5,color='#64748b')
    axs[0].set_ylabel('Vorticity error / teacher error');axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    save(fig,'accuracy_rollouts')

    fig,ax=plt.subplots(figsize=(8,3.4))
    for n in [1,4]:
        g=df[df.N==n].sort_values(['dataset','seed'])
        for j,(_,r) in enumerate(g.iterrows()):
            x=([1,4].index(n))+(j-(len(g)-1)/2)*.055
            ax.errorbar(x,r.R20,yerr=[[max(0,r.R20-r.CI_low)],[max(0,r.CI_high-r.R20)]],
                fmt='o' if r.dataset==1101 else 's',color=COLORS[n],capsize=3,
                markerfacecolor=COLORS[n] if r.dataset==1101 else 'white')
    ax.axhline(1,color='#475569',ls='--');ax.set_xticks([0,1],['One-step training','Four-step training'])
    ax.set_ylabel('Total-error ratio, steps 1-20');ax.grid(axis='y',alpha=.2)
    ax.set_title('Two training datasets x three initializations per recipe')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markeredgecolor='#475569',markerfacecolor='#475569',label='Training set A'),
        Line2D([0],[0],marker='s',color='none',markeredgecolor='#475569',markerfacecolor='white',label='Training set B')],
        loc='upper center',fontsize=8,title='Bars: conditional 95% intervals',title_fontsize=7.5)
    save(fig,'primary_seed_results')

    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for n,g in df.groupby('N'):
        for ax,col,denom in zip(axs,['kinetic_energy','enstrophy','correlation'],['reference_kinetic_energy','reference_enstrophy',None]):
            vals=np.array([(curves[name][col]/curves[name][denom]).to_numpy() if denom else curves[name][col].to_numpy() for name in g.name])
            t=teacher.physical_time
            ax.plot(t,np.median(vals,axis=0),color=COLORS[n],label=f'N={n}')
            ax.fill_between(t,vals.min(axis=0),vals.max(axis=0),color=COLORS[n],alpha=.14)
    for ax,col,denom in zip(axs,['kinetic_energy','enstrophy','correlation'],['reference_kinetic_energy','reference_enstrophy',None]):
        vals=teacher[col]/teacher[denom] if denom else teacher[col]
        ax.plot(teacher.physical_time,vals,color='#475569',ls='--',label='Teacher')
        if denom:ax.axhline(1,color='black',lw=.7)
        ax.set_xlabel('Physical forecast time');ax.grid(alpha=.2)
    axs[0].set_ylabel('Kinetic energy / reference');axs[1].set_ylabel('Enstrophy / reference')
    axs[2].set_ylabel('Field correlation with reference');axs[0].legend(fontsize=8)
    save(fig,'physical_structure')

    fig,ax=plt.subplots(figsize=(8,3.7))
    rs=ref['energy_spectrum'][:,:20].mean(axis=(0,1));ax.semilogy(np.arange(1,len(rs)),rs[1:],color='black',label='96-grid reference')
    ts=np.load(OUT/'baselines'/'teacher'/'diagnostics.npz')['mean_energy_spectrum_1_20']
    ax.semilogy(np.arange(1,len(ts)),ts[1:],color='#64748b',ls='--',label='32-grid teacher')
    for n,g in df.groupby('N'):
        vals=np.array([np.load(Path(p)/'diagnostics.npz')['mean_energy_spectrum_1_20'] for p in g.directory])
        k=np.arange(1,vals.shape[1]);ax.semilogy(k,np.median(vals,axis=0)[1:],color=COLORS[n],label=f'N={n}')
        ax.fill_between(k,vals.min(axis=0)[1:],vals.max(axis=0)[1:],color=COLORS[n],alpha=.14)
    ax.axvspan(13,24,color='#e2e8f0',alpha=.5,label='Beyond coarse-band maximum radius')
    ax.set_xlim(1,24);ax.set_ylim(max(rs[1:25].max()*1e-10,1e-12),rs[1:25].max()*2)
    ax.set_xlabel('Radial wavenumber');ax.set_ylabel('Kinetic energy per shell')
    ax.set_title('Time-averaged spectrum over the primary forecast window');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'energy_spectrum')

    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for n,g in df.groupby('N'):
        vals=np.array([np.load(Path(p)/'loss.npy').reshape(-1,300).mean(axis=1) for p in g.directory])
        updates=np.arange(1,vals.shape[1]+1)*300
        axs[0].semilogy(updates,np.median(vals,axis=0),color=COLORS[n],label=f'N={n}')
        axs[0].fill_between(updates,vals.min(axis=0),vals.max(axis=0),color=COLORS[n],alpha=.14)
        for _,r in g.iterrows():axs[1].scatter(r.common_validation_normalized_MSE,r.R20,color=COLORS[n],marker='o' if r.dataset==1101 else 's')
    axs[0].set_xlabel('Adam updates');axs[0].set_ylabel('Normalized training MSE per step');axs[0].legend()
    axs[1].set_xscale('log');axs[1].axhline(1,color='#475569',ls='--')
    axs[1].set_xlabel('Same-target, one-step validation MSE / scale squared');axs[1].set_ylabel('Primary error ratio')
    for ax in axs:ax.grid(alpha=.2)
    save(fig,'teacher_imitation')

    fig,ax=plt.subplots(figsize=(8,3.3));positions=np.arange(len(df.groupby(['N','dataset'])))
    for j,((n,dataset),g) in enumerate(df.groupby(['N','dataset'])):
        cost=np.median(g.deviation_MSE_fraction_1_20);cross=np.median(g.cross_term_fraction_1_20)
        ax.bar(j-.16,cost,width=.3,color='#f59e0b',label='Deviation cost' if j==0 else None)
        ax.bar(j+.16,cross,width=.3,color='#2563eb',label='Alignment term (negative helps)' if j==0 else None)
    ax.set_xticks(positions,[f'N={n}, set {"A" if d==1101 else "B"}' for n,d in df.groupby(['N','dataset']).groups])
    ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Fraction of teacher MSE, steps 1-20');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    save(fig,'error_alignment')

    # The first test example and data1101/seed0 models are chosen in advance.
    chosen={}
    for n in [1,4]:
        g=df[(df.N==n)&(df.dataset==1101)&(df.seed==0)]
        if len(g):chosen[n]=np.load(Path(g.iloc[0].directory)/'diagnostics.npz')['snapshots'][0]
    if len(chosen)==2:
        fine=ref['fine_snapshots'][0]
        coarse=np.load(OUT/'data'/'test.npz')['trajectory'][0]
        fig,axs=plt.subplots(4,3,figsize=(9,9))
        rows=[fine[[2,3,4],0],coarse[[5,20,100],0],chosen[1][[1,2,3],0],chosen[4][[1,2,3],0]]
        lim=np.max(abs(rows[0]));labels=['Fine reference','Rough teacher','One-step FNO','Four-step FNO']
        for i,row in enumerate(rows):
            for j,field in enumerate(row):
                im=axs[i,j].imshow(field.T,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,extent=(0,2*np.pi,0,2*np.pi))
                axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
                if i==0:axs[i,j].set_title(f'Physical time {[1,4,20][j]}')
                if j==0:axs[i,j].set_ylabel(labels[i])
        fig.suptitle('Predetermined test example 0; identical initial state and color scale')
        fig.subplots_adjust(left=.1,right=.89,top=.91,bottom=.025,wspace=.06,hspace=.12)
        coloraxis=fig.add_axes([.915,.23,.015,.5])
        fig.colorbar(im,cax=coloraxis,label='Vorticity')
        save(fig,'flow_snapshots',layout=False)

    discussion=json.loads((OUT/'DISCUSSION.json').read_text()) if (OUT/'DISCUSSION.json').exists() else {}
    lines=['# 2D forced-flow extension results','',f'Main outcomes: {len(all_df)}/12. Ratios below 1 favor the emulator. '
        f'Failed/nonfinite outcomes retained in tables: {sum(all_df.status!="completed")}. Curves show finite completed forecasts.',
        '', '## Questions and fixed method','',
        'Compare coarse-only one-step and four-step FNO training on the same H=8 trajectories. '
        'There are two independent 128-flow training datasets and three initializations each. '
        'All main models use width 16, modes 10, four blocks, and 12,000 Adam updates. '
        'Testing uses 64 flows, a 32-grid teacher, a 96-grid reference, and a 128-grid refinement. '
        'See [the protocol](../../TURBULENCE_EXPERIMENT_PLAN.md) for preflight amendments and limitations.',
        '', '## Main results','',pd.DataFrame(tables).to_markdown(index=False),'',
        'Seed ranges are descriptive; bootstrap intervals are conditional on a frozen model. '
        'The shared test set is not multiplied by the number of trained models.',
        '', '## Non-learning controls','',base[['name','R20','R100','R21_100','velocity_R20']].to_markdown(index=False),
        '', '## Paired training contrasts','',pd.DataFrame(contrasts).to_markdown(index=False)]
    for name,title in [('primary_seed_results','All trained models'),('accuracy_rollouts','Short and extended prediction'),
            ('physical_structure','Energy, enstrophy, correlation'),('energy_spectrum','Energy across scales'),
            ('teacher_imitation','Imitation and optimization'),('error_alignment','Squared-error decomposition'),('flow_snapshots','Predetermined example')]:
        if (FIG/f'{name}.png').exists():lines+=['',f'## {title}','',f'![{title}](figures/{name}.png)']
    if discussion:
        lines+=['','## Interpretation','']
        lines+=discussion.get('findings',[])
        lines+=['','## Report-ready paragraphs','']+discussion.get('report_paragraphs',[])
        lines+=['','## Limitations','']+['- '+x for x in discussion.get('limitations',[])]
    lines+=['','## Numerical reference validation','',json.dumps(json.loads((OUT/'data'/'reference_validation.json').read_text()),indent=2),
        '', 'Sources: Exponax 0.1.0 and PDEquinox 0.1.2 installed source; '
        'https://fkoehler.site/exponax/api/stepper/nonlinear/navier_stokes/ ; '
        'https://arxiv.org/abs/2510.23111 . All new numbers come from this project’s artifacts.']
    if (OUT/'long_reference_sensitivity.json').exists():
        lines+=['','## Longer reference check: fixed first-eight test subset','',
            'The primary reference check uses all 64 test flows; the following longer check uses only the predetermined first eight.',
            '',json.dumps(json.loads((OUT/'long_reference_sensitivity.json').read_text()),indent=2)]
    (OUT/'RESULTS.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps({'evaluated':len(df),'report':str(OUT/'RESULTS.md')}))


def records_as_dict(df):return df.to_dict('records')


if __name__=='__main__':main()
