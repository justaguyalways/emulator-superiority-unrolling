"""Tables and publication-ready figures; keep historical and new cohorts apart."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'extension_2026_11_22'
FIG=OUT/'figures'
GROUP={'extension_conv':'ConvNet amplitude control',
       'extension_fixedH5':'FNO shared five-step data',
       'extension_burgers':'UNet Burgers',
       'extension_calibration':'Runtime calibration'}
COLORS={'unit':'#2563eb','varied':'#d97706','N=1':'#2563eb',
        'N=2':'#d97706','N=5':'#059669','WIG N=2':'#9333ea'}


def collect():
    rows=[]
    for cohort,folder in [('historical',ROOT/'results'/'runs'),('new',OUT/'runs')]:
        for path in sorted(folder.glob('*/summary.json')):
            s=json.loads(path.read_text());c=s['config']
            metrics=pd.read_csv(path.parent/'metrics.csv')
            if cohort=='new' and (path.parent/'loss.npy').exists():
                losses=np.load(path.parent/'loss.npy')
                count=len(losses)//500
                pd.DataFrame({'update_step':np.arange(1,count+1)*500,
                    'mean_loss':losses[:count*500].reshape(count,500).mean(axis=1),
                    'mean_loss_per_step':losses[:count*500].reshape(count,500).mean(axis=1)/c['num_unrolled_steps']
                    }).to_csv(path.parent/'training_curve.csv',index=False)
            b=c['convection_difficulty']!=0
            row={'cohort':cohort,'name':s['name'],'group':GROUP.get(c['tag'],'historical'),
                'status':'completed' if np.isfinite(metrics.error_emulator).all() else 'rollout_nonfinite',
                'arch':c['network'].split(';')[0], 'pde':'Burgers' if b else 'advection',
                'solver':'Burgers P1' if b else ('explicit' if c['linear_difficulties'][1]==-.9 else 'implicit'),
                'mode':c['train_mode'],'N':c['num_unrolled_steps'],
                'H':c['num_unrolled_steps'] if c['temporal_horizon_train']<0 else c['temporal_horizon_train'],
                'amplitudes':'varied' if c['ic_train'].startswith('indamp') else 'unit',
                'seed':c['seed_init'],'training_updates':c['num_training_steps'],
                'test_samples':c['num_samples_test'],'test_horizon':c['temporal_horizon_test'],
                'loss_end':s['loss_end'],'normalized_loss':s['loss_end']/c['num_unrolled_steps'],
                'teacher_validation_loss':s.get('teacher_validation_loss'),
                'train_seconds':s['time_train_s'],'directory':str(path.parent),
                'primary_bootstrap_low':s.get('primary_test_bootstrap95',[None,None])[0],
                'primary_bootstrap_high':s.get('primary_test_bootstrap95',[None,None])[1],
                'reference_sensitivity':s.get('tighter_reference_primary_difference')}
            for h in [20,50,100,200]:
                if len(metrics)<h:continue
                a=metrics.error_emulator.to_numpy()[:h];b0=metrics.error_solver.to_numpy()[:h]
                with np.errstate(invalid='ignore',divide='ignore'):
                    integrated=float(a.sum()/b0.sum())
                row[f'R{h}']=integrated if np.isfinite(integrated) else math.inf
                ratio=metrics.superiority.to_numpy()[:h]
                row[f'mean_ratio_{h}']=float(ratio.mean()) if np.isfinite(ratio).all() else math.inf
                row[f'fraction_superior_{h}']=float(np.mean(ratio<1))
                row[f'emulator_error_mean_{h}']=float(np.mean(a))
                row[f'solver_error_mean_{h}']=float(np.mean(b0))
            for t in [1,2,3,5,10,20,50,100,200]:
                if len(metrics)>=t:row[f's{t}']=float(metrics.superiority.iloc[t-1])
            row['primary']=row['R20' if c['convection_difficulty'] else 'R100']
            rows.append(row)
        for path in sorted(folder.glob('*/failure.json')):
            if (path.parent/'summary.json').exists():continue
            s=json.loads(path.read_text());c=s['config'];nonlinear=c['convection_difficulty']!=0
            row={'cohort':cohort,'name':s['name'],'group':GROUP.get(c['tag'],'historical'),
                'status':'training_failed','failure_reason':s['reason'],
                'arch':c['network'].split(';')[0],'pde':'Burgers' if nonlinear else 'advection',
                'solver':'Burgers P1' if nonlinear else ('explicit' if c['linear_difficulties'][1]==-.9 else 'implicit'),
                'mode':c['train_mode'],'N':c['num_unrolled_steps'],
                'H':c['num_unrolled_steps'] if c['temporal_horizon_train']<0 else c['temporal_horizon_train'],
                'amplitudes':'varied' if c['ic_train'].startswith('indamp') else 'unit',
                'seed':c['seed_init'],'training_updates':s['completed_updates'],
                'test_samples':c['num_samples_test'],'test_horizon':c['temporal_horizon_test'],
                'loss_end':np.nan,'normalized_loss':np.nan,'teacher_validation_loss':np.nan,
                'train_seconds':s['training_seconds'],'directory':str(path.parent),'primary':math.inf,
                'reference_sensitivity':np.nan}
            for h in [20,50,100,200]:
                row[f'R{h}']=math.inf
                row[f'mean_ratio_{h}']=math.inf
                row[f'fraction_superior_{h}']=0
                row[f'emulator_error_mean_{h}']=math.inf
                row[f'solver_error_mean_{h}']=np.nan
            for t in [1,2,3,5,10,20,50,100,200]:row[f's{t}']=math.inf
            rows.append(row)
    df=pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT/'per_run_results.csv',index=False)
    return df


def fmt(x):
    if x is None or pd.isna(x):return '—'
    if not np.isfinite(x):return 'unstable'
    if x!=0 and abs(x)<.001:return f'{x:.2e}'
    return f'{x:.3f}' if abs(x)<100 else f'{x:.2g}'


def interval(values):
    a=np.asarray(values,dtype=float)
    unavailable=int(np.isnan(a).sum())
    a=a[~np.isnan(a)]
    if not len(a):return 'unavailable'
    a=np.where(np.isfinite(a),a,np.inf)
    text=f'{fmt(np.median(a))} [{fmt(np.min(a))}, {fmt(np.max(a))}]'
    return text+(f'; {unavailable} unavailable' if unavailable else '')


def table(df,keys,metrics):
    records=[]
    for k,g in df.groupby(keys,dropna=False,sort=True):
        if not isinstance(k,tuple):k=(k,)
        record=dict(zip(keys,k));record['n']=len(g)
        for metric in metrics:record[metric]=interval(g[metric])
        record['seeds with primary < 1']=f'{int((g.primary<1).sum())}/{len(g)}'
        records.append(record)
    return pd.DataFrame(records)


def plot_summary(df,group,filename):
    d=df[(df.cohort=='new')&(df.group==group)]
    if d.empty:return
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.4),sharey=True,layout='constrained')
    for ax,solver in zip(axs,['explicit','implicit']):
        a=d[d.solver==solver]
        categories=['unit','varied'] if group.startswith('Conv') else ['unit']
        for amp in categories:
            aa=a[a.amplitudes==amp]
            for n in [1,2,5]:
                v=aa.loc[aa.N==n,'primary'].to_numpy()
                if not len(v):continue
                finite=v[np.isfinite(v)]
                if not len(finite):continue
                x={1:0,2:1,5:2}[n]+({'unit':-.06,'varied':.06}[amp] if len(categories)>1 else 0)
                mid=np.median(v)
                if np.isfinite(mid):
                    ax.plot([x,x],[finite.min(),finite.max()],color=COLORS[amp],lw=1.5)
                    ax.plot(x,mid,'o',color=COLORS[amp],ms=6,
                            label=f'{amp} heights' if n==min(aa.N) else None)
                ax.scatter(np.full(len(finite),x),finite,s=12,color=COLORS[amp],alpha=.45)
                ax.annotate(f'n={len(v)}',(x,finite.max()),xytext=(0,6),
                            textcoords='offset points',ha='center',fontsize=7)
                if len(finite)<len(v):
                    ax.text(.02,.94,f'{len(v)-len(finite)} unstable in {amp}, N={n}',
                            transform=ax.transAxes,fontsize=7,va='top')
        ax.axhline(1,color='#475569',ls='--',lw=1,label='Equal error')
        ax.set_xticks([0,1,2],['1','2','5']);ax.set_xlabel('Steps used in the training loss (N)')
        ax.set_xlim(-.3,2.3)
        ax.set_title(f'{solver.capitalize()} teacher');ax.set_yscale('log')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Total-error ratio over steps 1–100\nBelow 1: emulator better')
    axs[0].legend(fontsize=8,loc='best')
    fig.suptitle(group,fontsize=12)
    for ext in ['png','pdf']:fig.savefig(FIG/f'{filename}.{ext}',dpi=220)
    plt.close(fig)


def plot_burgers(df):
    d=df[(df.cohort=='new')&(df.group=='UNet Burgers')]
    if d.empty:return
    fig,axs=plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
    for (mode,n),g in d.groupby(['mode','N']):
        label=f'N={n}' if mode=='sup' else f'WIG N={n}'
        color=COLORS[label]
        failed=sum(g.status=='training_failed')
        g=g[g.status!='training_failed']
        if g.empty:continue
        curves=[pd.read_csv(Path(p)/'metrics.csv') for p in g.directory]
        s=np.stack([x.superiority.to_numpy() for x in curves])
        e=np.stack([x.error_emulator.to_numpy() for x in curves])
        ts=np.arange(1,s.shape[1]+1)
        for ax,h in [(axs[0,0],20),(axs[0,1],100)]:
            vals=s[:,:h]
            ax.plot(ts[:h],np.median(vals,axis=0),color=color,
                    label=f'{label} ({len(g)} completed, {failed} failed)')
            ax.fill_between(ts[:h],np.min(vals,axis=0),np.max(vals,axis=0),color=color,alpha=.12)
        axs[1,0].plot(ts[:20],np.median(e[:,:20],axis=0),color=color,label=label)
        loss_curves=[pd.read_csv(Path(p)/'training_curve.csv') for p in g.directory]
        length=min(map(len,loss_curves));xs=loss_curves[0].update_step.to_numpy()[:length]
        blocks=np.stack([a.mean_loss_per_step.to_numpy()[:length] for a in loss_curves])
        axs[1,1].plot(xs,np.median(blocks,axis=0),color=color,label=label)
    baseline=curves[0].error_solver.to_numpy()
    axs[1,0].plot(np.arange(1,21),baseline[:20],color='#0f172a',ls='--',label='Teacher')
    for ax in axs[0]:
        ax.axhline(1,color='#475569',ls='--',lw=1);ax.set_ylabel('Superiority ratio')
    axs[0,0].set_title('Main test window: steps 1–20')
    axs[0,1].set_title('Separate stability test: steps 1–100');axs[0,1].set_yscale('log')
    axs[1,0].set_ylabel('Mean normalized RMSE');axs[1,0].set_title('Absolute prediction errors')
    axs[1,1].set_title('Training fit (loss divided by N)');axs[1,1].set_yscale('log')
    axs[1,1].set_xlabel('Optimizer updates');axs[1,1].set_ylabel('Mean training MSE per step')
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xlabel('Prediction step')
    for ax in axs.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18)
    axs[0,0].legend(fontsize=7);axs[1,0].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(FIG/f'burgers_unet.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.25),layout='constrained')
    for (mode,n),g in d.groupby(['mode','N']):
        g=g[g.status!='training_failed']
        if g.empty:continue
        label=f'N={n}' if mode=='sup' else f'WIG N={n}'
        curves=np.stack([pd.read_csv(Path(p)/'metrics.csv').superiority.to_numpy() for p in g.directory])
        for ax,h in zip(axs,[20,100]):
            x=np.arange(1,h+1);v=curves[:,:h]
            ax.plot(x,np.median(v,axis=0),color=COLORS[label],label=f'{label} (n={len(g)})')
            ax.fill_between(x,np.min(v,axis=0),np.max(v,axis=0),color=COLORS[label],alpha=.12)
    for ax,h in zip(axs,[20,100]):
        ax.axhline(1,color='#475569',ls='--',lw=1)
        ax.set_title('Main window: 1–20' if h==20 else 'Longer stability test: 1–100')
        ax.set_xlabel('Prediction step');ax.set_ylabel('Superiority ratio')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    axs[1].set_yscale('log');axs[0].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(FIG/f'burgers_accuracy_stability.{ext}',dpi=220)
    plt.close(fig)


def paired_contrasts(df):
    rows=[]
    d=df[(df.cohort=='new')&(df.group=='ConvNet amplitude control')]
    for (solver,n,seed),g in d.groupby(['solver','N','seed']):
        if set(g.amplitudes)!=set(['unit','varied']):continue
        unit=g[g.amplitudes=='unit'].iloc[0];var=g[g.amplitudes=='varied'].iloc[0]
        rows.append({'comparison':'Conv varied minus unit','solver':solver,'N':n,'seed':seed,
            'base':unit.primary,'treatment':var.primary,'difference':var.primary-unit.primary})
    d=df[(df.cohort=='new')&(df.group=='FNO shared five-step data')]
    for (solver,seed),g in d.groupby(['solver','seed']):
        if not any(g.N==1):continue
        base=g[g.N==1].iloc[0]
        for n in [2,5]:
            if not any(g.N==n):continue
            item=g[g.N==n].iloc[0]
            rows.append({'comparison':'FNO shared H: unrolled minus one-step','solver':solver,'N':n,'seed':seed,
                'base':base.primary,'treatment':item.primary,'difference':item.primary-base.primary})
    pd.DataFrame(rows).to_csv(OUT/'paired_seed_contrasts.csv',index=False)
    return rows


def plot_homogeneity():
    path=OUT/'homogeneity_results.csv'
    if not path.exists():return
    data=pd.read_csv(path)
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.5),layout='constrained',sharey=True)
    all_values=[]
    for ax,arch,tag,title in [(axs[0],'Conv','extension_conv','ConvNet: one-step training'),
                             (axs[1],'FNO','extension_fixedH5','FNO: one-step loss, shared H=5 data')]:
        d=data[(data.arch==arch)&(data.group==tag)&(data.solver=='explicit')&
               (data.N==1)&(data.amplitudes=='ind;1;True')]
        if len(d):
            all_values.extend(d.original_R100.tolist()+d.homogeneous_R100.tolist())
            for _,r in d.iterrows():
                ax.plot([0,1],[r.original_R100,r.homogeneous_R100],color='#94a3b8',alpha=.6,lw=1)
            for x,col,color in [(0,'original_R100','#2563eb'),(1,'homogeneous_R100','#d97706')]:
                a=d[col].to_numpy()
                ax.plot([x,x],[a.min(),a.max()],color=color,lw=3)
                ax.scatter([x],[np.median(a)],s=55,color=color,zorder=3)
                ax.annotate(f'{np.median(a):.3f}',(x,np.median(a)),xytext=(0,9),
                            textcoords='offset points',ha='center',fontsize=9)
            ax.text(.03,.04,f'{len(d)} initialization seeds',transform=ax.transAxes,fontsize=8)
        ax.axhline(1,color='#475569',ls='--',lw=1)
        ax.set_xticks([0,1],['Original model','Height-scaling rule'])
        ax.set_xlim(-.3,1.3)
        ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Total-error ratio, steps 1–100\nBelow 1: emulator better')
    for ax in axs:ax.set_ylim(0,max([1.]+[x for x in all_values if np.isfinite(x)])*1.17)
    fig.suptitle('Same learned weights and first prediction; different later behavior',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(FIG/f'homogeneity_intervention.{ext}',dpi=220)
    plt.close(fig)


def amplitude_stress(df):
    records=[]
    for _,row in df[df.cohort=='new'].iterrows():
        path=Path(row.directory)/'test_amplitude.csv'
        if not path.exists():continue
        for a in pd.read_csv(path).to_dict('records'):
            records.append({'name':row['name'],'group':row['group'],'arch':row.arch,
                'solver':row.solver,'N':row.N,'H':row.H,'training_heights':row.amplitudes,
                'seed':row.seed,**a})
    if not records:return
    d=pd.DataFrame(records);d.to_csv(OUT/'amplitude_stress_results.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.2),layout='constrained',sharey=True)
    for ax,n in zip(axs,[1,5]):
        a=d[(d.arch=='Conv')&(d.solver=='explicit')&(d.N==n)]
        for heights,g in a.groupby('training_heights'):
            agg=g.groupby('amplitude').integrated_ratio_1_100.agg(['median','min','max','size'])
            ax.plot(agg.index,agg['median'],'o-',color=COLORS[heights],label=f'{heights} training heights')
            ax.fill_between(agg.index,agg['min'],agg['max'],color=COLORS[heights],alpha=.12)
        ax.axhline(1,color='#475569',ls='--',lw=1)
        ax.axvline(1,color='#94a3b8',ls=':',lw=1)
        ax.set_title(f'ConvNet, N={n}');ax.set_xlabel('Test wave height')
        ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Total-error ratio, steps 1–100');axs[0].legend(fontsize=7)
    fig.suptitle('A small change in test height: explicit advection teacher',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(FIG/f'amplitude_stress.{ext}',dpi=220)
    plt.close(fig)


def markdown(df):
    new=df[df.cohort=='new']
    discussion_path=OUT/'DISCUSSION.json'
    discussion=json.loads(discussion_path.read_text()) if discussion_path.exists() else {}
    lines=['# Extension results: amplitude controls and nonlinear Burgers',
        '', 'Generated from completed run artifacts. Ratios below 1 favor the emulator.', '',
        '## Completion and provenance','',
        f'- Newly trained, evaluated configurations: **{len(new)}**, including one runtime calibration.',
        f'- Historical configurations retained separately: **{int((df.cohort=="historical").sum())}**.',
        '- The historical cohort is not pooled with the new cohort. Same nominal seed/configuration '
        'does not guarantee the same trained network across dependency versions.',
        '- Main comparisons use common data and initialization seeds within the new environment. '
        'Three initialization seeds measure initialization variability, not dataset uncertainty.',
        '- Full method and hypotheses: `../../EXPERIMENT_PLAN.md`. Raw per-run data: '
        '`per_run_results.csv`; paired contrasts: `paired_seed_contrasts.csv`.', '',
        '| Study | Completed | Planned |','|---|---:|---:|']
    for group,planned in [('ConvNet amplitude control',36),('FNO shared five-step data',18),('UNet Burgers',12)]:
        lines.append(f'| {group} | {sum(new.group==group)} | {planned} |')
    if discussion.get('findings'):
        lines+=['','## Main findings','']+['- '+x for x in discussion['findings']]
    lines += ['', '## Experiment 1A: Does amplitude diversity change the ConvNet advantage?', '',
        '**Question.** Does the report’s amplitude explanation also hold for a local convolutional '
        'network when it is trained over multiple prediction steps?', '',
        '**Design.** Both advection teachers; fixed height 1 versus uniform heights 0.5–1.5; '
        'N=1,2,5; three initialization seeds; 10,000 Adam updates and batch 32. '
        'H=N follows the midway protocol. The test set is the same 10 unit-height waves. '
        'There are 100 periodic grid points; explicit upwind uses CFL 0.9 and implicit '
        'upwind uses CFL 3.0, each evaluated against its exact advection solution. '
        'The different CFL values also mean different elapsed physical times per step; '
        'this is not an isolated causal comparison of explicit versus implicit integration.', '',
        '**Primary score.** Total mean RMSE over prediction steps 1–100, divided by the '
        'teacher’s corresponding total error. Reported as median [minimum, maximum] across seeds.', '']
    d=new[new.group=='ConvNet amplitude control']
    if len(d):lines.append(table(d,['solver','amplitudes','N'],['R100','s100']).to_markdown(index=False))
    else:lines.append('No completed cells yet.')
    lines+=['','![ConvNet amplitude comparison](figures/conv_amplitudes.png)', '',
        discussion.get('conv_conclusion',''), '',
        '**Interpretation boundary.** A median improvement is not a seed-robust improvement if the '
        'range crosses 1. A weaker effect after varied-height training is evidence against a '
        'general superiority claim, but does not prove no individual trained network can improve.', '',
        '## Experiment 1B: Is the effect due to unrolling or to extra training states?', '',
        '**Question.** In the original H=N study, a longer loss also supplied longer teacher '
        'trajectories. Does the unroll-length effect remain when the underlying trajectories are fixed?', '',
        '**Design.** FNO, unit-height waves, H=5 for every recipe, N=1,2,5, both teachers, '
        'three initialization seeds, and identical update count/schedule. N=1 and N=2 '
        'sample all sliding windows of the same trajectories. The number of windows and '
        'the weights assigned to time levels still differ; this is not an identical-window experiment.', '']
    d=new[new.group=='FNO shared five-step data']
    if len(d):lines.append(table(d,['solver','N'],['R100','s100']).to_markdown(index=False))
    else:lines.append('No completed cells yet.')
    lines+=['','![Fixed-data-horizon comparison](figures/fno_fixed_horizon.png)', '',
        discussion.get('fixedH_conclusion',''), '',
        '## Experiment 2: Does UNet superiority survive nonlinear evolution?', '',
        '**Question.** Does the emulator repeatedly outperform its teacher on Burgers, '
        'and does multi-step or solver-in-loop training change short-term accuracy and longer stability?', '',
        '**Design.** The paper’s 27,193-parameter UNet, 60 points, one-Picard-step teacher, '
        'converged-Picard reference, 500 training examples, 100,000 Adam updates, batch 32, '
        'peak learning rate 0.0003, 5,000 warmup updates. Supervised N=1,2,5 and WIG N=2 '
        'each use seeds 0/1/2. Training/test/shuffle data seeds are 10/20/30. '
        'Diffusion/convection difficulties are 0.2/-8 in APEBench units. The initial '
        'profiles have the first Fourier mode and a mean offset, normalized to maximum '
        'absolute height 1; broad initial-condition or parameter generalization is not tested. '
        'No model or hyperparameter is selected on test superiority.', '',
        '**Primary score.** Total mean nRMSE over steps 1–20, divided by the teacher’s total. '
        'The test has 100 fixed held-out examples; steps 21–100 are a distinct stability extension. '
        'The paper text specifies 100 test examples/20 steps; its released script uses 20/10. '
        'Our choice follows the text and explicitly documents the difference.', '']
    d=new[new.group=='UNet Burgers']
    if len(d):
        lines.append(table(d,['mode','N'],['R20','s3','s5','s10','s20','R100','s100']).to_markdown(index=False))
        lines+=['','Absolute errors (mean over steps 1–20; median [min,max] across seeds):','',
                table(d,['mode','N'],['emulator_error_mean_20','solver_error_mean_20','teacher_validation_loss']).to_markdown(index=False)]
        lines+=['','`teacher_validation_loss` above evaluates each recipe’s own objective '
            'on independent teacher-generated validation trajectories. Different recipes use '
            'different losses, so these values are not a common cross-recipe fit metric. '
            'The frozen-model diagnostic below supplies identical one-step states and targets.']
        sens=d.reference_sensitivity.abs().max()
        lines += ['',f'Tightening reference tolerance from 1e-5 to 1e-7 and the iteration cap '
                  f'from 100 to 200 changed the primary ratio by at most **{sens:.6g}** among completed models.']
    else:lines.append('No completed cells yet.')
    lines+=['','![UNet Burgers results](figures/burgers_unet.png)', '',
        discussion.get('burgers_conclusion',''), '',
        '**Interpretation boundary.** A favorable step 3 or 5 is a transient advantage; it '
        'does not establish a lower total error over steps 1–20. Worse teacher-imitation '
        'fit is an alternative explanation for a worse unrolled result. H=N also changes '
        'the training states, so this Burgers comparison alone does not isolate the loss. '
        'The converged reference solves the same discretized system; it is not an exact continuum PDE solution.', '',
        '## Additional mechanism check: remove height-dependent feedback without retraining', '',
        '**Question.** Does superiority survive if the same trained model is forced to scale '
        'its prediction in proportion to the wave height, as the linear equation does?', '',
        '**Intervention.** On linear advection only, use f_hom(u)=a f(u/a), where a is '
        'the maximum absolute input height. The weights and first prediction on the unit-height '
        'test inputs stay fixed. This is a post-training diagnostic; it is not a new learned model '
        'and is not applied to nonlinear Burgers.', '']
    hp=OUT/'homogeneity_results.csv'
    if hp.exists():
        h=pd.read_csv(hp)
        h=h[h.group!='extension_calibration']
        records=[]
        for (arch,solver,n,horizon,amp),g in h.groupby(['arch','solver','N','H','amplitudes']):
            records.append({'arch':arch,'solver':solver,'N':n,'H':horizon,
                'heights':'varied' if amp.startswith('indamp') else 'unit','n':len(g),
                'original R100':interval(g.original_R100),
                'height-scaled R100':interval(g.homogeneous_R100),
                'max first-step difference':f'{g.max_first_step_error_difference.max():.2g}'})
        lines.append(pd.DataFrame(records).to_markdown(index=False))
        lines+=['','![Same-weights scaling intervention](figures/homogeneity_intervention.png)']
    else:lines.append('No completed intervention checks yet.')
    if discussion.get('homogeneity_conclusion'):
        lines+=['',discussion['homogeneity_conclusion']]
    cp=OUT/'burgers_teacher_coverage.csv'
    if cp.exists():
        lines+=['','## What extra teacher states does Burgers unrolling expose?', '',
            'This analysis uses the same 500 teacher-generated training trajectories, without '
            'using neural test outcomes. Higher-mode energy measures how much the shape '
            'departs from its initial smooth single-wave form. The local gap applies both '
            'teacher and converged reference to the very same state.', '',
            pd.read_csv(cp).to_markdown(index=False), '',
            '![Training-state coverage](figures/burgers_teacher_coverage.png)']
    stress=OUT/'amplitude_stress_results.csv'
    if stress.exists():
        a=pd.read_csv(stress);a=a[(a.arch=='Conv')&(a.solver=='explicit')&a.N.isin([1,5])]
        records=[]
        for (n,heights,amp),g in a.groupby(['N','training_heights','amplitude']):
            records.append({'N':n,'training_heights':heights,'test_height':amp,'n':len(g),
                'R100':interval(g.integrated_ratio_1_100)})
        lines+=['','## Secondary test: ±5% changes in test wave height','',
            'These test heights were fixed in the runner before training. The same trained '
            'models are compared with correctly rescaled teacher and reference trajectories. '
            'This extends the report’s amplitude stress test to the fresh multi-step ConvNet study.', '',
            pd.DataFrame(records).to_markdown(index=False),'',
            '![Test-height robustness](figures/amplitude_stress.png)']
    secondary=OUT/'burgers_secondary_metrics.csv'
    if secondary.exists():
        lines+=['','## Secondary Burgers metric and per-example checks','',
            'Absolute RMSE changes how examples are weighted relative to normalized RMSE. '
            'These are secondary diagnostics and do not replace the predefined primary score. '
            'The per-example win fraction is computed within each trained model; the 100 '
            'test examples are shared across seeds and must not be counted as 300 independent examples.', '',
            pd.read_csv(secondary).to_markdown(index=False)]
    common=OUT/'common_teacher_fit.csv'
    if common.exists():
        f=pd.read_csv(common)
        f=f[f.pde=='Burgers'];records=[]
        for (mode,n),g in f.groupby(['mode','N']):
            records.append({'mode':mode,'N':n,'n':len(g),
                'same_initial_states_MSE':interval(g.common_initial_MSE),
                'same_five_time_levels_MSE':interval(g.common_H5_MSE)})
        # Keep scientific-notation precision for small one-step MSEs.
        for record,(_,g) in zip(records,f.groupby(['mode','N'])):
            for dst,col in [('same_initial_states_MSE','common_initial_MSE'),
                            ('same_five_time_levels_MSE','common_H5_MSE')]:
                a=g[col].to_numpy()
                record[dst]=f'{np.median(a):.3g} [{a.min():.3g}, {a.max():.3g}]'
        if len(records):
            lines+=['','## Common teacher-imitation validation: identical targets across recipes','',
                'Every frozen model is evaluated on the identical 100 held-out Burgers '
                'profiles (seed 110) and their teacher states at times 0–4. This uses '
                'one-step MSE throughout, without the high-fidelity test reference or '
                'any checkpoint selection. Small errors here measure imitation, not superiority.', '',
                pd.DataFrame(records).to_markdown(index=False)]
    if discussion.get('report_paragraphs'):
        lines+=['','## Report-ready paragraphs','']
        for paragraph in discussion['report_paragraphs']:lines.extend([paragraph,''])
    lines += ['',
        '## What is genuinely new relative to the midway report?', '',
        '- Fresh ConvNet comparisons over multiple training steps with both fixed and diverse heights.',
        '- A common-trajectory-horizon FNO comparison that separates part of the data-coverage confound.',
        '- A three-initialization UNet Burgers study, with a larger test set than the project’s preliminary FNO runs.',
        '- Whole-window scores, absolute errors, conditional paired bootstrap intervals, independent '
        'teacher-imitation validation losses, longer-rollout stability, and tighter-reference checks.',
        '- A documented runtime sensitivity check. The original report’s amplitude finding itself '
        'is existing work and should not be described as a new discovery here.', '',
        '## Sources and reproducibility','',
        '- Midway report: [midway_report_story.pdf](../../reports/midway_report_story.pdf).',
        '- Paper: https://arxiv.org/html/2510.23111v2 (Appendix B, architectures and Burgers protocol).',
        '- Authors’ code: https://github.com/tum-pbs/emulator-superiority ; pinned commit '
        '`461eb07d2714c958d43d4821abc78f2d31efff97`.',
        '- `validation.json` records forward/gradient checks, exact resumed-Adam agreement, '
        'P1 derivative finite differences, and reference sensitivity.',
        '- `historical_manifest.json` records the initial 137-run cohort. Each new summary '
        'records source hashes, dependencies, parameters, samples, and actual completed updates.',
        '- Resume/reproduce: `.venv/bin/python src/run_extension_pool.py --workers 6`; '
        'regenerate analysis: `.venv/bin/python src/analyze_extension.py`.',
        '- Failed training runs and nonfinite rollouts must be reported, not filtered out. '
        'No significance or universal physical-accuracy claim follows from three initialization seeds.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'legend.frameon':False,'axes.edgecolor':'#94a3b8'})
    df=collect()
    if df.empty:return
    paired_contrasts(df)
    new=df[df.cohort=='new']
    aggregate=table(new,['group','solver','mode','amplitudes','N'],['primary'])
    aggregate.to_csv(OUT/'aggregate_results.csv',index=False)
    plot_summary(df,'ConvNet amplitude control','conv_amplitudes')
    plot_summary(df,'FNO shared five-step data','fno_fixed_horizon')
    plot_burgers(df)
    plot_homogeneity()
    amplitude_stress(df)
    markdown(df)
    print(json.dumps({'new_runs':len(new),'historical_runs':int((df.cohort=='historical').sum()),
        'report':str(OUT/'RESULTS.md')}))


if __name__=='__main__':main()
