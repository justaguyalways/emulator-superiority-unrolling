"""Create a self-contained, readable PDF after all prescribed cells finish."""
import argparse
from html import escape
import json
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, PageBreak, KeepTogether)

from analyze_extension import OUT,FIG,interval,table


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/tmp/emusup_results/Experiment_Results.pdf')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    df=pd.read_csv(OUT/'per_run_results.csv');new=df[df.cohort=='new']
    expected={'ConvNet amplitude control':36,'FNO shared five-step data':18,'UNet Burgers':12}
    incomplete={k:(sum(new.group==k),n) for k,n in expected.items() if sum(new.group==k)!=n}
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f'Incomplete study, not publishing final report: {incomplete}')
    discussion_path=OUT/'DISCUSSION.json'
    discussion=json.loads(discussion_path.read_text()) if discussion_path.exists() else {}
    if not discussion and not args.allow_incomplete:raise RuntimeError('Write and review the final discussion first')
    fontdir=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    pdfmetrics.registerFont(TTFont('DejaVu',str(fontdir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(fontdir/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
    styles={
        'title':ParagraphStyle('title',fontName='DejaVu-Bold',fontSize=23,leading=29,textColor=colors.HexColor('#0f172a'),spaceAfter=16),
        'h1':ParagraphStyle('h1',fontName='DejaVu-Bold',fontSize=16,leading=21,textColor=colors.HexColor('#0f172a'),spaceAfter=12,keepWithNext=True),
        'body':ParagraphStyle('body',fontName='DejaVu',fontSize=9.3,leading=14,spaceAfter=9),
        'small':ParagraphStyle('small',fontName='DejaVu',fontSize=7.7,leading=11,spaceAfter=7,textColor=colors.HexColor('#475569')),
        'cell':ParagraphStyle('cell',fontName='DejaVu',fontSize=7,leading=10),
        'th':ParagraphStyle('th',fontName='DejaVu-Bold',fontSize=7,leading=10,textColor=colors.white),
    }
    story=[];width=A4[0]-84
    def para(text,style='body'):
        safe=escape(text.replace('–','-').replace('−','-').replace('—','-'))
        safe=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',safe)
        story.append(Paragraph(safe,styles[style]))
    def image(name,maxheight=265):
        path=FIG/name
        if not path.exists():return
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        scale=min(width/w,maxheight/h)
        story.append(Image(str(path),width=w*scale,height=h*scale))
        story.append(Spacer(1,9))
    def tab(headers,records,weights=None):
        if not len(records):para('Results are not yet complete.','small');return
        vals=[[Paragraph(escape(str(x)),styles['th']) for x in headers]]
        labels={'sup':'Data only','div_wig':'Solver guided','unit':'Fixed 1',
                'varied':'0.5-1.5','explicit':'Explicit','implicit':'Implicit'}
        vals += [[Paragraph(escape(labels.get(str(x),str(x)).replace('–','-')),styles['cell'])
                  for x in row] for row in records]
        weights=weights or [1]*len(headers)
        t=Table(vals,colWidths=[width*x/sum(weights) for x in weights],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e3a5f')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
            ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#cbd5e1'))]))
        story.append(t);story.append(Spacer(1,11))
    def section(title):
        story.append(PageBreak());para(title,'h1')
    def grouped_rows(group,keys,metrics):
        d=new[new.group==group]
        out=[]
        for key,g in d.groupby(keys,sort=True):
            if not isinstance(key,tuple):key=(key,)
            out.append([*key,*[interval(g[m]) for m in metrics],f'{sum(g.primary<1)}/{len(g)}'])
        return out

    para('When does an emulator\noutperform its teacher?','title')
    para('Extension of the 10-749 midway report | Experiments 1 and 2 | 22 November 2026','small')
    para('An emulator is a neural network trained to imitate a numerical solver. We asked '
         'whether an apparent accuracy advantage survives a broader training distribution, '
         'a controlled change in the training loss, and nonlinear wave evolution.')
    para('A ratio below 1 means the emulator is more accurate than its training solver. '
         'The main score adds prediction errors over an entire specified window; '
         'individual favorable time steps are reported separately.')
    tab(['Study','Question','New configurations'],[
        ['ConvNet waves','Fixed versus varied wave heights; N=1,2,5','36'],
        ['FNO waves','Same five-step training trajectories; N=1,2,5','18'],
        ['UNet Burgers','Nonlinear evolution and solver-in-loop training','12']], [1.1,2.4,.8])
    para(f'All {len(new)} new outcomes include one runtime calibration. '
         'The 137 historical runs are retained as a separate cohort. All comparisons '
         'in the main tables use fresh runs in the same numerical environment.','small')
    for text in discussion.get('findings',['Study in progress; conclusions will be reviewed after all runs finish.']):
        para('• '+text)
    para('Numbers in brackets are the minimum and maximum across initialization seeds, '
         'not a confidence interval. The datasets are shared across those seeds.','small')

    section('1A. Does wave-height diversity change superiority?')
    para('We trained the same ConvNet with waves of height exactly 1 or heights sampled '
         'uniformly between 0.5 and 1.5. Both solvers, all three initialization seeds, '
         'and N=1,2,5 used the same 10,000-update Adam recipe and batch size 32. '
         'The test uses 10 fixed unit-height waves. Training horizon H=N follows the midway setup.')
    para('Advection uses 100 periodic grid points: explicit upwind at CFL 0.9 or implicit '
         'upwind at CFL 3.0, each compared with its exact advection reference. Each training '
         'set has 300 profiles. The solver settings also have different time steps, '
         'so comparisons between solvers are descriptive rather than an isolated method ablation.','small')
    records=grouped_rows('ConvNet amplitude control',['solver','amplitudes','N'],['R100','s100'])
    tab(['Teacher','Heights','N','Total-error ratio, 1–100','Ratio at step 100','Seeds better'],
        records,[.8,.7,.35,1.7,1.7,.65])
    image('conv_amplitudes.png',230)
    para(discussion.get('conv_conclusion','Results not yet complete.'))

    section('1B. Keep the training trajectories fixed')
    para('Longer training losses previously also supplied longer teacher trajectories. '
         'Here every FNO recipe receives the same five-step trajectories (H=5), '
         'with the same training/data seeds, batch size, and update count. '
         'Only the loss unroll length N changes. N=1 samples single-step windows; '
         'N=5 uses all five targets per trajectory.')
    tab(['Teacher','N','Total-error ratio, 1–100','Ratio at step 100','Seeds better'],
        grouped_rows('FNO shared five-step data',['solver','N'],['R100','s100']),[.8,.3,1.7,1.7,.65])
    image('fno_fixed_horizon.png',235)
    para(discussion.get('fixedH_conclusion','Results not yet complete.'))
    para('The number of sliding windows differs. N=1 and N=5 assign equal expected weight '
         'to the five target time levels; N=2 gives the middle levels more weight. '
         'The compute cost also grows with N. This controls the trajectory dataset, '
         'not every source of optimization or sampling variation.','small')

    section('2. Nonlinear Burgers with the paper’s UNet')
    para('We used the 27,193-parameter UNet, 60 spatial points, 500 training examples, '
         '100,000 Adam updates, batch size 32, peak rate 0.0003, and 5,000 warmup updates. '
         'The cheap teacher makes one nonlinear correction pass (a Picard iteration); '
         'the reference repeats those corrections until the answer settles. '
         'Testing uses 100 held-out examples: steps 1–20 are primary, 21–100 test longer stability.')
    para('APEBench diffusion/convection difficulties are 0.2/−8. Initial profiles contain '
         'the first Fourier mode and a random mean offset, normalized to maximum absolute height 1.','small')
    para('We retain this project’s training/test/shuffle data seeds 10/20/30 rather than '
         'the authors’ released 100/10000/1000000. This is an extension on the project’s '
         'data split, not a numerical replication of the published table.','small')
    tab(['Recipe','N','Total-error ratio, 1–20','Total-error ratio, 1–100','Seeds better (1–20)'],
        grouped_rows('UNet Burgers',['mode','N'],['R20','R100']),[.7,.3,1.7,1.7,.8])
    image('burgers_accuracy_stability.png',220)
    para(discussion.get('burgers_conclusion','Results not yet complete.'))
    para('WIG means gradients pass through the teacher during solver-in-loop training. '
         'The reference is a converged solve of the same discretized equations, '
         'not an exact continuum solution. H=N also changes which teacher states are available.','small')

    section('Mechanism: change scaling, keep the learned weights')
    para('For a linear equation, doubling the input height must double the next state. '
         'We enforce this rule after training: divide the wave by its maximum height, '
         'apply the same network, then multiply its answer by that height. '
         'The weights and the first prediction on unit-height test inputs are unchanged.')
    image('homogeneity_intervention.png',250)
    para(discussion.get('homogeneity_conclusion','Results not yet complete.'))
    para('Why this check works','h1')
    para('Suppose a model exactly copies a linear teacher L on every unit-height single wave v. '
         'The scaled version must then obey f_scaled(a v)=a f(v)=a L(v)=L(a v). '
         'Because the teacher keeps the wave within this same shape family, '
         'the scaled model follows the teacher at every subsequent step. Its superiority ratio is 1.')
    para('Real networks only approximately fit finitely many phases, so equality need not be exact. '
         'The experiment measures that deviation. This argument is restricted to '
         'single-mode linear advection; the scaling rule is not valid for Burgers.','small')

    section('Secondary checks: input robustness and error metric')
    para('A model can excel at the original test height and fail when it changes slightly. '
         'We reevaluated each trained wave model at heights 0.95, 1.00, and 1.05, '
         'with teacher and reference trajectories rescaled consistently. These heights '
         'were fixed before the training sweeps.')
    image('amplitude_stress.png',220)
    stress=OUT/'amplitude_stress_results.csv'
    if stress.exists():
        a=pd.read_csv(stress);a=a[(a.arch=='Conv')&(a.solver=='explicit')&a.N.isin([1,5])]
        rows=[]
        for (n,heights),g in a.groupby(['N','training_heights']):
            rows.append([n,heights,*[interval(g[g.amplitude==v].integrated_ratio_1_100)
                                    for v in [.95,1.,1.05]]])
        tab(['N','Training heights','Test height 0.95','Test height 1.00','Test height 1.05'],
            rows,[.25,.8,1.55,1.55,1.55])
    secondary=OUT/'burgers_secondary_metrics.csv'
    if secondary.exists():
        s=pd.read_csv(secondary)
        para('For Burgers, absolute RMSE and normalized RMSE weight examples differently. '
             'The following secondary checks retain the original primary score and show '
             'whether its conclusion depends on that weighting.')
        rows=[]
        for (mode,n),g in s.groupby(['mode','N']):
            rows.append([mode,n,interval(g.nRMSE_R20),interval(g.absolute_RMSE_R20),
                interval(g.test_examples_with_lower_total_nRMSE_fraction)])
        tab(['Recipe','N','Normalized-error ratio','Absolute-error ratio','Fraction of test examples better'],
            rows,[.6,.25,1.5,1.5,1.5])
        para('The per-example fraction is calculated separately for each trained model. '
             'All seeds share the same 100 test examples; they are not 300 independent examples. '
             'Bootstrap intervals in the raw results resample entire test trajectories, '
             'preserving temporal dependence, and are conditional on the trained model.','small')

    section('How do the training states change?')
    para('Extending Burgers trajectories changes the examples presented to the network. '
         'We measured shape complexity and the teacher’s local error on all 500 training examples, '
         'independently of neural test results.')
    image('burgers_teacher_coverage.png',245)
    coverage=pd.read_csv(OUT/'burgers_teacher_coverage.csv')
    tab(['Input time','Higher-mode energy fraction','Teacher local nRMSE','Median Picard iterations'],
        [[int(r.teacher_time),f'{r.mean_higher_mode_energy_fraction:.3f}',
          f'{r.mean_one_step_teacher_reference_nRMSE:.3f}',int(r.median_picard_iterations_to_reference_tolerance)]
         for _,r in coverage.iterrows()],[.65,1.5,1.2,1.3])
    para('Higher-mode energy excludes the constant mean mode. Local error applies the '
         'cheap and converged solvers to the identical current state; it is not a comparison '
         'between already diverged trajectories.','small')
    para('This shows why longer trajectories can change the learning problem: later states '
         'contain more small-scale structure and the cheap solver makes a larger local error. '
         'During an unrolled prediction the network feeds back its own states. '
         'These teacher-state measurements describe the target trajectory and do not '
         'prove which mechanism causes the final accuracy change.')

    section('Teacher imitation and numerical reliability')
    para('A network can lose to the solver simply because it has not learned to copy it '
         'accurately. Training losses from different recipes are not directly comparable: '
         'they contain different prediction lengths and targets. We therefore tested '
         'every frozen model against identical one-step teacher targets, using 100 '
         'separate validation profiles and the same five input time levels.')
    common=OUT/'common_teacher_fit.csv'
    if common.exists():
        f=pd.read_csv(common);f=f[f.pde=='Burgers'];rows=[]
        for (mode,n),g in f.groupby(['mode','N']):
            def scientific_interval(column):
                a=g[column].to_numpy()
                return f'{np.median(a):.3g} [{a.min():.3g}, {a.max():.3g}]'
            rows.append([mode,n,scientific_interval('common_initial_MSE'),
                         scientific_interval('common_H5_MSE')])
        tab(['Recipe','N','Initial-state teacher MSE','Common five-level teacher MSE'],
            rows,[.7,.3,1.6,1.6])
    para('These are imitation errors, not errors against the converged reference. '
         'They use validation seed 110 and do not select checkpoints or hyperparameters. '
         'A small imitation error alone does not guarantee a stable repeated rollout.','small')
    v=json.loads((OUT/'validation.json').read_text())
    b=new[new.group=='UNet Burgers']
    sensitivity=b.reference_sensitivity.abs().max() if len(b) else np.nan
    tab(['Check','Observed result'],[
        ['ConvNet / UNet forward agreement',f"{v['Conv;34;10;relu']['max_forward_difference']:.2g} / {v['UNet;12;2;relu']['max_forward_difference']:.2g} maximum difference"],
        ['Gradient agreement',f"Relative differences {v['Conv;34;10;relu']['relative_gradient_difference']:.2g} / {v['UNet;12;2;relu']['relative_gradient_difference']:.2g}"],
        ['Resumed versus uninterrupted Adam',f"Parameter difference {v['chunk_resume_parameter_difference']:.1g}"],
        ['Teacher derivative finite difference',f"Relative difference {v['P1_directional_derivative']['relative_difference']:.3g}"],
        ['Tighter reference, final Burgers models',f'Maximum primary-ratio change {sensitivity:.3g}']], [1.4,2.1])
    gap=v['reference_tolerance_check']['max_reference_RMSE_fraction_of_teacher_gap']*100
    state_difference=v['reference_tolerance_check']['max_state_difference']
    para('The tighter reference uses tolerance 1e-7 and up to 200 iterations instead of '
         f'1e-5 and 100. On validation trajectories, its maximum state difference was '
         f'{state_difference:.3g}; its RMS difference was at most {gap:.4f}% of the '
         'teacher/reference error gap. Final-model primary-score sensitivity is shown above. '
         'The full validation history is retained in validation.json.','small')

    section('What can be added to the course report?')
    for p in discussion.get('report_paragraphs',['Final discussion pending completion.']):para(p)
    para('Limits of the conclusions','h1')
    for p in discussion.get('limitations',[]):para('• '+p)
    section('Reproducibility and sources')
    audit=json.loads((OUT/'audit.json').read_text())
    tab(['Final audit','Observed outcome'],[
        ['Unique primary configurations',f"{audit['observed_primary_cells']} / {audit['planned_primary_cells']}"],
        ['Full prescribed training budget',f"{audit['completed_full_budget']} / {audit['planned_primary_cells']}"],
        ['Training failures / nonfinite primary rollouts',
         f"{len(audit['training_failures_retained'])} / {len(audit['nonfinite_rollouts_retained'])}"],
        ['Common primary numerical environment',f"{audit['common_primary_provenance_versions']} provenance version"],
        ['Teacher error curves within each test setup',
         f"Maximum absolute difference {audit['maximum_teacher_baseline_error_difference']:.1g}"],
        ['Historical summaries unchanged',audit['unchanged_historical_summaries']]], [1.5,1.5])
    para('The audit recomputes primary scores from per-step CSVs and checks the expected '
         'grid, budgets, data seeds, source hashes, and saved model/optimizer artifacts. '
         'Interrupted workers resumed saved Adam states. The executed numerical sources '
         'were archived before a post-run JSON serialization fix; the numerical algorithm '
         'and recorded outcomes are unchanged.','small')
    tab(['Model / execution detail','Setting'],[
        ['ConvNet', 'Conv;34;10;relu - 31,757 parameters'],
        ['FNO', 'FNO;12;18;4;gelu - 32,527 parameters'],
        ['UNet', 'UNet;12;2;relu - 27,193 parameters'],
        ['Training updates', 'Advection 10,000; Burgers 100,000'],
        ['Numerical runtime', 'Python 3.12.14; JAX 0.4.38; CPU'],
        ['Core simulation dependencies', 'APEBench 0.1.1; Exponax 0.1.0; Trainax 0.0.2']], [1.2,1.8])
    para('The project contains EXPERIMENT_PLAN.md; results/extension_2026_11_22/RESULTS.md; '
         'per_run_results.csv; paired_seed_contrasts.csv; homogeneity_results.csv; '
         'validation.json; the original 137-run manifest; per-model configurations, '
         'metrics, test-example errors, weights, optimizer checkpoints, and loss histories.','small')
    para('Generate the tables and figures with src/analyze_extension.py. '
         'Run or resume the prescribed grid with src/run_extension_pool.py. '
         'The current dependency and author-source identities are recorded in each run summary.','small')
    para('Background: Koehler and Thuerey, Neural Emulator Superiority (arXiv:2510.23111v2), '
         'Appendix B; authors’ repository tum-pbs/emulator-superiority, pinned commit '
         '461eb07d2714c958d43d4821abc78f2d31efff97. The paper text specifies 100 Burgers '
         'test examples over 20 steps; the released script uses 20 examples over 10 steps. '
         'This study follows the text and uses project data seeds 10/20/30.','small')
    para('Paper: https://arxiv.org/abs/2510.23111\n'
         'Author code: https://github.com/tum-pbs/emulator-superiority','small')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('DejaVu',7)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(42,22,'10-749 | Emulator superiority: extension experiments')
        canvas.drawRightString(A4[0]-42,22,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=42,rightMargin=42,
        topMargin=38,bottomMargin=38,title='Emulator Superiority: Extension Experiments',
        author='10749 course project')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    from pypdf import PdfReader
    reader=PdfReader(str(dest))
    print(json.dumps({'file':str(dest),'pages':len(reader.pages),'bytes':dest.stat().st_size,
        'incomplete':incomplete,'page_text_lengths':[len(p.extract_text() or '') for p in reader.pages]}))


if __name__=='__main__':main()
