"""Create the final, self-contained 2D extension report after all outcomes exist."""
from html import escape
import json
from pathlib import Path
import re
import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from analyze_turbulence import OUT,FIG,interval


def main():
    df=pd.read_csv(OUT/'per_model_results.csv');assert len(df)==12,'Complete all prescribed outcomes first'
    discussion=json.loads((OUT/'DISCUSSION.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text());assert audit['passed']
    validation=json.loads((OUT/'validation.json').read_text())
    refval=json.loads((OUT/'data'/'reference_validation.json').read_text())
    baselines=pd.read_csv(OUT/'baseline_results.csv')
    font=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    pdfmetrics.registerFont(TTFont('DejaVu',str(font/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(font/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
    styles={
        'title':ParagraphStyle('title',fontName='DejaVu-Bold',fontSize=22,leading=28,textColor=colors.HexColor('#0f172a'),spaceAfter=14),
        'heading':ParagraphStyle('heading',fontName='DejaVu-Bold',fontSize=16,leading=21,spaceAfter=12,keepWithNext=True,textColor=colors.HexColor('#0f172a')),
        'body':ParagraphStyle('body',fontName='DejaVu',fontSize=9.2,leading=13.5,spaceAfter=9),
        'small':ParagraphStyle('small',fontName='DejaVu',fontSize=7.7,leading=10.7,spaceAfter=8,textColor=colors.HexColor('#475569')),
        'cell':ParagraphStyle('cell',fontName='DejaVu',fontSize=7.1,leading=10),
        'th':ParagraphStyle('th',fontName='DejaVu-Bold',fontSize=7.1,leading=10,textColor=colors.white)}
    story=[];width=A4[0]-84
    def p(text,style='body'):
        text=escape(str(text).replace('–','-').replace('−','-').replace('—','-'))
        text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
        story.append(Paragraph(text,styles[style]))
    def section(title):story.append(PageBreak());p(title,'heading')
    def table(headers,rows,weights=None):
        values=[[Paragraph(escape(str(v)),styles['th']) for v in headers]]
        values += [[Paragraph(escape(str(v)),styles['cell']) for v in row] for row in rows]
        weights=weights or [1]*len(headers)
        t=Table(values,colWidths=[width*w/sum(weights) for w in weights],repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e3a5f')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
            ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,10)])
    def image(name,height=260):
        from PIL import Image as PILImage
        path=FIG/f'{name}.png'
        with PILImage.open(path) as im:w,h=im.size
        scale=min(width/w,height/h)
        story.extend([Image(str(path),width=w*scale,height=h*scale),Spacer(1,9)])
    def conclusion(key):
        for text in discussion.get(key,[]):p(text)

    p('Can a coarse-trained emulator\nbeat its 2D flow solver?','title')
    p('CMU 10-749 | Two-dimensional turbulent-flow extension | 24 November 2026','small')
    p('An emulator is a neural network trained to imitate a numerical solver. '
      'Here it learns only from a rough simulation of continuously stirred fluid. '
      'We compare its repeated predictions and its teacher against a more accurate simulation.')
    p('This moves the project beyond single waves and nonlinear 1D Burgers. '
      'The flow contains interacting swirls at several scales, with naturally varying heights. '
      'The earlier single-wave, fixed-height explanation does not directly determine the outcome.')
    table(['Question','How we tested it'],[
        ['Does superiority survive in 2D?', 'Rough-only FNO training; 64 held-out flows and a refined reference.'],
        ['Does longer training help?', 'One-step versus four-step losses on identical eight-step teacher trajectories.'],
        ['Is an apparent win physically meaningful?', 'Energy, enstrophy, spectra, correlation, and simple forgetting/smoothing controls.']], [1,1.9])
    p('Score = emulator error / teacher error. Below 1 favors the emulator; 1 means equal error. '
      'The main score sums normalized vorticity errors over all 20 predefined forecast steps. '
      'Individual favorable steps do not replace this score.')
    conclusion('findings')
    p('Twelve main models: two independent training datasets x three initializations x two training recipes. '
      'The same 64 test flows are shared across models. This is a modest-Re forced-flow benchmark, '
      'not a high-Re or industrial turbulence claim.','small')

    section('What exactly was simulated and learned?')
    p('The state is vorticity: how strongly the fluid rotates at each location. '
      'Advection moves that rotation around, viscosity smooths it, drag removes energy, '
      'and a fixed forcing pattern continually adds energy. Boundaries wrap around a periodic square.')
    table(['Component','Fixed setting'],[
        ['Equation', '2D incompressible Navier-Stokes with Kolmogorov forcing'],
        ['Physical parameters','Box side 2*pi; viscosity 0.01; scalar drag -0.1; forcing mode 4, scale 1'],
        ['Teacher', '32 x 32 grid; ETDRK2; 10 internal steps per recorded interval 0.2'],
        ['Reference', '96 x 96 grid; ETDRK4; 20 internal steps per recorded interval'],
        ['Refinement', '128 x 128; 40 internal steps; all 64 flows for primary window'],
        ['Training data','Two datasets of 128 independent warmed-up flows, eight teacher steps each'],
        ['Validation / testing','32 teacher-validation flows / 64 test flows, independent seeds'],
        ['Network','Residual FNO, width 16, 10 modes, four blocks; 410,753 parameters'],
        ['Training','12,000 Adam updates/model; batch 8; initializations 0,1,2'],
        ['Forecast windows','Steps 1-20 (time 0.2-4) primary; 21-100 (time 4.2-20) extended']], [1,2.7])
    p('Every initial flow is first warmed up for 60 time units using the rough solver. '
      'Training inputs, targets, and normalization come entirely from rough data. '
      'The network receives the current vorticity and the known forcing pattern; '
      'no fine-solver targets, per-field height normalization, or test-based checkpoint selection are used.')
    p('At forecast start, test-flow RMS vorticity ranges from 2.48 to 4.44 and maximum absolute '
      'vorticity from 8.84 to 15.39. The flow distribution has naturally varying strengths '
      'rather than a fixed unit-height constraint.','small')
    p('The fine solver starts from the identical coarse field, interpolated in Fourier space. '
      'Primary errors use the same resolved band, |kx| and |ky| at most 9. '
      'Full-fine-field errors additionally include the small-scale reference power that coarse predictions cannot represent. '
      'Unresolved scales contain on average '
      f"{100*refval['unresolved_enstrophy_fraction_mean_1_20']:.2f}% of reference enstrophy and "
      f"{100*refval['unresolved_kinetic_energy_fraction_mean_1_20']:.3f}% of kinetic energy in the main window.",'small')
    p('The smaller fitting pilot underfit the rough solver. The final larger architecture was chosen '
      'using only separate rough-solver validation data. The protocol records that amendment. '
      'Sliding-window weights and compute differ between N=1 and N=4 even though trajectories are shared.','small')

    section('Main result: every dataset and initialization')
    p('Each entry is the median [minimum, maximum] across three initialization seeds. '
      'The two rows per recipe keep training-dataset variation visible.')
    rows=[]
    for (n,dataset),g in df.groupby(['N','dataset']):
        rows.append([n,'A' if dataset==1101 else 'B',interval(g.R20),interval(g.velocity_R20),interval(g.full_field_R20),f'{sum(g.R20<1)}/{len(g)}'])
    table(['N','Training data','Primary vorticity ratio','Velocity ratio','Full-field ratio','Models better'],rows,[.25,.7,1.5,1.5,1.5,.65])
    teacher_error=float(baselines[baselines.name=='teacher'].iloc[0].teacher_mean_nRMSE_1_20)
    values={n:float(g.emulator_mean_nRMSE_1_20.median()) for n,g in df.groupby('N')}
    p(f'Mean normalized vorticity error over the primary window: teacher {teacher_error:.3f}; '
      f'one-step median {values[1]:.3f}; four-step median {values[4]:.3f}.','small')
    image('primary_seed_results',230)
    conclusion('primary_conclusion')
    p('Bootstrap bars resample whole test trajectories, preserving dependence across time. '
      'They describe test-case uncertainty conditional on each trained model. They do not '
      'turn six trained models into six independent datasets or establish universal superiority.','small')

    section('What happens when predictions are repeated?')
    image('accuracy_rollouts',240)
    rows=[]
    for n,g in df.groupby('N'):
        rows.append([n,interval(g.R20),interval(g.R21_100),interval(g.R100),interval(g.mean_correlation_21_100)])
    table(['N','Steps 1-20','Steps 21-100','Steps 1-100','Later correlation'],rows,[.3,1.4,1.4,1.4,1.4])
    p('Non-learning controls help interpret the ratios. Persistence keeps the initial flow unchanged. '
      'The zero model predicts no vorticity. Linear flow includes forcing, viscosity and drag '
      'but removes nonlinear advection. None uses fine data.')
    control_names={'teacher':'Rough teacher','zero':'Zero field','persistence':'Unchanged initial flow',
        'linear_flow':'Forced linear flow','time_refined32':'Time-refined 32-grid','space_refined48':'Refined 48-grid'}
    table(['Control','Primary ratio','Full 100-step ratio','Later-window ratio'],
        [[control_names.get(r['name'],r['name']),f"{r['R20']:.3f}",f"{r['R100']:.3f}",f"{r['R21_100']:.3f}"] for r in baselines.to_dict('records')],[1.2,1,1,1])
    conclusion('time_conclusion')
    p('In a chaotic flow, pointwise predictions can eventually lose alignment with the reference. '
      'A smooth or zero field can then have less squared error than a detailed, displaced flow. '
      'Later-window ratios must therefore be interpreted alongside physical structure.','small')

    section('Does the emulator preserve the flow’s structure?')
    image('physical_structure',190)
    image('energy_spectrum',220)
    conclusion('physics_conclusion')
    p('Kinetic energy measures motion; enstrophy measures squared rotation and is more sensitive '
      'to small-scale structure. A spectrum shows how energy is distributed across swirl sizes. '
      'Shaded bands show the observed model range; they are not confidence intervals. '
      'The coarse band is square in Fourier space and has maximum radial extent about 12.7.','small')

    section('What explains the change in error?')
    image('teacher_imitation',210)
    rows=[]
    for n,g in df.groupby('N'):
        v=g.common_validation_normalized_MSE.to_numpy()
        rows.append([n,f'{np.median(v):.3g} [{v.min():.3g}, {v.max():.3g}]',interval(g.test_example_win_fraction)])
    table(['N','Same-input one-step imitation MSE / scale squared','Fraction of test flows better'],rows,[.3,2,1.5])
    image('error_alignment',170)
    conclusion('mechanism_conclusion')
    p('The decomposition uses squared field errors: model error = teacher error + model deviation. '
      'The deviation adds a nonnegative error cost; its alignment with teacher error may cancel '
      'or increase that error. These terms exactly explain an MSE change, while the primary '
      'score averages normalized root errors and is reported separately.','small')

    section('A fixed example: how the swirls evolve')
    p('Test example 0, training dataset 1101, initialization 0. This example and these '
      'models were fixed before seeing performance. All panels use the same color scale. '
      'The fine row displays 96-grid fields; teacher and emulator rows display their actual 32-grid predictions.')
    image('flow_snapshots',590)
    p('Images illustrate one trajectory; numerical conclusions use all 64 test flows and every prescribed model.','small')

    section('What can be added to the course report?')
    conclusion('report_paragraphs')
    p('Limits of the conclusion','heading')
    for text in discussion.get('limitations',[]):p('• '+text)

    section('Validation, compute, and reproducibility')
    long_ref=json.loads((OUT/'long_reference_sensitivity.json').read_text())
    table(['Check','Observed result'],[
        ['Unique main outcomes / full training budget',f"{audit['observed_outcomes']}/12 outcomes; {audit['full_budget_completed']}/12 full budgets"],
        ['Training failures / nonfinite forecasts',f"{len(audit['training_failures_retained'])} / {len(audit['nonfinite_forecasts_retained'])}"],
        ['Data, source and model hashes','All match; one main training provenance version'],
        ['Primary-score recalculation',f"Maximum difference {audit['maximum_recomputed_primary_score_difference']:.2g}"],
        ['Taylor-Green analytic decay',f"Relative error {validation['Taylor_Green_relative_error_after_time_2']:.3g}"],
        ['Forced linear analytic solution',f"Maximum error {validation['forced_Stokes_max_error']:.3g}"],
        ['Resumed / uninterrupted Adam',f"Maximum parameter difference {validation['resumed_Adam_parameter_difference']:.1g}"],
        ['Network gradient finite difference',f"Relative difference {validation['FNO_directional_derivative_relative_difference']:.3g}"],
        ['96-to-128 reference refinement',f"Mean difference {100*refval['refinement_fraction_of_teacher_gap']:.4f}% of teacher error gap"],
        ['Final-model reference sensitivity',f"Maximum primary-ratio change {df.reference_primary_change.abs().max():.3g}"],
        ['Long-reference check, first 8 flows',f"Maximum later-window ratio change {long_ref['maximum_model_later_ratio_change_on_subset']:.3g}"]],[1.3,1.5])
    p('On the fixed eight-flow long-horizon subset, the reference refinement difference '
      f"averages {100*long_ref['reference_refinement_fraction_of_teacher_gap_later']:.2f}% of the teacher error gap. "
      'Individual chaotic fields are more sensitive at long times. The zero-field later-window '
      f"ratio remains {long_ref['zero_later_ratio_subset_reference128']:.3f} with the 128-grid reference. "
      'This subset check does not replace the all-64-flow primary validation.','small')
    timing=OUT/'inference_timing.json'
    if timing.exists():
        t=json.loads(timing.read_text())
        labels={'FNO':'FNO emulator','teacher':'32-grid teacher','refined48':'48-grid numerical solver','reference96':'96-grid reference'}
        table(['Method','Milliseconds per field / forecast step'],[[labels[r['method']],f"{r['milliseconds_per_field_step']:.3f}"] for r in t['rows']],[1,1.5])
        p('Timing uses compiled synchronized CPU execution, batch eight, 20 forecast steps '
          'and five repeats, with compilation excluded and training workers finished. '
          'The emulator architecture is identical at inference for both recipes.','small')
    p('The main study uses 144,000 training updates on CPU. Recorded per-model training '
      'times are approximately 5.1 minutes for N=1 and 16.7 minutes for N=4 (medians), '
      'with three workers running in parallel. Those worker durations include compilation '
      'and CPU contention; they are not the elapsed time across disconnections.','small')
    p('The preflight analytic check exposed a drag-coefficient mismatch in Exponax 0.1.0. '
      'The extension corrects its linear symbol to apply scalar drag once; all main data '
      'were regenerated before training. Pre-correction artifacts and validation history '
      'are retained separately. The final checks above use the corrected equation.','small')
    p('Reproduction: TURBULENCE_EXPERIMENT_PLAN.md; src/turbulence_data.py; '
      'src/train_turbulence.py; src/run_turbulence_pool.py; src/evaluate_turbulence.py; '
      'src/analyze_turbulence.py; src/audit_turbulence.py. Results include training and '
      'reference data, final weights and Adam checkpoints, losses, per-test-flow metrics, '
      'source snapshots and numerical validation. Existing 1D study results remain separate.','small')
    p('Sources: Koehler and Thuerey, Neural Emulator Superiority, https://arxiv.org/abs/2510.23111 ; '
      'Exponax Navier-Stokes documentation, https://fkoehler.site/exponax/api/stepper/nonlinear/navier_stokes/ ; '
      'installed Exponax 0.1.0 and PDEquinox 0.1.2 source. This 2D study and its numerical results are new project work.','small')
    path=OUT/'Turbulence_2D_Results.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('DejaVu',7);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(42,22,'10-749 | 2D forced-flow emulator study');canvas.drawRightString(A4[0]-42,22,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=38,
        title='2D Turbulent-Flow Emulator Study',author='10749 course project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    pdf=PdfReader(path)
    print(json.dumps({'file':str(path),'pages':len(pdf.pages),'bytes':path.stat().st_size,
        'page_text_lengths':[len(p.extract_text() or '') for p in pdf.pages]}))


if __name__=='__main__':main()
