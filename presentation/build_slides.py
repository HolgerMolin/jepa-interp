"""Build a sourced, editable draft deck and matching PDF from saved results.

Run with /scratch/users/hmolin/jepa-interp/.venv/bin/python.
Presentation-only dependencies live outside the research environment.
"""
import sys
from pathlib import Path
sys.path.append('/scratch/users/hmolin/jepa-interp/presentation-deps')
import argparse
import hashlib
import html
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PPoly
from PIL import Image,ImageDraw
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ASSETS=HERE/'assets'
W,H=960,540
INK='#171717';MUTED='#696969';BG='#FFFFFF';TEAL='#355F85';ORANGE='#355F85';PURPLE='#355F85';LIGHT='#E5E5E5'
COLORS=[TEAL,TEAL,TEAL]
TASKS=['direction','speed','acceleration'];NAMES=['Direction','Speed','Acceleration'];UNITS=['°','m/s','m/s²']
SOURCES={}


def read(relative):
    p=ROOT/relative;SOURCES[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def figsave(fig,name):
    fig.savefig(ASSETS/(name+'.png'),dpi=190,bbox_inches='tight',facecolor='white')
    fig.savefig(ASSETS/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return 'assets/'+name+'.png'


def make_figures():
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'normal',
        'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,
        'axes.titlepad':14,'axes.grid':False,'legend.frameon':False,'figure.constrained_layout.use':True})
    probes=read('results/probe_results.json')
    fig,axes=plt.subplots(1,3,figsize=(14,5.6),sharey=True)
    for ax,t,name,unit,color in zip(axes,TASKS,NAMES,UNITS,COLORS):
        rows=probes[t]['layers'];xx=[r['layer'] for r in rows];yy=[r['test']['r2'] for r in rows]
        ax.plot(xx,yy,color=color,lw=2)
        ax.scatter([1],[yy[0]],color=color,s=35,zorder=5)
        ax.annotate(f'{yy[0]:.2f}',(1,yy[0]),xytext=(7,-23),textcoords='offset points',fontsize=13)
        ax.set(title=name,xticks=[1,8,16,24],ylim=(0,1.05),xlim=(.5,24.5))
    axes[0].set_ylabel('Held-out R²')
    fig.supxlabel('Encoder block',fontsize=14)
    figsave(fig,'layerwise')
    null=read('results/nullspace/scores.json');fig,axes=plt.subplots(1,3,figsize=(14,5.6))
    for ax,t,name,color in zip(axes,TASKS,NAMES,COLORS):
        rows=null[t]['rounds'];xx=[r['removed_dimensions'] for r in rows];yy=[r['test']['r2'] for r in rows]
        ax.plot(xx,yy,color=color,lw=2.5);ax.scatter(xx[-1],yy[-1],color=color,s=45)
        ax.set(title=name,ylim=(-.05,1.05),xlim=(0,256))
        ax.annotate(str(xx[-1]),(xx[-1],yy[-1]),xytext=(-15,16),textcoords='offset points',fontsize=12)
    axes[0].set_ylabel('Readout R²');fig.supxlabel('Cumulative rank removed',fontsize=14)
    figsave(fig,'nullspace')
    steering=read('results/steering/scores.json');fig,axes=plt.subplots(1,3,figsize=(14,5.6),sharey=True)
    for ax,t,name,color in zip(axes,TASKS,NAMES,COLORS):
        rows=steering[t]['sweep'];xx=[r['probe_count'] for r in rows]
        baseline=steering[t]['mean_no_steering_target_mae']
        yy=[100*(1-r['mean_evaluator_target_mae']/baseline) for r in rows]
        ax.plot(xx,yy,'o-',color=color,lw=2.2,ms=4)
        ax.axhline(0,color=MUTED,ls=':',lw=1,label='No edit')
        ax.set(xscale='log',title=name,xticks=[1,4,16,64,256],ylim=(-3,103),yticks=[0,25,50,75,100])
        ax.set_xticklabels(['1','4','16','64','256'])
    axes[0].set_ylabel('Readout error reduction (%)')
    axes[0].legend(fontsize=11,loc='lower right');fig.supxlabel('Construction probes',fontsize=14)
    figsave(fig,'multi_probe')
    fig=plt.figure(figsize=(14,4.8),layout='none')
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.92,wspace=.25)
    for k,(t,name,color) in enumerate(zip(TASKS,NAMES,COLORS)):
        path=ROOT/f'results/spline/model_{t}.npz';SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        m=np.load(path);f=PPoly(m['spline_coefficients'],m['spline_knots']);u=np.linspace(m['spline_knots'][0],m['spline_knots'][-1],250);z=f(u)
        ax=fig.add_subplot(1,3,k+1,projection='3d');ax.plot(*z[:,:3].T,color=color,lw=2.5)
        ax.scatter(*m['centroids'][:,:3].T,color=color,s=24);ax.set(title=name+' (periodic)' if t=='direction' else name,xlabel='PC1',ylabel='PC2',zlabel='PC3')
        ax.grid(False)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False;axis.pane.set_edgecolor('white')
        ax.tick_params(labelsize=9)
        ax.zaxis.labelpad=2
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('PC3',rotation=90)
        if t!='direction':
            from matplotlib.ticker import MaxNLocator
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(nbins=4))
        ax.view_init(22,-55) if t=='direction' else ax.view_init(20,55)
    figsave(fig,'spline_geometry')
    spline=read('results/spline/scores.json')
    reductions=read('results/spline_offset_v1/readout_error_reduction.json')['tasks']
    fig,axes=plt.subplots(1,3,figsize=(14,3.8))
    for ax,t,name,color in zip(axes,TASKS,NAMES,COLORS):
        vals=[reductions[t]['chord_interior_reduction_percent'],reductions[t]['spline_interior_reduction_percent']]
        ax.scatter(vals,[1,0],color=[MUTED,color],s=70)
        for y,value in zip([1,0],vals):ax.annotate(f'{value:.1f}%',(value,y),xytext=(-5,12),textcoords='offset points',ha='right',fontsize=12)
        ax.set_yticks([1,0],['Chord','Spline'])
        ax.set(title=name,xlabel='Readout error reduction (%)',xlim=(0,103),xticks=[0,50,100],ylim=(-.6,1.7))
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    figsave(fig,'spline_chords')
    it=read('results/iterated_velocity_v1/effects.json')['splits']['D'];fig,ax=plt.subplots(figsize=(14,5.6))
    caps=[1,2,4,8,16,32]
    for kind,color,label in [('velocity',TEAL,'Velocity edit'),('random',MUTED,'Matched random')]:
        rows=[it[f'{kind}_k{k:03d}_+0.50'] for k in caps];v=np.array([r['predicted']['velocity_slope_gain']['mean'] for r in rows])*100
        ci=np.array([r['predicted']['velocity_slope_gain']['ci95'] for r in rows])*100
        ax.errorbar(caps,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='o-',color=color,label=label,capsize=3,lw=2)
    ax.set(xscale='log',ylabel='Velocity response gain (%)',xlabel='Maximum eligible probes per token',xticks=caps,ylim=(-1,21));ax.set_xticklabels(list(map(str,caps)));ax.legend(fontsize=12)
    figsave(fig,'linear_predictor')
    cv=read('results/covariance_velocity_v1/effects.json')['D']
    rows=[cv['mean_+0.50'],cv['linear_+0.50']]
    fig,ax=plt.subplots(figsize=(14,5.6))
    for i,(r,color) in enumerate(zip(rows,[MUTED,TEAL])):
        st=r['predicted']['velocity_slope_gain'];value=st['mean']*100;ci=np.array(st['ci95'])*100
        ax.barh(i,value,height=.42,color=color)
        ax.errorbar(value,i,xerr=[[value-ci[0]],[ci[1]-value]],fmt='none',ecolor=INK,capsize=4,lw=1.2)
        ax.text(ci[1]+.45,i,f'{value:.1f}%',va='center',fontsize=17)
    ax.set(yticks=[0,1],yticklabels=['Spline','Iterated linear'],xlabel='Requested velocity change achieved (%)',xlim=(0,22),ylim=(-.65,1.65),xticks=[0,5,10,15,20])
    ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=14)
    figsave(fig,'predictor_comparison')
    diag=read('results/spline_residual_diagnostics_v1/scores.json');poly=read('results/covariance_velocity_v1/span_quadratic_scores.json')
    names=['RBF probe','Quadratic probe']
    vals=[diag['curve_span_removed']['rbf']['D']['r2'],poly['64']['D']['r2']]
    fig,ax=plt.subplots(figsize=(14,5.6))
    ax.barh(range(2),vals,color=TEAL,height=.42)
    for i,v in enumerate(vals):ax.text(v+.025,i,f'{v:.2f}',va='center',fontsize=17)
    ax.set(yticks=range(2),yticklabels=names,xlim=(0,1.05),ylim=(-.65,1.65),xlabel='Held-out speed prediction R²')
    ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=14)
    figsave(fig,'residual_diagnostics')
    # Illustrative offset-preserving edit, not measured activations.
    fig,ax=plt.subplots(figsize=(14,5.6))
    curve=lambda x: .3+.15*x+.2*np.sin(x)
    xs=np.linspace(0,4.9,150);source=.8;target=3.8;offset=1.1
    before=np.array([source,curve(source)+offset])
    after=np.array([target,curve(target)+offset])
    ax.plot(xs,curve(xs),color=MUTED,lw=2)
    ax.plot([source,source],[curve(source),before[1]],color=MUTED,ls='--',lw=1.3)
    ax.plot([target,target],[curve(target),after[1]],color=TEAL,ls='--',lw=1.3)
    ax.scatter([source,target],[curve(source),curve(target)],color=MUTED,s=28)
    ax.annotate('',xy=after,xytext=before,arrowprops=dict(arrowstyle='->',color=TEAL,lw=2.2,shrinkA=7,shrinkB=7))
    ax.scatter(*before,color=MUTED,s=65,zorder=5);ax.scatter(*after,color=TEAL,s=65,zorder=5)
    ax.text(source,before[1]+.16,'Before',ha='center',fontsize=17)
    ax.text(target,after[1]+.16,'After',ha='center',fontsize=17,color=TEAL)
    ax.text(source,curve(source)-.2,'Source centroid',ha='center',va='top',fontsize=15)
    ax.text(target,curve(target)-.2,'Target centroid',ha='center',va='top',fontsize=15)
    ax.text(source-.12,(curve(source)+before[1])/2,'offset',ha='right',fontsize=15,color=MUTED)
    ax.text(target+.12,(curve(target)+after[1])/2,'same offset',fontsize=15,color=TEAL)
    ax.text(.5,-.02,'z′ = z + f(target) − f(source)',transform=ax.transAxes,ha='center',fontsize=20)
    ax.set(xlim=(-.45,5.5),ylim=(-.25,2.55));ax.axis('off')
    figsave(fig,'spline_edit_rules')
    # Real, unmodified input frames; no synthetic reconstruction or edited video.
    import av
    fig,axes=plt.subplots(3,4,figsize=(12,6.5))
    for row,t in enumerate(TASKS):
        manifest=ROOT/f'vjepa-physics-takehome-4E00/data/{t}/manifest.jsonl';entry=json.loads(manifest.read_text().splitlines()[len(manifest.read_text().splitlines())//2])
        path=manifest.parent/entry['video'];SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with av.open(str(path)) as video:frames=[f.to_ndarray(format='rgb24') for f in video.decode(video=0)]
        for col,k in enumerate([0,5,10,15]):
            axes[row,col].imshow(frames[k]);axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(f'Frame {k}',fontsize=14)
            if col==0:axes[row,col].text(-.08,.5,NAMES[row],transform=axes[row,col].transAxes,rotation=90,va='center',ha='right',fontsize=15)
    figsave(fig,'example_frames')
    return probes,null,steering,spline


def content(probes,null,steering,spline):
    return [{'title': 'Physics in VJEPA2',
      'section': 'V-JEPA PHYSICS',
      'layout': 'cover',
      'takeaway': 'Accurate decoding → redundant representations → limited downstream control',
      'notes': 'Opening (30 s). We reproduce the requested experimental progression, then ask a stronger '
               'question: do representation edits control predicted motion? The main deck targets approximately '
               '15 minutes; appendices support discussion.',
      'sources': ['vjepa-physics-takehome-4E00/README.md'],
      'context': '',
      'author': 'Holger Molin'},
     {'title': 'Data splits',
      'section': 'SETUP',
      'layout': 'splits',
      'context': 'Disjoint video splits within each physical variable',
      'table_headers': ['Split', 'Purpose', 'Direction', 'Speed / accel.'],
      'table_rows': [['A', 'Fit probes and splines', '768', '768'],
                     ['B', 'Select hyperparameters', '220', '256'],
                     ['C', 'Train independent readouts', '220', '256'],
                     ['D', 'Evaluate probes and edits', '292', '256']],
      'takeaway': 'Separate fitting, selection, independent readout training, and evaluation.',
      'notes': 'Data splits (45 s). Counts are clips per variable: direction 1,500, speed 1,536, acceleration '
               '1,536; total 4,572. A fits probes and splines; B selects regularization, layer, stopping and '
               'spline resolution. C trains independent evaluators; evaluator regularization uses '
               'cross-validation within C. D evaluates held-out clips. Later analyses reuse D, so those '
               'follow-ups are exploratory rather than untouched confirmatory tests. The predictor extension '
               'uses a speed-restricted subset of D.',
      'sources': ['results/splits.json',
                  'results/steering/protocol.json',
                  'results/velocity_causal_v1/protocol.json']},
     {'title': 'Linear probing across layers',
      'section': 'PART 1 • LAYER-WISE PROBING',
      'layout': 'figure',
      'figure': 'assets/layerwise.png',
      'takeaway': 'B selects block 19: 3.22° direction • 0.068 m/s speed • 0.198 m/s² acceleration',
      'caption': 'Ridge on mean-pooled tokens, fitted on A; regularization and common block selected on B. '
                 'Held-out R² on D; direction uses sine/cosine targets. All panels share the same R² scale.',
      'notes': 'Layer-wise results (75 s). Direction predicts sine and cosine; scalar labels retain physical '
               'units. Common layer selection maximizes mean B R² subject to adequate decoding for all tasks. '
               'Already at block 1, D R² is .838/.982/.976 for direction/speed/acceleration. Thus we reproduce '
               'accurate decoding, but not a sharp delayed direction-emergence pattern on this simpler dataset. '
               'Avoid presenting this as a contradiction of the paper. Static, normalization and solver controls '
               'are in the appendix.',
      'sources': ['results/probe_results.json',
                  'results/validation_selection.json',
                  'results/diagnostics/AUDIT.md'],
      'context': 'No distinct physics emergence zone on these clips: all three variables are readable from block '
                 '1.'},
     {'title': 'Iterative nullspace probes',
      'section': 'PART 1 • ITERATIVE NULLSPACES',
      'layout': 'figure',
      'figure': 'assets/nullspace.png',
      'takeaway': 'Near-baseline stopping requires removing 220 / 223 / 248 dimensions.',
      'caption': 'Fit → remove readout row space → refit. Curves show fresh probes, not the original probe after '
                 'ablation.',
      'notes': 'Nullspaces (65 s). The x-axis is cumulative independent rank, not probe count: direction can '
               'remove two dimensions per round. A fits and B selects alpha/stopping. Maximum 256 dimensions; '
               'three near-baseline rounds trigger stopping. Terminal D R² .022/.035/.019. Direction MAE remains '
               '77.4° versus constant 91.3°, so this is not exact erasure. Counts depend on estimator and '
               'stopping rule; they are not intrinsic dimensions. No rank-matched random removal in this '
               'original pooled experiment.',
      'sources': ['results/nullspace/scores.json', 'results/nullspace/protocol.json'],
      'context': 'Fresh linear readout after each removal'},
     {'title': 'Multi-probe steering',
      'section': 'PART 1 • MULTI-PROBE STEERING',
      'layout': 'figure',
      'figure': 'assets/multi_probe.png',
      'takeaway': 'At 16 probes, readout error reduction is 95.9% for direction, 95.0% for speed, and 94.5% for '
                  'acceleration.',
      'caption': 'Independent C-trained evaluator on D clips; every sweep point is normalized by the same '
                 'unedited target-error baseline for that variable.',
      'notes': 'Multi-probe steering (75 s). Minimum-change least-squares edits in the A-probe subspace preserve '
               'its orthogonal complement. Direction targets every 45 degrees; scalars use low/mid/high targets. '
               'Readout error reduction is 100 × (1 − mean edited absolute target error / mean unedited absolute '
               'target error), using the same D clips and requested targets in numerator and denominator. This '
               'is a ratio of aggregate errors, not a mean of per-example percentages. Zero means no '
               'improvement; 100% means the target is reached exactly; negative values mean worse than no edit. '
               'It is not R² and does not measure downstream velocity response. At 16 probes the reductions are '
               '95.85%, 95.01%, and 94.50%. Sixteen is a fixed illustrative setting, not a D-selected optimum. '
               'Full sequences increase edit sizes and can reduce readout error reduction. These are pooled '
               'independent-readout results; later encoder blocks and predictor are not run. Construction '
               'success follows from satisfying the probe constraints; transfer to the independent evaluator is '
               'the test.',
      'sources': ['results/steering/scores.json', 'results/steering/protocol.json'],
      'context': 'Readout error reduction vs no edit · higher is better\n'
                 '0% = no improvement · 100% = target reached'},
     {'title': 'Readout agreement does not establish causality',
      'section': 'INTERPRETATION',
      'layout': 'argument',
      'context': '',
      'paragraphs': ['Edits move an independently trained probe’s estimates toward the target.',
                     'Both probes can use the same correlated features. Their agreement does not show that the '
                     'predictor uses those features.',
                     'Causal test: edit velocity, run the predictor, and measure the change in predicted '
                     'position.'],
      'takeaway': 'The correlation persists across independent readouts; a causal role in predicted motion '
                  'remains untested by this experiment.',
      'notes': 'Interpretation (45 s). Multi-probe steering confirms that target-directed edits transfer from '
               'the construction probes to an independent C-trained evaluator on D clips. This is stronger than '
               'satisfying the construction probes, but both can depend on the same correlates. The intervention '
               'causes a change in the external evaluator output; it does not by itself establish that the '
               'physical variable controls the model’s predicted motion. This pooled experiment does not run the '
               'remaining encoder blocks or predictor. The later prefix-token experiment addresses that missing '
               'downstream test by editing velocity and measuring predicted position through a frozen readout. '
               'That test still requires attention to readout validity and off-target effects; failure of the '
               'pooled test to establish causality is not proof of noncausality.',
      'sources': ['results/steering/protocol.json',
                  'results/steering/scores.json',
                  'results/velocity_causal_v1/protocol.json']},
     {'title': 'Centroid splines',
      'section': 'PART 2 • SPLINE CONSTRUCTION',
      'layout': 'figure',
      'figure': 'assets/spline_geometry.png',
      'takeaway': 'A-only PCA64 + binned centroids; B selects 8–16 knots.',
      'caption': 'First three PCs of the 64D centroid fits. Steering adds the source-to-target centroid '
                 'difference and preserves the offset.',
      'notes': 'Spline construction (80 s). Fits are label-conditioned centroids, not a learned full activation '
               'manifold. A fits the 64D PCA and centroid curves; B selects 12 direction knots, 8 speed knots '
               'and 12 acceleration knots by reconstruction error at known labels. PCA retains about 98% of '
               'activation variance. Direction periodicity is imposed; direction data combine two motion '
               'regimes. Every spline result presented uses an offset-preserving centroid-difference edit with '
               'the known source label. The same offset from the source centroid is retained at the target, and '
               'the PCA orthogonal complement is unchanged. This does not guarantee membership in the full '
               'activation manifold. The appendix illustrates this edit. Camera views: direction elevation 22°, '
               'azimuth −55°; speed and acceleration elevation 20°, azimuth 55°.',
      'sources': ['results/spline/validation.json',
                  'results/spline/model_direction.npz',
                  'results/spline_offset_v1/protocol.json'],
      'context': '64-dimensional PCA fit; first three components shown'},
     {'title': 'Spline versus linear steering',
      'section': 'PART 2 • HELD-OUT SPLINE EVALUATION',
      'layout': 'steering_table',
      'table_headers': ['Method', 'Direction', 'Speed', 'Acceleration'],
      'table_rows': [['Spline', '94.3%', '87.8%', '89.6%'], ['Linear (16 probes)', '95.9%', '95.0%', '94.5%']],
      'takeaway': 'Linear steering gives greater readout error reduction for all three variables; both methods '
                  'improve substantially over no edit.',
      'notes': 'Pooled readout comparison (60 s). Readout error reduction is 100 × (1 − mean edited absolute '
               'target error / mean unedited absolute target error), using the same D clips and requested '
               'targets in numerator and denominator. This is a ratio of aggregate errors, not a mean of '
               'per-example percentages. Zero means no improvement; 100% means the target is reached exactly; '
               'negative values mean worse than no edit. It is not R² and does not measure downstream velocity '
               'response. Spline reductions are 94.33% direction, 87.84% speed and 89.59% acceleration; the '
               'fixed 16-probe linear values are 95.85%, 95.01% and 94.50%. The unedited baseline errors are 90 '
               'degrees, 1.567887 m/s and 4.081567 m/s². Raw MAEs remain in the experiment report. The spline '
               'adds the centroid difference f(target) − f(known source) through the saved 64D PCA basis, '
               'preserving the source offset and giving no edit when source equals target. A/B fit and select '
               'the PCA/curves; frozen C evaluators score D. No method is selected on D. These are exploratory '
               'readout-transfer results, not proof of causal control or preservation of unrelated scene '
               'properties.',
      'sources': ['results/spline_offset_v1/REPORT.md',
                  'results/spline_offset_v1/scores.json',
                  'results/steering/scores.json',
                  'results/spline_offset_v1/readout_error_reduction.json'],
      'context': 'Independent-readout error reduction vs no edit · higher is better\n'
                 '0% = no improvement · 100% = target reached',
      'caption': 'Offset-preserving spline and linear edits use identical requested targets, D clips, C-trained '
                 'evaluators, and unedited error baselines.'},
     {'title': 'Steering through the predictor',
      'section': 'EXTENSION • A STRONGER INTERVENTION TEST',
      'layout': 'pipeline',
      'transition': 'Earlier: one full-clip pooled vector → external probe\n'
                    'Here: 1,024 prefix tokens → remaining encoder + predictor',
      'token_mapping': 'Fit a probe per token and make token-specific edits.',
      'steps': ['Frames 0–7',
                'Layer 19\nvelocity edit',
                'Blocks 20–24\n+ final norm',
                'Frozen\npredictor',
                'Position\nreadout'],
      'bullets': ['Future targets: frames 8–15',
                  'Expected displacement: Δp(t) = Δv · (t − t₀)',
                  'Response gain = measured velocity change / requested change'],
      'takeaway': 'Gain 1 = the requested physical slope change. Gain 0 = no velocity-like response.',
      'notes': 'Predictor assay (90 s). Prefix is physically truncated before embeddings/attention; replacing '
               'future pixels with noise leaves features/predictions unchanged. A fits per-token vx,vy and '
               'position probes, B chooses fits. C-only position ridge is shared across future horizons and '
               'trained on unedited predictor outputs. Baseline farthest D error8.17px vs hold-last21.26px; '
               'measured prefix extrapolation .78px. For causal edits restrict speeds to[.75,3.5],184 clips each '
               'B/D; ±.25/±.50m/s along original direction. Farthest requested+.5 effect≈5px, but primary gain '
               'is free-intercept time slope, not final displacement/5. This tests the frozen '
               'encoder–predictor–readout, not generated video.',
      'sources': ['results/velocity_causal_v1/REPORT.md',
                  'results/velocity_causal_v1/intervention_protocol.json'],
      'context': ''},
     {'title': 'Iterated velocity steering',
      'section': 'EXTENSION • ITERATED TOKEN STEERING',
      'layout': 'figure',
      'figure': 'assets/linear_predictor.png',
      'takeaway': 'Matched one-probe gain 3.45% → all eligible probes 17.91%.',
      'caption': '+0.50 m/s • 184 D clips • 95% clip-bootstrap intervals. Eligible probes per token: median 3, '
                 'range 1–23.',
      'notes': 'Iterated token steering (90 s). Each token gets its own sequence; original position-probe '
               'directions are protected. Steering eligibility requires both velocity components to have B R² ≥ '
               '0.5. Token fitting stops after three near-baseline probes or a 256-dimensional cap. There are '
               '4,274 eligible probes across 1,024 tokens; the median is 3 and maximum 23. The cap-32 condition '
               'therefore saturates the eligible set. Random controls match per-token edit norm and preserve '
               'construction position readouts. Independent context-position drift grows to 0.367 px. The '
               'initial historical joint four-output fit gave 3.33%; the matched one-probe condition shown here '
               'gives 3.45%. This supports an intervention response in the measured pipeline, not “18% of all '
               'causal information.” Full erasure has substantial position confounding; see the appendix.',
      'sources': ['results/iterated_velocity_v1/REPORT.md', 'results/iterated_velocity_v1/effects.json'],
      'context': 'Δspeed = +0.50 m/s · 184 D clips · 95% bootstrap intervals'},
     {'title': 'Predicted motion: spline versus linear',
      'section': 'EXTENSION • SPEED STEERING',
      'layout': 'figure',
      'figure': 'assets/predictor_comparison.png',
      'takeaway': 'Spline steering achieves 5.39% of the requested velocity change; iterated linear steering '
                  'achieves 17.91%.',
      'caption': 'Velocity change is inferred from the time slope of predicted-position readouts. Methods differ '
                 'in token mapping and position constraints.',
      'notes': 'Main predictor comparison (60 s). The spline edit adds f(s + Δs) − f(s), using known source '
               'speed and broadcasting the same channel displacement to every token. It achieves 5.39% of the '
               'requested velocity change. Iterated linear steering achieves 17.91%, with separate '
               'token-specific directions and position-readout constraints. Thus this compares the two '
               'implemented methods; it does not isolate the effect of using a linear versus nonlinear function '
               'class. Gain is the measured change in velocity, inferred from predicted positions, divided by '
               'the requested change. The plotted intervals are 95% clip-bootstrap intervals conditional on the '
               'fitted models. Unsuccessful spline variants have been omitted from the presentation; their full '
               'results remain in the experiment reports.',
      'sources': ['results/covariance_velocity_v1/effects.json',
                  'results/centroid_velocity_v1/REPORT.md',
                  'results/iterated_velocity_v1/REPORT.md'],
      'context': 'Requested +0.50 m/s · 184 D clips · 95% bootstrap intervals'},
     {'title': 'Conclusions',
      'section': 'CONCLUSIONS',
      'layout': 'conclusions',
      'cards': [['Readout', 'All three variables are accurately and redundantly represented.'],
                ['Evaluation', 'Independent-readout accuracy alone does not establish causal control.'],
                ['Predictor response', 'Iterated linear edits produce the strongest measured motion response.']],
      'bullets': [],
      'takeaway': 'We have evidence of causal sensitivity, not complete or physically validated counterfactual '
                  'control.',
      'notes': 'Conclusion (60 s). Accurate readout and redundant representations are established on these '
               'clips. All spline results presented use the centroid-difference edit that preserves the source '
               'offset. The fixed linear setting achieves greater pooled readout error reduction, but that '
               'metric alone does not validate downstream steering or preservation of unrelated scene '
               'properties. In the speed-only predictor test, iterated linear edits give 17.91% response versus '
               '5.39% for spline edits. These methods differ in token mapping and position constraints. Avoid '
               'claiming that splines universally fail, that the offset-preserving method guarantees manifold '
               'membership, or that 18% of causal information has been captured. The predictor response is '
               'measured through a frozen position readout, with limited counterfactual validation.',
      'sources': ['vjepa-physics-takehome-4E00/README.md',
                  'results/covariance_velocity_v1/REPORT.md',
                  'results/spline_offset_v1/REPORT.md'],
      'context': ''},
     {'title': 'Spline steering preserves the offset',
      'section': 'APPENDIX A • METHODS',
      'layout': 'figure',
      'takeaway': 'Carry the original offset to the target centroid; requesting no change gives exactly zero '
                  'edit.',
      'notes': 'The diagram illustrates the single spline edit used throughout this presentation. In PCA '
               'coordinates z, add f(target) − f(source): z_new = z + f(target) − f(source). The resulting '
               'offset z_new − f(target) is exactly z − f(source). Lift this difference through the PCA basis to '
               'edit the activation; the orthogonal component is unchanged. The method uses the known source '
               'label, and requesting that same label gives exactly zero edit. In the predictor experiment, the '
               'lifted channel displacement is added to every token. This preserves the centroid-curve offset, '
               'not necessarily all unrelated information or membership in the full activation manifold. The '
               'diagram is illustrative rather than a measured PCA projection.',
      'sources': ['misc/evaluate_pooled_spline_offset.py',
                  'misc/centroid_velocity.py',
                  'results/spline_offset_v1/protocol.json'],
      'context': 'Add the source-to-target centroid difference · known source label · schematic PCA space',
      'figure': 'assets/spline_edit_rules.png'},
     {'title': 'Position-readout limitations',
      'section': 'APPENDIX • EVALUATION LIMITS',
      'layout': 'table',
      'table_headers': ['Unedited baseline / diagnostic', 'Farthest D error', 'What it tells us'],
      'table_rows': [['Predictor → C-trained position ridge', '8.17 px', 'Beats holding the last position'],
                     ['Hold last observed position', '21.26 px', 'Static reference'],
                     ['Measured-prefix extrapolation', '0.78 px', 'Simple physics remains much better']],
      'takeaway': 'An intervention effect is established in the measured pipeline; natural counterfactual '
                  'validity remains open.',
      'notes': 'The three baselines use all 256 D clips. The frozen C-trained position readout applied to '
               'unedited predictions has a farthest-frame error of 8.17 px. This beats holding the last observed '
               'position (21.26 px), but measured-prefix trajectory extrapolation is much better (0.78 px). '
               'Evaluators were not trained on edited examples, so out-of-distribution readout behavior remains '
               'possible. Bootstrap intervals condition on fitted models and omit training-seed uncertainty. No '
               'edited video was generated.',
      'sources': ['results/velocity_causal_v1/REPORT.md', 'results/iterated_velocity_v1/REPORT.md'],
      'context': ''},
     {'title': 'Spline curvature comparison',
      'section': 'APPENDIX',
      'layout': 'figure',
      'figure': 'assets/spline_chords.png',
      'context': 'Offset-preserving edits · interior waypoints · readout error reduction vs no edit',
      'takeaway': 'Spline interpolation gives greater readout error reduction than the straight chord for all '
                  'three variables.',
      'notes': 'Both paths preserve the same source offset, share endpoints and use the same 21-waypoint grid. '
               'Readout error reduction is 100 × (1 − mean edited absolute target error / mean unedited absolute '
               'target error), using the same D clips and requested targets in numerator and denominator. This '
               'is a ratio of aggregate errors, not a mean of per-example percentages. Zero means no '
               'improvement; 100% means the target is reached exactly; negative values mean worse than no edit. '
               'It is not R² and does not measure downstream velocity response. Here the denominator uses the '
               'interior waypoint targets from this experiment, rather than the fixed-target baseline on the '
               'main slide. Spline/chord reductions are 94.29%/83.91% direction, 91.31%/88.20% speed, and '
               '90.68%/89.07% acceleration. The chord is straight interpolation in PCA space, not the '
               'multi-probe linear baseline. Neither path comparison runs the predictor.',
      'sources': ['results/spline_offset_v1/scores.json',
                  'results/spline_offset_v1/protocol.json',
                  'results/spline_offset_v1/readout_error_reduction.json']}]


FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('Deck',FONT));pdfmetrics.registerFont(TTFont('DeckBold',FONT_BOLD))


def wrap(text,size,width,bold=False):
    font='DeckBold' if bold else 'Deck';lines=[]
    for para in text.split('\n'):
        words=para.split();line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if line and pdfmetrics.stringWidth(candidate,font,size)>width:lines.append(line);line=word
            else:line=candidate
        lines.append(line)
    return lines


class Deck:
    def __init__(self):
        self.prs=Presentation();self.prs.slide_width=Inches(13.333333);self.prs.slide_height=Inches(7.5)
        self.pdf=canvas.Canvas(str(HERE/'vjepa_physics_draft.pdf'),pagesize=(W,H))
        self.pdf.setTitle('V-JEPA physics — draft presentation');self.bounds=[];self.text_records=[]
    def box(self,x,y,w,h,fill=LIGHT,line=None,radius=False):
        sh=self.slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
        sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill.lstrip('#'))
        if line:sh.line.color.rgb=RGBColor.from_string(line.lstrip('#'))
        else:sh.line.fill.background()
        self.pdf.setFillColor(fill);self.pdf.setStrokeColor(line or fill)
        self.pdf.rect(x,H-y-h,w,h,stroke=bool(line),fill=1)
        self.bounds.append([self.index,'box',x,y,w,h])
    def txt(self,text,x,y,w,size=20,color=INK,bold=False,maxh=None):
        lines=wrap(text,size,w,bold);leading=size*1.25;height=leading*len(lines)+3
        if maxh is not None and height>maxh:raise ValueError(f'Slide{self.index} text overflow: {text[:65]} {height}>{maxh}')
        tb=self.slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(height))
        tf=tb.text_frame;tf.clear();tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False
        for k,line in enumerate(lines):
            p=tf.paragraphs[0] if k==0 else tf.add_paragraph();p.text=line;p.font.name='DejaVu Sans';p.font.size=Pt(size);p.font.bold=bold
            p.font.color.rgb=RGBColor.from_string(color.lstrip('#'));p.space_before=p.space_after=Pt(0);p.line_spacing=Pt(leading)
        self.pdf.setFont('DeckBold' if bold else 'Deck',size);self.pdf.setFillColor(color)
        for k,line in enumerate(lines):self.pdf.drawString(x,H-y-size-k*leading,line)
        self.bounds.append([self.index,'text',x,y,w,height]);self.text_records.append({'slide':self.index,'text':text,'lines':lines,'box':[x,y,w,height]})
        return height
    def pic(self,path,x,y,w,h):
        path=HERE/path;im=Image.open(path);ratio=min(w/im.width,h/im.height);pw,ph=im.width*ratio,im.height*ratio;px=x+(w-pw)/2;py=y+(h-ph)/2
        self.slide.shapes.add_picture(str(path),Pt(px),Pt(py),width=Pt(pw),height=Pt(ph))
        self.pdf.drawImage(str(path),px,H-py-ph,width=pw,height=ph)
        self.bounds.append([self.index,'image',px,py,pw,ph])
    def table(self,headers,rows,x,y,w,rowh=44,size=17,widths=None):
        widths=widths or [w/len(headers)]*len(headers)
        for i,row in enumerate([headers]+rows):
            xx=x
            for cell,cw in zip(row,widths):
                self.txt(cell,xx+8,y+i*rowh+9,cw-16,size,color=INK,bold=i==0,maxh=rowh-10)
                xx+=cw
            self.box(x,y+(i+1)*rowh-1,w,.7,fill=INK if i==0 else LIGHT)
    def add(self,s,index):
        self.index=index;self.slide=self.prs.slides.add_slide(self.prs.slide_layouts[6]);self.box(0,0,W,H,BG)
        title_size=28
        while len(wrap(s['title'],title_size,880,False))>1 and title_size>22:title_size-=1
        if s['layout']!='cover':self.txt(s['title'],44,29,872,title_size,maxh=50)
        if s.get('context'):self.txt(s['context'],45,77,870,12,color=MUTED,maxh=33)
        layout=s['layout']
        if layout=='cover':
            for text,y,size,color in [(s['title'],210,38,INK),(s['author'],277,22,MUTED)]:
                width=pdfmetrics.stringWidth(text,'Deck',size)
                self.txt(text,(W-width)/2,y,width+4,size,color=color)
        elif layout=='splits':
            self.table(s['table_headers'],s['table_rows'],44,143,872,rowh=54,size=18,widths=[80,422,154,216])
            self.txt('Clips per variable. Follow-up analyses reuse D and are exploratory.',52,448,856,14,color=MUTED)
        elif layout=='argument':
            for text,y in zip(s['paragraphs'],[150,250,380]):
                self.txt(text,45,y,850,23,maxh=90)
        elif layout=='setup':
            yy=125
            for b in s['bullets']:yy+=self.txt(b,45,yy,870,20,maxh=62)+16
            for i,(title,body) in enumerate(s['cards']):
                x=45+i*224;self.txt(title,x,353,202,20);self.txt(body,x,391,202,16,color=MUTED)
            self.txt('Follow-up analyses reuse D and are exploratory.',45,470,865,15,color=MUTED)
        elif layout=='figure':
            self.pic(s['figure'],37,119,886,379)
        elif layout=='steering_table':
            self.table(s['table_headers'],s['table_rows'],44,190,872,rowh=68,size=18,widths=[282,166,184,240])
        elif layout=='comparison':
            self.pic(s['figure'],38,114,884,232)
            self.table(s['table_headers'],s['table_rows'],44,364,872,rowh=41,size=14,widths=[272,180,180,240])
        elif layout=='pipeline':
            self.txt(s['transition'],45,85,870,16,maxh=45)
            for i,label in enumerate(s['steps']):
                x=42+i*180;self.box(x,145,158,80,BG,line='#A0A0A0');self.txt(label,x+12,160,136,18,maxh=65)
                if i<4:self.txt('→',x+161,160,22,21,color=TEAL)
            self.txt(s['token_mapping'],45,252,870,18,maxh=50)
            yy=335
            for b in s['bullets']:yy+=self.txt(b,45,yy,870,20,maxh=65)+17
        elif layout=='conclusions':
            for i,(title,body) in enumerate(s['cards']):
                self.txt(f'{i+1}.',45,132+i*84,40,23)
                self.txt(body,93,132+i*84,795,23,maxh=64)
            yy=420
            for b in s['bullets']:yy+=self.txt(b,45,yy,870,15,color=MUTED,maxh=42)+13
        elif layout=='methods':
            for i,(title,equation,body) in enumerate(s['blocks']):
                x=44+i*449;self.txt(title,x,136,402,22)
                self.txt(equation,x,204,402,24,maxh=100);self.txt(body,x,336,402,18,maxh=130)
        elif layout=='table':
            widths=[374,190,312] if len(s['table_headers'])==3 else None
            if s['title'] in ['Additional spline experiments','Position-readout limitations']:widths=[278,245,353]
            self.table(s['table_headers'],s['table_rows'],42,112,876,rowh=50 if s['title']=='Direction-probe controls' else 56,size=15,widths=widths)
            if s['title']=='Position-readout limitations':self.txt('Baselines: 256 D clips.',44,476,870,12,color=MUTED,maxh=20)
        elif layout=='checklist':
            yy=121
            for title,body in s['blocks']:
                self.txt(title,44,yy,872,19,bold=True);yy+=30
                yy+=self.txt(body,44,yy,872,16,maxh=70)+16
        elif layout=='references':
            for i,(title,body) in enumerate(s['blocks']):
                x=42+i*449;self.txt(title,x,123,429,20,bold=True);self.txt(body,x,163,429,16,maxh=280)
        if layout!='cover':self.txt(str(index),900,515,30,10,color=MUTED)
        self.slide.notes_slide.notes_text_frame.text=s['notes']+'\n\nInterpretation: '+s.get('takeaway','')+'\n\nFigure context: '+s.get('caption','')+'\n\nSources:\n'+'\n'.join(s['sources'])
        self.pdf.showPage()
    def save(self):
        self.prs.save(HERE/'vjepa_physics_draft.pptx');self.pdf.save()
        overflow=[b for b in self.bounds if b[2]<-.1 or b[3]<-.1 or b[2]+b[4]>W+.1 or b[3]+b[5]>H+.1]
        if overflow:raise ValueError(('Off-slide content',overflow))
        (HERE/'layout_checks.json').write_text(json.dumps({'slide_count':len(self.prs.slides),'main_slides':12,'backup_slides':len(self.prs.slides)-12,'off_slide_elements':overflow,'elements_checked':len(self.bounds)},indent=2))


def notes_html(slides):
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>V-JEPA draft — speaker notes</title><style>body{font:17px/1.6 system-ui;max-width:960px;margin:45px auto;color:#172f43}section{border-top:2px solid #087f8c;padding:20px 0}small{color:#526877}li{margin:7px 0}code{font-size:14px}a{color:#087f8c}</style></head><body><h1>V-JEPA physics — draft speaker notes</h1><p>12 main slides for approximately 15 minutes, plus 3 backup slides. Text and diagrams are editable in PowerPoint. Scientific plots are generated from saved results.</p>']
    for i,s in enumerate(slides,1):
        parts.append(f'<section><small>{html.escape(s["section"])}</small><h2>{i}. {html.escape(s["title"])}</h2><p><strong>{html.escape(s["takeaway"])}</strong></p><p>{html.escape(s["notes"])}</p><ul>')
        for source in s['sources']:parts.append(f'<li><code>{html.escape(source)}</code></li>')
        parts.append('</ul></section>')
    parts.append('</body></html>');(HERE/'speaker_notes.html').write_text('\n'.join(parts))


def preview():
    directory=HERE/'previews';directory.mkdir(exist_ok=True);doc=pymupdf.open(HERE/'vjepa_physics_draft.pdf');thumbs=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1,1),alpha=False);p=directory/f'slide_{i+1:02d}.png';pix.save(p)
        im=Image.open(p);im.thumbnail((480,270));tile=Image.new('RGB',(500,300),'#DEE6EB');tile.paste(im,(10,10));ImageDraw.Draw(tile).text((12,282),f'{i+1:02d}',fill=INK);thumbs.append(tile)
    for start,end,name in [(0,12,'main_contact_sheet.jpg'),(12,len(doc),'backup_contact_sheet.jpg')]:
        sheet=Image.new('RGB',(1500,300*((end-start+2)//3)),'white')
        for j,im in enumerate(thumbs[start:end]):sheet.paste(im,((j%3)*500,(j//3)*300))
        sheet.save(directory/name,quality=90)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-content',action='store_true');args=parser.parse_args()
    inputs=make_figures();path=HERE/'slides.json'
    if args.refresh_content or not path.exists():path.write_text(json.dumps(content(*inputs),indent=2,ensure_ascii=False))
    slides=json.loads(path.read_text());assert len(slides)==15
    deck=Deck()
    for i,s in enumerate(slides,1):deck.add(s,i)
    deck.save();notes_html(slides);preview()
    (HERE/'provenance.json').write_text(json.dumps({'sources_sha256':SOURCES,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes':'Charts summarize saved artifacts, including the pooled-PCA offset-preservation rerun. Building the deck does not rerun experiments. PDF and PowerPoint share layout; PDF is not a native PowerPoint rendering.'},indent=2))
    print(f'Built editable PowerPoint, PDF, speaker notes, {len(slides)} slide previews and source provenance.',flush=True)


if __name__=='__main__':main()
