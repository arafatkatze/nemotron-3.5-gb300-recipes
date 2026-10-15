"""Build 32 exportable plot variants and a self-contained, offline HTML selector."""
from pathlib import Path
import json,base64,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'design-gallery';ASSETS=OUT/'assets';ASSETS.mkdir(parents=True,exist_ok=True)
D=json.loads((ROOT/'evidence/measurements.json').read_text())
STYLES=[
 dict(name='Electric',note='Cline-inspired · dark / bold / blue',bg='#080b10',ink='#f5f5f3',muted='#a7abb5',accent='#536cf4',secondary='#747e91',grid='#252c38',font='Arial',shape='solid'),
 dict(name='Paper',note='Anthropic-inspired · warm / serif / restrained',bg='#faf9f5',ink='#242520',muted='#77786c',accent='#be6b4e',secondary='#b0b1a4',grid='#deded3',font='Georgia',shape='lollipop'),
 dict(name='Swiss',note='Editorial grid · sharp / black / red',bg='#ffffff',ink='#111111',muted='#717171',accent='#ee3d2c',secondary='#222222',grid='#e4e4e4',font='Arial',shape='vertical'),
 dict(name='Terminal',note='Engineering log · mono / phosphor / precise',bg='#101915',ink='#d7eddc',muted='#8ca892',accent='#94e5ac',secondary='#4f725a',grid='#2b4032',font='DejaVu Sans Mono',shape='terminal'),
 dict(name='Blueprint',note='Technical drawing · cobalt / fine rules',bg='#163b64',ink='#f2f7fc',muted='#adc4db',accent='#99d7fb',secondary='#608fbb',grid='#31577d',font='DejaVu Sans Mono',shape='blueprint'),
 dict(name='Terracotta',note='Magazine spread · soft / oversized / tactile',bg='#f7e9dc',ink='#432e27',muted='#91766a',accent='#ad4b35',secondary='#cba28c',grid='#e3cec0',font='Georgia',shape='warm'),
 dict(name='Signal',note='Minimal studio · sparse / lavender / typographic',bg='#f6f5fb',ink='#262334',muted='#827b95',accent='#7253d7',secondary='#c1b6dd',grid='#e1dceb',font='Arial',shape='signal'),
 dict(name='Newsprint',note='Research journal · monochrome / hatch / serif',bg='#f1f0e9',ink='#232323',muted='#72726b',accent='#292929',secondary='#a6a69f',grid='#d1d1ca',font='Georgia',shape='newsprint')]
ARTIFACTS=[('aggregate','Aggregate output'),('improvement','Matched improvement'),('single','Single-stream decode'),('intelligence','Intelligence × speed')]
first=[x for x in D['reasoning/fp16-sr-mtp1-summary.json'] if x['concurrency']==256];second=D['confirm/confirmation-summary.json']
control=[x for x in D['reasoning/fp32-control-summary.json'] if x['concurrency']==256]
def pooled(rows):return sum(x['completed_tokens'] for x in rows)/sum(x['wall_seconds'] for x in rows)
RATE=[x['aggregate_output_tok_s'] for x in first+second]
REF=[('GLM-5.3-Flash',65,42),('Qwen3.8-Flash-Next',62,40),('MiMo-V2.6-Flash',118,38),('DeepSeek V4.1 Flash',223,39.5),('Qwen3.8 27B',77,34),('MiniMax-M3',100,29.2),('Hy3',82,25.3),('Inkling Small',124,25.6),('Nemotron 3 Ultra',197,23),('Ling 3.0 Flash',311,20),('Muse Glimmer',118,17.4),('Qwen3.5 122B',113,15.5),('Nemotron 3 Super',139,12.8),('gpt-oss-120b',160,11.6),('Mistral Small 4',177,11.2)]
CAPTIONS={
 'aggregate':'Six runs, two independently started serving processes, one DGX Station GB300. C256; 512 input / 512 output tokens; NVFP4 weights; FP8 KV; 16K configured context; reasoning included. Aggregate streaming HTTP throughput includes prefill/client overhead; warmup excluded.',
 'improvement':'Matched C256 comparison. Pooled total output / elapsed time over three initial-instance runs per arm: FP32 recurrent state 9,711.96 tok/s; FP16 state plus stochastic rounding 11,815.99 tok/s (+21.664%). Same model, reasoning, runtime, MTP1 and workload. This isolates the combined state/rounding intervention, not each component separately.',
 'single':'Earlier Triton/MTP3: 540.65 decode tok/s. Latest MTP1: 390.29 and 397.65. Each probe: 8,192 input / 128 output tokens. Earlier quality suite:20 checks; latest:23 cases repeated3times. These are different configurations. Latest bar uses pooled common-decode time; the label retains the observed range.',
 'intelligence':'Editorial reconstruction: gray reference coordinates and intelligence scores were visually estimated from the supplied chart, not an exact downloaded dataset. Reference speeds use100K input; our measured540.65tok/s uses8K. The highlighted intelligence score is inherited from that reference, not measured on our local configuration. Not a matched speed/quality comparison.'}
plt.rcParams['svg.fonttype']='none'

def setup(s,title,sub,big=None):
 plt.rcParams['font.family']=[s['font'],'DejaVu Sans']
 fig=plt.figure(figsize=(12,6.75),dpi=160,facecolor=s['bg'])
 title_size=30 if s['font']!='DejaVu Sans Mono' else 27
 if s['shape']=='swiss':title_size=34
 fig.text(.06,.88,title,color=s['ink'],fontsize=title_size,weight='bold' if s['font']=='Arial' else 'normal')
 fig.text(.062,.82,sub,color=s['muted'],fontsize=15)
 if s['shape']=='newsprint':
  fig.add_artist(plt.Line2D([.06,.94],[.94,.94],color=s['ink'],lw=1.2,transform=fig.transFigure))
  fig.add_artist(plt.Line2D([.06,.94],[.77,.77],color=s['ink'],lw=.6,transform=fig.transFigure))
 if s['shape']=='blueprint':
  fig.add_artist(plt.Line2D([.04,.96],[.95,.95],color=s['grid'],lw=.7,transform=fig.transFigure))
 return fig

def axes(fig,s,pos=(.11,.18,.83,.55),direction='y'):
 ax=fig.add_axes(pos,facecolor=s['bg']);ax.set_axisbelow(True)
 for k in ['top','right','left']:ax.spines[k].set_visible(False)
 ax.spines['bottom'].set_color(s['grid']);ax.tick_params(length=0,pad=12,labelsize=14,colors=s['muted'])
 ax.grid(axis=direction,color=s['grid'],lw=.7)
 if s['shape']=='blueprint':ax.grid(axis='both',color=s['grid'],lw=.7)
 return ax

def export(fig,slug):
 for ext in ['png','svg']:
  f=ASSETS/f'{slug}.{ext}';fig.savefig(f,facecolor=fig.get_facecolor(),metadata={'Creator':'Nemotron recipe design gallery'})
  if ext=='svg':f.write_text('\n'.join(x.rstrip() for x in f.read_text().splitlines())+'\n')
 plt.close(fig)

items=[]
for si,s in enumerate(STYLES,1):
 for key,title in ARTIFACTS:
  slug=f'{si:02d}-{s["name"].lower()}-{key}'
  if key=='aggregate':
   fig=setup(s,'11,840 tok/s','Aggregate output · 256 concurrent streams')
   ax=axes(fig,s);ax.set_ylim(0,14500);ax.set_xlim(-.65,5.65)
   colors=[s['accent']]*3+[s['secondary']]*3
   if s['shape'] in ['lollipop','signal']:
    ax.vlines(range(6),0,RATE,color=colors,linewidth=2 if s['shape']=='lollipop' else 5)
    ax.scatter(range(6),RATE,s=180 if s['shape']=='lollipop' else 240,color=colors,zorder=3)
   else:
    bars=ax.bar(range(6),RATE,color=colors,width=.58 if s['shape']!='terminal' else .76)
    if s['shape']=='newsprint':
     for b in bars[3:]:b.set_hatch('///');b.set_edgecolor(s['ink']);b.set_linewidth(0)
   for i,v in enumerate(RATE):ax.text(i,v+450,f'{v:,.0f}',color=s['ink'],ha='center',fontsize=16,weight='bold' if s['font']=='Arial' else 'normal')
   ax.set_xticks([1,4],['Server 1','Server 2']);ax.set_yticks([0,6000,12000],['0','6K','12K'])
  elif key in ['improvement','single']:
   if key=='improvement':
    fig=setup(s,'21.7% more output','Matched throughput · 256 concurrent streams')
    vals=[pooled(control),pooled(first)];labels=['FP32 state','FP16 + rounding'];texts=[f'{v:,.0f}' for v in vals];maximum=13000
   else:
    fig=setup(s,'Single-stream decode','Two configurations')
    old=D['controls/triton-mtp3-8k-summary.json']['common_decode_tok_s'];r=[D['reasoning/fp16-sr-mtp1-8k-summary.json']['common_decode_tok_s'],D['confirm/confirmation-8k-summary.json']['common_decode_tok_s']]
    vals=[old,2/sum(1/v for v in r)];labels=['MTP3 · earlier','MTP1 · latest'];texts=['541','390–398'];maximum=600
   colors=[s['secondary'],s['accent']] if key=='improvement' else [s['accent'],s['secondary']]
   if s['shape'] in ['vertical','newsprint','signal']:
    ax=axes(fig,s,pos=(.15,.18,.75,.53));ax.set_ylim(0,maximum*1.18);ax.set_xlim(-.7,1.7)
    if s['shape']=='signal':ax.vlines([0,1],0,vals,colors=colors,lw=3);ax.scatter([0,1],vals,color=colors,s=220)
    else:
     bars=ax.bar([0,1],vals,color=colors,width=.55)
     if s['shape']=='newsprint':bars[0].set_hatch('///');bars[0].set_edgecolor(s['ink']);bars[0].set_linewidth(0)
    for i,v in enumerate(vals):ax.text(i,v+maximum*.045,texts[i],ha='center',fontsize=25,color=s['ink'],weight='bold' if s['font']=='Arial' else 'normal')
    ax.set_xticks([0,1],labels);ax.set_yticks([0,maximum],['0',f'{maximum:,} tok/s'])
   else:
    ax=axes(fig,s,pos=(.29,.20,.65,.48),direction='x');ax.grid(False);ax.set_xlim(0,maximum*1.02);ax.set_ylim(-.65,1.65)
    if s['shape'] in ['lollipop','blueprint']:
     ax.hlines([1,0],0,vals,color=colors,lw=2);ax.scatter(vals,[1,0],color=colors,s=220,zorder=3)
     for y,v,text in zip([1,0],vals,texts):ax.text(v-maximum*.035,y+.22,text,ha='right',color=s['ink'],fontsize=24)
    else:
     ax.barh([1,0],vals,color=colors,height=.42)
     for y,v,text in zip([1,0],vals,texts):ax.text(v-maximum*.035,y,text,ha='right',va='center',color=s['bg'] if s['shape']=='terminal' else ('#ffffff' if s['shape'] in ['solid','warm'] else s['ink']),fontsize=24,weight='bold')
    ax.set_yticks([1,0],labels);ax.tick_params(axis='y',labelsize=19,labelcolor=s['ink']);ax.set_xticks([0,maximum],['0',f'{maximum:,} tok/s'])
  else:
   fig=setup(s,'Intelligence × speed','Open-weight models · local run highlighted')
   ax=axes(fig,s,pos=(.10,.16,.83,.56),direction='both');ax.set_xlim(0,650);ax.set_ylim(5,47)
   ax.scatter([r[1] for r in REF],[r[2] for r in REF],color=s['secondary'],s=65,zorder=3)
   offsets={'GLM-5.3-Flash':(6,8),'DeepSeek V4.1 Flash':(8,8),'MiniMax-M3':(8,5),'Ling 3.0 Flash':(8,-13),'Nemotron 3 Ultra':(8,-7)}
   for name,x,y in REF:
    if name in offsets:ax.annotate(name,(x,y),xytext=offsets[name],textcoords='offset points',fontsize=11.5,color=s['ink'])
   ax.scatter([540.6509512165002],[26.5],s=200,facecolors='none',edgecolors=s['accent'],lw=2.5,zorder=4)
   ax.annotate('Nemotron 3.5 Super VL\n541 tok/s · local 8K',(540.6509512165002,26.5),xytext=(-8,22),textcoords='offset points',ha='right',color=s['accent'],fontsize=16,weight='bold')
   ax.set_xticks([0,200,400,600]);ax.set_yticks([10,20,30,40]);ax.set_xlabel('Output tokens / second',color=s['muted'],fontsize=13,labelpad=12);ax.set_ylabel('Intelligence index',color=s['muted'],fontsize=13,labelpad=12)
  export(fig,slug)
  svg=(ASSETS/f'{slug}.svg').read_bytes();items.append(dict(style=si,key=key,title=title,slug=slug,svg=base64.b64encode(svg).decode(),caption=CAPTIONS[key]))
(OUT/'gallery-data.json').write_text(json.dumps({'styles':STYLES,'artifacts':ARTIFACTS,'items':items},indent=2))
print('Created',len(items),'figures × PNG/SVG; gallery data ready.')

# Keep both direct and root entry points self-contained.
template=(OUT/'template.html').read_text()
payload=json.loads((OUT/'gallery-data.json').read_text())
page=template.replace('__GALLERY_DATA__',json.dumps(payload,separators=(',',':')))
(OUT/'index.html').write_text(page)
(ROOT/'index.html').write_text(page)
