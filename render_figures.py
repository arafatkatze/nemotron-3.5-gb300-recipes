"""Minimal blog figures. Full conditions live in captions.md, outside the images."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).parent
D=json.loads((ROOT/'evidence/measurements.json').read_text())
THEMES={
 'dark':dict(bg='#080b10',ink='#f5f5f3',muted='#a7abb5',track='#191e28',accent='#536cf4',control='#737b8d',rule='#303745'),
 'light':dict(bg='#faf9f5',ink='#181815',muted='#73736a',track='#eae8e1',accent='#c66f50',control='#a3a69a',rule='#d8d7cf')}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'svg.fonttype':'none'})

def canvas(title,theme,sub=None):
 t=THEMES[theme];fig=plt.figure(figsize=(12,6.75),dpi=160,facecolor=t['bg'])
 ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1200);ax.set_ylim(675,0);ax.axis('off')
 ax.text(62,76,title,fontsize=31,weight='bold',color=t['ink'],va='center')
 if sub:ax.text(64,123,sub,fontsize=16,color=t['muted'],va='center')
 return fig,ax,t

def bar(ax,x,y,w,h,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',facecolor=color,edgecolor='none'))

def export(fig,name,theme):
 folder=ROOT/'figures' if theme=='dark' else ROOT/'figures/light'
 folder.mkdir(parents=True,exist_ok=True)
 for ext in ['png','svg']:
  f=folder/f'{name}.{ext}';fig.savefig(f,dpi=160,facecolor=THEMES[theme]['bg'])
  if ext=='svg':f.write_text('\n'.join(s.rstrip() for s in f.read_text().splitlines())+'\n')
 plt.close(fig)

def pooled(rows):return sum(r['completed_tokens'] for r in rows)/sum(r['wall_seconds'] for r in rows)
first=[x for x in D['reasoning/fp16-sr-mtp1-summary.json'] if x['concurrency']==256]
second=D['confirm/confirmation-summary.json']
control=[x for x in D['reasoning/fp32-control-summary.json'] if x['concurrency']==256]
values=[x['aggregate_output_tok_s'] for x in first+second]

for theme in THEMES:
 fig,ax,t=canvas('Aggregate output',theme,'256 streams · one DGX Station GB300')
 ax.text(60,305,'11,840',fontsize=64,weight='bold',color=t['ink'],va='center')
 ax.text(66,370,'tokens / second',fontsize=22,color=t['muted'],va='center')
 ax.text(66,498,'6 runs',fontsize=21,color=t['ink'],weight='bold')
 ax.text(66,537,'2 fresh servers',fontsize=18,color=t['muted'])
 ax.plot([460,460],[205,555],color=t['rule'],linewidth=.8)
 x0,step,bw,base,scale=530,97,58,540,325/12500
 for i,v in enumerate(values):
  x=x0+i*step;h=v*scale
  bar(ax,x,base-h,bw,h,t['accent'] if i<3 else t['control'])
  ax.text(x+bw/2,base-h-20,f'{v:,.0f}',fontsize=14.5,ha='center',color=t['ink'],weight='medium')
 ax.plot([515,1140],[base,base],color=t['rule'],linewidth=.8)
 ax.text(655,590,'Server 1',ha='center',fontsize=17,color=t['muted'])
 ax.text(946,590,'Server 2',ha='center',fontsize=17,color=t['muted'])
 export(fig,'01-six-runs',theme)

 fig,ax,t=canvas('21.7% more output',theme,'256 concurrent streams')
 top,bottom=245,420;x,w,h,maxv=337,790,68,12500
 cv,nv=pooled(control),pooled(first)
 for y,label,value,color in [(top,'FP32 state',cv,t['control']),(bottom,'FP16 + rounding',nv,t['accent'])]:
  ax.text(62,y+h/2,label,fontsize=21,color=t['ink'],va='center')
  bar(ax,x,y,w,h,t['track']);bar(ax,x,y,w*value/maxv,h,color)
  ax.text(x+w*value/maxv-22,y+h/2,f'{value:,.0f}',fontsize=23,weight='bold',color='#ffffff' if theme=='dark' else '#181815',ha='right',va='center')
 ax.plot([x,x+w],[568,568],color=t['rule'],linewidth=.8)
 ax.text(x,601,'0',fontsize=15,color=t['muted'],ha='center')
 ax.text(x+w,601,'12,500 tok/s',fontsize=15,color=t['muted'],ha='right')
 export(fig,'02-matched-state-comparison',theme)

 fig,ax,t=canvas('Single-stream decode',theme,'Two configurations')
 old=D['controls/triton-mtp3-8k-summary.json']['common_decode_tok_s']
 recent=[D['reasoning/fp16-sr-mtp1-8k-summary.json']['common_decode_tok_s'],D['confirm/confirmation-8k-summary.json']['common_decode_tok_s']]
 center=2/sum(1/v for v in recent)
 x,w,h,maxv=337,790,68,600
 for y,label,value,value_label,color in [(245,'MTP3 · earlier',old,'541',t['accent']),(420,'MTP1 · latest',center,'390–398',t['control'])]:
  ax.text(62,y+h/2,label,fontsize=21,color=t['ink'],va='center')
  bar(ax,x,y,w,h,t['track']);bar(ax,x,y,w*value/maxv,h,color)
  ax.text(x+w*value/maxv-22,y+h/2,value_label,fontsize=23,weight='bold',color='#ffffff' if theme=='dark' else '#181815',ha='right',va='center')
 ax.plot([x,x+w],[568,568],color=t['rule'],linewidth=.8)
 ax.text(x,601,'0',fontsize=15,color=t['muted'],ha='center')
 ax.text(x+w,601,'600 tok/s',fontsize=15,color=t['muted'],ha='right')
 export(fig,'03-single-stream-configurations',theme)

metrics={'six_run_min':min(values),'six_run_max':max(values),'six_run_pooled_output_tok_s':pooled(first+second),'matched_control_c256':pooled(control),'matched_candidate_c256':pooled(first)}
metrics['matched_c256_gain_percent']=(pooled(first)/pooled(control)-1)*100
(ROOT/'evidence/derived-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print('Rendered3figures in dark and light themes;PNG andSVG for each.')
