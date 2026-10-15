"""Rebuild standalone blog figures from the inspected measurement snapshot."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).parent
D = json.loads((ROOT / 'evidence' / 'measurements.json').read_text())
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID, ACCENT = '#151515', '#666666', '#e5e5e5', '#a18b36'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'text.color': INK, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': INK,
                     'svg.fonttype': 'none', 'figure.facecolor': 'white',
                     'axes.facecolor': 'white'})

def base(big, subtitle, note):
    fig = plt.figure(figsize=(12, 7.5), dpi=160)
    fig.text(.075, .89, big, fontsize=35, weight='bold')
    fig.text(.077, .83, subtitle, fontsize=14, color=MUTED)
    fig.text(.077, .055, note, fontsize=10.5, color=MUTED, linespacing=1.6)
    ax = fig.add_axes([.12, .25, .81, .49])
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both', length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=.7)
    return fig, ax

def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', facecolor='white')
        if ext == 'svg':
            target = OUT / f'{name}.{ext}'
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)

first = [x for x in D['reasoning/fp16-sr-mtp1-summary.json'] if x['concurrency'] == 256]
second = D['confirm/confirmation-summary.json']
six = first + second
values = [x['aggregate_output_tok_s'] for x in six]
fig, ax = base('11,840 tok/s', 'Aggregate HTTP output · 256 streams · one DGX Station GB300',
    'Six runs across two fresh instances · 512 input / 512 output tokens per request · reasoning enabled\n'
    'Our weights: NVFP4. Actual headline: 10,500 tok/s with FP8; exact workload undisclosed. Not a matched comparison.')
ax.bar(range(6), values, width=.64, color=[INK]*3+[ACCENT]*3)
ax.axhline(10500, color=MUTED, linestyle=(0, (5, 4)), linewidth=1.2)
ax.text(5.45, 10100, 'Actual reported headline: 10,500', ha='right', va='top', fontsize=11, color=MUTED,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 3})
for i, y in enumerate(values): ax.text(i, y + 260, f'{y:,.0f}', ha='center', fontsize=12, weight='bold')
ax.set_xticks(range(6), ['Instance 1\nrun 1', 'Instance 1\nrun 2', 'Instance 1\nrun 3',
                         'Instance 2\nrun 1', 'Instance 2\nrun 2', 'Instance 2\nrun 3'])
ax.set_ylim(0, 14000)
ax.set_yticks([0, 4000, 8000, 12000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.set_ylabel('Output tokens / second', labelpad=12)
save(fig, '01-six-runs')

control = D['reasoning/fp32-control-summary.json']
candidate = D['reasoning/fp16-sr-mtp1-summary.json']
def pooled(rows):
    return sum(x['completed_tokens'] for x in rows)/sum(x['wall_seconds'] for x in rows)
fig, ax = base('21.7% more throughput', 'Matched reasoning-enabled comparison · aggregate HTTP output',
    'Each bar pools three runs: total completed output tokens / total elapsed wall time. Error bars show run min–max.\n'
    'NVFP4 weights · FP8 KV · MTP1 · FlashInfer · 512 input / 512 output · FP16 state also enables stochastic rounding.')
for j, (rows, label, color) in enumerate([(control, 'FP32 recurrent state', INK), (candidate, 'FP16 state + stochastic rounding', ACCENT)]):
    for i, c in enumerate([128, 256]):
        r = [x for x in rows if x['concurrency'] == c]
        v = pooled(r)
        lo, hi = min(x['aggregate_output_tok_s'] for x in r), max(x['aggregate_output_tok_s'] for x in r)
        x = i + (j-.5)*.34
        ax.bar(x, v, width=.29, color=color, label=label if i == 0 else None)
        ax.errorbar(x, v, yerr=[[v-lo], [hi-v]], color=INK, capsize=4, linewidth=1.1)
        ax.text(x, hi+350, f'{v:,.0f}', ha='center', fontsize=12, weight='bold')
ax.set_xticks([0, 1], ['128 concurrent streams', '256 concurrent streams'])
ax.set_ylim(0, 15000)
ax.set_yticks([0, 4000, 8000, 12000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.set_ylabel('Output tokens / second', labelpad=12)
ax.legend(frameon=False, loc='upper left', fontsize=11)
save(fig, '02-matched-state-comparison')

old = D['controls/triton-mtp3-8k-summary.json']['common_decode_tok_s']
new = D['reasoning/fp16-sr-mtp1-8k-summary.json']['common_decode_tok_s']
fresh = D['confirm/confirmation-8k-summary.json']['common_decode_tok_s']
fig, ax = base('541 tok/s', 'Earlier single-stream peak · separate configuration from the 11.8K aggregate winner',
    'Our probes: 8,192 input / 128 output tokens · common-decode metric · one probe per instance.\n'
    'Earlier MTP3 passed a smaller 20-check suite. Latest MTP1: 23 distinct cases × 3 repeats. Actual workload undisclosed.')
ax.grid(False)
ax.grid(axis='x', color=GRID, linewidth=.7)
vals = [old, new, fresh]
labels = ['Earlier Triton + MTP3', 'Latest MTP1 · instance 1', 'Latest MTP1 · instance 2']
ax.barh([2, 1, 0], vals, height=.52, color=[INK, ACCENT, ACCENT])
for y, v in zip([2, 1, 0], vals):ax.text(v+10, y, f'{v:.1f}', va='center', weight='bold')
ax.set_yticks([2, 1, 0], labels)
ax.set_xlim(0, 650)
ax.set_xticks([0, 200, 400, 600])
ax.axvline(455, color=MUTED, linestyle=(0, (5, 4)), linewidth=1.2)
ax.text(.98, 1.04, 'Dashed line: Actual claim of 455', transform=ax.transAxes,
        ha='right', fontsize=11, color=MUTED)
ax.set_xlabel('Single-stream decode tokens / second', labelpad=12)
ax.set_position([.32, .25, .61, .49])
save(fig, '03-single-stream-configurations')

metrics = {'six_run_min': min(values), 'six_run_max': max(values),
           'six_run_pooled_output_tok_s': pooled(six),
           'matched_control_c256': pooled([x for x in control if x['concurrency']==256]),
           'matched_candidate_c256': pooled(first)}
metrics['matched_c256_gain_percent'] = (metrics['matched_candidate_c256']/metrics['matched_control_c256']-1)*100
(ROOT/'evidence'/'derived-metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics, indent=2))
