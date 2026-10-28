import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'原始证据/prefill_pr_v1';OUT=ROOT/'图表';OUT.mkdir(exist_ok=True)
x=json.loads((RAW/'serve_9b_final_v1/analysis.json').read_text())
labels=['8K / 16 / C1','8K / 128 / C1','8K / 128 / C4','8K / 128 / C17','2K / 128 / C4','256 / 512 / C17','8K / 1024 / C4']
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for ax,key,title in zip(axes,['mean_ttft_ms','output_throughput'],['Mean time to first token (lower is better)','Output throughput (higher is better)']):
 for mode,shift,color,label in [('native',-.16,'#8594a4','Baseline'),('candidate',.16,'#227c9d','Candidate')]:
  med=np.array([r['arms'][mode][key]['median']/r['arms']['native'][key]['median']*100 for r in x['rows']])
  lo=np.array([r['arms'][mode][key]['min']/r['arms']['native'][key]['median']*100 for r in x['rows']])
  hi=np.array([r['arms'][mode][key]['max']/r['arms']['native'][key]['median']*100 for r in x['rows']])
  ax.errorbar(med,np.arange(len(labels))+shift,xerr=[med-lo,hi-med],fmt='o',capsize=3,color=color,label=label)
 ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(100,color='#bbb',ls='--');ax.set_xlabel('Baseline median = 100; whiskers = observed min/max');ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
axes[0].legend(loc='lower left');fig.suptitle('Qwen3.5-9B FP8 on RTX4090 — native card-swapped comparison',fontsize=13)
fig.supxlabel('Input / output / concurrency. Six runs per arm from two starts; last control: two runs. Whiskers are not confidence intervals.',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'serving_results.{ext}',dpi=180)
plt.close(fig)
a=json.loads((RAW/'qualification/baseline/timings.json').read_text());b=json.loads((RAW/'qualification/candidate/timings.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
lab=[f"{r['M']} x {r['N']} x {r['K']}" for r in a];positions=np.arange(len(a))
for data,shift,c,label in [(a,-.17,'#8594a4','Baseline'),(b,.17,'#227c9d','Candidate')]:
 med=np.array([r['median_us'] for r in data]);lo=np.array([min(r['samples_us']) for r in data]);hi=np.array([max(r['samples_us']) for r in data])
 axes[0].barh(positions+shift,med,height=.3,color=c,label=label,xerr=[med-lo,hi-med],capsize=2)
axes[0].set_yticks(positions,lab);axes[0].invert_yaxis();axes[0].set_xlabel('Microseconds; median and range of 3 samples');axes[0].set_title('Native GEMM, >=1 GiB rotating weights');axes[0].legend()
axes[1].bar(['5 stages','3 stages'],[80,48],color=['#8594a4','#227c9d']);axes[1].set_ylabel('Shared memory per CTA (KiB)');axes[1].set_ylim(0,95);axes[1].set_title('128 x 128 x 64 tile launch resources')
for i,v in enumerate([80,48]):axes[1].text(i,v+2,str(v),ha='center')
fig.suptitle('Same tile and accumulation operator; reduced pipeline storage')
fig.supxlabel('Shared-memory resource reduction permits higher theoretical residency; achieved occupancy was not measured.',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'kernel_results.{ext}',dpi=180)
plt.close(fig)
# Separate instrumented timeline: do not use these durations for speedup claims.
fig,axes=plt.subplots(2,1,figsize=(10,3.6),layout='constrained')
for ax,mode in zip(axes,['baseline','candidate']):
 t=json.loads((RAW/f'native_profile/{mode}/launch.trace.json').read_text())
 events=[e for e in t['traceEvents'] if e.get('cat')=='kernel'];start=min(e['ts'] for e in events)
 ax.broken_barh([((e['ts']-start)/1000,e['dur']/1000) for e in events],(0,1),facecolors='#8594a4' if mode=='baseline' else '#227c9d')
 ax.set_yticks([.5],[mode]);ax.set_ylim(-.2,1.2);ax.set_xlabel('Milliseconds from first GPU launch (separate process per arm)')
fig.suptitle('Native prefill GEMM trace: M8192 / N24576 / K4096')
fig.supxlabel('Three instrumented launches; timing includes profiler effects. Use unprofiled microbenchmarks for performance.',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'native_launch_timeline.{ext}',dpi=180)
