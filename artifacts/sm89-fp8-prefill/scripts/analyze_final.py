"""Analyze crossover runs; all repeats are nested within two starts per arm."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); assert (root/'COMPLETE.json').exists()
assert not list(root.glob('INVALID*')) and not (root/'FAILED.json').exists()
metrics=['output_throughput','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_tpot_ms','p95_itl_ms','mean_e2el_ms','gpu_joules_per_output_token']
names=['prefill8k_o16_c1','long8k_c1','long8k_c4','long8k_c17','medium2k_c4','decode_control_c17','long8k_o1024_c4']
rows=[];requests=0;measurements=0
for name in names:
    arms={};cards={}
    for mode in ['native','candidate']:
        records=[]
        for folder in sorted(root.glob(f'*_{mode}_gpu*_tp1')):
            gpu=int(folder.name.split('_gpu')[1][0]);bycard=[]
            paths=[folder/(name+'.json')] if name=='long8k_o1024_c4' else sorted(folder.glob(name+'_r[123].json'))
            assert len(paths)==(1 if name=='long8k_o1024_c4' else 3)
            for path in paths:
                d=json.loads(path.read_text());assert d['failed']==0 and d['completed']==d['num_prompts']
                requests+=d['completed'];measurements+=1
                e=json.loads(path.with_name(path.stem+'.energy_validated.json').read_text())
                d['gpu_joules_per_output_token']=e['validated_joules_per_output_token']
                bycard.append({k:d[k] for k in metrics})
            records+=bycard;cards.setdefault(gpu,{})[mode]={k:statistics.median(x[k] for x in bycard) for k in metrics}
        arms[mode]={k:dict(median=statistics.median(x[k] for x in records),min=min(x[k] for x in records),max=max(x[k] for x in records),samples=[x[k] for x in records]) for k in metrics}
    changes={k:100*(arms['candidate'][k]['median']/arms['native'][k]['median']-1) for k in metrics}
    percard={g:{k:100*(v['candidate'][k]/v['native'][k]-1) for k in metrics} for g,v in cards.items()}
    rows.append(dict(workload=name,arms=arms,change_pct=changes,per_gpu_change_pct=percard))
    print(name, {k:round(changes[k],2) for k in ['output_throughput','mean_ttft_ms','mean_tpot_ms','gpu_joules_per_output_token']})
result=dict(requests=requests,measurements=measurements,scope='Two starts per arm, one per card; three nested repeats per start except long-output control (one). Median/min/max across six runs (control: two). No independence or confidence-interval claim.',rows=rows)
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
assert requests==1520 and measurements==76
