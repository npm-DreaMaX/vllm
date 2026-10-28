import json
import sys
from pathlib import Path

root=Path(sys.argv[1])
rows=[]
for name in ['prefill8k_o16_c1','long8k_c1','long8k_c4','long8k_c17','medium2k_c4','decode_control_c17']:
    paths=[root/f'{i}_{mode}_gpu0_tp1/{name}.json' for i,mode in enumerate(['native','candidate'])]
    if not all(p.exists() for p in paths):continue
    a,b=[json.loads(p.read_text()) for p in paths]
    assert a['failed']==b['failed']==0
    metrics=['output_throughput','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_tpot_ms']
    row=dict(workload=name,baseline={k:a[k] for k in metrics},candidate={k:b[k] for k in metrics},
             change_pct={k:100*(b[k]/a[k]-1) for k in metrics})
    rows.append(row)
    print(name,{k:round(v,2) for k,v in row['change_pct'].items()})
(root/'analysis.json').write_text(json.dumps({'scope':'Decision screen, one run per arm/workload; not final repeat evidence','rows':rows},indent=2)+'\n')
