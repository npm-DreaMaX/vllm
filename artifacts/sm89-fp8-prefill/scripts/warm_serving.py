"""Sustained HTTP warmup, avoiding benchmark CLI import gaps between batches."""
import json,time

def warm(server,h,concurrency,seconds=45):
 body=dict(model=h.MODEL_ID,prompt=[[42]*256 for _ in range(concurrency)],
           max_tokens=512,temperature=0,ignore_eos=True,stream=False)
 start=time.monotonic();rows=[];h.PHASE=f'{server["mode"]}_sustained_warmup'
 while time.monotonic()-start<seconds:
  tick=time.monotonic()
  response=json.loads(h.request(server['port'],'/v1/completions',body))
  assert len(response['choices'])==concurrency,response.keys()
  assert response['usage']['completion_tokens']==512*concurrency,response['usage']
  rows.append(dict(seconds=time.monotonic()-tick,usage=response['usage']))
 result=dict(minimum_wall_seconds=seconds,elapsed_seconds=time.monotonic()-start,
             concurrency=concurrency,prompt_tokens=256,output_tokens=512,batches=rows)
 (server['folder']/'sustained_warmup.json').write_text(json.dumps(result,indent=2))
 h.event('sustained_warmup_complete',mode=server['mode'],elapsed_seconds=result['elapsed_seconds'],batches=len(rows))
