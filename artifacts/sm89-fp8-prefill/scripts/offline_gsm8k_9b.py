"""Full official GSM8K evaluation with a fixed offline request queue."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import vllm
from vllm import LLM
ROOT=Path('/workspace/research/value_reboot_20261027')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['baseline','candidate'],required=True);args=p.parse_args();args.out.mkdir(exist_ok=False)
 manifest=json.loads((ROOT/'prefill_pr_v1/ready.json').read_text())
 core=Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so';sha=hashlib.sha256(core.read_bytes()).hexdigest();assert sha==manifest[args.mode+'_core_sha256']
 spec=importlib.util.spec_from_file_location('official_gsm8k',ROOT/'vllm_fp8/tests/evals/gsm8k/gsm8k_eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
 kwargs=dict(model=str(ROOT.parent.parent/'models/Qwen3.5-9B'),dtype='bfloat16',quantization='fp8_per_tensor',tensor_parallel_size=1,language_model_only=True,
             max_model_len=10240,max_num_seqs=17,max_num_batched_tokens=8192,kv_cache_memory_bytes=6*1024**3,seed=42,
             enable_prefix_caching=False,enable_chunked_prefill=True,async_scheduling=False,generation_config='vllm',
             attention_config=dict(backend='FLASH_ATTN'),compilation_config=dict(max_cudagraph_capture_size=64))
 (args.out/'protocol.json').write_text(json.dumps(dict(core_sha256=sha,kwargs=kwargs,eval='Official offline GSM8K full1319,5shot,chat,thinking false,max512,greedy; fixed request queue and synchronous scheduler',purpose='Hold scheduling constant while comparing model outputs; online timing/quality evaluated separately'),indent=2))
 llm=LLM(**kwargs)
 prompts,labels=ev._build_gsm8k_prompts(1319,5)
 original_chat=llm.chat
 def capture(*a,**kw):
  outputs=original_chat(*a,**kw)
  with (args.out/'responses.jsonl').open('w') as f:
   for index,(o,label) in enumerate(zip(outputs,labels)):
    text=o.outputs[0].text
    f.write(json.dumps(dict(index=index,label=label,answer=ev.get_answer_value(text),text=text,token_ids=list(o.outputs[0].token_ids),finish_reason=o.outputs[0].finish_reason,stop_reason=o.outputs[0].stop_reason),ensure_ascii=False)+'\n')
  return outputs
 llm.chat=capture
 result=ev.evaluate_gsm8k_offline(llm,num_questions=1319,num_shots=5,max_tokens=512,temperature=0,use_chat_completions=True,chat_template_kwargs=dict(enable_thinking=False))
 (args.out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 llm.chat=original_chat
 llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
