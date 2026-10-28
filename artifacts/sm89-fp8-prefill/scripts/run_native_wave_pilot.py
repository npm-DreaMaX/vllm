"""Real pinned-main native-library A/B; a decision screen before full qualification."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT=Path('/workspace/research/value_reboot_20261027')
sys.path.insert(0,str(ROOT.parent/'mlp_redesign_20261027'))
import run_context_final_ab as h

h.OUT=ROOT/'native_wave_08b_pilot_v1'
h.MODEL=h.ROOT/'models/Qwen3.5-0.8B'
h.MODEL_ID='Qwen/Qwen3.5-0.8B'
h.MODEL_REVISION='2fc06364715b967f1860aea9cf38778875588b17'


def start(gpu,mode,rd):
    manifest=json.loads((ROOT/'wave_native_ready.json').read_text())
    runtime=ROOT/('native_baseline_runtime' if mode=='native' else 'vllm_fp8')
    sha=manifest['baseline_core_sha256' if mode=='native' else 'candidate_core_sha256']
    folder=h.OUT/f'round{rd}_gpu{gpu}_{mode}';folder.mkdir()
    env=os.environ.copy()
    for key in ['ENGINE_OPT_MODE','ADA_SKINNY_MODE','FP8_WAVE_MODE']:env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(runtime),
        RESEARCH_RUNTIME_ROOT=str(runtime),RESEARCH_CORE_SHA256=sha,
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
        VLLM_USE_RUST_FRONTEND='0',VLLM_USE_RUST_BENCH='0',VLLM_USE_V2_MODEL_RUNNER='1',
        VLLM_GPU_SYNC_CHECK='error',VLLM_CACHE_ROOT=str(ROOT/'native_wave_cache'/mode/'vllm'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'native_wave_cache'/mode/'inductor'))
    trace=folder/'trace';trace.mkdir()
    profiler={'profiler':'torch','torch_profiler_dir':str(trace),'torch_profiler_with_stack':False,
        'torch_profiler_record_shapes':True,'ignore_frontend':True,
        'detailed_trace_annotation':True,'delay_iterations':16,'max_iterations':8}
    port=18610+gpu
    args=['taskset','-c','8-15' if gpu==0 else '16-23',h.PY,str(ROOT/'native_main_cli.py'),
        'serve',str(h.MODEL),'--served-model-name',h.MODEL_ID,'--host','127.0.0.1','--port',str(port),
        '--language-model-only','--quantization','fp8_per_tensor','--dtype','bfloat16',
        '--tensor-parallel-size','1','--max-model-len','10240','--max-num-seqs','128',
        '--max-num-batched-tokens','8192','--gpu-memory-utilization','0.9',
        '--kv-cache-memory-bytes',str(12*1024**3),'--seed','42','--no-enable-prefix-caching',
        '--enable-chunked-prefill','--generation-config','vllm','--attention-backend','FLASH_ATTN',
        '--profiler-config',json.dumps(profiler),'--compilation-config',
        json.dumps({'cudagraph_capture_sizes':[1,2,4,8,16,24,32,48,64,96,128]})]
    (folder/'server_command.json').write_text(json.dumps({'argv':args,'environment':
        {k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','RESEARCH_RUNTIME_ROOT','RESEARCH_CORE_SHA256','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR']},
        'main_pin':'88d3995ea18de47ba5c306789aa615474ad74592','model_revision':h.MODEL_REVISION,
        'scope':'Actual current-main source-built native A/B; single-repeat swapped-card pilot, not final repeat qualification.'},indent=2)+'\n')
    log=(folder/'server.log').open('w')
    process=subprocess.Popen(args,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(gpu=gpu,mode=mode,folder=folder,port=port,process=process,log=log,source=runtime,environment=env)


def bench(s,name,count,input_len,output_len,concurrency,profile=False):
    folder=s['folder']
    args=[h.PY,str(ROOT/'bench_energy_cli.py'),'bench','serve','--backend','vllm',
        '--base-url',f'http://127.0.0.1:{s["port"]}','--endpoint','/v1/completions',
        '--model',h.MODEL_ID,'--tokenizer',str(h.MODEL),'--dataset-name','random',
        '--random-input-len',str(input_len),'--random-output-len',str(output_len),
        '--random-range-ratio','0','--ignore-eos','--temperature','0','--request-rate','inf',
        '--seed','42','--num-prompts',str(count),'--num-warmups','0','--max-concurrency',str(concurrency),
        '--ready-check-timeout-sec','0','--probe-request-rate','0',
        '--percentile-metrics','ttft,tpot,itl,e2el','--metric-percentiles','50,95,99',
        '--save-result','--save-detailed','--result-dir',str(folder),
        '--result-filename',name+'.json','--disable-tqdm']
    if profile:args.append('--profile')
    env=s['environment'].copy();env.update(BENCH_ENERGY_GPU_INDEX=str(s['gpu']),BENCH_ENERGY_JSON=str(folder/(name+'.energy.json')))
    (folder/(name+'_command.json')).write_text(json.dumps(args,indent=2)+'\n')
    h.event('bench_start',gpu=s['gpu'],mode=s['mode'],name=name,count=count,input_len=input_len,output_len=output_len,concurrency=concurrency)
    (folder/(name+'_before.prom')).write_text(h.request(s['port'],'/metrics'))
    with (folder/(name+'.log')).open('w') as log:
        result=subprocess.run(args,cwd=s['source'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    if result.returncode:raise RuntimeError(f'Benchmark failed: {folder}/{name}.log')
    d=json.loads((folder/(name+'.json')).read_text())
    assert d['completed']==count and d['failed']==0,d
    assert set(d['input_lens'])=={input_len} and set(d['output_lens'])=={output_len}
    energy=json.loads((folder/(name+'.energy.json')).read_text())
    assert energy['requests_started']==energy['requests_finished']==energy['successful_requests']==count,energy
    assert energy['energy_j']>0,energy
    energy['validated_output_tokens']=sum(d['output_lens'])
    energy['validated_joules_per_output_token']=energy['energy_j']/energy['validated_output_tokens']
    (folder/(name+'.energy_validated.json')).write_text(json.dumps(energy,indent=2)+'\n')
    (folder/(name+'_after.prom')).write_text(h.request(s['port'],'/metrics'))
    h.event('bench_complete',gpu=s['gpu'],mode=s['mode'],name=name,output_throughput=d['output_throughput'],
            mean_tpot_ms=d['mean_tpot_ms'],gpu_joules_per_output_token=energy['validated_joules_per_output_token'])


def quality_preflight(s):
    prompts=['Answer with a digit only: 2+2=?','Answer with one word: What is the capital of France?']
    rows=[]
    for prompt in prompts:
        body={'model':h.MODEL_ID,'messages':[{'role':'user','content':prompt}],
            'temperature':0,'seed':42,'max_tokens':64,'stream':False,
            'chat_template_kwargs':{'enable_thinking':False}}
        response=json.loads(h.request(s['port'],'/v1/chat/completions',body))
        text=response['choices'][0]['message']['content'];rows.append({'prompt':prompt,'response':response})
        assert text and not re.search(r'!{8,}|\?{8,}',text),text
    (s['folder']/'quality_preflight.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert re.search(r'\b4\b',rows[0]['response']['choices'][0]['message']['content']),rows[0]


def main():
    deadline=time.monotonic()+7200
    while not (ROOT/'wave_native_ready.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Native build')
        time.sleep(5)
    h.OUT.mkdir(exist_ok=False)
    (h.OUT/'protocol.json').write_text(json.dumps({'scope':'Swapped native-library pilot; one repeat per arm/card. Full repeat/quality qualification remains required.',
        'main_pin':'88d3995ea18de47ba5c306789aa615474ad74592','quantization':'Official fp8_per_tensor online quantization, BF16 activations/output. Both arms identical precision/storage.',
        'energy':'Device cumulative-energy counter differences over formal requests; excludes separate warmups/client startup.',
        'workloads':{'S65':[512,512,65,256],'L65':[8192,1024,65,128],'S16_control':[512,512,16,128]},
        'graph_note':'Concurrency65 pads to96 tokens with the declared graph sizes.'},indent=2)+'\n')
    thread=threading.Thread(target=h.monitor,daemon=True);thread.start();servers=[]
    try:
        for rd,modes in [(1,['native','candidate']),(2,['candidate','native'])]:
            h.PHASE=f'round{rd}_startup';servers=[start(gpu,mode,rd) for gpu,mode in enumerate(modes)]
            h.parallel(servers,h.ready);h.parallel(servers,quality_preflight)
            for name,count,inp,out,c in [('energy_calibration',4,256,64,4),('warmup_S65',128,512,128,65),
                                        ('S65',256,512,512,65),('L65',128,8192,1024,65),
                                        ('S16_control',128,512,512,16)]:
                h.PHASE=f'round{rd}_{name}';h.parallel(servers,bench,name,count,inp,out,c)
            h.PHASE=f'round{rd}_profile';h.parallel(servers,bench,'profile_decode65',65,128,64,65,profile=True)
            for s in servers:h.stop(s)
            servers=[]
        (h.OUT/'COMPLETE.json').write_text('{"complete":true}\n')
    finally:
        for s in servers:h.stop(s)
        h.STOP.set();thread.join(timeout=10)


if __name__=='__main__':main()
