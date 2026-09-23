#!/usr/bin/env python3
"""Portable adapter; the frozen runtime is never rewritten."""
from __future__ import annotations
import argparse, gzip, hashlib, inspect, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'runtime'), str(ROOT/'scripts')]

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')

def path_value(value):
    p=Path(value).expanduser()
    return Path(os.path.abspath(p if p.is_absolute() else ROOT/p))

def make_config():
    c=json.loads((ROOT/'configs/p161.json').read_text())
    base=os.environ.get('OPENAI_API_BASE',c['api_base']).rstrip('/')
    model=os.environ.get('OPENAI_MODEL','AUTO')
    key=os.environ.get('OPENAI_API_KEY') or 'EMPTY'
    c.update(api_base=base,api_key=key,model_name=model,
             observer_api_base=base,observer_api_key=key,observer_model_name=model,
             local_qwen_model_path=str(path_value(os.environ.get('QWEN_MODEL_PATH',c['local_qwen_model_path']))),
             local_qwen_python=str(path_value(os.environ.get('QWEN_PYTHON',c['local_qwen_python']))),
             local_qwen_cuda_visible_devices=os.environ.get('COSEEK_GPU',''),
             skeleton_cache_dir=str(ROOT/'runs/_skeleton_cache'))
    return c

def verify_frozen():
    manifest=json.loads((ROOT/'provenance/runtime.sha256.json').read_text())
    bad=[name for name,h in manifest.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    if bad: raise ValueError('Frozen runtime changed: '+', '.join(bad[:5]))
    return len(manifest)

def safe_config(c):
    return {k: '<redacted>' if 'api_key' in k else v for k,v in c.items()}

def doctor():
    c=make_config(); failures=[]
    print('Frozen source files:',verify_frozen())
    for command in ['ffmpeg','ffprobe','nvidia-smi']:
        found=shutil.which(command);print(command,found or 'MISSING')
        if not found: failures.append(command)
    import run_mlvu  # imports the actual agent and tool registry
    print('Main Python:',sys.version.split()[0],'; frozen agent imports OK')
    p=Path(c['local_qwen_model_path'])
    if not (p/'config.json').is_file() or not list(p.glob('*.safetensors')):failures.append('Qwen weights')
    print('Qwen model files:', 'OK' if 'Qwen weights' not in failures else 'MISSING')
    check='import torch,transformers; from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor; print("Qwen Python OK; torch",torch.__version__,"transformers",transformers.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"'
    try:
        r=subprocess.run([c['local_qwen_python'],'-c',check],timeout=90,env={**os.environ,'CUDA_VISIBLE_DEVICES':c['local_qwen_cuda_visible_devices']})
        if r.returncode:failures.append('Qwen environment')
    except (OSError,subprocess.TimeoutExpired):failures.append('Qwen Python unavailable')
    if not c['local_qwen_cuda_visible_devices']:failures.append('COSEEK_GPU not selected')
    print('API configuration:', 'present (not contacted)' if c['api_key'] and c['api_base'] else 'missing')
    if not c['api_key'] or not c['api_base']:failures.append('API configuration')
    print('Failures:', ', '.join(failures) or 'none')
    return int(bool(failures))

def load_cases(args):
    rows=json.loads(path_value(args.cases).read_text())
    if not isinstance(rows,list) or not rows:raise ValueError('cases must be a nonempty JSON array')
    ids=set()
    for row in rows:
        ident=row.get('id','')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',ident) or ident in {'.','..'} or ident in ids:raise ValueError('Each id must be unique and filename-safe')
        ids.add(ident)
        if not isinstance(row.get('question'),str) or not row['question'].strip():raise ValueError('Missing question')
        if not 2<=len(row.get('choices',[]))<=5 or not all(isinstance(x,str) and x.strip() for x in row['choices']):raise ValueError('Provide 2-5 nonempty choices')
        row['video']=str(path_value(row['video']))
        if not Path(row['video']).is_file():raise ValueError('Video not found: '+row['video'])
        if row.get('answer') is not None and row['answer'] not in 'ABCDE'[:len(row['choices'])]:raise ValueError('answer must match one supplied choice or be omitted')
        if row.get('subtitles'):
            row['subtitles']=str(path_value(row['subtitles']))
            if not Path(row['subtitles']).is_file():raise ValueError('Subtitle file not found')
            if Path(row['subtitles']).suffix.lower()!='.srt':raise ValueError('Convert subtitle annotations to timestamped SRT first')
    return rows

def episode(row, out, preflight=False):
    from transport import BoundedTransport, ExperimentTransportFailure, ReplayContractError, sanitized_error_details
    import run_mlvu as native
    from videoseek.core.minimal_global_fsm import should_use_minimal_global_fsm
    class PreflightComplete(BaseException):pass
    class BudgetCensored(BaseException):pass
    c=make_config();verify_frozen()
    if not preflight and not (c['api_key'] and c['api_base']):raise ValueError('Set OPENAI_API_BASE and OPENAI_API_KEY in .env')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['COSEEK_USAGE_LOG']=str(out/'api_usage.jsonl')
    state=SimpleNamespace(active=False,agent=None)
    started=time.monotonic()
    summary={'id':row['id'],'status':'failed','ground_truth':row.get('answer')}
    class AuditedTransport(BoundedTransport):
        def call(self,*args,**kwargs):
            v=inspect.signature(BoundedTransport.call).bind(self,*args,**kwargs);v.apply_defaults();v=v.arguments
            if v['model_name']!=c['model_name'] or v['api_base'].rstrip('/')!=c['api_base']:raise ReplayContractError('All strong-model calls must use the configured internal service')
            if v['temperature']!=1.0 or v['seed']!=42:raise ReplayContractError('Frozen sampling settings changed')
            if any(f.function in {'_observe_with_local_qwen','_observe_local_qwen'} for f in inspect.stack()):raise ReplayContractError('Local observer attempted API fallback')
            if preflight:
                def project(x):
                    if isinstance(x,dict):
                        if x.get('type')=='image_url':return {'type':'image_url','sha256':hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()}
                        return {k:project(y) for k,y in x.items()}
                    return [project(y) for y in x] if isinstance(x,list) else x
                dump(out/'initial_input.json',project(v['messages']))
                raise PreflightComplete()
            if self.started is None:self.start_suffix()
            usage=native.summarize_api_usage(native.read_api_usage_log(out/'api_usage.jsonl'))
            if usage.get('api_total_tokens',0)>=150000:raise BudgetCensored('Reported-token dispatch limit reached')
            n=self.semantic_calls+1
            request={k:v[k] for k in ['messages','model_name','reasoning_effort','seed','temperature','max_tokens','tools','tool_choice','return_json']}
            request['memory']=state.agent.observation_memory
            (out/'requests').mkdir(exist_ok=True)
            with gzip.open(out/'requests'/f'{n:03}.json.gz','wt',encoding='utf-8') as f:json.dump(request,f,ensure_ascii=False,default=str)
            r=super().call(*args,**kwargs)
            dump(out/'responses'/f'{n:03}.json',r.model_dump())
            return r
    transport=AuditedTransport(state,out)
    try:
        state.agent=native.VideoSeekAgent(config=c,video_path=row['video'],subtitle_path=row.get('subtitles'),output_dir=str(out),tools=c['tools'],verbose=False)
        for i,sub in enumerate(state.agent.subtitles,1):sub.update(subtitle_id=f'S{i:04d}',source_marked=True)
        state.agent.config['_question_option_letters']='ABCDE'[:len(row['choices'])]
        query=native.build_query(row['question'],row['choices'])
        summary['branch']='global_planner' if should_use_minimal_global_fsm(query) else 'legacy_adaptive'
        dump(out/'sanitized_config.json',safe_config(c))
        from vllm_adapter import thinking_mode
        dump(out/'api_profile.json',{'provider':'vllm','model':c['model_name'],'base':c['api_base'],'thinking':thinking_mode(),'json_mode':os.environ.get('VLLM_JSON_MODE','true'),'gpt_reasoning_effort_sent':False,'completion_limit_parameter':'max_tokens'})
        dump(out/'run_contract.json',{'version':'P161-vLLM-20260923','branch':summary['branch'],'question':query,'video_path':row['video'],'video_size':Path(row['video']).stat().st_size,'config_sha256':hashlib.sha256(json.dumps(safe_config(c),sort_keys=True).encode()).hexdigest()})
        transport.install()
        trajectory=state.agent.run(query).to_dict();dump(out/'trajectory.json',trajectory)
        dump(out/'messages.json',state.agent.messages)
        pred_raw=trajectory.get('final_answer','');g=state.agent.observation_memory.get('p130_global')
        pred=native.normalize_pred(pred_raw,row['choices'])
        if pred not in 'ABCDE'[:len(row['choices'])] or not pred:raise ReplayContractError('Invalid final answer')
        terminal='native_answer'
        summary.update(status='completed',answer=pred,terminal=terminal,finish_reason=trajectory.get('finish_reason'),correct=pred==row['answer'] if row.get('answer') else None)
    except PreflightComplete:summary.update(status='preflight_passed')
    except BudgetCensored:summary.update(status='budget_censored')
    except (KeyboardInterrupt,SystemExit):raise
    except BaseException as e:
        summary.update(status='failed',error_type=type(e).__name__,diagnostic=sanitized_error_details(e,c['api_key']))
    finally:
        if state.agent is not None:dump(out/'final_memory.json',state.agent.observation_memory)
        summary.update(wall_s=round(time.monotonic()-started,3),transport=transport.summary(),usage=native.summarize_api_usage(native.read_api_usage_log(out/'api_usage.jsonl')))
        transport.close();dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['status'] in ['completed','preflight_passed'] else 1

def main():
    p=argparse.ArgumentParser(description='Frozen CoSeek P161 + I-frame/local overview')
    p.add_argument('command',choices=['run','preflight','doctor','verify','probe','_episode'])
    p.add_argument('--cases',default='examples/cases.json');p.add_argument('--output');p.add_argument('--case-json');p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if a.command=='verify':print('Verified frozen files:',verify_frozen());return 0
    if a.command=='probe':
        from probe_vllm import probe
        return probe(make_config(),path_value(a.output or 'runs/api-probe-'+time.strftime('%Y%m%d-%H%M%S')))
    if a.command=='doctor':return doctor()
    if a.command=='_episode':return episode(json.loads(Path(a.case_json).read_text()),Path(a.output),a.preflight)
    rows=load_cases(a);verify_frozen()
    if a.command=='run':
        c=make_config()
        if not c['local_qwen_cuda_visible_devices']:raise ValueError('Set COSEEK_GPU explicitly to an available GPU for the small search model')
        from vllm_adapter import resolve_model
        os.environ['OPENAI_MODEL']=resolve_model(c['api_base'],c['api_key'],c['model_name'])
    out=path_value(a.output or 'runs/'+time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True,exist_ok=False);dump(out/'cases.json',rows)
    failures=0
    from resident_owner import ResidentOwner
    with ResidentOwner(make_config(),out,enabled=a.command=='run'):
      for row in rows:
          case_file=out/'inputs'/f'{row["id"]}.json';dump(case_file,row)
          case_out=out/row['id']
          cmd=[sys.executable,str(Path(__file__).resolve()),'_episode','--case-json',str(case_file),'--output',str(case_out)]
          if a.command=='preflight':cmd.append('--preflight')
          with (out/f'{row["id"]}.log').open('w') as log:
              # A separate process group also cleans up Qwen workers after failure/timeout.
              proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
              try:proc.wait(timeout=10900)
              except subprocess.TimeoutExpired:
                  dump(case_out/'summary.json',{'id':row['id'],'status':'episode_timeout'})
              finally:
                  import signal
                  try:os.killpg(proc.pid,signal.SIGTERM)
                  except ProcessLookupError:pass
                  try:proc.wait(timeout=10)
                  except subprocess.TimeoutExpired:
                      os.killpg(proc.pid,signal.SIGKILL);proc.wait()
          summary=json.loads((case_out/'summary.json').read_text()) if (case_out/'summary.json').exists() else {'id':row['id'],'status':'process_failed','returncode':proc.returncode}
          failures+=summary['status'] not in ['completed','preflight_passed']
          with (out/'results.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
          print(row['id'],summary['status'],summary.get('answer',''),flush=True)
          attempts=case_out/'transport_attempts.jsonl'
          if a.command=='run' and attempts.exists() and any(json.loads(line).get('http_status')==429 for line in attempts.read_text().splitlines()):time.sleep(900)
    print('Results:',out,'; failed/skipped:',failures)
    return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
