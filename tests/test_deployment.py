import importlib.util,json,os,sys,tempfile,unittest,threading
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import portable as p
from vllm_adapter import build_request,resolve_model,validate_response

class ConfigTests(unittest.TestCase):
 def test_all_remote_roles_change_without_search_or_budget_changes(self):
  base=json.loads((R/'configs/p161.json').read_text())
  with patch.dict(os.environ,{'OPENAI_MODEL':'company-qwen','OPENAI_API_BASE':'http://example.invalid:8077/v1/','OPENAI_API_KEY':'test','COSEEK_GPU':'5'},clear=True):c=p.make_config()
  self.assertEqual((c['model_name'],c['observer_model_name']),('company-qwen','company-qwen'))
  self.assertEqual(c['api_base'],c['observer_api_base']);self.assertEqual(c['api_key'],c['observer_api_key'])
  for key in ['max_steps','max_tokens','overview_sampling_mode','dual_path_overview_enabled','adaptive_api_token_hard_limit','localize_qwen_coarse_max_frames','local_qwen_max_new_tokens']:
   self.assertEqual(c.get(key),base.get(key),key)
  self.assertFalse(c['local_qwen_fallback_to_api']);self.assertGreater(p.verify_frozen(),100)
 def test_multichoice_subtitle_validation(self):
  with tempfile.TemporaryDirectory() as d:
   video=Path(d)/'x.mp4';video.touch();sub=Path(d)/'x.srt';sub.write_text('1\n00:00:01,000 --> 00:00:02,000\nhello\n')
   case=Path(d)/'cases.json';case.write_text(json.dumps([dict(id='x',video=str(video),subtitles=str(sub),question='Q',choices=['a','b','c','d','e'],answer='E')]))
   self.assertEqual(p.load_cases(SimpleNamespace(cases=str(case)))[0]['answer'],'E')
 def test_interpreter_path_does_not_resolve_venv_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   link=Path(d)/'python';link.symlink_to(sys.executable);self.assertEqual(p.path_value(str(link)),link)
 def test_request_keeps_images_but_omits_gpt_parameters(self):
  messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,TEST'}}]}]
  with patch.dict(os.environ,{'VLLM_ENABLE_THINKING':'false'},clear=True):q=build_request('q',messages,100,42,1.,return_json=True)
  self.assertIs(q['messages'],messages);self.assertNotIn('reasoning_effort',q);self.assertNotIn('max_completion_tokens',q)
  self.assertEqual(q['extra_body']['chat_template_kwargs']['enable_thinking'],False)
  self.assertEqual(q['response_format'],{'type':'json_object'})
 def test_server_default_not_silently_disabled(self):
  with patch.dict(os.environ,{},clear=True):q=build_request('q',[],100,42,1.)
  self.assertNotIn('extra_body',q)
 def test_rejects_incomplete_or_reasoning_only_output(self):
  for choice in [dict(message={'content':'<think>reason</think>{}'},finish_reason='stop'),dict(message={'content':None,'reasoning':'x'},finish_reason='stop'),dict(message={'content':'{"x":'},finish_reason='length')]:
   with self.assertRaises(ValueError):validate_response({'choices':[choice]})

class WireTests(unittest.TestCase):
 def test_real_sdk_wire_model_discovery_and_usage_logging(self):
  try:import openai,litellm
  except ImportError:self.skipTest('Run in main environment for SDK integration test')
  seen=[]
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def send(self,data):
    b=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
   def do_GET(self):self.send({'object':'list','data':[{'id':'internal-qwen','object':'model','created':0,'owned_by':'test'}]})
   def do_POST(self):
    q=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append(q)
    self.send({'id':'test','object':'chat.completion','created':0,'model':q['model'],'choices':[{'index':0,'message':{'role':'assistant','content':'{"ok":true}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':9,'completion_tokens':5,'total_tokens':14}})
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  base=f'http://127.0.0.1:{server.server_port}/v1'
  try:
   self.assertEqual(resolve_model(base,'EMPTY'),'internal-qwen')
   with self.assertRaises(ValueError):resolve_model(base,'EMPTY','gpt-5')
   from transport import BoundedTransport,TRANSPORT_BUDGET
   self.assertEqual(TRANSPORT_BUDGET['http_timeout_s'],400);self.assertEqual(TRANSPORT_BUDGET['max_episode_retries'],2)
   with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'COSEEK_USAGE_LOG':str(Path(d)/'usage.jsonl'),'VLLM_ENABLE_THINKING':'false'}):
    tr=BoundedTransport(SimpleNamespace(active=False),Path(d));tr.start_suffix()
    try:response=tr.call('internal-qwen',[{'role':'user','content':'Return JSON'}],base,api_key='EMPTY',max_tokens=128,return_json=True)
    finally:tr.close()
    self.assertEqual(json.loads(response.choices[0].message.content),{'ok':True})
    self.assertEqual(json.loads((Path(d)/'usage.jsonl').read_text())['total_tokens'],14)
   self.assertEqual(seen[0]['model'],'internal-qwen');self.assertEqual(seen[0]['max_tokens'],128)
   self.assertNotIn('reasoning_effort',seen[0]);self.assertNotIn('extra_body',seen[0]);self.assertEqual(seen[0]['chat_template_kwargs'],{'enable_thinking':False})
  finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main()
