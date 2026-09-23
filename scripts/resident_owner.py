"""Own one existing resident-Qwen service for the portable queue."""
import os,sys,json,subprocess,tempfile,time,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ResidentOwner:
    def __init__(self,config,out,enabled=True):
        self.config=config;self.out=out;self.enabled=enabled;self.proc=None;self.private=None;self.log=None;self.previous=os.environ.get('COSEEK_QWEN_SOCKET')
    def __enter__(self):
        if not self.enabled:return self
        from videoseek.resident_qwen import health,fingerprint
        if self.previous:
            status=health(self.previous)
            if not status.get('worker_alive') or status.get('fingerprint')!=fingerprint(self.config):raise RuntimeError('External resident worker does not match this model/GPU configuration')
            return self
        self.private=Path(tempfile.mkdtemp(prefix='coseek-'))
        config=self.private/'qwen.json';endpoint=self.private/'q.sock'
        config.write_text(json.dumps({k:v for k,v in self.config.items() if k.startswith('local_qwen_') and 'api_key' not in k}));config.chmod(0o600)
        env=dict(os.environ,PYTHONPATH=str(ROOT/'runtime'),CUDA_VISIBLE_DEVICES=self.config['local_qwen_cuda_visible_devices'])
        self.log=(self.out/'resident.log').open('w')
        self.proc=subprocess.Popen([sys.executable,'-m','videoseek.resident_qwen','--config',str(config),'--socket',str(endpoint),'--status',str(self.out/'resident_status.json'),'--parent-pid',str(os.getpid())],env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        try:
            while time.monotonic()-start<300:
                if self.proc.poll() is not None:raise RuntimeError('Search model startup failed; see resident.log')
                if endpoint.exists():
                    status=health(endpoint)
                    if status.get('worker_alive') and status.get('fingerprint')==fingerprint(self.config):
                        os.environ['COSEEK_QWEN_SOCKET']=str(endpoint);return self
                time.sleep(1)
            raise TimeoutError('Search model startup exceeded 300 seconds')
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,*unused):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait()
            if self.log:self.log.close()
            if self.private:shutil.rmtree(self.private)
        if self.previous is None:os.environ.pop('COSEEK_QWEN_SOCKET',None)
        else:os.environ['COSEEK_QWEN_SOCKET']=self.previous
