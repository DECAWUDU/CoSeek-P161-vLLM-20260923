"""Batch-owned local IPC transport for the existing stateless Qwen worker.

Only model weights/processor survive requests. No Agent, messages or evidence
state is stored here. A configured socket is mandatory: never fall back to
starting a second model when the resident service is unavailable.
"""
from __future__ import annotations
import hashlib, json, os, signal, socket, socketserver, threading, time, uuid
from pathlib import Path
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

def identity(config):
    return {
        'python': str(config.get('local_qwen_python') or ''),
        'model': str(Path(config.get('local_qwen_model_path') or '').expanduser()),
        'gpu': str(config.get('local_qwen_cuda_visible_devices') or ''),
        'device_map': str(config.get('local_qwen_device_map') or 'auto'),
        'max_memory': str(config.get('local_qwen_max_memory') or ''),
        'dtype': str(config.get('local_qwen_torch_dtype') or 'bfloat16'),
        'no_cpu_offload': bool(config.get('local_qwen_no_cpu_offload', True)),
    }

def fingerprint(config):
    return hashlib.sha256(json.dumps(identity(config),sort_keys=True).encode()).hexdigest()

def exchange(endpoint, payload, timeout):
    payload = dict(payload, id=uuid.uuid4().hex)
    data = json.dumps(payload,ensure_ascii=False).encode()+b'\n'
    if len(data)>MAX_MESSAGE_BYTES: raise ValueError('Resident request too large')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout);sock.connect(str(endpoint));sock.sendall(data)
        with sock.makefile('rb') as stream: line=stream.readline(MAX_MESSAGE_BYTES+1)
    if not line or len(line)>MAX_MESSAGE_BYTES: raise RuntimeError('Invalid resident response size')
    response=json.loads(line)
    if response.get('id')!=payload['id']: raise RuntimeError('Resident response ID mismatch')
    if response.get('error'): raise RuntimeError('Resident Qwen request failed: '+str(response['error']))
    return response

def health(endpoint, timeout=2):
    return exchange(endpoint,{'cmd':'health'},timeout)

def request(endpoint, config, parts, *, output_dir=None):
    tokens=int(config.get('local_qwen_max_new_tokens') or 768)
    started=time.monotonic()
    # Same per-generation deadline as the original worker. Small IPC grace
    # lets its timeout/error return before the socket client expires.
    response=exchange(endpoint,{'cmd':'generate','fingerprint':fingerprint(config),'parts':parts,'max_new_tokens':tokens},int(config.get('local_qwen_timeout_s') or 900)+10)
    if output_dir:
        p=Path(output_dir)/'_resident_qwen_requests.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        row={k:response[k] for k in ['id','service_pid','worker_pid','fingerprint']}
        row.update(max_new_tokens=tokens,wall_s=round(time.monotonic()-started,3))
        # One O_APPEND write; concurrent local callers do not share buffers.
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:os.write(fd,(json.dumps(row)+'\n').encode())
        finally:os.close(fd)
    return str(response.get('raw') or '')

def serve(config, endpoint, status_path, worker_factory=None):
    if worker_factory is None:
        from .observer import _PersistentQwenWorker
        worker_factory=_PersistentQwenWorker
    endpoint=Path(endpoint);status_path=Path(status_path)
    endpoint.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    # The owner supplies a unique, private directory. Never remove another socket.
    if endpoint.exists(): raise RuntimeError('Resident socket already exists')
    status_path.parent.mkdir(parents=True,exist_ok=True)
    worker=worker_factory(config,output_dir=str(status_path.parent))
    state_lock=threading.RLock();generation_lock=threading.Lock()
    state={'status':'loading','service_pid':os.getpid(),'fingerprint':fingerprint(config),'identity':identity(config),'requests':0,'errors':0,'started_unix':time.time()}
    closing=threading.Event()
    def publish(**values):
        with state_lock:
            state.update(values,updated_unix=time.time())
            tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status_path)
    def stop_worker():
        proc=worker.proc
        worker.stop(kill=True)
        if proc is not None:
            try:proc.wait(timeout=5)
            except Exception:pass
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            payload={}
            try:
                self.request.settimeout(10)
                line=self.rfile.readline(MAX_MESSAGE_BYTES+1)
                if not line or len(line)>MAX_MESSAGE_BYTES:raise ValueError('Invalid request size')
                payload=json.loads(line);cmd=payload.get('cmd')
                if cmd=='health':
                    with state_lock:response=dict(state,worker_alive=worker.proc is not None and worker.proc.poll() is None)
                elif cmd=='generate':
                    if payload.get('fingerprint')!=fingerprint(config):raise ValueError('Resident model/device config mismatch')
                    tokens=payload.get('max_new_tokens');parts=payload.get('parts')
                    if not isinstance(tokens,int) or isinstance(tokens,bool) or tokens<=0 or not isinstance(parts,list):raise ValueError('Invalid generation request')
                    if not generation_lock.acquire(timeout=int(config.get('local_qwen_timeout_s') or 900)):raise TimeoutError('Resident generation queue timed out')
                    try:
                        if closing.is_set():raise RuntimeError('Resident service stopping')
                        # A disconnected/expired caller must not restart a model.
                        self.request.setblocking(False)
                        try:
                            if self.request.recv(1,socket.MSG_PEEK)==b'':return
                        except BlockingIOError:pass
                        finally:self.request.settimeout(10)
                        if worker.proc is None or worker.proc.poll() is not None:raise RuntimeError('Resident worker unavailable; owner must restart service')
                        publish(status='busy')
                        try:raw=worker.request(parts,max_new_tokens=tokens)
                        except BaseException:
                            stop_worker();publish(status='failed',errors=state['errors']+1);raise
                        publish(status='ready',requests=state['requests']+1,worker_pid=worker.proc.pid)
                        response=dict(raw=raw,service_pid=os.getpid(),worker_pid=worker.proc.pid,fingerprint=fingerprint(config))
                    finally:generation_lock.release()
                else:raise ValueError('Unknown resident command')
                response['id']=payload.get('id');self.wfile.write((json.dumps(response,ensure_ascii=False)+'\n').encode())
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as exc:
                try:self.wfile.write((json.dumps({'id':payload.get('id'),'error':type(exc).__name__+': '+str(exc)[:2000]})+'\n').encode())
                except OSError:pass
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=True
        block_on_close=False
    publish()
    server=None
    try:
        start=time.monotonic();worker.ensure_started()
        server=Server(str(endpoint),Handler);os.chmod(endpoint,0o600)
        publish(status='ready',worker_pid=worker.proc.pid,load_wall_s=round(time.monotonic()-start,3))
        server.serve_forever(poll_interval=.2)
    finally:
        closing.set()
        if server is not None:server.server_close()
        stop_worker();endpoint.unlink(missing_ok=True);publish(status='stopped')

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--socket',required=True);p.add_argument('--status',required=True);p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args()
    # Linux parent-death signal also releases the model if the scheduler crashes.
    import ctypes
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('Cannot set parent-death signal')
    if os.getppid()!=a.parent_pid:raise RuntimeError('Resident owner disappeared before startup')
    def stop(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    serve(json.loads(Path(a.config).read_text()),a.socket,a.status)
if __name__=='__main__':main()
