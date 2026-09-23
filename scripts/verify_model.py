#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model_dir');p.add_argument('--weights',action='store_true');a=p.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'provenance/model_files.sha256.json').read_text());bad=[];checked=0
for name,meta in manifest.items():
    if not a.weights and name.endswith('.safetensors'):continue
    f=Path(a.model_dir)/name
    if not f.is_file():bad.append(name+' missing');continue
    h=hashlib.sha256()
    with f.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    checked+=1
    if h.hexdigest()!=meta['sha256']:bad.append(name+' differs')
print('Checked',checked,'files; differences:',bad)
raise SystemExit(int(bool(bad)))
