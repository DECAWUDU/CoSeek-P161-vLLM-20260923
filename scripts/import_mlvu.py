#!/usr/bin/env python3
"""Convert an existing MLVU subset to portable cases; no question rewriting."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('subset');p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.data_root).expanduser().resolve();rows=[]
for item in json.loads(Path(a.subset).read_text()):
    rel=item['video_path'];video=Path(rel)
    if not video.is_absolute():video=root/rel.split('data/mlvu/',1)[-1]
    if not video.is_file():raise SystemExit('Missing video: '+str(video))
    for i,q in enumerate(item['conversations']):
        if len(q['choices'])!=4:raise SystemExit('Frozen contract requires 4 choices')
        row={'id':str(item['video_id'])+'_'+str(i),'video':str(video),'question':q['question'],'choices':q['choices'],'question_type':q.get('question_type','')}
        if q.get('answer') in q['choices']:row['answer']=chr(65+q['choices'].index(q['answer']))
        rows.append(row)
Path(a.output).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print('Exported',len(rows),'questions')
