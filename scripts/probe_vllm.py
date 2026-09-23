"""Small explicit API checks; no local GPU and no user video is uploaded."""
import base64,io,json,os,time
from pathlib import Path
from vllm_adapter import resolve_model,build_request,validate_response,thinking_mode
def probe(config,out):
    import httpx
    from openai import OpenAI
    from PIL import Image,ImageDraw
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'endpoint':config['api_base'],'thinking':thinking_mode(),'checks':[],'scope':'text, JSON, and synthetic multi-image compatibility; not a QA benchmark'}
    try:
        model=resolve_model(config['api_base'],config['api_key'],config['model_name']);report['model']=model
        cases=[('text',[{'role':'user','content':'Reply only OK.'}],False)]
        cases.append(('json',[{'role':'user','content':'Return JSON only: {"ok":true}.'}],True))
        content=[{'type':'text','text':'Inspect the images. Return JSON with first_color and last_color using lowercase basic English color names for the background of the first and last image.'}]
        count=int(os.environ.get('VLLM_PROBE_IMAGES','32'))
        if not 2<=count<=64:raise ValueError('VLLM_PROBE_IMAGES must be 2..64')
        for i in range(count):
            color='red' if i==0 else 'blue' if i==count-1 else (100+i,100+i,100+i)
            image=Image.new('RGB',(96,96),color);ImageDraw.Draw(image).text((4,4),str(i+1),fill='white');buf=io.BytesIO();image.save(buf,format='PNG')
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}})
        cases.append((f'multimodal_{count}',[{'role':'user','content':content}],True))
        with httpx.Client(trust_env=False,timeout=400) as client:
            api=OpenAI(base_url=config['api_base'],api_key=config['api_key'],max_retries=0,http_client=client)
            for name,messages,want_json in cases:
                started=time.monotonic();q=build_request(model,messages,2048,42,1.0,return_json=want_json)
                raw=api.chat.completions.create(**q).model_dump()
                (out/f'{name}.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
                validate_response(raw)
                text=raw['choices'][0]['message']['content'];parsed=json.loads(text) if want_json else None
                if name=='text' and text.strip().rstrip('.')!='OK':raise ValueError('Text instruction not followed')
                if name=='json' and parsed!={'ok':True}:raise ValueError('JSON check failed')
                if name.startswith('multimodal') and (parsed.get('first_color'),parsed.get('last_color'))!=('red','blue'):raise ValueError('Synthetic image order/color check failed')
                if not raw.get('usage') or not all(isinstance(raw['usage'].get(k),int) for k in ('prompt_tokens','completion_tokens','total_tokens')):raise ValueError('Usage missing; token budget cannot be enforced')
                (out/f'{name}.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
                report['checks'].append({'name':name,'status':'passed','seconds':time.monotonic()-started,'usage':raw['usage']})
                (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        report['status']='passed'
    except Exception as exc:
        from transport import sanitized_error_details
        report.update(status='failed',error_type=type(exc).__name__,diagnostic=sanitized_error_details(exc,config['api_key']))
        if isinstance(exc,ValueError):report['detail']=str(exc)[:500]
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
    return int(report['status']!='passed')
