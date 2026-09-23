"""Deployment-only request adaptation; does not change CoSeek evidence logic."""
import os

def thinking_mode():
    value=os.environ.get('VLLM_ENABLE_THINKING','server').lower()
    if value not in {'true','false','server'}:
        raise ValueError('VLLM_ENABLE_THINKING must be true, false, or server')
    return None if value=='server' else value=='true'

def build_request(model_name,messages,max_tokens,seed,temperature,tools=None,tool_choice=None,return_json=False):
    # Qwen thinking is a chat-template option, not GPT reasoning_effort.
    request=dict(model=model_name,messages=messages,max_tokens=max_tokens,seed=seed,temperature=temperature)
    thinking=thinking_mode()
    if thinking is not None:request['extra_body']={'chat_template_kwargs':{'enable_thinking':thinking}}
    if tools:request['tools']=tools
    if tool_choice is not None and tools:request['tool_choice']=tool_choice
    if return_json and os.environ.get('VLLM_JSON_MODE','true').lower()=='true':
        request['response_format']={'type':'json_object'}
    return {k:v for k,v in request.items() if v is not None}

def validate_response(payload):
    choices=payload.get('choices') or []
    if not choices:raise ValueError('vLLM returned no choices')
    message=choices[0].get('message') or {}
    content=message.get('content')
    if choices[0].get('finish_reason')=='length':
        raise ValueError('vLLM completion truncated; check thinking/output/context budgets')
    if message.get('tool_calls'):return
    if not isinstance(content,str) or not content.strip():
        raise ValueError('Empty final content; check reasoning parser and completion budget')
    if '<think>' in content or '</think>' in content:
        raise ValueError('Reasoning leaked into content; configure vLLM reasoning parser instead of treating it as a tool receipt')

def resolve_model(base,key,configured='AUTO'):
    import httpx
    from openai import OpenAI
    with httpx.Client(trust_env=False,timeout=15) as client:
        api=OpenAI(base_url=base,api_key=key,max_retries=0,http_client=client)
        models=api.models.list().data
    ids=[m.id for m in models]
    if configured and configured!='AUTO':
        if configured not in ids:raise ValueError('Configured model not served; available IDs: '+repr(ids))
        return configured
    if len(ids)!=1:raise ValueError('Set OPENAI_MODEL to one exact served ID: '+repr(ids))
    return ids[0]
