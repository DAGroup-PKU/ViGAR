"""Responses-compatible curation client; no embedded credentials or implicit retries."""
import http.client
import json
import os
import time
from urllib.parse import urlparse

def request(path, body=None, timeout=600):
    endpoint=urlparse(os.environ['GOALWAM_VLM_BASE_URL'])
    if endpoint.scheme != 'https': raise ValueError('HTTPS endpoint required')
    key=os.environ['GOALWAM_VLM_API_KEY']
    payload=None if body is None else json.dumps(body,ensure_ascii=False).encode()
    conn=http.client.HTTPSConnection(endpoint.hostname,endpoint.port,timeout=timeout)
    start=time.monotonic()
    try:
        conn.request('GET' if body is None else 'POST',path,body=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        response=conn.getresponse();raw=response.read();result=json.loads(raw)
        stats=dict(http=response.status,request_bytes=len(payload or b''),response_bytes=len(raw),seconds=round(time.monotonic()-start,3))
        if response.status != 200: raise RuntimeError('Curation endpoint returned HTTP '+str(response.status))
        return result,stats
    finally: conn.close()

def output_text(response):
    if response.get('status')!='completed': raise ValueError('Response incomplete')
    return ''.join(c['text'] for item in response.get('output',[]) for c in item.get('content',[]) if c.get('type')=='output_text')
