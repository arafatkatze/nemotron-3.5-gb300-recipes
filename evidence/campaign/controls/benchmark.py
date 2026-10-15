import asyncio,hashlib,json,pathlib,re,statistics,sys,time
from functools import lru_cache
import aiohttp
from transformers import AutoTokenizer
P=pathlib.Path('/work'); BASE='http://127.0.0.1:8015'; MODEL='nemotron-3.5-super-vl'
TOK=AutoTokenizer.from_pretrained('/model',trust_remote_code=True,local_files_only=True)

def save(n,x): (P/n).write_text(json.dumps(x,indent=2))
@lru_cache(maxsize=600)
def prompt(i,length=512):
    # Keep the native chat framing and explicit final instruction. The supplied token IDs freeze exact input lengths.
    full=TOK.apply_chat_template([{'role':'user','content':f'Document {i}: '+('A reliable service records requests, validates inputs, processes jobs and handles failures. '*1200)+'\nWrite an extended, detailed engineering tutorial on designing this service, with examples, failure analysis, and testing. Continue for at least 1500 words.'}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    full=full["input_ids"] if hasattr(full,"keys") else full
    assert len(full)>=length and all(isinstance(x,int) for x in full)
    return full[:length-110]+full[-110:]

async def post(s,route,body):
    async with s.post(BASE+route,json=body) as r:
        body=await r.text();assert r.status==200,(r.status,body[:1500]);return json.loads(body)

async def request(s,i,length=512,out=512,start=None):
    body=dict(model=MODEL,prompt=prompt(i,length),max_tokens=out,temperature=0,seed=1234,ignore_eos=True,stream=True,stream_options={'include_usage':True})
    if start is not None:await start.wait()
    begin=time.perf_counter();first=None;text='';usage=None;finish=None;arrivals=[]
    async with s.post(BASE+'/v1/completions',json=body) as r:
        assert r.status==200,(r.status,(await r.text())[:1500])
        while True:
            raw=await r.content.readline()
            if not raw:break
            if not raw.startswith(b'data:'):continue
            raw=raw[5:].strip()
            if raw==b'[DONE]':break
            d=json.loads(raw)
            if d.get('usage'):usage=d['usage']
            for c in d.get('choices',[]):
                if c.get('text'):
                    now=time.perf_counter()
                    if first is None:first=now
                    arrivals.append(now);text+=c['text']
                if c.get('finish_reason'):finish=c['finish_reason']
    end=time.perf_counter();assert usage and usage['completion_tokens']==out,(usage,finish)
    assert usage['prompt_tokens']==length,(usage,length)
    assert first is not None and len(text)>100
    ids=TOK.encode(text,add_special_tokens=False)
    maxrun=run=1
    for a,b in zip(ids,ids[1:]):
        run=run+1 if a==b else 1;maxrun=max(maxrun,run)
    assert maxrun<32 and len(set(ids))>=min(30,out//4),'Degenerate repeated output'
    return dict(index=i,begin=begin,first=first,end=end,ttft=first-begin,latency=end-begin,usage=usage,finish=finish,text=text,sha256=hashlib.sha256(text.encode()).hexdigest(),chunks=len(arrivals))

async def batch(s,c,length=512,out=512):
    gate=asyncio.Event();jobs=[asyncio.create_task(request(s,i,length,out,gate)) for i in range(c)]
    await asyncio.sleep(0);begin=time.perf_counter();gate.set();rows=await asyncio.gather(*jobs);end=time.perf_counter()
    total=sum(r['usage']['completion_tokens'] for r in rows);first=min(r['first'] for r in rows);last=max(r['end'] for r in rows)
    return dict(concurrency=c,input_tokens=length,output_tokens=out,requests=c,completed_tokens=total,wall_seconds=end-begin,aggregate_output_tok_s=total/(end-begin),common_decode_tok_s=(total-c)/(last-first),ttft_p50=statistics.median(r['ttft'] for r in rows),ttft_max=max(r['ttft'] for r in rows),latency_p50=statistics.median(r['latency'] for r in rows),latency_max=max(r['latency'] for r in rows),rows=rows)

async def run(arm):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240),connector=aiohttp.TCPConnector(limit=300)) as s:
        questions=[('Return only the integer result of 84 * 3 / 2.','126'),('Return only the integer value of sum([1, 2, 3]).','6'),('Return only the capital of France.','Paris'),('Return only the integer result of (17 + 23) * 5.','200'),('Return only a JSON array containing 2, 3, 5 in that order.','[2,3,5]'),('Return only the lowercase word true if all([True, True]) is true.','true')]
        checks=[]
        for q,expected in questions:
            d=await post(s,'/v1/chat/completions',dict(model=MODEL,messages=[dict(role='user',content=q)],temperature=0,max_tokens=128,seed=1234,chat_template_kwargs={'enable_thinking':False}))
            answer=d['choices'][0]['message'].get('content') or ''
            ok=re.sub(r'\s','',answer).rstrip('.').lower()==expected.lower()
            checks.append(dict(question=q,expected=expected,answer=answer,passed=ok,response=d))
        save(arm+'-quality-private.json',checks)
        assert all(x['passed'] for x in checks),'Strict answer quality gate failed'
        deterministic=await batch(s,4,512,128);save(arm+'-deterministic-private.json',deterministic)
        ref=P/'baseline-deterministic-private.json'
        if arm!='baseline':
            reference=json.loads(ref.read_text());matches=[a['text']==b['text'] for a,b in zip(reference['rows'],deterministic['rows'])]
            save(arm+'-equivalence.json',dict(exact_text_matches=matches,all_match=all(matches)))
            # Diagnostic only: baseline itself changes prose across batch shapes.
            # Ground-truth structured numerical tests below are the qualification gate.
        numeric=[]
        cases=[]
        for i in range(12):
            unit=137+11*i;qty=3+i;shipping=79;paid=20000
            total=unit*qty+shipping
            cases.append((f"A buyer purchases {qty} items at {unit} cents each and pays {shipping} cents for shipping. They hand over {paid} cents. Return only JSON with integer keys items_cost, total, and change. Do not add explanation or markdown.",dict(items_cost=unit*qty,total=total,change=paid-total)))
        for i in range(8):
            values=[i+3,2*i,7,i+3,11-i]
            result=sorted(set(x*x for x in values if x%2))
            cases.append((f"Evaluate this Python expression exactly: sorted(set(x*x for x in {values!r} if x%2)). Return only the JSON array result, no explanation or markdown.",result))
        async def numerical(q,expected):
            d=await post(s,'/v1/chat/completions',dict(model=MODEL,messages=[dict(role='user',content=q)],temperature=0,max_tokens=256,seed=1234,chat_template_kwargs={'enable_thinking':False}))
            answer=d['choices'][0]['message'].get('content') or ''
            try:parsed=json.loads(answer);passed=parsed==expected
            except Exception:passed=False
            return dict(question=q,expected=expected,answer=answer,passed=passed,response=d)
        for j in range(0,len(cases),4):numeric+=await asyncio.gather(*(numerical(q,e) for q,e in cases[j:j+4]))
        save(arm+'-numerical-quality-private.json',numeric)
        if arm=='control':
            save('control-quality-reference.json',numeric)
        else:
            reference=json.loads((P/'control-quality-reference.json').read_text())
            regressions=[i for i,(a,b) in enumerate(zip(reference,numeric)) if a['passed'] and not b['passed']]
            save(arm+'-quality-comparison.json',dict(reference_correct=sum(x['passed'] for x in reference),candidate_correct=sum(x['passed'] for x in numeric),total=len(numeric),newly_failing_cases=regressions))
            assert not regressions,'Numerical quality regression against control' 
        save(arm+'-gate-method.json',dict(version=3,exact_answer_checks=6,numerical_checks=len(numeric),numerical_correct=sum(x['passed'] for x in numeric),qualification='six exact smoke checks; every control-passing numerical case must remain correct; all errors retained',prose_identity='diagnostic only; unoptimized baseline changes wording across batch sizes',precision='unchanged official NVFP4 weights, FP32 SSM, FP8 KV'))
        # Every concurrency receives a same-shape warmup, excluded from timing.
        results=[]
        for c in [128,256]:
            await batch(s,c,512,128)
            for repeat in range(3):
                result=await batch(s,c);result['repeat']=repeat;results.append(result)
                save(arm+'-measurements-private.json',results)
                save(arm+'-summary.json',[{k:v for k,v in x.items() if k!='rows'} for x in results])
        # Fresh long-input latency is a separate scope, never mixed with decode results.
        long=await batch(s,1,8192,128);save(arm+'-8k-private.json',long)
        save(arm+'-8k-summary.json',{k:v for k,v in long.items() if k!='rows'})
        return results
if __name__=='__main__':asyncio.run(run(sys.argv[1]))
