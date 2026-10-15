import asyncio,hashlib,json,pathlib,re,statistics,sys,time
from functools import lru_cache
import aiohttp
P = None; BASE = None; MODEL = None; TOK = None; ARGS = None

def save(n,x): (P/n).write_text(json.dumps(x,indent=2))
@lru_cache(maxsize=600)
def prompt(i,length=512):
    # Keep the native chat framing and explicit final instruction. The supplied token IDs freeze exact input lengths.
    full=TOK.apply_chat_template([{'role':'user','content':f'Document {i}: '+('A reliable service records requests, validates inputs, processes jobs and handles failures. '*1200)+'\nWrite an extended, detailed engineering tutorial on designing this service, with examples, failure analysis, and testing. Continue for at least 1500 words.'}],tokenize=True,add_generation_prompt=True,enable_thinking=True)
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
    return dict(concurrency=c,thinking_enabled=True,input_tokens=length,output_tokens=out,requests=c,completed_tokens=total,wall_seconds=end-begin,aggregate_output_tok_s=total/(end-begin),common_decode_tok_s=(total-c)/(last-first),ttft_p50=statistics.median(r['ttft'] for r in rows),ttft_max=max(r['ttft'] for r in rows),latency_p50=statistics.median(r['latency'] for r in rows),latency_max=max(r['latency'] for r in rows),rows=rows)

async def run(arm):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240),connector=aiohttp.TCPConnector(limit=300,force_close=True)) as s:
        questions=[('Return only the integer result of 84 * 3 / 2.','126'),('Return only the integer value of sum([1, 2, 3]).','6'),('Return only the capital of France.','Paris'),('Return only the integer result of (17 + 23) * 5.','200'),('Return only a JSON array containing 2, 3, 5 in that order.','[2,3,5]'),('Return only the lowercase word true if all([True, True]) is true.','true')]
        checks=[]
        for q,expected in questions:
            d=await post(s,'/v1/chat/completions',dict(model=MODEL,messages=[dict(role='user',content=q)],temperature=0,max_tokens=2048,seed=1234,chat_template_kwargs={'enable_thinking':True}))
            answer=d['choices'][0]['message'].get('content') or ''
            ok=re.sub(r'\s','',answer).rstrip('.').lower()==expected.lower()
            checks.append(dict(question=q,expected=expected,answer=answer,passed=ok,response=d))
        save(arm+'-quality-private.json',checks)
        assert all(x['passed'] for x in checks),'Strict answer quality gate failed'
        deterministic=await batch(s,4,512,128);save(arm+'-deterministic-private.json',deterministic)
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
        filler='The archive contains routine service notes about queues, retries, and monitoring. '
        for k,pos in enumerate([0,250,500]):
            value=9137+113*k
            document=filler*pos+f'\nThe exact stored reference OMEGA-{k} has value {value}.\n'+filler*(500-pos)
            cases.append((document+f'\nLook up the stored value for OMEGA-{k}. Return only JSON with the key value, no explanation or markdown.',dict(value=value)))
        cases=cases*3  # Repeat the fixed suite to check stochastic-rounding stability.
        async def numerical(q,expected):
            d=await post(s,'/v1/chat/completions',dict(model=MODEL,messages=[dict(role='user',content=q)],temperature=0,max_tokens=2048,seed=1234,chat_template_kwargs={'enable_thinking':True}))
            answer=d['choices'][0]['message'].get('content') or ''
            try:parsed=json.loads(answer);passed=parsed==expected
            except Exception:passed=False
            return dict(question=q,expected=expected,answer=answer,passed=passed,response=d)
        for j in range(0,len(cases),4):numeric+=await asyncio.gather(*(numerical(q,e) for q,e in cases[j:j+4]))
        save(arm+'-numerical-quality-private.json',numeric)
        assert all(x['passed'] for x in numeric), 'Qualified recipe requires all 69 observations to pass'
        if ARGS.reference is None:
            save('control-quality-reference.json',numeric)
        else:
            reference=json.loads(ARGS.reference.read_text())
            assert len(reference)==len(numeric), 'Mismatched reference length'
            assert all(a['question']==b['question'] for a,b in zip(reference,numeric)), 'Mismatched quality cases'
            regressions=[i for i,(a,b) in enumerate(zip(reference,numeric)) if a['passed'] and not b['passed']]
            save(arm+'-quality-comparison.json',dict(reference_correct=sum(x['passed'] for x in reference),reference_total=len(reference),candidate_correct=sum(x['passed'] for x in numeric),total=len(numeric),newly_failing_cases=regressions))
            assert not regressions,'Numerical quality regression against control'
            if arm=='fp32-control':save('control-quality-reference.json',numeric)
        save(arm+'-gate-method.json',dict(version=5,thinking_enabled=True,exact_answer_checks=6,numerical_checks=len(numeric),unique_numerical_cases=23,numerical_repeats=3,numerical_correct=sum(x['passed'] for x in numeric),qualification='six exact smoke checks; every control-passing numerical case must remain correct; all errors retained',prose_identity='diagnostic only; unoptimized baseline changes wording across batch sizes',precision='official NVFP4 weights and FP8 KV; SSM dtype and stochastic rounding recorded in each launch; TCP force_close'))
        if ARGS.check_integrity:
            async def identity_check(i,gate):
                expected=f'CHECK-{i:03d}'
                q=f'Your assigned code is {expected}. '+('This is a record integrity check. '*40)+'Return only your assigned code exactly, with no explanation or markdown.'
                await gate.wait();began=time.perf_counter()
                d=await post(s,'/v1/chat/completions',dict(model=MODEL,messages=[dict(role='user',content=q)],temperature=0,max_tokens=512,seed=1234,chat_template_kwargs={'enable_thinking':True}))
                ended=time.perf_counter();answer=(d['choices'][0]['message'].get('content') or '').strip()
                return dict(index=i,expected=expected,answer=answer,passed=answer==expected,begin=began,end=ended,usage=d.get('usage'))
            gate=asyncio.Event();jobs=[asyncio.create_task(identity_check(i,gate)) for i in range(256)]
            await asyncio.sleep(0);gate.set();checks256=await asyncio.gather(*jobs)
            events=sorted([(x['begin'],1) for x in checks256]+[(x['end'],-1) for x in checks256])
            active=peak=0
            for _,delta in events:active+=delta;peak=max(peak,active)
            save(arm+'-parallel-integrity-private.json',checks256)
            save(arm+'-parallel-integrity-summary.json',dict(requests=256,passed=sum(x['passed'] for x in checks256),peak_client_inflight=peak))
            assert peak==256,'Did not observe all256requests concurrently in flight'
            assert all(x['passed'] for x in checks256),'Concurrent request integrity failure'
        # Every concurrency receives a same-shape warmup, excluded from timing.
        results=[]
        for c in ARGS.concurrencies:
            await batch(s,c,512,128)
            for repeat in range(ARGS.repeats):
                result=await batch(s,c);result['repeat']=repeat;results.append(result)
                save(arm+'-measurements-private.json',results)
                save(arm+'-summary.json',[{k:v for k,v in x.items() if k!='rows'} for x in results])
        # Fresh long-input latency is a separate scope, never mixed with decode results.
        long=await batch(s,1,8192,128);save(arm+'-8k-private.json',long)
        save(arm+'-8k-summary.json',{k:v for k,v in long.items() if k!='rows'})
        return results
def main():
    import argparse
    parser=argparse.ArgumentParser(description='Pinned Nemotron recipe: quality gates and streaming HTTP throughput. Prompts are synthetic; output includes reasoning tokens.')
    parser.add_argument('--base-url',default='http://127.0.0.1:8015')
    parser.add_argument('--tokenizer',default='/model',help='Local pinned model/tokenizer directory')
    parser.add_argument('--model',default='nemotron-3.5-super-vl')
    parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/results'))
    parser.add_argument('--arm',choices=['control','candidate','confirmation'],default='candidate')
    parser.add_argument('--reference',type=pathlib.Path,help='Prior control-quality-reference.json; all passing reference cases must remain correct')
    parser.add_argument('--concurrencies',type=int,nargs='+',default=[128,256])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--check-integrity',action='store_true',help='Add the measured 256-request unique-identity check')
    args=parser.parse_args()
    if args.repeats<1 or any(c<1 or c>256 for c in args.concurrencies):parser.error('Use positive repeats and concurrency 1..256')
    if args.reference is not None and not args.reference.is_file():parser.error('Reference file does not exist')
    if not pathlib.Path(args.tokenizer).is_dir():parser.error('Pinned local model/tokenizer directory does not exist')
    from transformers import AutoTokenizer
    global P,BASE,MODEL,TOK,ARGS
    ARGS=args;P=args.output;P.mkdir(parents=True,exist_ok=True)
    BASE=args.base_url.rstrip('/');MODEL=args.model
    TOK=AutoTokenizer.from_pretrained(args.tokenizer,trust_remote_code=True,local_files_only=True)
    save(args.arm+'-run-config.json',dict(base_url=BASE,model=MODEL,tokenizer=args.tokenizer,concurrencies=args.concurrencies,repeats=args.repeats,quality_cases=23,quality_repeats=3))
    asyncio.run(run(args.arm))
if __name__=='__main__':main()
