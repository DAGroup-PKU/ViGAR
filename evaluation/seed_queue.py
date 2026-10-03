"""Atomic shared-filesystem seed batches; no evaluator or model semantics changed."""
import contextlib
import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def atomic(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)


@contextlib.contextmanager
def mutex(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+120
    while True:
        try:p.mkdir();break
        except FileExistsError:
            if time.monotonic()>deadline:raise TimeoutError('Merge lock retained; inspect owner: '+str(p))
            time.sleep(.1)
    try:
        atomic(p/'owner.json',dict(host=os.uname().nodename,pid=os.getpid()))
        yield
    finally:
        (p/'owner.json').unlink();p.rmdir()


class SeedQueue:
    def __init__(self,root,checkpoint,plan,canonical,batch_size=5):
        self.root=Path(root);self.ck=Path(checkpoint).name;self.plan=plan
        self.q=self.root/self.ck/'seed_queue_v1';self.canonical=canonical
        self.batch_size=batch_size
        self.paired=load(plan.get('paired_episode_by_checkpoint',{}).get(self.ck,plan['paired_episode_file']))
        assert batch_size>=1 and plan.get('goal_refresh_mode')=='sync_current'
        contract=dict(checkpoint=str(checkpoint),planner=plan.get('planner'),batch_size=batch_size,
                      paired_sha256=hashlib.sha256(json.dumps(self.paired,sort_keys=True).encode()).hexdigest(),
                      goal_refresh_mode=plan['goal_refresh_mode'])
        with mutex(self.q/'contract.lock'):
            p=self.q/'CONTRACT.json'
            if p.exists():assert load(p)==contract, 'Queue contract changed'
            else:atomic(p,contract)
        self.cells=[(t,s) for t in plan['tasks'] for s in ('clean','random')]
        for t,s in self.cells:
            rows=self.paired[s+'/'+t]
            assert len(rows)==50 and len({r['seed'] for r in rows})==50
            assert all(isinstance(r['instruction'],str) for r in rows)

    def validate(self,task,split,records):
        allowed={r['seed']:r['instruction'] for r in self.paired[split+'/'+task]}
        seen=set()
        for r in records:
            assert r['status']=='evaluated' and r['task']==task
            assert r['seed'] not in seen and r['seed'] in allowed
            assert r['instruction']==allowed[r['seed']], 'Instruction mismatch'
            assert isinstance(r['success'],bool)
            seen.add(r['seed'])
        return seen

    def cell(self,t,s):return self.q/'cells'/s/t

    def initialize(self,t,s):
        c=self.cell(t,s)
        if (c/'progress.json').exists():return
        with mutex(c/'merge.lock'):
            if (c/'progress.json').exists():return
            p=self.canonical(t,s)/'summary.json'
            old=load(p) if p.exists() else {}
            rows=[r for r in old.get('records',[]) if r.get('status')=='evaluated']
            self.validate(t,s,rows)
            if old.get('evaluated') is not None:assert old['evaluated']==len(rows)
            atomic(c/'frozen_summary.json',old)
            atomic(c/'frozen_records.json',rows)
            atomic(c/'progress.json',dict(seeds=[r['seed'] for r in rows],successes=sum(r['success'] for r in rows)))

    def claims(self,owner):
        for t,s in self.cells:self.initialize(t,s)
        # Long/hard tasks go first; dynamic batches avoid a final task-sized tail.
        def priority(cell):
            t,s=cell;done=load(self.cell(t,s)/'progress.json')['seeds']
            weight=4 if t in ('put_bottles_dustbin','stack_bowls_three','stack_blocks_three') else 1
            return -(50-len(done))*weight
        for t,s in sorted(self.cells,key=priority):
            entries=self.paired[s+'/'+t]
            for start in range(0,50,self.batch_size):
                c=self.cell(t,s);done=set(load(c/'progress.json')['seeds'])
                wanted=[r for r in entries[start:start+self.batch_size] if r['seed'] not in done]
                if not wanted:continue
                b=c/f'batch{start:03d}';b.mkdir(parents=True,exist_ok=True)
                try:(b/'claim').mkdir()
                except FileExistsError:continue
                atomic(b/'claim/owner.json',dict(owner=owner,host=os.uname().nodename,pid=os.getpid(),time=time.time()))
                # Each fixed bucket has exactly one writer, including across nodes.
                done=set(load(c/'progress.json')['seeds'])
                wanted=[r for r in entries[start:start+self.batch_size] if r['seed'] not in done]
                if not wanted:
                    atomic(b/'COMPLETED.json',dict(already_complete=True));continue
                if (b/'paired.json').exists():
                    wanted=load(b/'paired.json')[s+'/'+t]
                    assert all(r in entries[start:start+self.batch_size] for r in wanted)
                else:atomic(b/'paired.json',{s+'/'+t:wanted})
                yield dict(task=t,split=s,entries=wanted,result=b/'result',paired=b/'paired.json',batch=b)

    def commit(self,job):
        t,s,b=job['task'],job['split'],job['batch'];d=load(job['result']/'summary.json')
        rows=[r for r in d['records'] if r.get('status')=='evaluated']
        got=self.validate(t,s,rows)
        assert got<={r['seed'] for r in job['entries']}
        atomic(b/'valid_records.json',rows)
        with mutex(self.cell(t,s)/'merge.lock'):
            records=load(self.cell(t,s)/'frozen_records.json')
            for p in sorted(self.cell(t,s).glob('batch*/valid_records.json')):records+=load(p)
            self.validate(t,s,records)
            order={r['seed']:i for i,r in enumerate(self.paired[s+'/'+t])}
            records.sort(key=lambda r:order[r['seed']])
            n=len(records);ok=sum(r['success'] for r in records);rate=ok/n if n else None
            summary=dict(protocol='strict sync current-obs I2I / no expert goal / no oracle switch',
                tasks={t:dict(evaluated=n,successes=ok,success_rate=rate,rejected_expert_seeds=0)},
                evaluated=n,successes=ok,micro_success_rate=rate,macro_success_rate=rate,
                errors=[],records=records,complete=n==50,
                scheduler='seed_queue_v1',batch_size=self.batch_size,
                source_artifacts=str(self.cell(t,s)))
            atomic(self.canonical(t,s)/'summary.json',summary)
            atomic(self.cell(t,s)/'progress.json',dict(seeds=[r['seed'] for r in records],successes=ok))
        if d.get('errors') or not d.get('complete') or len(rows)!=len(job['entries']):
            atomic(b/'FAILED.json',dict(errors=d.get('errors'),evaluated=len(rows)))
            raise RuntimeError('Batch incomplete; valid results committed, inspect '+str(b))
        atomic(b/'COMPLETED.json',dict(seeds=sorted(got),time=time.time()))

    def complete(self):
        for t,s in self.cells:
            p=self.canonical(t,s)/'summary.json'
            if not p.exists():return False
            d=load(p)
            if not d['complete'] or d['errors'] or self.validate(t,s,d['records'])!={r['seed'] for r in self.paired[s+'/'+t]}:return False
        return True
