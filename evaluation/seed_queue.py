import json
import os
import shutil
import socket
import uuid
from pathlib import Path


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class SeedQueue:
    def __init__(self, root, checkpoint, plan):
        self.root = Path(root)/Path(checkpoint).name
        self.plan = plan
        self.cells = [(task, split) for task in plan['tasks'] for split in plan['splits']]
        self.host = socket.gethostname()

    def result(self, task, split):
        return self.root/'results'/split/task

    def complete(self, task, split):
        path = self.result(task, split)/'summary.json'
        if not path.exists(): return False
        summary = json.loads(path.read_text())
        records = [r for r in summary['records'] if r.get('status') == 'evaluated']
        expected = self.plan['episodes_per_split']
        return (summary.get('complete') is True and not summary.get('errors') and
                summary.get('evaluated') == expected and len(records) == expected and
                len({r['seed'] for r in records}) == expected and
                all(r['task'] == task and isinstance(r.get('success'), bool) for r in records))

    def all_complete(self):
        return all(self.complete(task, split) for task, split in self.cells)

    def has_active_claims(self):
        for path in (self.root/'claims').glob('*/*/owner.json'):
            try: owner = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError): continue
            if owner['host'] != self.host or process_alive(owner['pid']): return True
        return False

    def claims(self, owner):
        for task, split in self.cells:
            if self.complete(task, split): continue
            lock = self.root/'claims'/split/task
            lock.parent.mkdir(parents=True, exist_ok=True)
            try:
                lock.mkdir()
            except FileExistsError:
                guard = lock.with_name(lock.name+'.reclaim')
                try: guard.mkdir()
                except FileExistsError: continue
                try:
                    try: record = json.loads((lock/'owner.json').read_text())
                    except (FileNotFoundError, json.JSONDecodeError): continue
                    if record['host'] != self.host or process_alive(record['pid']): continue
                    stale = lock.with_name(lock.name+'.stale.'+uuid.uuid4().hex)
                    try: lock.rename(stale)
                    except FileNotFoundError: continue
                    shutil.rmtree(stale)
                    try: lock.mkdir()
                    except FileExistsError: continue
                finally:
                    guard.rmdir()
            token = uuid.uuid4().hex
            atomic(lock/'owner.json', dict(host=self.host, pid=os.getpid(), owner=owner, token=token))
            try:
                if not self.complete(task, split):
                    yield dict(task=task, split=split, result=self.result(task, split))
            finally:
                if lock.exists():
                    record = json.loads((lock/'owner.json').read_text())
                    if record['token'] == token:
                        (lock/'owner.json').unlink()
                        lock.rmdir()

    def summary(self, checkpoint):
        if not self.all_complete(): raise RuntimeError('Evaluation is incomplete')
        records = []
        paired = {}
        for task, split in self.cells:
            summary = json.loads((self.result(task, split)/'summary.json').read_text())
            rows = [r for r in summary['records'] if r.get('status') == 'evaluated']
            records.extend(dict(r, split=split) for r in rows)
            paired[f'{split}/{task}'] = [dict(seed=r['seed'], instruction=r['instruction']) for r in rows]
        totals = {}
        for split in self.plan['splits']:
            rows = [r for r in records if r['split'] == split]
            successes = sum(r['success'] for r in rows)
            totals[split] = dict(episodes=len(rows), successes=successes, success_rate=successes/len(rows))
        atomic(self.root/'episodes.json', paired)
        result = dict(complete=True, checkpoint=str(checkpoint), episodes=len(records), totals=totals)
        atomic(self.root/'summary.json', result)
        return result
