import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path
from seed_queue import SeedQueue,atomic,load

class QueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
        paired={s+'/task':[dict(seed=i,instruction='instruction-'+str(i)) for i in range(50)] for s in ['clean','random']}
        atomic(self.r/'paired.json',paired)
        self.plan=dict(tasks=['task'],paired_episode_file=str(self.r/'paired.json'),goal_refresh_mode='sync_current')
        self.canon=lambda t,s:self.r/'canonical'/s/t
        self.q=SeedQueue(self.r,'iter_000050000',self.plan,self.canon)
    def tearDown(self):self.tmp.cleanup()
    def row(self,i,success=False):return dict(task='task',status='evaluated',seed=i,instruction='instruction-'+str(i),success=success)
    def finish(self,j):
        atomic(j['result']/'summary.json',dict(records=[self.row(r['seed']) for r in j['entries']],complete=True,errors=[]))
        self.q.commit(j)
    def test_concurrent_exact_once_and_frozen_preservation(self):
        old=[self.row(i,True) for i in range(7)]
        atomic(self.canon('task','clean')/'summary.json',dict(records=old))
        def worker(k):
            ids=[]
            for j in self.q.claims(str(k)):
                ids += [(j['split'],r['seed']) for r in j['entries']];self.finish(j)
            return ids
        with concurrent.futures.ThreadPoolExecutor(16) as pool:ids=sum(pool.map(worker,range(16)),[])
        self.assertEqual(len(ids),93);self.assertEqual(len(set(ids)),93)
        self.assertTrue(self.q.complete())
        self.assertEqual(load(self.canon('task','clean')/'summary.json')['records'][:7],old)
        self.assertEqual(list(self.q.claims('again')),[])
    def test_instruction_mismatch_rejected(self):
        j=next(self.q.claims('a'));r=self.row(0);r['instruction']='wrong'
        atomic(j['result']/'summary.json',dict(records=[r],complete=True,errors=[]))
        with self.assertRaises(AssertionError):self.q.commit(j)
    def test_incomplete_batch_preserves_valid_records(self):
        j=next(self.q.claims('a'));atomic(j['result']/'summary.json',dict(records=[self.row(0)],complete=False,errors=['test failure']))
        with self.assertRaises(RuntimeError):self.q.commit(j)
        self.assertEqual(load(self.canon('task','clean')/'summary.json')['evaluated'],1)
        self.assertFalse(self.q.complete())
    def test_duplicate_records_rejected(self):
        with self.assertRaises(AssertionError):self.q.validate('task','clean',[self.row(1),self.row(1)])
    def test_batch_contract_change_rejected(self):
        with self.assertRaises(AssertionError):SeedQueue(self.r,'iter_000050000',self.plan,self.canon,batch_size=1)

if __name__=='__main__':unittest.main()
