import ast,math,unittest
from pathlib import Path
from lookahead_contract import select_stage,policy_case

class Contract(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(select_stage([0,120],[120,200],101,.15),0)
        self.assertEqual(select_stage([0,120],[120,200],102,.15),1)
        self.assertEqual(select_stage([0,120],[120,200],119,0),0)
    def test_final_no_redirect(self):
        self.assertEqual(select_stage([0,120],[120,200],199,.15),1)
    def test_ceil_and_single(self):
        self.assertEqual(select_stage([0,11],[11,20],8,.15),0)
        self.assertEqual(select_stage([0,11],[11,20],9,.15),1)
        self.assertEqual(select_stage([0],[11],10,.15),0)
    def test_coupled_case(self):
        a=dict(end_frame_exclusive=120,target_frame=119,phase_text='first',goal_path='first.png')
        b=dict(end_frame_exclusive=200,target_frame=199,phase_text='second',goal_path='second.png')
        self.assertIs(policy_case([a,b],102),b)
    def test_actual_shared_implementation(self):
        p=Path(__file__).resolve().parents[2]/'i2i/training_runtime/cosmos_framework/data/vfm/local_datasets/episode_image_edit_dataset.py'
        self.assertTrue(p.is_file(), 'Packaged stage-selector source is required')
        tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_subgoal_target_segment_index')
        ns={'math':math,'_DEFAULT_NEXT_SUBGOAL_TAIL_FRACTION':.15}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
        for lengths in [(11,9),(120,80),(1,1),(15,16,17)]:
            starts=[];ends=[];last=0
            for n in lengths:starts.append(last);last+=n;ends.append(last)
            segs=tuple((s,e,False,'','') for s,e in zip(starts,ends))
            for ratio in [0,.15,1]:
                for frame in range(last):self.assertEqual(select_stage(starts,ends,frame,ratio),ns[fn.name](segs,frame,ratio))
if __name__=='__main__':unittest.main()
