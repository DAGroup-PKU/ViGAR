"""Restore exact policy-trajectory EEF/camera streams; never use subgoal planner episode IDs."""
import concurrent.futures,hashlib,json,os
from pathlib import Path
import h5py,numpy as np,pyarrow.parquet as pq
from reference.geometry import world_pose_to_canonical,ACTION_MASK,STATE_MASK
DATA=Path(os.environ['VIGAR_ANNOTATED_DATASET'])
OUT=Path(os.environ['VIGAR_DATASET49_WORK'])
def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 rows=json.loads((DATA/'family_mapping_v1.json').read_text())['episodes']
 states={};actions={}
 for f in sorted((DATA/'data').rglob('*.parquet')):
  table=pq.read_table(f,columns=['episode_index','frame_index','observation.state','action']).to_pandas()
  for eid,t in table.groupby('episode_index'):
   assert int(eid)not in states
   t=t.sort_values('frame_index');assert t.frame_index.tolist()==list(range(len(t)))
   states[int(eid)]=np.stack(t['observation.state']).astype(np.float32)
   actions[int(eid)]=np.stack(t['action']).astype(np.float32)
 assert len(states)==len(rows)==2500
 (OUT/'metadata').mkdir(parents=True,exist_ok=True)
 def one(r):
  eid=int(r['episode_index']);src=Path(r['source_hdf5']);target=OUT/'metadata'/f'episode_{eid:06d}.npz'
  with h5py.File(src)as f:
   q=np.asarray(f['joint_action/vector'],dtype=np.float32)
   assert np.array_equal(q,states[eid]),('State alignment',eid)
   cam=np.asarray(f['observation/head_camera/cam2world_gl'],dtype=np.float64)
   # The reference stores camera-local EEFs with an identity camera anchor.
   # This is only valid across target rows when camera extrinsics are constant.
   assert np.allclose(cam,cam[0],atol=1e-6,rtol=0),('Moving camera requires world-frame conversion',eid)
   s=np.zeros((len(q),49),dtype=np.float32);s[:,48]=1
   for side,j,g,e in [('left',0,7,26),('right',8,15,34)]:
    s[:,j:j+6]=np.asarray(f[f'joint_action/{side}_arm'],dtype=np.float32)
    for group,idx in [('joint_action',g),('endpose',e+7)]:
     grip=np.asarray(f[f'{group}/{side}_gripper'],dtype=np.float32)
     assert np.isfinite(grip).all() and grip.min()>=-1e-4 and grip.max()<=1.0001
     s[:,idx]=(1-np.clip(grip,0,1))*100
    s[:,e:e+7]=np.stack([world_pose_to_canonical(p,c)for p,c in zip(f[f'endpose/{side}_endpose'],cam)])
   a=actions[eid];same=bool(np.array_equal(a,q));next_prefix=bool(np.array_equal(a[:-1],q[1:]))
   # Do not fabricate next-pose targets until the command alignment is audited.
   tmp=target.with_suffix('.npz.tmp')
   with tmp.open('wb')as h:np.savez_compressed(h,state49=s,joints14=q,recorded_action14=a,cam2world_gl=cam,state_valid_mask=STATE_MASK,action_valid_mask=ACTION_MASK)
   os.replace(tmp,target)
   return dict(episode=eid,task=r['family'],frames=len(q),source=str(src),state_sha256=sha(q),canonical_sha256=sha(s),action_equals_state=same,action_matches_next_state_prefix=next_prefix,final_action_matches_final_state=bool(np.array_equal(a[-1],q[-1])),file=str(target))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  results=list(pool.map(one,rows))
 summary=dict(episodes=len(results),frames=sum(x['frames']for x in results),state_alignment_verified=True,action_equals_state=sum(x['action_equals_state']for x in results),action_matches_next_state_prefix=sum(x['action_matches_next_state_prefix']for x in results),last_command_equals_last_state=sum(x['final_action_matches_final_state']for x in results),training_ready=False,reason='Recovered state geometry only; EEF action alignment, normalizers, native training integration and paired tests must pass before launch')
 (OUT/'metadata_manifest.json').write_text(json.dumps(results,indent=2))
 (OUT/'METADATA_RECOVERED.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
