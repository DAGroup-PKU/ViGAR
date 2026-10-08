"""Create an isolated 49-D view of the exact 2500 policy trajectories."""
import json,os
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from reference.geometry import ACTION_MASK,STATE_MASK
ROOT=Path(os.environ['VIGAR_DATASET49_WORK'])
OLD=Path(os.environ['VIGAR_ANNOTATED_DATASET'])
def main():
 proof=json.loads((ROOT/'METADATA_RECOVERED.json').read_text());assert proof['episodes']==2500 and proof['action_equals_state']==2500
 out=ROOT/'dataset49';(out/'meta/episodes/chunk-000').mkdir(parents=True,exist_ok=True);(out/'data/chunk-000').mkdir(parents=True,exist_ok=True)
 info=json.loads((OLD/'meta/info.json').read_text());base=(OLD/'data').resolve().parent
 info.update(eef_gripper_frame='astribot_s1',action_dim_mask=ACTION_MASK.tolist(),state_dim_mask=STATE_MASK.tolist(),data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',video_path=str(base/'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'))
 for k in ('observation.state','action'):info['features'][k]['shape']=[49]
 for k in ('observation.state_valid_mask','action_valid_mask'):info['features'][k]=dict(dtype='bool',shape=[49],names=None)
 episodes={int(e['episode_index']):e for f in (base/'meta/episodes').glob('chunk-*/*.parquet')for e in pq.read_table(f).to_pylist()}
 manifests=json.loads((OLD/'family_mapping_v1.json').read_text())['episodes'];meta=[]
 for r in manifests:
  eid=int(r['episode_index']);dst=out/'data/chunk-000'/f'file-{eid:03d}.parquet'
  with np.load(ROOT/'metadata'/f'episode_{eid:06d}.npz')as z:
   s=z['state49'];n=len(s);action=np.zeros_like(s);action[:-1]=s[1:]
   sm=np.broadcast_to(STATE_MASK,s.shape).copy();am=np.broadcast_to(ACTION_MASK,s.shape).copy();am[-1]=False
   table=pa.table({'episode_index':pa.array([eid]*n,pa.int64()),'frame_index':pa.array(range(n),pa.int64()),'timestamp':np.arange(n,dtype=np.float32)/30,'observation.state':pa.array(s.tolist(),pa.list_(pa.float32(),49)),'action':pa.array(action.tolist(),pa.list_(pa.float32(),49)),'observation.state_valid_mask':pa.array(sm.tolist(),pa.list_(pa.bool_(),49)),'action_valid_mask':pa.array(am.tolist(),pa.list_(pa.bool_(),49))})
   pq.write_table(table,dst,compression='zstd')
  ep={k:v for k,v in episodes[eid].items()if not k.startswith('stats/')}
  ep.update({'data/chunk_index':0,'data/file_index':eid,'dataset_from_index':sum(x['length']for x in meta),'dataset_to_index':sum(x['length']for x in meta)+n,'length':n,'tasks':[r['task']],'annotation':json.dumps(dict(task=dict(command=dict(en=r['task'])),fps=30,duration=n/30,resolution=[240,320],segments=[]))})
  meta.append(ep)
  if eid%250==0:print('CONVERTED',eid,flush=True)
 pq.write_table(pa.Table.from_pylist(meta),out/'meta/episodes/chunk-000/file-000.parquet')
 pq.write_table(pa.table({'task':[r['task']for r in manifests]}),out/'meta/tasks.parquet')
 (out/'meta/info.json').write_text(json.dumps(info,indent=2))
 manifest={'robotwin_clean2500':dict(robot_type='robotwin_aloha_agilex',dataset_type='lerobot',data_path=str(out),fps=30,sampling_rate=1,num_episodes=2500)}
 (ROOT/'dataset49_manifest.json').write_text(json.dumps(manifest,indent=2))
 receipt=dict(complete=True,episodes=2500,frames=sum(x['length']for x in meta),action_alignment='explicit next measured state for rows0..N-2; unknown final command masked, never fabricated hold',source=str(OLD),raw_data_untouched=True,video_copy=False)
 (out/'READY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
