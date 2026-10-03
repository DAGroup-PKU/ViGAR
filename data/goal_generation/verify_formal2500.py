"""Verify the immutable strict-quality 2500-episode cache on CPFS."""
import hashlib,json,os
from collections import Counter,defaultdict
from pathlib import Path

R=Path(os.environ['VIGAR_GOAL_GENERATION_WORK'])
DATASET=Path(os.environ['VIGAR_DATASET_V7'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 c=R/'cache';m=json.loads((c/'manifest.json').read_text());f=json.loads((c/'FROZEN.json').read_text())
 assert sha(c/'manifest.json')==f['manifest_sha256']and m['complete']and m['provisional']is False
 assert m['episodes']==2500 and m['slots']==len(m['cases'])==3691 and m['selection_policy']=='strict_then_bounded_mild'
 assert sha(DATASET/'meta/annotations.json')==m['annotations_sha256']
 tasks=defaultdict(set);models=Counter();keys=set()
 for x in m['cases']:
  assert x['key']not in keys;keys.add(x['key']);p=c/x['goal_path'];assert sha(p)==x['goal_sha256']
  b=p.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n'and int.from_bytes(b[16:20],'big')==320 and int.from_bytes(b[20:24],'big')==384
  g=x['vlm_grade'];faults=set(g.get('hard_faults')or())
  if x['admission_tier']=='strict':assert g['decision']=='pass'and g['stage_observable']is True and g['auto_eligible']is True and not faults
  else:
   allowed={'pass','uncertain','reject'}if faults=={'deformation'}else{'pass','uncertain'}
   assert x['admission_tier']=='mild'and g['decision']in allowed and g['stage_observable']is True and faults in (set(),{'deformation'})
   assert g['semantic_match']>=3 and g['object_identity_color']>=3 and g['object_target_geometry']>=3
   assert g['robot_contact_geometry']>=2 and g['physical_plausibility']>=2 and g['background_preservation']>=2
  tasks[x['task']].add(x['episode_index']);models[x['score_model']]+=1
 assert len(tasks)==50 and all(len(v)==50 for v in tasks.values()) and dict(models)==m['selected_score_models']
 (c/'TRANSFER_VERIFIED.json').write_text(json.dumps(dict(manifest_sha256=f['manifest_sha256'],episodes=2500,slots=3691,tasks=50,
  score_models=dict(models),all_sha_verified=True,strict_quality=True)))
 print('H200_FORMAL2500_CACHE_VERIFIED',dict(models),flush=True)
if __name__=='__main__':main()
