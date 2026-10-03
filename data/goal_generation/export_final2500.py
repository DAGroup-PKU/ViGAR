import hashlib,json,shutil,os
from collections import Counter,defaultdict
from pathlib import Path
E=Path(os.environ['VIGAR_GOAL_ADMISSION_ROOT']);OUT=Path(os.environ['GOALWAM_GENERATED_GOAL_CACHE']);BASE=Path(os.environ['VIGAR_GOAL_GENERATION_WORK']);DATASET=os.environ['VIGAR_DATASET_V7'];ANNOT='6d5f5d5548a4520a6b1d8ec6d56913ea2d4c8d88f141fe14a67217d64a4b7189'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def main():
 a=E/'FINAL_ADMISSION_HUMAN_COMPLETE.json';x=json.loads(a.read_text());assert x['training_ready']and x['episodes']==2500 and len(x['selected'])==3691
 base={c['key']:c for c in json.loads((BASE/'cases.json').read_text())};assert set(base)=={v['key']for v in x['selected']};rows=[];tasks=defaultdict(set);tiers=Counter();models=Counter()
 for s in sorted(x['selected'],key=lambda z:z['key']):
  b=base[s['key']];src=Path(s['goal_path']);assert sha(src)==s['goal_sha256'];dst=OUT/'goals'/(s['key']+'.png');dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert sha(dst)==s['goal_sha256']
  else:shutil.copy2(src,dst)
  tasks[b['task']].add(b['episode_index']);tiers[s['admission_tier']]+=1;models[s['score_model']]+=1;rows.append(dict(b,goal_path='goals/'+s['key']+'.png',goal_sha256=s['goal_sha256'],selected_candidate=s['candidate'],selection_method=s['admission_tier'],score_model=s['score_model'],score_sha256=s['score_sha256'],human_approved=s['human_approved']))
 assert len(tasks)==50 and all(len(v)==50 for v in tasks.values())
 manifest=dict(schema='goalwam-i2i-fixedgoal-cache/v2',complete=True,provisional=False,cases=rows,checkpoint='targeted8k_plus_historical',weight_variant='regular',dataset=DATASET,annotations_sha256=ANNOT,source_admission_path=str(a),source_admission_sha256=sha(a),split='clean',tasks=50,episodes=2500,slots=3691,selection_policy='strict_then_bounded_mild_plus_two_explicit_human_overrides',admission_tiers=dict(tiers),selected_score_models=dict(models),human_overrides=2,no_gt_fallback=True,training_started=False)
 write(OUT/'manifest.json',manifest);write(OUT/'FROZEN.json',dict(manifest_sha256=sha(OUT/'manifest.json'),episodes=2500,slots=3691,all_goal_sha_verified=True,human_overrides=2,no_gt_fallback=True));print('FINAL2500_CACHE_FROZEN',dict(tiers),dict(models))
if __name__=='__main__':main()
