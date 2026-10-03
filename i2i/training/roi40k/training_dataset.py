"""Original unified sampler, real target metadata, explicit 1000-row fallback."""
from collections import Counter
from functools import lru_cache
import gzip
import json
import os
import torch
from protocol import DATASET,PREVIOUS,WORK,OUTPUT,write_json,append_json,digest
from sampling import UnifiedIndex
from fixed_ee_roi import VIEWS,ee_point,project,rectangular_canvas,image_digest

class SelectedDataset:
    def __init__(self,base,records,geometry):
        self.base=base;self.records=records;self.geometry=geometry
        self.index=UnifiedIndex(base.episodes,base.samples,records)
        self.counts=Counter()

    def __len__(self):return len(self.base)

    @lru_cache(maxsize=128)
    def mask(self,eid,frame):
        if eid not in self.geometry:
            if not self.records[eid].get('source_record'):raise ValueError(f'Unexpected missing ROI: {eid}')
            return torch.zeros(384,320), 'missing_metadata_global_loss'
        target=self.geometry[eid][frame];views={}
        for name,(camera,*_) in VIEWS.items():
            c=target['cameras'][camera]
            views[name]=dict(source_hw=c['source_hw'],ee_pixels={arm:project(ee_point(p),c['extrinsic_cv'],c['intrinsic_cv'])
                for arm,p in target['poses'].items()})
        mask,_=rectangular_canvas(views)
        return mask, 'ee_window' if mask.any() else 'offscreen_global_loss'

    def __getitem__(self,index):
        item=dict(self.base[self.index.resolve(index)])
        eid=int(item['episode_id']);frame=int(item['target_frame_index'])
        if int(item['n_orig_video_frames'])!=self.records[eid]['frame_count']:raise ValueError('Training frame count drift')
        mask,status=self.mask(eid,frame)
        self.counts[status]+=1
        item['_loss_roi_pair']=torch.stack([torch.zeros_like(mask),mask])
        n=sum(self.counts.values())
        if n<=2 or n%1000==0:
            worker=torch.utils.data.get_worker_info()
            append_json(OUTPUT/'dataset_audit'/f'rank{os.environ.get("RANK","0")}.worker{worker.id if worker else 0}.jsonl',
                dict(episode=eid,frame=frame,task=self.records[eid]['task'],domain=self.records[eid]['source_config'],
                    status=status,counts=self.counts,roi_fraction=float(mask.mean()),
                    gt_actual_decoder_sha256=image_digest(item['images'][1]),source_video=item['__url__']))
        return item

def get_dataset(**kwargs):
    from cosmos_framework.data.vfm.local_datasets import episode_image_edit_dataset as upstream
    rows=json.loads((DATASET/'meta/episode_manifest.json').read_text())['episodes']
    records={int(r['episode_index']):r for r in rows}
    geometry={}
    with gzip.open(WORK/'state_aligned_target_metadata.jsonl.gz','rt') as f:
        for r in map(json.loads,f):
            if not r['full_state_alignment']['state_alignment_verified']:raise ValueError('Unverified geometry')
            geometry[r['episode_index']]={t['frame']:t for t in r['targets']}
    missing=set(records)-set(geometry)
    if len(missing)!=1000 or any(not records[e].get('source_record') for e in missing):raise ValueError('Unexpected ROI coverage')
    loader=upstream._load_metadata_cache
    cache=loader(PREVIOUS/'dataset_index/.cache/episode_image_edit_metadata.json')
    info=json.loads((DATASET/'meta/info.json').read_text())
    for r in rows:
        if r.get('source_record'):
            eid=int(r['episode_index'])
            key=info['video_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid,video_key='observation.images.concat_view_384x320')
            cache[key]=dict(width=320,height=384,fps=30.,total_frames=r['frame_count'])
    upstream._load_metadata_cache=lambda _:dict(cache)
    writer=upstream._write_metadata_cache;upstream._write_metadata_cache=lambda *_a,**_k:None
    try:base=upstream.get_episode_image_edit_dataset(**kwargs)
    finally:upstream._load_metadata_cache=loader;upstream._write_metadata_cache=writer
    ds=SelectedDataset(base,records,geometry)
    if int(os.environ.get('RANK','0'))==0:
        draws=Counter(ds.index.by_episode[base.samples[ds.index.resolve(i)][0]] for i in range(100000))
        if any(draws[(task,'seg1_clean')]!=200 or draws[(task,'rand250')]!=1800 for task in ds.index.tasks):raise ValueError('Sampling drift')
        write_json(OUTPUT/'dataset_audit/contract.json',dict(tasks=50,episodes=len(rows),roi_episodes=len(geometry),
            missing_roi_global_loss=len(missing),clean_random='1:9',target='expert GT',same_model=True,
            target_mode=base.target_mode,manifest_sha256=digest(DATASET/'meta/episode_manifest.json'),
            annotation_sha256=digest(DATASET/'meta/annotations.json'),mask_head=96,mask_wrist=48,relative_weight=4))
        print('EE_WINDOW_DATASET_READY original_sampling_1_to_9 missing_1000_global_loss',flush=True)
    return ds
