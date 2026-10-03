"""Eight independent GPU workers; one pinned regular-weight model per GPU."""
import json, os, subprocess, sys, time
from pathlib import Path
RELEASE=Path(__file__).resolve().parents[2]/'i2i/inference_runtime'
OUT=Path(os.environ['VIGAR_GOAL_GENERATION_WORK'])
CHECKPOINT=Path(os.environ['GOALWAM_I2I_CHECKPOINT'])
from curate import sha,write
GPUS=os.environ.get('CURATION_GPU_LIST','0,1,2,3,4,5,6,7').split(',')

def worker(rank):
    import numpy as np
    from PIL import Image
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.context_managers import distributed_init,model_init
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils import distributed
    from evaluate_episode_image_edit_nano import _keep_eval_callbacks
    from robotwin_i2i_goal import generate_goal
    import torch.distributed as dist
    contract=json.loads((OUT/'contract.json').read_text())
    logical_checkpoint=str(CHECKPOINT)
    assert contract['split']=='clean' and contract['checkpoint']==logical_checkpoint
    assert sha(CHECKPOINT/'model/.metadata')==contract['checkpoint_metadata_sha256']
    contract_sha=sha(OUT/'contract.json')
    assert sha(OUT/'cases.json')==json.loads((OUT/'PREPARED.json').read_text())['cases_sha256']
    cases=json.loads((OUT/'cases.json').read_text())[rank::len(GPUS)]
    with distributed_init():distributed.init()
    overrides=['job.project=vigar','job.group=clean_goal_curation',f'job.name=clean_curation_gpu{rank}',
       'job.wandb_mode=disabled',f'checkpoint.load_path={CHECKPOINT}','checkpoint.load_training_state=false',
       'checkpoint.only_load_scheduler_state=false','checkpoint.keys_to_skip_loading=[]',
       'checkpoint.dcp_async_mode_enabled=false','checkpoint.load_ema_to_reg=false',
       'checkpoint.load_ema_to_reg_single_net=false','model.config.ema.enabled=false',
       'trainer.run_validation=false','trainer.run_validation_on_start=false','trainer.max_iter=1',
       'model.config.compile.enabled=false','model.config.parallelism.data_parallel_shard_degree=1',
       'model.config.parallelism.data_parallel_replicate_degree=1']
    config=load_experiment_from_toml(RELEASE/'episode_image_edit_nano.toml',extra_overrides=overrides)
    config.validate();config.freeze();trainer=config.trainer.type(config);_keep_eval_callbacks(trainer)
    with model_init():model=instantiate(config.model)
    model=model.to('cuda',memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format);trainer.checkpointer.load(model)
    trainer.callbacks.on_train_start(model,iteration=10000);model.eval()
    print('CLEAN_MODEL_READY',rank,str(CHECKPOINT),flush=True)
    try:
        for case in cases:
            assert case['split']=='clean'
            folder=OUT/case['key'];ready=folder/'generated.json'
            if ready.exists():
                previous=json.loads(ready.read_text());assert previous['contract_sha256']==contract_sha
                assert all(sha(folder/c['path'])==c['sha256'] for c in previous['case']['candidates'])
                continue
            image=np.asarray(Image.open(folder/'input_rgb.png').convert('RGB')).copy()
            assert sha(folder/'input_rgb.png')==case['input_rgb.png_sha256']
            for c in case['candidates']:
                receipt=folder/(c['id']+'.receipt.json');path=folder/c['path']
                if receipt.exists():
                    prev=json.loads(receipt.read_text());assert prev['contract_sha256']==contract_sha
                    assert prev['seed']==c['seed'] and sha(path)==prev['sha256']
                    c['sha256']=prev['sha256'];continue
                started=time.monotonic()
                goal=generate_goal(model,image,case['prompt'],seed=c['seed'])
                assert goal.dtype==np.uint8 and goal.shape==(384,320,3)
                tmp=path.with_suffix('.tmp');Image.fromarray(goal).save(tmp,format='PNG');tmp.replace(path)
                c['sha256']=sha(path)
                write(receipt,dict(contract_sha256=contract_sha,seed=c['seed'],sha256=c['sha256'],
                    input_sha256=case['input_rgb.png_sha256'],gt_sent_to_i2i=False,seconds=time.monotonic()-started))
            write(ready,dict(contract_sha256=contract_sha,case=case))
            print('CLEAN_GENERATED',rank,case['key'],flush=True)
        write(OUT/f'gpu{rank}.completed.json',dict(slots=len(cases),contract_sha256=contract_sha))
    finally:
        trainer.checkpointer.finalize();dist.destroy_process_group()

def main():
    if len(sys.argv)>1:worker(int(sys.argv[1]));return
    while not (OUT/'PREPARED.json').exists():time.sleep(30)
    (OUT/'logs').mkdir(exist_ok=True)
    processes=[];logs=[]
    for i,gpu in enumerate(GPUS):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,RANK='0',LOCAL_RANK='0',
             WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='0',
             IMAGINAIRE_OUTPUT_ROOT=str(OUT/f'model_outputs/gpu{i}'))
        log=(OUT/f'logs/generate_gpu{i}.log').open('a');logs.append(log)
        processes.append(subprocess.Popen([sys.executable,__file__,str(i)],env=env,stdout=log,stderr=subprocess.STDOUT))
    write(OUT/'generation_pids.json',[p.pid for p in processes])
    codes=[p.wait() for p in processes]
    for log in logs:log.close()
    if any(codes):write(OUT/'GENERATION_FAILED.json',dict(codes=codes));raise SystemExit(1)
    write(OUT/'GENERATION_COMPLETE.json',dict(slots=3691,candidates=14764,split='clean'))

if __name__=='__main__':main()
