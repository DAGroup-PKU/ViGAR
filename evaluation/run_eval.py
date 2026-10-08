import argparse,concurrent.futures,fcntl,json,os,signal,subprocess,sys,time,urllib.request,shutil,socket
from pathlib import Path

import tempfile

import yaml
import threading
from seed_queue import SeedQueue

OUT=Path(os.environ['NATIVE_EVAL_ROOT']);PLAN=json.loads((OUT/'PLAN.json').read_text())
SCRIPTS=Path(__file__).resolve().parent;SOURCE=Path(PLAN['source']);VENDOR=SCRIPTS/'planner_service'
NODE=int(os.environ.get('EVAL_NODE_RANK','0'))
SIMPY=os.environ['NATIVE_SIM_PYTHON']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)



def devices():
    import torch
    count = torch.cuda.device_count()
    required = PLAN['gpus_per_node']
    if count < required:
        raise RuntimeError(f"Evaluation needs {required} visible GPUs; found {count}")
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    return [value.strip() for value in mask.split(',')[:required]] if mask else [str(i) for i in range(required)]


def run_pair(checkpoint,pair):
    gpu_ids=devices();planner_gpu,policy_gpu=gpu_ids[2*pair:2*pair+2]
    root=OUT/Path(checkpoint).name/('node'+str(NODE));attempt=root/f'pair{pair}'/str(time.time_ns())
    attempt.mkdir(parents=True);processes=[];streams=[]
    base=int(os.environ.get('NATIVE_BASE_PORT',PLAN['base_port']))
    port=int(base)+NODE*100+pair*10
    if os.environ.get('NATIVE_STRICT_IDLE')=='1' or PLAN.get('strict_idle'):
        gpuids=f'{planner_gpu},{policy_gpu}'
        def idle():
            apps=subprocess.check_output(['nvidia-smi','-i',gpuids,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            rows=subprocess.check_output(['nvidia-smi','-i',gpuids,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
            return not apps and len(rows)==2 and all(int(a)<=1 and int(b)==0 for a,b in (r.split(',') for r in rows))
        while True:
            if idle():
                time.sleep(10)
                if idle():break
            time.sleep(30)
    def env(gpu,paths):
        e=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',
            MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port+1000),PYTHONPATH=':'.join(map(str,paths)),
            WANDB_MODE='disabled',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1',
            COSMOS_TRAINING='1',TOKENIZERS_PARALLELISM='false',IMAGINAIRE_OUTPUT_ROOT=str(attempt/'model_outputs'))
        e.pop('WANDB_API_KEY',None);return e
    spawn_lock=threading.Lock()
    def _spawn(command,name,e,cwd=None):
        f=(attempt/name).open('w');streams.append(f)
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=e,cwd=cwd,start_new_session=True)
        processes.append(p);write(attempt/'pids.json',[dict(pid=p.pid,args=p.args) for p in processes]);return p
    def spawn(*a,**kw):
        with spawn_lock:return _spawn(*a,**kw)
    try:
        release=Path(PLAN['planner_release'])
        planner_service=Path(PLAN.get('planner_service_dir',str(VENDOR)))
        planner_env=env(planner_gpu,[release,SCRIPTS,planner_service])
        if PLAN.get('fast_inference'):planner_env['VIGAR_PLANNER_BACKEND_CONFIG']=str(SCRIPTS.parent/'configs/planner_backend_fast.json')
        planner=spawn([sys.executable,str(planner_service/'serve_planner.py'),'--checkpoint',PLAN['planner'],
            '--sft-toml',str(release/'episode_image_edit_nano.toml'),'--port',str(port)],'planner.log',
            planner_env,release)
        policy_env=env(policy_gpu,[SOURCE,SOURCE/'third_party/cosmos_runtime'])
        policy_env['MASTER_PORT']=str(port+1001)
        if PLAN.get('fast_inference'):policy_env['VIGAR_FAST_INFERENCE']='1'
        policy=spawn([sys.executable,'-m','recipes.simulation.robotwin.vigar.server','--checkpoint',checkpoint,'--weights','ema',
            '--host','127.0.0.1','--port',str(port+1),'--output',str(attempt/'policy_outputs')],
            'policy.log',policy_env,SOURCE)
        sys.path.insert(0,str(VENDOR));from robotwin_wire import Client
        deadline=time.monotonic()+900
        while True:
            if any(p.poll() is not None for p in (planner,policy)):raise RuntimeError('A model server exited during startup')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port+1}/health',timeout=2) as resp:health=json.load(resp)
                assert health['checkpoint']==checkpoint and health['action_dim']==49 and health['action_horizon']==48
                cli=Client(port,timeout=3)
                try:ping=cli.call(dict(cmd='ping'))
                finally:cli.close()
                assert ping['checkpoint']==PLAN['planner']
                break
            except (OSError,ConnectionError,TimeoutError):
                if time.monotonic()>deadline:raise TimeoutError('Model startup timeout')
                time.sleep(3)
        write(attempt/'status.json',dict(state='ready'))
        stop_lanes=threading.Event()
        def run_lane(lane):
            queue=SeedQueue(OUT,checkpoint,PLAN)
            claims=queue.claims(f'{NODE}:{pair}:{lane}')
            try:
                for job in claims:
                    if stop_lanes.is_set():return
                    task,split,result=job['task'],job['split'],job['result']
                    cfg=dict(workspace=PLAN['simroot'],robotwin_root=PLAN['simroot'],
                        server_url=f'http://127.0.0.1:{port+1}',tasks=[task],
                        task_config='demo_clean' if split=='clean' else 'demo_randomized',
                        episodes=PLAN['episodes_per_split'],instruction_count=PLAN['episodes_per_split'],
                        start_seed=PLAN['start_seed'],max_seed_attempts=PLAN['max_seed_attempts'],
                        instruction_type='unseen',action_type='qpos',replan_steps=32,max_policy_steps=None,
                        generation_seed=9000,request_timeout=600,save_video=False,output=str(result),
                        sampling=dict(num_steps=10,guidance=1.0,shift=2.0),
                        async_planner=True,planner_port=port,policy_obs_order='bgr',joint_smoothing_window=1,
                        goal_refresh_mode=PLAN['goal_refresh_mode'])
                    if PLAN.get('paired_episode_file'):
                        cfg.update(paired_episode_file=PLAN['paired_episode_file'],paired_split=split)
                    config=attempt/f'{task}.{split}.yaml';config.write_text(yaml.safe_dump(cfg))
                    e=env(policy_gpu,[SCRIPTS,SOURCE,SOURCE/'third_party/cosmos_runtime',VENDOR])
                    e.update(PATH=str(Path(SIMPY).parent)+':'+e['PATH'],
                        LD_LIBRARY_PATH=str(Path(SIMPY).parent.parent/'lib')+':/usr/local/nvidia/lib64:/usr/local/nvidia/lib',
                        VIGAR_RENDER_DEVICE='cuda:0',
                        WARP_CACHE_PATH=tempfile.mkdtemp(prefix='warp-',dir=attempt.resolve()))
                    sim=spawn([SIMPY,str(SCRIPTS/'native_async_rollout.py'),'--config',str(config)],
                              f'{task}.{split}.log',e,PLAN['simroot'])
                    code=sim.wait()
                    shutil.rmtree(e['WARP_CACHE_PATH'])
                    if code or not queue.complete(task,split):
                        raise RuntimeError(f'Incomplete evaluation: {task}/{split}; see {attempt}')
            except BaseException:
                stop_lanes.set();raise
            finally:
                claims.close()
        with concurrent.futures.ThreadPoolExecutor(PLAN['simulators_per_model_pair']) as lanes:
            list(lanes.map(run_lane,range(PLAN['simulators_per_model_pair'])))
        write(attempt/'status.json',dict(state='completed'))
    except Exception as e:
        write(attempt/'status.json',dict(state='failed',message=str(e)));raise
    finally:
        for p in reversed(processes):
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in streams:f.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--pair',type=int)
    parser.add_argument('--reset-locks',action='store_true')
    args=parser.parse_args()
    if not 0 <= NODE < PLAN['nodes']:
        raise ValueError(f"EVAL_NODE_RANK must be between 0 and {PLAN['nodes']-1}")
    if args.checkpoint is not None:
        if args.pair is None or not 0 <= args.pair < PLAN['gpus_per_node']//2:
            parser.error('Invalid model pair')
        run_pair(args.checkpoint,args.pair);return
    devices()
    lock=(OUT/f'eval.node{NODE}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.reset_locks and NODE != 0:
        raise ValueError('Reset locks from node 0 before starting the other nodes')
    for checkpoint in PLAN['checkpoints']:
        queue=SeedQueue(OUT,checkpoint,PLAN)
        if queue.all_complete():
            queue.summary(checkpoint);continue
        if args.reset_locks:
            for owner in (queue.root/'claims').glob('*/*/owner.json'):
                record=json.loads(owner.read_text())
                if record['host']==socket.gethostname():
                    from seed_queue import process_alive
                    if process_alive(record['pid']):
                        raise RuntimeError('An evaluation worker is still running')
            shutil.rmtree(queue.root/'claims',ignore_errors=True)
            for old_failure in queue.root.glob('node*.failed.json'): old_failure.unlink()
        failure=queue.root/f'node{NODE}.failed.json'
        failure.unlink(missing_ok=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(PLAN['gpus_per_node']//2) as pool:
                codes=list(pool.map(lambda pair:subprocess.call([
                    sys.executable,__file__,'--checkpoint',checkpoint,'--pair',str(pair)]),
                    range(PLAN['gpus_per_node']//2)))
            if any(codes):raise RuntimeError('An evaluation worker failed; see its log')
            while not queue.all_complete():
                failed=list(queue.root.glob('node*.failed.json'))
                if failed:raise RuntimeError(f'Another node failed: {failed[0]}')
                if not queue.has_active_claims():
                    time.sleep(1)
                    if not queue.all_complete() and not queue.has_active_claims():
                        raise RuntimeError('Unfinished tasks have no active workers; stop old workers and use --reset-locks')
                time.sleep(5)
            queue.summary(checkpoint)
        except BaseException as error:
            write(failure,dict(error=str(error)));raise
    if NODE==0:
        write(OUT/'summary.json',dict(complete=True,checkpoints=PLAN['checkpoints'],
              episodes=PLAN['episodes_per_checkpoint']*len(PLAN['checkpoints'])))


if __name__=='__main__':main()
