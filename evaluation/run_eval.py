"""Native49D policy + I2I planner: eight independent model pairs across two nodes."""
import concurrent.futures,fcntl,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path

import yaml
import threading
from seed_queue import SeedQueue

OUT=Path(os.environ['NATIVE_EVAL_ROOT']);PLAN=json.loads((OUT/'PLAN.json').read_text())
SCRIPTS=Path(__file__).resolve().parent;SOURCE=Path(PLAN['source']);VENDOR=SCRIPTS/'i2i_service'
NODE=int(os.environ['EVAL_NODE_RANK'])
SIMPY=os.environ['NATIVE_SIM_PYTHON']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)



def canonical_result(checkpoint,task,split):
    node=(PLAN['tasks'].index(task)%8)//4
    return OUT/Path(checkpoint).name/('node'+str(node))/'results'/split/task

def task_complete(checkpoint,task):
    for split in ('clean','random'):
        p=canonical_result(checkpoint,task,split)/'summary.json'
        if not p.exists():return False
        d=json.loads(p.read_text())
        if not d['complete'] or d['errors'] or d['evaluated']!=50:return False
    return True

def claim_tasks(checkpoint):
    queue=OUT/'task_claims'/Path(checkpoint).name;queue.mkdir(parents=True,exist_ok=True)
    def remaining(task):
        count=0
        for split in ('clean','random'):
            p=canonical_result(checkpoint,task,split)/'summary.json'
            count+=50-(json.loads(p.read_text())['evaluated'] if p.exists() else 0)
        return count
    for task in sorted(PLAN['tasks'],key=lambda t:-remaining(t)):
        if task_complete(checkpoint,task):continue
        claim=queue/(task+'.claim')
        try:claim.mkdir()
        except FileExistsError:continue
        # mkdir is atomic across the shared filesystem; flock was only local here.
        try:
            if not task_complete(checkpoint,task):yield task
        finally:
            claim.rmdir()

def run_pair(checkpoint,pair):
    root=OUT/Path(checkpoint).name/('node'+str(NODE));attempt=root/f'pair{pair}'/str(time.time_ns())
    attempt.mkdir(parents=True);processes=[];streams=[]
    base=PLAN.get('node_base_ports',{}).get(str(NODE),int(os.environ.get('NATIVE_BASE_PORT','60100')))
    port=int(base)+NODE*100+pair*10
    if os.environ.get('NATIVE_STRICT_IDLE')=='1' or PLAN.get('strict_idle'):
        gpuids=f'{pair*2},{pair*2+1}'
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
        planner_service=Path(PLAN.get('i2i_service_dir',str(VENDOR)))
        write(attempt/'I2I_BACKEND_REQUEST.json',dict(service=str(planner_service),policy_checkpoint=checkpoint))
        planner=spawn([sys.executable,str(planner_service/'serve_i2i_goal.py'),'--checkpoint',PLAN['planner'],
            '--sft-toml',str(release/'episode_image_edit_nano.toml'),'--port',str(port)],'planner.log',
            env(pair*2,[release,SCRIPTS,planner_service]),release)
        policy_env=env(pair*2+1,[SOURCE,SOURCE/'third_party/goalwam'])
        policy_env['MASTER_PORT']=str(port+1001)
        policy=spawn([sys.executable,'-m','recipes.simulation.robotwin.goalwam.server','--checkpoint',checkpoint,'--weights','ema',
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
        write(attempt/'MODEL_READY.json',dict(policy=health,planner=ping))
        stop_lanes=threading.Event()
        def run_lane(lane):
            try:
                queue=SeedQueue(OUT,checkpoint,PLAN,lambda task,split:canonical_result(checkpoint,task,split))
                batch_count=0
                for job in queue.claims(f'{NODE}:{pair}:{attempt.name}:lane{lane}'):
                    if stop_lanes.is_set():return
                    task=job['task']
                    for split in [job['split']]:
                        result=job['result'];summary=result/'summary.json'
                        cfg=dict(workspace=PLAN['simroot'],robotwin_root=PLAN['simroot'],server_url=f'http://127.0.0.1:{port+1}',
                            tasks=[task],task_config='demo_clean' if split=='clean' else 'demo_randomized',episodes=len(job["entries"]),
                            instruction_count=50,start_seed=400000,max_seed_attempts=1000,instruction_type='unseen',
                            action_type='qpos',replan_steps=32,max_policy_steps=None,generation_seed=9000,request_timeout=600,
                            save_video=False,output=str(result),sampling=dict(num_steps=10,guidance=1.0,shift=2.0),
                            async_i2i=True,i2i_port=port,policy_obs_order='legacy_bgr',joint_smoothing_window=1,
                            goal_refresh_mode=PLAN.get('goal_refresh_mode','async_latest'))
                        if PLAN.get('paired_episode_file'):
                            cfg.update(paired_episode_file=str(job['paired']),paired_split=split)
                        config=attempt/f'{task}.{split}.{job["batch"].name}.yaml';config.write_text(yaml.safe_dump(cfg))
                        e=env(pair*2+1,[SCRIPTS,SOURCE,SOURCE/'third_party/goalwam',VENDOR])
                        e.update(PATH=str(Path(SIMPY).parent)+':'+e['PATH'],LD_LIBRARY_PATH=str(Path(SIMPY).parent.parent/'lib')+':/usr/local/nvidia/lib64:/usr/local/nvidia/lib',
                            GOALWAM_RENDER_DEVICE='cuda:0')
                        while True:
                            before=len(json.loads(summary.read_text())['records']) if summary.exists() else 0
                            sim=spawn([SIMPY,str(SCRIPTS/'native_async_rollout.py'),'--config',str(config)],f'{task}.{split}.{job["batch"].name}.{time.time_ns()}.log',e,PLAN['simroot'])
                            code=sim.wait()
                            if code!=0:
                                if summary.exists():queue.commit(job)
                                raise RuntimeError(f'Simulator failed: {task}/{split}')
                            data=json.loads(summary.read_text());assert not data['errors']
                            if data['complete']:break
                            assert len(data['records'])>before, 'Simulator made no progress'
                        queue.commit(job)
                        trace=Path(str(result)+'.async_goals.jsonl')
                        traces=[json.loads(l) for l in trace.read_text().splitlines()]
                        assert traces and all(x['goal_refresh_mode']=='sync_current' and
                            x['source_observation_version']==x['current_observation_version'] and
                            not x['expert_goals'] and not x['oracle_stage_switching'] for x in traces)
                        write(job['batch']/'SYNC_VERIFIED.json',dict(queries=len(traces),policy_checkpoint=checkpoint))
                        batch_count+=1
                        if os.environ.get('SEED_QUEUE_MAX_BATCHES') and batch_count>=int(os.environ['SEED_QUEUE_MAX_BATCHES']):break
                    if os.environ.get('SEED_QUEUE_MAX_BATCHES') and batch_count>=int(os.environ['SEED_QUEUE_MAX_BATCHES']):break
            except BaseException:
                stop_lanes.set();raise
        with concurrent.futures.ThreadPoolExecutor(2) as lane_pool:
            list(lane_pool.map(run_lane,range(2)))
        write(attempt/'COMPLETED.json',dict(complete=True))
    except Exception as e:
        write(attempt/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:
        for p in reversed(processes):
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in streams:f.close()


def main():
    if len(sys.argv)>1:
        run_pair(sys.argv[1],int(sys.argv[2]));return
    lock=(OUT/('eval.node'+str(NODE)+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for checkpoint in PLAN['checkpoints']:
        if not (Path(checkpoint)/'complete.json').exists():raise FileNotFoundError(Path(checkpoint)/'complete.json')
        if all(task_complete(checkpoint,t) for t in PLAN['tasks']):continue
        slots=PLAN['pair_slots'][str(NODE)]
        with concurrent.futures.ThreadPoolExecutor(len(slots)) as pool:
            codes=list(pool.map(lambda p:subprocess.call([sys.executable,__file__,checkpoint,str(p)]),slots))
        if any(codes):raise RuntimeError('Evaluation pair failed; preserve all records')
        write(OUT/Path(checkpoint).name/('eval.node'+str(NODE)+'.done.json'),dict(complete=True))
        while not all((OUT/Path(checkpoint).name/('eval.node'+str(n)+'.done.json')).exists() for n in (0,1)):time.sleep(10)
        assert all(task_complete(checkpoint,t) for t in PLAN['tasks'])
        if NODE!=0:continue
        records=[]
        for task in PLAN['tasks']:
            for split in ('clean','random'):
                d=json.loads((canonical_result(checkpoint,task,split)/'summary.json').read_text())
                records.extend(dict(r,split=split) for r in d['records'] if r['status']=='evaluated')
        assert len(records)==5000
        write(OUT/Path(checkpoint).name/'COMPLETED.json',dict(complete=True,episodes=5000,checkpoint=checkpoint,totals={sp:dict(episodes=2500,success=sum(r['success'] for r in records if r['split']==sp)) for sp in ('clean','random')}))
    if NODE==0:write(OUT/'COMPLETED.json',dict(complete=True,checkpoints=PLAN['checkpoints'],episodes=5000*len(PLAN['checkpoints'])))
if __name__=='__main__':main()
