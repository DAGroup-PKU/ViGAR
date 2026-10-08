"""Opt-in simulation optimizations; preserve physics, RNG and observation boundaries."""
import functools
import inspect
import textwrap
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def install(defer_render=False,cache_every=1):
    from envs._base_task import Base_Task
    if defer_render:
        original=Base_Task._update_render
        source=textwrap.dedent(inspect.getsource(original))
        assert source.count('self.scene.update_render()')==1
        source=source.replace('self.scene.update_render()',
            "if getattr(self, '_speed_capture_depth', 0) or self.render_freq:\n        self.scene.update_render()")
        scope=dict(original.__globals__);exec(compile(source,'<deferred_render>','exec'),scope)
        Base_Task._update_render=scope['_update_render']
        for name in ['get_obs','_take_picture']:
            fn=getattr(Base_Task,name)
            def wrap(fn):
                @functools.wraps(fn)
                def call(self,*a,**kw):
                    self._speed_capture_depth=getattr(self,'_speed_capture_depth',0)+1
                    try:return fn(self,*a,**kw)
                    finally:self._speed_capture_depth-=1
                return call
            setattr(Base_Task,name,wrap(fn))
    if cache_every>1:
        original=Base_Task.close_env;count=[0]
        @functools.wraps(original)
        def close(self,clear_cache=False):
            if clear_cache:count[0]+=1
            return original(self,clear_cache=bool(clear_cache and count[0]%cache_every==0))
        Base_Task.close_env=close

class AsyncWriter:
    def __init__(self):self.pool=ThreadPoolExecutor(1);self.pending=[]
    def savez(self,path,**arrays):
        # Copy only caller-owned arrays; the network/physics never shares mutable buffers.
        copied={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in arrays.items()}
        self.pending.append(self.pool.submit(np.savez_compressed,path,**copied))
        if len(self.pending)>4:self.pending.pop(0).result()
    def close(self):
        try:
            for p in self.pending:p.result()
        finally:self.pool.shutdown(wait=True)
