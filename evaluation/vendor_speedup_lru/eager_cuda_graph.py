"""Capture the original eager kernels, without changing CFG batching or GEMMs."""
import hashlib
from collections import OrderedDict
from enum import Enum
import torch
from torch.utils._pytree import tree_flatten,tree_unflatten


class EagerGraphForward:
    def __init__(self, original, max_graphs=16):
        if max_graphs < 2:raise ValueError('At least two graphs are required for CFG')
        self.original=original;self.max_graphs=max_graphs;self.cache=OrderedDict()
        self.evictions=0
        self.replays=0;self.bypasses=0
        self.bypass_types=set()

    def __call__(self,*args,**kwargs):
        leaves,spec=tree_flatten((args,kwargs));signature=[]
        for value in leaves:
            if isinstance(value,torch.Tensor):
                row=('tensor',str(value.device),str(value.dtype),tuple(value.shape),tuple(value.stride()))
                if value.device.type=='cpu':
                    row+= (hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),)
                signature.append(row)
            elif value is None or isinstance(value,(bool,int,float,str,torch.dtype,torch.device,Enum,slice,range)):
                signature.append((type(value).__name__,str(value)))
            else:
                typename=type(value).__module__+'.'+type(value).__qualname__
                if typename not in self.bypass_types:
                    self.bypass_types.add(typename)
                    print('EAGER_GRAPH_UNSUPPORTED_LEAF',typename,sorted(vars(value)) if hasattr(value,'__dict__') else '',flush=True)
                self.bypasses+=1;return self.original(*args,**kwargs)
        key=(str(spec),tuple(signature),torch.is_autocast_enabled('cuda'),str(torch.get_autocast_dtype('cuda')))
        if key not in self.cache:
            if torch.is_grad_enabled():
                self.bypasses+=1;return self.original(*args,**kwargs)
            if len(self.cache)>=self.max_graphs:
                # Wait for the last replay/output clone before releasing its graph pool.
                # Keep exact input shapes and kernels; never reuse a mismatched graph.
                torch.cuda.synchronize()
                _,entry=self.cache.popitem(last=False)
                del entry
                self.evictions+=1
                print('EAGER_CUDA_GRAPH_EVICTED',self.evictions,flush=True)
            static=[v.detach().clone() if isinstance(v,torch.Tensor) else v for v in leaves]
            aa,kk=tree_unflatten(static,spec)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self.original(*aa,**kk)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=self.original(*aa,**kk)
            self.cache[key]=(graph,static,output)
            print('EAGER_CUDA_GRAPH_CAPTURED',len(self.cache),flush=True)
        graph,static,output=self.cache[key]
        self.cache.move_to_end(key)
        for dst,src in zip(static,leaves):
            if isinstance(src,torch.Tensor) and src.is_cuda:dst.copy_(src)
        graph.replay();self.replays+=1
        # Cond/uncond may reuse the same graph. Never expose an output buffer
        # that a later replay could overwrite before CFG combines both values.
        flat,out_spec=tree_flatten(output)
        return tree_unflatten([v.clone() if isinstance(v,torch.Tensor) else v for v in flat],out_spec)


def install(model,max_graphs=16):
    assert not model.training
    from cosmos_framework.model.vfm.mot.attention import SplitInfo
    from torch.utils._pytree import register_pytree_node
    def flatten_info(value):
        names=tuple(sorted(vars(value)))
        return [getattr(value,n) for n in names],names
    def unflatten_info(values,names):
        obj=SplitInfo.__new__(SplitInfo)
        obj.__dict__.update(zip(names,values))
        return obj
    try:register_pytree_node(SplitInfo,flatten_info,unflatten_info)
    except ValueError as e:
        if 'already registered' not in str(e):raise
    wrapper=EagerGraphForward(model.net.language_model.forward,max_graphs)
    model.net.language_model.forward=wrapper
    return wrapper
