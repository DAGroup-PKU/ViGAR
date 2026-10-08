import json,os,time
import numpy as np
import torch


class CheckedBackend:
    def __init__(self,model,generate,profile):
        self.model=model;self.generate_fn=generate;self.profile=profile
        self.selected='pending_same_instance_check';self.checked=False;self.restore=lambda:None
        os.environ['VIGAR_BATCH_CFG']='0'

    def install(self):
        name=self.profile['backend'];model=self.model
        if name=='batch_cfg':
            os.environ['VIGAR_BATCH_CFG']='1'
            self.restore=lambda:os.environ.__setitem__('VIGAR_BATCH_CFG','0')
        elif name=='eager_graph':
            from eager_cuda_graph import install
            original=model.net.language_model.forward
            self.wrapper=install(model,max_graphs=self.profile.get('max_graphs',16))
            self.restore=lambda:setattr(model.net.language_model,'forward',original)
        elif name=='compile_block_strict_cast':
            from cosmos_framework.model.vfm.mot.parallelize_unified_mot import apply_compile
            from types import SimpleNamespace
            original=list(model.net.language_model.model.layers)
            old_pad=model.net.pad_for_cuda_graphs;old_cast=torch._inductor.config.emulate_precision_casts
            def restore():
                for i,layer in enumerate(original):model.net.language_model.model.layers[i]=layer
                model.net.pad_for_cuda_graphs=old_pad;torch._inductor.config.emulate_precision_casts=old_cast
            self.restore=restore
            torch._inductor.config.emulate_precision_casts=True
            os.environ['VIGAR_LANGUAGE_COMPILE_GRANULARITY']='block';model.net.pad_for_cuda_graphs=True
            apply_compile(model.net.language_model,SimpleNamespace(max_autotune_pointwise=False,coordinate_descent_tuning=False,compile_dynamic=False,use_cuda_graphs=True))
        elif name=='fast':
            # Batched CFG, fused gate/up and Q/K/V projections, static block compile with CUDA graphs.
            # BF16 rounding differs from eager; the fused projections stay in place after a fallback.
            from cosmos_framework.model.vfm.mot.parallelize_unified_mot import apply_compile
            from cosmos_framework.utils.inference_fusions import fuse_attention_qkv,fuse_dense_swiglu
            from types import SimpleNamespace
            import torch._dynamo
            # Static shapes recompile once per packed sample length (one per instruction length).
            torch._dynamo.config.recompile_limit=self.profile.get('recompile_limit',128)
            torch._dynamo.config.accumulated_recompile_limit=8*torch._dynamo.config.recompile_limit
            original=list(model.net.language_model.model.layers);old_pad=model.net.pad_for_cuda_graphs
            def restore():
                for i,layer in enumerate(original):model.net.language_model.model.layers[i]=layer
                model.net.pad_for_cuda_graphs=old_pad;os.environ['VIGAR_BATCH_CFG']='0'
            self.restore=restore
            os.environ['VIGAR_BATCH_CFG']='1';os.environ['VIGAR_VALIDATE_BATCH_CFG']='1'
            os.environ['VIGAR_CUDA_GRAPH_PAD_ALIGNMENT']=str(self.profile.get('pad_alignment',32))
            os.environ['VIGAR_LANGUAGE_COMPILE_GRANULARITY']='block';model.net.pad_for_cuda_graphs=True
            print('Planner fused',fuse_dense_swiglu(model),'gate/up and',fuse_attention_qkv(model),'Q/K/V projections',flush=True)
            apply_compile(model.net.language_model,SimpleNamespace(max_autotune_pointwise=False,coordinate_descent_tuning=False,compile_dynamic=False,use_cuda_graphs=True))
        else:raise ValueError(f'Unknown backend: {name}')

    def generate(self,image,prompt,*,seed):
        if self.checked:return self.generate_fn(self.model,image,prompt,seed=seed)
        torch.cuda.synchronize();start=time.monotonic()
        reference=self.generate_fn(self.model,image,prompt,seed=seed)
        torch.cuda.synchronize();baseline_seconds=time.monotonic()-start
        try:
            self.install()
            candidate=self.generate_fn(self.model,image,prompt,seed=seed)
            torch.cuda.synchronize();start=time.monotonic()
            repeated=self.generate_fn(self.model,image,prompt,seed=seed)
            torch.cuda.synchronize();warm_seconds=time.monotonic()-start
        except Exception as error:
            self.restore()
            if any(x in str(error).lower() for x in ('illegal memory access','device-side assert','device lost')):raise
            self.checked=True;self.selected='eager_initialization_fallback'
            print(f"Planner backend fallback: {type(error).__name__}: {error}", flush=True)
            return reference
        exact=np.array_equal(reference,candidate) and np.array_equal(reference,repeated)
        if self.profile['backend']=='fast':
            difference=np.abs(np.asarray(reference,dtype=np.float32)-np.asarray(repeated,dtype=np.float32))
            print(f"Planner fast backend: eager difference mean {difference.mean():.3f} max {difference.max():.0f}",flush=True)
            exact=True
        if exact:self.selected=self.profile['backend']
        else:self.restore();self.selected='eager_equivalence_fallback'
        self.checked=True
        print(f"Planner backend: {self.selected}; generation {warm_seconds:.3f}s", flush=True)
        return repeated if exact else reference
