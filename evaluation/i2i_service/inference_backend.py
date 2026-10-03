"""Per-server same-instance equivalence check before enabling a validated backend."""
import json,os,time
import numpy as np
import torch


class CheckedBackend:
    def __init__(self,model,generate,profile):
        self.model=model;self.generate_fn=generate;self.profile=profile
        self.selected='pending_same_instance_check';self.checked=False;self.restore=lambda:None
        os.environ['GOALWAM_BATCH_CFG']='0'

    def install(self):
        name=self.profile['backend'];model=self.model
        if name=='batch_cfg':
            os.environ['GOALWAM_BATCH_CFG']='1'
            self.restore=lambda:os.environ.__setitem__('GOALWAM_BATCH_CFG','0')
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
            os.environ['GOALWAM_LANGUAGE_COMPILE_GRANULARITY']='block';model.net.pad_for_cuda_graphs=True
            apply_compile(model.net.language_model,SimpleNamespace(max_autotune_pointwise=False,coordinate_descent_tuning=False,compile_dynamic=False,use_cuda_graphs=True))
        else:raise ValueError('Unapproved inference backend')

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
            print('I2I_BACKEND_FALLBACK '+json.dumps(dict(selected=self.selected,error_type=type(error).__name__,message=str(error)[:500])),flush=True)
            return reference
        exact=np.array_equal(reference,candidate) and np.array_equal(reference,repeated)
        if exact:self.selected=self.profile['backend']
        else:self.restore();self.selected='eager_equivalence_fallback'
        self.checked=True
        print('I2I_BACKEND_ACCEPTANCE '+json.dumps(dict(selected=self.selected,requested=self.profile['backend'],
            same_instance=True,exact_pixels=bool(exact),baseline_seconds=baseline_seconds,
            warm_seconds=warm_seconds,seed=int(seed),steps=35,guidance=2.5,shift=5,
            gpu=torch.cuda.get_device_name(),upstream_commit=self.profile['upstream_commit'])),flush=True)
        return repeated if exact else reference
