"""Observe raw gradients before native clipping and finite sanitization."""

import cosmos_framework.callbacks.grad_clip as native_grad_clip
import torch
import torch.distributed as dist
from cosmos_framework.utils.callback import Callback

from .evaluator import cpu_tree, selected_parameters


class UpdateAudit(Callback):
    def __init__(self, record=True):
        self.record = record
        self.gradients, self.before, self.clip_norms = {}, {}, {}
        self.original_clip = native_grad_clip._clip_grad

        def observed_clip(*args, **kwargs):
            result = self.original_clip(*args, **kwargs)
            self.clip_norms = cpu_tree(dict(global_norm=result[0], per_mesh=result[1]))
            return result

        native_grad_clip._clip_grad = observed_clip

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        if self.record:
            self.gradients = selected_parameters(model, gradients=True)
            self.before = selected_parameters(model)
        valid = torch.ones((), device="cuda", dtype=torch.int32)
        for parameter in model.net.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
                valid *= torch.isfinite(gradient).all().to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise FloatingPointError("Nonfinite ViGAR gradient before clipping")

    def close(self):
        native_grad_clip._clip_grad = self.original_clip
