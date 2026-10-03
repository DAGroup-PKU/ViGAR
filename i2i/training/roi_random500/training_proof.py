"""Low-frequency evidence of real optimizer work; no GPU smoke workload."""
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
from cosmos_framework.utils.callback import Callback
from protocol import append_json


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def optimizer_groups(container):
    # Cosmos VFM wraps one or more torch optimizers in OptimizersContainer.
    return [group for optimizer in container.optimizers for group in optimizer.param_groups]


class TrainingProof(Callback):
    def __init__(self, output, batch, world):
        super().__init__()
        self.output, self.batch, self.world = Path(output), batch, world
        self.saved = []
        self.started = time.monotonic()

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        if iteration >= 5 and iteration % 100:
            return
        self.saved = []
        for group in optimizer_groups(optimizer):
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = local(param.grad).detach().flatten()
                value = local(param).detach().flatten()
                if grad.numel() and value.numel():
                    # Sample the coordinate with the largest gradient, not an arbitrary zero shard.
                    index = int(grad.abs().argmax().item())
                    if not torch.isfinite(grad[index]):
                        raise RuntimeError('Nonfinite sampled optimizer gradient')
                    if float(grad[index].abs()) > 0:
                        self.saved.append((param, index, value[index].clone(), float(grad[index].abs())))
                if len(self.saved) == 4:
                    break
            if len(self.saved) == 4:
                break

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration=0):
        if iteration >= 5 and iteration % 100:
            return
        delta = sum(float((local(p).detach().flatten()[i] - before).abs()) for p, i, before, _ in self.saved)
        evidence = torch.tensor([sum(row[3] for row in self.saved), delta], device='cuda', dtype=torch.float64)
        dist.all_reduce(evidence)
        row = dict(optimizer_step=iteration + 1, rank=dist.get_rank(), sampled_grad_abs_sum=float(evidence[0]),
                   sampled_update_abs_sum=float(evidence[1]), world_size=self.world,
                   configured_global_batch=self.batch * self.world,
                   lr=float(optimizer_groups(optimizer)[0]['lr']), elapsed_seconds=time.monotonic() - self.started,
                   max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        append_json(self.output / f'optimizer.rank{dist.get_rank()}.jsonl', row)
        if dist.get_rank() == 0:
            print('QUALITY_OPTIMIZER_UPDATE', json.dumps(row), flush=True)
        # A zero sampled delta can be bfloat16 quantization, not a failed optimizer.
        # The receipt is evidence for inspection; sampling alone must not abort training.

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration > 5 and iteration % 100:
            return
        if dist.get_rank() == 0:
            row = dict(iteration=iteration, loss=float(local(loss).detach().mean()),
                       elapsed_seconds=time.monotonic() - self.started)
            append_json(self.output / 'loss.jsonl', row)
            print('QUALITY_TRAIN_PROGRESS', json.dumps(row), flush=True)
