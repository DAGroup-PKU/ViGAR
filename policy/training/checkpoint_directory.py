"""Collective checkpoint directory reservation on a shared filesystem."""
from pathlib import Path

def reserve_checkpoint_directory(path, dist):
    path = Path(path)
    error = [None]
    if dist.get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f'Checkpoint directory reservation failed: {error[0]}')
    dist.barrier()
