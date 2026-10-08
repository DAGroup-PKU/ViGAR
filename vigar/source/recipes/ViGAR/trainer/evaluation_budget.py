"""Independent evaluation budgets, with compatibility for historical recipes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationBudget:
    loss_per_rank: int
    generation_per_rank: int
    visual_max: int

    @property
    def windows_per_rank(self):
        return max(self.loss_per_rank, self.generation_per_rank)


def resolve_evaluation_budget(
    world_size,
    *,
    count=32,
    visual_count=8,
    eval_loss_per_rank=None,
    generation_per_rank=None,
    generation_wandb_max=None,
):
    for name, value in dict(
        count=count,
        visual_count=visual_count,
        eval_loss_per_rank=eval_loss_per_rank,
        generation_per_rank=generation_per_rank,
        generation_wandb_max=generation_wandb_max,
    ).items():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if eval_loss_per_rank is None or generation_per_rank is None:
        if count % world_size:
            raise ValueError("Legacy data.eval_count must be divisible by world size for FSDP evaluation")
    loss = count // world_size if eval_loss_per_rank is None else eval_loss_per_rank
    generation = count // world_size if generation_per_rank is None else generation_per_rank
    visuals = visual_count if generation_wandb_max is None else generation_wandb_max
    return EvaluationBudget(loss, generation, min(visuals, generation * world_size))
