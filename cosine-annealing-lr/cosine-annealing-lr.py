import math

def cosine_annealing_schedule(base_lr: float, min_lr: float, total_steps: int, current_step: int) -> float:
    """
    Returns the cosine-annealed learning rate for the requested step.
    """
    lr = min_lr + 0.5 *(base_lr - min_lr) * (1 + math.cos(math.pi * current_step / total_steps))
    return lr