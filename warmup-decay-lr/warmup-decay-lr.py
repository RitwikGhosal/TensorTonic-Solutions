def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    lr = base_lr
    if current_step < warmup_steps:
        lr = base_lr * (current_step / warmup_steps)
    else:
        lr = base_lr * ((total_steps - current_step) / (total_steps - warmup_steps))
    return lr