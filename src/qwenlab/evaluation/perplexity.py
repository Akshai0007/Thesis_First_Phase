"""Perplexity = exp(average cross-entropy loss per predicted token). Lower = the model is less "surprised" by
the held-out text. The average is weighted by the number of predicted tokens in each batch (so a batch with
fewer real targets does not count the same as a full one)."""
import math

import torch


def perplexity_from_loss(loss: float) -> float:
    return math.exp(min(loss, 50.0))      # capped only to avoid overflow if a run has diverged


@torch.no_grad()
def evaluate_lm(model, batch_fn, n_batches: int, device: str, dtype=torch.float32) -> dict:
    """Run `n_batches` batches from batch_fn() -> (input_ids, labels, loss_mask) through the model in eval mode.
    Returns {"loss": nats per token, "perplexity": exp(loss), "tokens": number of predicted tokens}."""
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for _ in range(n_batches):
        input_ids, labels, loss_mask = batch_fn()
        input_ids, labels = input_ids.to(device), labels.to(device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
            n = int((loss_mask.bool() & (labels != -100)).sum())
        else:
            n = int((labels != -100).sum())
        with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                            dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(input_ids, labels=labels, loss_mask=loss_mask)
        total_nll += float(loss) * n
        total_tokens += n
    if was_training:
        model.train()
    mean = total_nll / max(total_tokens, 1)
    return {"loss": mean, "perplexity": perplexity_from_loss(mean), "tokens": total_tokens}
