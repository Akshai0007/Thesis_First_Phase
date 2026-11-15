"""Loss by position inside the context window. If extra context helps, the loss falls as the position grows
(early positions have little to condition on); a flat curve means the model is not using long context.
Used for the context-window comparison between models / sequence lengths: compare the late-position buckets."""
import torch
import torch.nn.functional as F


def position_edges(seq_len: int):
    edges, e = [0], 16
    while e < seq_len:
        edges.append(e)
        e *= 2
    edges.append(seq_len)
    return edges


@torch.no_grad()
def loss_by_position(model, batch_fn, n_batches: int, device, dtype=torch.float32, max_rows_at_once_tokens: int = 32768):
    """-> {"0-16": mean loss, "16-32": ..., "overall": ...}. Needs the model's logits (labels=None path)."""
    was_training = model.training
    model.eval()
    sums, count, seq_len = None, 0, None
    for _ in range(n_batches):
        x, y, _ = batch_fn()
        seq_len = x.size(1)
        rows = max(1, max_rows_at_once_tokens // seq_len)
        for i in range(0, x.size(0), rows):
            xi, yi = x[i:i + rows].to(device), y[i:i + rows].to(device)
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=dtype,
                                enabled=dtype != torch.float32):
                logits, _ = model(xi)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yi.reshape(-1), reduction="none")
            per_pos = ce.view(xi.size(0), seq_len).sum(0).double().cpu()
            sums = per_pos if sums is None else sums + per_pos
            count += xi.size(0)
    if was_training:
        model.train()
    edges = position_edges(seq_len)
    out = {f"{a}-{b}": float(sums[a:b].sum() / (count * (b - a))) for a, b in zip(edges[:-1], edges[1:])}
    out["overall"] = float(sums.sum() / (count * seq_len))
    return out
