"""Any Hugging Face causal LM, presented to qwenlab exactly like the hand-written `qwen` model.

Contract (what trainer / evaluation / generation / init already rely on, nothing more):
    forward(input_ids, labels=None, attn_mask=None, loss_mask=None) -> (logits | None, loss | None)
    .num_layers (int, read by init.py), .cfg (ModelConfig), a state_dict that save/load round-trips.

Select with  {"model": {"name": "hf", "arch": "qwen3", ...shape fields as usual...}}.  The model is built from a config
(random weights, never pretrained) and then initialised by qwenlab's own `init` scheme, same as the hand-written model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from ..registry import MODELS
from .arch_presets import build_hf_config


def _ce_sum_count(logits, labels, mask):
    """(sum of token losses, number of counted tokens) with qwenlab's loss_mask / -100 conventions."""
    if mask is not None:
        labels = labels.masked_fill(~mask.bool(), -100)
    total = F.cross_entropy(logits.float(), labels, ignore_index=-100, reduction="sum")
    return total, (labels != -100).sum()


@MODELS.register("hf")
class HFCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        import transformers
        self.cfg = cfg
        self.num_layers = cfg.num_layers                    # read by init.py (depth-scaled init)
        hf_cfg = build_hf_config(cfg)
        try:
            self.hf = transformers.AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=cfg.attn_impl)
        except TypeError:                                   # very old transformers without that keyword
            self.hf = transformers.AutoModelForCausalLM.from_config(hf_cfg)
        if cfg.gradient_checkpointing:
            self.hf.gradient_checkpointing_enable()
        self._chunk_ok = None                               # decided on first use: can the loss be chunked for this model?

    # ------------------------------------------------------------------ helpers
    def _backbone_and_head(self):
        """(decoder, output-projection weight) when this architecture exposes them, else None (full-logits path)."""
        try:
            dec, head = self.hf.get_decoder(), self.hf.get_output_embeddings()
            if dec is not None and head is not None and hasattr(head, "weight"):
                return dec, head.weight
        except Exception:
            pass
        return None

    def forward(self, input_ids, labels=None, attn_mask=None, loss_mask=None):
        b, s = input_ids.shape
        if s > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {s} is longer than the model's context window "
                             f"({self.cfg.max_seq_len}, the length it is trained for); shorten the input")
        if attn_mask is not None:
            raise NotImplementedError("the hf model supports causal attention only (attn_mask must be None)")

        chunk = self.cfg.loss_chunk_tokens
        if labels is None:
            return self.hf(input_ids=input_ids).logits, None

        parts = self._backbone_and_head() if chunk else None
        if parts is None:                                   # simple path: full logits, one cross-entropy
            logits = self.hf(input_ids=input_ids).logits
            total, n = _ce_sum_count(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                     None if loss_mask is None else loss_mask.reshape(-1))
            return None, total / n.clamp_min(1)

        dec, weight = parts
        hidden = dec(input_ids=input_ids).last_hidden_state
        cap = getattr(self.hf.config, "final_logit_softcapping", None)
        x = hidden.reshape(-1, hidden.size(-1))
        y = labels.reshape(-1)
        m = None if loss_mask is None else loss_mask.reshape(-1)

        def piece(xc, yc, mc):
            lg = xc @ weight.t()
            if cap:
                lg = torch.tanh(lg / cap) * cap
            return _ce_sum_count(lg, yc, mc)

        tot = x.new_zeros((), dtype=torch.float32)
        cnt = torch.zeros((), dtype=torch.long, device=x.device)
        use_ckpt = self.cfg.loss_chunk_checkpoint and torch.is_grad_enabled()
        for i in range(0, x.size(0), chunk):
            xc, yc = x[i:i + chunk], y[i:i + chunk]
            mc = None if m is None else m[i:i + chunk]
            if use_ckpt:
                t, c = checkpoint(piece, xc, yc, mc, use_reentrant=False)
            else:
                t, c = piece(xc, yc, mc)
            tot, cnt = tot + t, cnt + c
        return None, tot / cnt.clamp_min(1)
