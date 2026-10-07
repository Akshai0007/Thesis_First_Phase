"""Hand-written Qwen2.5-style decoder-only transformer.

Every component below was written from raw PyTorch after reading the actual
HF `transformers` source for Qwen2 (modeling_qwen2.py) -- not from memory of
"how transformers generally work". The specific choices that make this
*Qwen*-style rather than generic-modern-style, each confirmed against that
source:
  - QKV projections carry a BIAS (query/key/value), but the output
    projection does NOT. Verified directly against the installed
    transformers source (modeling_qwen2.py): Qwen2Attention.__init__
    HARDCODES `bias=True` on q_proj/k_proj/v_proj and `bias=False` on
    o_proj -- this is NOT read from any config field in this library
    version, so `Qwen2Config(attention_bias=...)` has no effect on it at
    all. Our `qkv_bias` field exists mainly for documentation clarity --
    real Qwen2/2.5 is always bias=True here, full stop.
  - RoPE theta defaults to 1,000,000 (not the more common 10,000).
  - Grouped-query attention with repeat_kv (not MHA).
  - RMSNorm (pre-norm), SwiGLU MLP, tied input/output embeddings (0.5B size).

Deliberately simplified relative to an earlier revision of this file: QK-norm
and a decoupled head_dim (num_heads * head_dim != hidden_size) were briefly
supported to also cover Qwen3, but have been removed -- this project targets
Qwen2.5-0.5B only, and that extra generality existed solely for Qwen3.

Module names (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
input_layernorm, post_attention_layernorm) are also chosen to MATCH real
Qwen2's naming exactly, on purpose -- it is what makes state-dict transplant
to/from the real HF model (see hf_bridge.py) a straight key-for-key copy
instead of a fragile renaming exercise, and it's the mechanism the
equivalence tests in tests/test_hf_equivalence.py rely on (proven: 0.000e+00
logit difference against real HF Qwen2ForCausalLM when given identical
weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS
from ..config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._is_norm = True  # used by init.py to skip normal-init and fill with 1.0 instead

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype))


def precompute_rope(head_dim: int, max_seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                       # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                 # (seq, head_dim) -- HF's layout, NOT interleaved
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (batch, heads, seq, head_dim); cos,sin: (seq, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


class QwenAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads

        # Names match real Qwen2 exactly -- see module docstring.
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              is_causal=attn_mask is None)
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class QwenMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = QwenAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = QwenMLP(cfg)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@MODELS.register("qwen")
class QwenForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_layers = cfg.num_layers      # read by init.py for depth-scaled init
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([QwenDecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Qwen (2, 2.5, 3) always uses RoPE -- no "learned position" option
        # exists for this architecture, unlike the separate from-scratch
        # project's configurable model, so this is unconditional.
        rope_freqs = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", rope_freqs[0], persistent=False)
        self.register_buffer("rope_sin", rope_freqs[1], persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Default-fallback init only -- apply_init() in init.py is the real,
        # distribution-based init used in practice (see init.py's docstring
        # for why: this project's explicit requirement is proper random
        # distributions, never a repeated constant). This fallback just
        # avoids leaving freshly-constructed Linear/Embedding layers at
        # PyTorch's own raw default if apply_init is never called.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attn_mask=None, loss_mask=None):
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos[:s].to(x.dtype), self.rope_sin[:s].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        x = self.norm(x)

        if labels is None:
            return self.lm_head(x), None

        loss = self._chunked_lm_loss(x, labels, loss_mask) if self.cfg.loss_chunk_tokens else \
            F.cross_entropy(self.lm_head(x).reshape(-1, self.cfg.vocab_size), labels.reshape(-1),
                             ignore_index=-100)
        return None, loss

    def _chunked_lm_loss(self, x, labels, loss_mask):
        """Compute the LM head + cross-entropy in chunks along the sequence dim
        instead of materializing (batch, seq, vocab=151936) logits all at once
        -- that tensor alone is ~600MB per 1k tokens at fp32/batch=1, and grows
        with batch size; chunking keeps peak memory bounded regardless of
        batch*seq, at the cost of a python-level loop instead of one big matmul.
        """
        chunk = self.cfg.loss_chunk_tokens
        # .reshape(), not .view(): labels/loss_mask are frequently produced by
        # slicing (e.g. arr[:, 1:] in causal_lm.py's packing, or the padding
        # logic in prompt_response.py) which yields a non-contiguous tensor --
        # .view() would raise on that, .reshape() copies only when it must.
        # Caught by a trainer test that ran a real batch through this path for
        # the first time; every earlier task test only checked list values,
        # never exercised the actual loss computation.
        x_flat, labels_flat = x.reshape(-1, x.size(-1)), labels.reshape(-1)
        mask_flat = loss_mask.reshape(-1) if loss_mask is not None else None
        total_loss, total_count = x.new_zeros(()), 0
        for i in range(0, x_flat.size(0), chunk):
            xc = x_flat[i:i + chunk]
            lc = labels_flat[i:i + chunk]
            logits_c = self.lm_head(xc)
            if mask_flat is not None:
                mc = mask_flat[i:i + chunk]
                lc = lc.masked_fill(~mc, -100)
            per_tok = F.cross_entropy(logits_c, lc, ignore_index=-100, reduction="sum")
            n_valid = (lc != -100).sum()
            total_loss = total_loss + per_tok
            total_count += n_valid.item()
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.cfg.max_seq_len:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        self.train()
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
