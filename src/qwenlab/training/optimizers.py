"""Optimizers, chosen by name in the config (`train.optimizer`, `train.optimizer_params`). Each builder takes
(model, train_cfg) and returns an object that behaves like a torch optimizer. Adding one = one function with
@OPTIMIZERS.register("name"); the trainer does not change.

"adamw" is exactly the optimizer the pipeline has always used (decay / no-decay groups). "muon_adamw" is NEW and
UNTESTED on real runs: Muon (orthogonalised momentum) on hidden 2-D weights, AdamW on embeddings / norms / biases."""
import itertools

import torch

from ..registry import OPTIMIZERS


def adamw_groups(model, train_cfg):
    """Weight decay only on >=2-D weight matrices; none on norms, biases and (optionally) the embeddings / lm_head."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = "embed_tokens" in name or "lm_head" in name
        if p.dim() < 2 or (train_cfg.no_decay_embeddings and is_embedding):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


@OPTIMIZERS.register("adamw")
def build_adamw(model, train_cfg):
    decay, no_decay = adamw_groups(model, train_cfg)
    groups = [{"params": decay, "weight_decay": train_cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = {}
    if getattr(train_cfg, "fused_adamw", False) and torch.cuda.is_available():
        kw["fused"] = True
    return torch.optim.AdamW(groups, lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2),
                             eps=train_cfg.eps, **kw)


# ----------------------------------------------------------------------------- Muon
_A, _B, _C = 3.4445, -4.7750, 2.0315


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    X = G.to(torch.bfloat16) if G.is_cuda and torch.cuda.is_bf16_supported() else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = _A * X + (_B * A + _C * (A @ A)) @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Momentum + Newton-Schulz orthogonalisation for 2-D weights. scale_mode "match_adamw_rms" (0.2*sqrt(max(rows, cols)))
    lets an AdamW learning rate transfer; "spectral" is max(1, rows/cols)**0.5. Re-tune lr whenever you change scale_mode."""

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5,
                 scale_mode="match_adamw_rms"):
        if scale_mode not in ("match_adamw_rms", "spectral"):
            raise ValueError(f"unknown scale_mode {scale_mode!r}")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps, scale_mode=scale_mode))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                o = newton_schulz(upd, g["ns_steps"]).to(p.dtype)
                rows, cols = p.shape
                scale = 0.2 * max(rows, cols) ** 0.5 if g["scale_mode"] == "match_adamw_rms" else max(1.0, rows / cols) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"] * scale)


class OptimizerBundle:
    """Several torch optimizers used as one: the trainer sets `g["lr"] = base_lr * schedule * g.get("lr_scale", 1)`."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        return list(itertools.chain.from_iterable(o.param_groups for o in self.optimizers))

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"bundle": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["bundle"]):
            o.load_state_dict(s)


@OPTIMIZERS.register("muon_adamw")
def build_muon_adamw(model, train_cfg):
    """params: momentum (0.95), ns_steps (5), scale_mode, muon_lr_mult (1.0: Muon lr = lr * mult), muon_weight_decay."""
    pr = train_cfg.optimizer_params
    muon_params, rest_model_names = [], set()
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() == 2 and "embed_tokens" not in name and "lm_head" not in name:
            muon_params.append(p)
            rest_model_names.add(id(p))
    decay, no_decay = adamw_groups(model, train_cfg)
    decay = [p for p in decay if id(p) not in rest_model_names]       # matrices handled by Muon are not in AdamW
    muon = Muon([{"params": muon_params, "lr_scale": pr.get("muon_lr_mult", 1.0)}], lr=train_cfg.lr,
                momentum=pr.get("momentum", 0.95), weight_decay=pr.get("muon_weight_decay", train_cfg.weight_decay),
                ns_steps=pr.get("ns_steps", 5), scale_mode=pr.get("scale_mode", "match_adamw_rms"))
    groups = [g for g in ({"params": decay, "weight_decay": train_cfg.weight_decay},
                          {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
    adam = torch.optim.AdamW(groups, lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps)
    return OptimizerBundle([muon, adam])
