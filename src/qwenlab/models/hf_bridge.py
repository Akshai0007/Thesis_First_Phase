"""Weight transplant between our hand-written QwenForCausalLM and HF's
Qwen2ForCausalLM / Qwen3ForCausalLM. Works because module names were chosen
to match exactly (see qwen.py's docstring) -- the only real difference is HF
wrapping everything except lm_head under a top-level "model." prefix.
"""
import re


def hf_state_dict_to_ours(hf_state_dict: dict) -> dict:
    out = {}
    for k, v in hf_state_dict.items():
        if k == "lm_head.weight":
            out["lm_head.weight"] = v
            continue
        m = re.match(r"^model\.(.*)$", k)
        if not m:
            continue  # anything unexpected is dropped, not silently misapplied
        out[m.group(1)] = v
    return out


def ours_state_dict_to_hf(our_state_dict: dict) -> dict:
    out = {}
    for k, v in our_state_dict.items():
        if k == "lm_head.weight":
            out["lm_head.weight"] = v
            continue
        if k.startswith("rope_"):
            continue  # buffers, not real weights -- HF recomputes these itself
        out[f"model.{k}"] = v
    return out
