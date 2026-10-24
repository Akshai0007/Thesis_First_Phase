"""Write a trained model in Hugging Face format (hf_model/: config.json + model.safetensors + tokenizer files)
so it can be loaded with AutoModelForCausalLM.from_pretrained(repo, subfolder="hf_model") in the comparison
notebooks. Qwen2-architecture models only (that is what the hf_bridge converts)."""
import shutil
from pathlib import Path

import torch

from .models.hf_bridge import ours_state_dict_to_hf


def export_hf(model, mcfg, tokenizer, out_dir, log=print) -> float:
    """Returns the largest logit difference between our model and the reloaded Hugging Face model (should be ~0)."""
    if mcfg.name != "qwen":
        raise ValueError(f"export_hf supports the 'qwen' model only, not {mcfg.name!r}")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_cfg = Qwen2Config(
        vocab_size=mcfg.vocab_size, hidden_size=mcfg.hidden_size, intermediate_size=mcfg.intermediate_size,
        num_hidden_layers=mcfg.num_layers, num_attention_heads=mcfg.num_heads, num_key_value_heads=mcfg.num_kv_heads,
        max_position_embeddings=mcfg.max_seq_len, rope_theta=mcfg.rope_theta, rms_norm_eps=mcfg.rms_norm_eps,
        tie_word_embeddings=mcfg.tie_embeddings, attention_bias=mcfg.qkv_bias,
        eos_token_id=tokenizer.eos_id, pad_token_id=tokenizer.pad_id)
    hf = Qwen2ForCausalLM(hf_cfg)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    missing, unexpected = hf.load_state_dict(ours_state_dict_to_hf(state), strict=False)
    missing = [k for k in missing if k != "lm_head.weight" or not mcfg.tie_embeddings]
    assert not missing and not unexpected, f"weight conversion mismatch: missing {missing}, unexpected {unexpected}"
    hf.tie_weights()
    hf.save_pretrained(out_dir, safe_serialization=True)

    tok = getattr(tokenizer, "tok", None)                  # Hugging Face tokenizer -> save it; SentencePiece -> copy the file
    if tok is not None:
        tok.save_pretrained(out_dir)
    elif hasattr(tokenizer, "sp") and getattr(tokenizer, "path", None):
        shutil.copy(tokenizer.path, out_dir / Path(tokenizer.path).name)

    # prove the export: reload from disk and compare logits with our model on random ids
    reloaded = Qwen2ForCausalLM.from_pretrained(out_dir).eval()
    ids = torch.randint(0, mcfg.vocab_size, (1, min(32, mcfg.max_seq_len)))
    was_training, dev = model.training, next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        ours = model(ids.to(dev))[0].float().cpu()
        theirs = reloaded(ids).logits.float()
    if was_training:
        model.train()
    diff = float((ours - theirs).abs().max())
    log(f"[export] hf_model written to {out_dir} | max logit difference vs reloaded model: {diff:.2e}")
    return diff
