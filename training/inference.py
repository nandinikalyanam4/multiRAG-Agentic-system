"""
Lazy-load LoRA model for local generation. Used by lora_rag agent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_model = None
_tokenizer = None


def get_local_model(adapters_path: Optional[Path] = None):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from config import settings
    path = adapters_path or getattr(settings, "lora_adapters_path", None)
    if not path:
        path = ROOT / "training" / "adapters"
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        return None, None
    base = "distilgpt2"
    _tokenizer = AutoTokenizer.from_pretrained(str(path))
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(base)
    _model = PeftModel.from_pretrained(_model, str(path))
    _model.eval()
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 120) -> Optional[str]:
    """Generate with local LoRA model. Returns None if not available."""
    model, tokenizer = get_local_model()
    if model is None or tokenizer is None:
        return None
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)
