"""
Load a model in 4-bit or 8-bit (when GPU/bitsandbytes available) and run one inference.
Demonstrates quantization for deployment. Saves nothing; just runs and prints timing.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"
DEFAULT_MODEL = "distilgpt2"


def main():
    parser = argparse.ArgumentParser(description="Run quantized model inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model or path to adapters")
    parser.add_argument("--adapters", type=str, default=None, help="Path to LoRA adapters (optional)")
    parser.add_argument("--prompt", type=str, default="Instruction: Answer briefly.\nInput: What is RAG?\nOutput:", help="Prompt text")
    parser.add_argument("--load-4bit", action="store_true", help="Load in 4-bit (needs bitsandbytes, GPU)")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install: pip install -r requirements-training.txt")
        raise SystemExit(1)

    load_kw = {}
    if args.load_4bit:
        try:
            load_kw["load_in_4bit"] = True
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.adapters or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
            model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb)
        except Exception as e:
            print(f"4-bit load failed ({e}). Run without --load-4bit (e.g. CPU).")
            model = AutoModelForCausalLM.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.adapters:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapters)
        except ImportError:
            print("peft not installed; skipping adapter load")
        except Exception as e:
            print(f"Adapter load failed: {e}")

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("Generated:", text[:500])
    print(f"Time: {elapsed:.2f}s")
    return text


if __name__ == "__main__":
    main()
