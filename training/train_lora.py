"""
Train LoRA adapters on a small LM using SFT data.
Uses transformers + peft. Saves adapters to training/adapters/.
Run after prepare_sft_data.py. Works on CPU (slow) or GPU.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"
ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"
DEFAULT_DATA = DATA_DIR / "sft_train.jsonl"
# Small model so it runs on CPU in reasonable time
DEFAULT_MODEL = "distilgpt2"


def format_example(record: dict) -> str:
    """Turn instruction/input/output into one text for causal LM."""
    inst = record.get("instruction", "")
    inp = record.get("input", "")
    out = record.get("output", "")
    return f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for SFT")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to sft_train.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--output", type=str, default=str(ADAPTERS_DIR), help="Where to save adapters")
    parser.add_argument("--steps", type=int, default=15, help="Training steps")
    parser.add_argument("--batch", type=int, default=1, help="Per-device batch size")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError as e:
        print("Install training deps: pip install -r requirements-training.txt")
        raise SystemExit(1) from e

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}. Run: python training/prepare_sft_data.py")
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if len(dataset) == 0:
        print("No examples in data file.")
        raise SystemExit(1)

    def add_text(example):
        return {"text": format_example(example)}

    dataset = dataset.map(add_text)

    tokenized = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format("torch")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],  # distilgpt2
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        save_strategy="steps",
        save_steps=args.steps,
        logging_steps=2,
        report_to="none",
    )

    from transformers import Trainer, DataCollatorForLanguageModeling
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"Adapters saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
