"""
Distillation: use OpenAI as teacher to generate target answers for SFT data.
Reads training/data/sft_train.jsonl (instruction + input), calls API for 'output', writes teacher_sft.jsonl.
Run after prepare_sft_data.py. Requires OPENAI_API_KEY.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"
IN_PATH = DATA_DIR / "sft_train.jsonl"
OUT_PATH = DATA_DIR / "teacher_sft.jsonl"


def main():
    if not IN_PATH.exists():
        print(f"Run first: python training/prepare_sft_data.py")
        raise SystemExit(1)

    from config import settings
    from openai import OpenAI
    if not settings.openai_api_key:
        print("Set OPENAI_API_KEY in .env or environment")
        raise SystemExit(1)

    client = OpenAI(api_key=settings.openai_api_key)
    records = []
    with open(IN_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_records = []
    for r in records:
        prompt = f"{r.get('instruction','')}\n\n{r.get('input','')}\n\nOutput:"
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        output = resp.choices[0].message.content.strip()
        out_records.append({"instruction": r["instruction"], "input": r["input"], "output": output})

    with open(OUT_PATH, "w") as f:
        for o in out_records:
            f.write(json.dumps(o) + "\n")
    print(f"Wrote {len(out_records)} teacher examples to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
