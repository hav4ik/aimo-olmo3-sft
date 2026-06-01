"""
Verify the Olmo 3 chat template renders + tokenizes correctly on
tulu-3-sft-personas-math samples. Run before any YAML training work.

  docker run --rm -v $PWD/runs:/runs axolotl-olmo3-sft:0.1.0 \
      python /runs/scripts/verify_chat_template.py
"""
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "allenai/Olmo-3-1025-7B"          # verified 7B base (NOT 1125)
DATASET_ID = "allenai/tulu-3-sft-personas-math"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"Tokenizer class: {tok.__class__.__name__}")
print(f"EOS: {tok.eos_token!r}  BOS: {tok.bos_token!r}  PAD: {tok.pad_token!r}")
print("Chat template (first 300 chars):")
print((tok.chat_template or "")[:300])
print("---")

has_gen_markers = "{% generation %}" in (tok.chat_template or "")
print(f"Has {{% generation %}} markers (needed for assistant_only_loss): {has_gen_markers}")
print()

ds = load_dataset(DATASET_ID, split="train")
print(f"Dataset size: {len(ds)}")
print(f"Fields: {ds.column_names}")

for i in range(3):
    msgs = ds[i]["messages"]
    print(f"\n=== Example {i}: {len(msgs)} turns ({[m['role'] for m in msgs]}) ===")
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    print(f"Rendered (first 400 chars):\n{rendered[:400]}")
    ids = tok.apply_chat_template(msgs, tokenize=True)
    print(f"Total tokens: {len(ids)}")
