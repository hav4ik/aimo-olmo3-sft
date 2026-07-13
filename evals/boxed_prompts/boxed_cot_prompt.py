"""Prompt template for no-tool, chain-of-thought (CoT) \\boxed{} *solution* problems.

Byte-exact with the Cascade2 ``math_notool`` training distribution used for the Olmo-3 Thinking SFT
dataset (``chankhavu/smolmo-sft-olmocore-pretokenized`` / ``chankhavu/smolmo-sft-v2-seqlen64k``).

Provenance (measured over the training source ``math_notool``):
  - SYSTEM prompt: 100% uniform (also identical for math_v4_cot / math_v4_tir_nc).
  - USER instruction "Please reason step by step, and put your final answer within \\boxed{{}}.":
    6,362 / 7,386 of math_notool, PREPENDED before the problem.
  - The double brace ``\\boxed{{}}`` is the exact form the model was trained on (an upstream
    str.format artifact). The model is robust to single ``\\boxed{}`` too, but this matches training.
"""

SYSTEM_PROMPT = (
    "You are an expert mathematical assistant. Provide rigorous, complete proofs. "
    "You are not allowed to use tools."
)

# The boxed instruction is a PLAIN string literal holding the literal double brace `\boxed{{}}` —
# the exact bytes the model was trained on. IMPORTANT: build by concatenation only. Do NOT pass this
# through str.format()/`%`/f-string formatting — that would collapse `{{}}` to `{}` and break the
# byte-exact match.
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}."


def build_user_content(problem: str) -> str:
    """User-turn content: instruction prepended, blank line, then the problem (math_notool layout)."""
    return BOXED_INSTRUCTION + "\n\n" + problem


def build_messages(problem: str) -> list:
    """Build the chat ``messages`` for a no-tool CoT \\boxed{} solution problem.

    Feed the result to the Olmo-3 chat template with ``add_generation_prompt=True`` (it appends
    ``<|im_start|>assistant\\n<think>``).

    Example::

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
        messages = build_messages("Find all integers n such that n+1 divides n^2 + 1.")
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(problem)},
    ]


if __name__ == "__main__":
    # self-check: confirm the literal double brace survives
    m = build_messages("<PROBLEM>")
    assert m[0]["content"] == SYSTEM_PROMPT
    assert m[1]["content"] == "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n<PROBLEM>"
    assert "\\boxed{{}}" in m[1]["content"], "double brace must be preserved"
    print(m[0]["content"])
    print("---")
    print(m[1]["content"])
