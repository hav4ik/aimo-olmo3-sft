"""Prompt template for TOOL-USE (Python execution) \\boxed{} problems.

Byte-exact with the Cascade2 ``math_withtool`` training distribution (task_type ``"tool"``) used for the
Olmo-3 Thinking SFT dataset (``chankhavu/smolmo-sft-olmocore-pretokenized`` / ``chankhavu/smolmo-sft-v2-seqlen64k``).

This is the tool-augmented counterpart of ``boxed_cot_prompt.py``: the model may call a stateful Python
interpreter (``stateful_python_code_exec``) during its reasoning, then box a final numeric/closed-form answer.

Provenance (measured over 42,311 ``math_withtool`` examples — see notes inline):
  - SYSTEM ``content``:   100% uniform (1 distinct / 42,311) — the tool system prompt below.
  - SYSTEM ``functions``: 100% uniform (1 distinct / 42,311, len 397) — the single ``stateful_python_code_exec``
    function spec below. This rides on the SYSTEM message as a separate ``functions`` field; the Olmo-3 chat
    template renders it as `` <functions>{functions}</functions>`` appended to the system turn.
  - USER instruction: phrasing VARIES (no single canonical wording). Placement distribution:
    prefix 29% / both 26% / embedded-or-none 28% / suffix 17%; dominant separator is ``"\\n\\n"``.
  - Brace form here is the SINGLE brace ``\\boxed{}`` (NOTE: the no-tool ``math_notool`` set used the literal
    double brace ``\\boxed{{}}`` — they differ; this file matches the tool distribution, single brace).

Render with the Olmo-3 chat template + ``add_generation_prompt=True``::

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    messages = build_tool_messages("Compute the 15th prime gap above 10^6.")
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #   <|im_start|>system\n{SYSTEM_CONTENT} <functions>{FUNCTIONS}</functions><|im_end|>\n
    #   <|im_start|>user\n{user}<|im_end|>\n
    #   <|im_start|>assistant\n<think>
"""

# --- SYSTEM message (uniform across 100% of math_withtool) ------------------------------------------

SYSTEM_CONTENT = (
    "You are an expert mathematical assistant. Provide rigorous, complete solutions. "
    "You are provided with function signatures within <functions></functions> XML tags. "
    "You may call one or more functions to assist with the user query. "
    "Output any function calls within <function_calls></function_calls> XML tags. "
    "Don't make assumptions about what values to plug into functions."
)

# The tool spec carried in the SYSTEM message's `functions` field (uniform, len 397). Kept as the exact
# JSON STRING the model was trained on (compact separators, double-quoted) — do NOT re-serialize, as
# json.dumps key order / whitespace would change the bytes.
FUNCTIONS = (
    '[{"type": "function", "function": {"name": "stateful_python_code_exec", '
    '"description": "Call this function to execute Python code in a stateful Jupyter notebook '
    'environment. Python will respond with the output of the execution or time out after 120.0 seconds.", '
    '"parameters": {"type": "object", "properties": {"code": {"type": "string", '
    '"description": "Code to execute"}}, "required": ["code"]}}}]'
)
assert len(FUNCTIONS) == 397, len(FUNCTIONS)  # matches the 397-char training spec exactly

# --- USER instruction --------------------------------------------------------------------------------
# Phrasing varies in training; this is the most balanced wording (appears at high frequency in BOTH the
# prefix and suffix positions, and mirrors the no-tool default). Single brace, matching math_withtool.
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Real instruction variants observed in the training distribution — sample from these to reproduce the
# natural phrasing spread (all single-brace \boxed{}). Roughly uniform frequency.
INSTRUCTION_VARIANTS = (
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "Solve the following math problem. Put your answer inside \\boxed{}.",
    "Solve the problem and put your answer in \\boxed{}.",
    "Provide your final answer in \\boxed{}.",
    "Your final answer should be placed in \\boxed{}.",
    "Please place your final answer inside \\boxed{}.",
    "Put your final answer in \\boxed{}.",
    "Your answer should be in \\boxed{}.",
)

SEP = "\n\n"  # dominant separator between instruction and problem (17,305 / 19,968 measured)


def build_user_content(problem: str, instruction: str = BOXED_INSTRUCTION,
                       placement: str = "prefix") -> str:
    """Assemble the user-turn content for a tool-use \\boxed{} problem.

    placement:
      - "prefix"  : instruction, blank line, problem        (most common single placement, ~29%)
      - "suffix"  : problem, blank line, instruction        (~17%)
      - "both"    : instruction, problem, instruction       (~26%)
      - "none"    : problem only (instruction relied on system/format conventions, ~28%)
    """
    if placement == "prefix":
        return instruction + SEP + problem
    if placement == "suffix":
        return problem + SEP + instruction
    if placement == "both":
        return instruction + SEP + problem + SEP + instruction
    if placement == "none":
        return problem
    raise ValueError(f"placement must be prefix/suffix/both/none, got {placement!r}")


def build_tool_messages(problem: str, instruction: str = BOXED_INSTRUCTION,
                        placement: str = "prefix") -> list:
    """Build the chat ``messages`` for a tool-use (Python exec) \\boxed{} problem.

    The SYSTEM message carries BOTH ``content`` and ``functions`` — the ``functions`` field is what the
    Olmo-3 chat template turns into the `` <functions>...</functions>`` block, enabling tool calls. Feed
    the result through ``apply_chat_template(..., add_generation_prompt=True)``.
    """
    return [
        {"role": "system", "content": SYSTEM_CONTENT, "functions": FUNCTIONS},
        {"role": "user", "content": build_user_content(problem, instruction, placement)},
    ]


if __name__ == "__main__":
    m = build_tool_messages("<PROBLEM>")
    assert m[0]["content"] == SYSTEM_CONTENT
    assert m[0]["functions"] == FUNCTIONS and len(m[0]["functions"]) == 397
    assert m[1]["content"] == "Please reason step by step, and put your final answer within \\boxed{}.\n\n<PROBLEM>"
    # suffix / both / none variants
    assert build_user_content("P", placement="suffix").endswith("\\boxed{}.")
    assert build_user_content("P", placement="none") == "P"
    print("SYSTEM:\n" + m[0]["content"])
    print("\nFUNCTIONS (len %d):\n%s" % (len(m[0]["functions"]), m[0]["functions"]))
    print("\nUSER (prefix):\n" + m[1]["content"])
    print("\nUSER (suffix):\n" + build_user_content("<PROBLEM>", placement="suffix"))
