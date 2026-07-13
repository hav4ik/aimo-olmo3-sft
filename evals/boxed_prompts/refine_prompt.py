"""Refine prompts (refine_tool / refine_notool) — a NEW task, NOT in the training distribution.

Given a triplet ``{problem, solution, evaluation}`` the model is asked to produce an *improved*
solution in the same ``## Solution`` format as the proof/solution task. Concretely this is the
proof-solution prompt with two extra inputs appended:

    ## Problem            (as in the proof prompt)
    ## Previous Solution  (a prior attempt)
    ## Evaluation         (an assessment of that attempt — e.g. the 0/1/6/7 analysis output)

Byte-reuse: the ENTIRE proof framing — task description, the 0/0.5/1 grading rubric, and the
``## Solution`` output-format spec — is reused VERBATIM from
``cascade2_prompts.PROOF_USER_PREFIX`` (which is byte-exact with the training distribution). Only the
two new input section separators and the closing refine instruction are new (and are NOT claimed to be
in training).

Systems (identical to the proof task, so refine_* parallels proof_*):
  refine_notool : cascade2 SYSTEM_PROMPT (no tools)
  refine_tool   : tool_boxed SYSTEM_CONTENT + FUNCTIONS schema (the hybrid tool setup)

Assemble by CONCATENATION only — never run through str.format()/% (the reused rubric contains literal
double braces ``\\boxed{{}}``).
"""

import json

from cascade2_prompts import PROOF_USER_PREFIX
from cascade2_prompts import SYSTEM_PROMPT as SYSTEM_NO_TOOL
from tool_boxed_prompt import FUNCTIONS
from tool_boxed_prompt import SYSTEM_CONTENT as SYSTEM_WITH_TOOL

# New (non-training) glue. The proof prefix already ends with "## Problem\n"; the problem is appended,
# then these two labelled input sections, then the refine instruction.
REFINE_PREV_SOLUTION_SEP = "\n\n## Previous Solution\n"
REFINE_EVALUATION_SEP = "\n\n## Evaluation\n"
REFINE_SUFFIX = (
    "\n\nThe ## Previous Solution above is an earlier attempt at this problem, and ## Evaluation is "
    "an assessment of that attempt's quality. Produce an improved solution: address every error, gap, "
    "and omission identified in the evaluation, and otherwise strengthen the rigor, completeness, and "
    "clarity of the argument. Present your improved final solution using the ## Solution format "
    "described above."
)

TOOLS = json.loads(FUNCTIONS)


def build_refine_user(problem: str, solution: str, evaluation: str) -> str:
    """The refine USER turn: proof prefix (verbatim) + problem + previous solution + evaluation + instr."""
    return (
        PROOF_USER_PREFIX
        + problem
        + REFINE_PREV_SOLUTION_SEP
        + solution
        + REFINE_EVALUATION_SEP
        + evaluation
        + REFINE_SUFFIX
    )


def build_refine_messages(problem: str, solution: str, evaluation: str, use_tools: bool = False) -> list:
    """Chat messages for the refine task. ``use_tools`` selects the tool (hybrid) vs no-tool system."""
    system = SYSTEM_WITH_TOOL if use_tools else SYSTEM_NO_TOOL
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_refine_user(problem, solution, evaluation)},
    ]


if __name__ == "__main__":
    P, S, E = "<PROBLEM>", "<PREV_SOLUTION>", "<EVALUATION>"
    user = build_refine_user(P, S, E)
    # The proof framing is reused verbatim, so the byte-exact rubric/format must still be present.
    assert user.startswith(PROOF_USER_PREFIX)
    assert "score it according to the following criteria" in user      # 0/0.5/1 rubric (reused)
    assert "## Solution // Your final solution should start" in user    # output-format spec (reused)
    # The two new inputs land in order, after the problem.
    assert ("## Problem\n" + P + REFINE_PREV_SOLUTION_SEP + S + REFINE_EVALUATION_SEP + E) in user
    assert user.endswith(REFINE_SUFFIX)
    notool = build_refine_messages(P, S, E, use_tools=False)
    tool = build_refine_messages(P, S, E, use_tools=True)
    assert notool[0]["content"] == SYSTEM_NO_TOOL
    assert tool[0]["content"] == SYSTEM_WITH_TOOL
    assert notool[1]["content"] == tool[1]["content"]  # same user turn; only the system differs
    print(f"OK — refine prompt structural checks passed (user-len={len(user)})")
