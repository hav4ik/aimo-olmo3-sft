"""Cascade2 (math_proof) prompt templates — proof solution / evaluation / analysis.

Byte-exact with the Olmo-3 Thinking SFT training distribution (source ``math_proof`` of
``chankhavu/smolmo-sft-v2-seqlen64k``). Every template here was verified by reconstructing 22,980
real training user-messages BYTE-EXACT (0 mismatches: sol_plain 1,165 / sol_verify 9,990 /
eval 6,096 / analysis 5,729).

Same SYSTEM prompt for all three tasks (and identical to the no-tool boxed prompt in
``boxed_cot_prompt.py``). The task differs entirely in the USER turn.

  - PROOF solution   : build_proof_messages(problem, self_verify=False|True)   -> {problem}
        NOTE: BOTH variants EMBED the 0/0.5/1 evaluation rubric and tell the model its solution will
        be graded by it (so it self-optimizes). They are NOT "no-rubric vs rubric"; they differ only
        in the REQUIRED OUTPUT:
        self_verify=False : output a "## Solution" block only (rubric-aware, but writes NO explicit
                            self-evaluation and NO boxed self-score).
        self_verify=True  : output "## Solution" + a "## Self Evaluation" block ending in a
                            \\boxed{{...}} self-score (0 / 0.5 / 1), after re-examining its own solution.
        (There is NO rubric-free proof prompt in math_proof. For a plain "solve & box the answer"
        prompt with no rubric, use boxed_cot_prompt.py — the math_notool template.)
  - EVALUATION       : build_evaluation_messages(problem, solution)  -> {problem, solution}
        score the solution 0 / 0.5 / 1 in \\boxed{{}} (with a written evaluation first).
  - ANALYSIS         : build_analysis_messages(problem, solution)    -> {problem, solution}
        grade the solution 0 / 1 / 6 / 7 in \\boxed{{}} (with a written analysis first).

NOTE: the boilerplate contains literal double braces ``\\boxed{{}}`` / ``\\boxed{{...}}`` (the exact
trained bytes). The build_* functions assemble by CONCATENATION — do NOT run the results through
str.format()/%, which would collapse ``{{}}`` -> ``{}``.

Feed the returned ``messages`` to the Olmo-3 chat template with add_generation_prompt=True.
"""

SYSTEM_PROMPT = 'You are an expert mathematical assistant. Provide rigorous, complete proofs. You are not allowed to use tools.'

# --- PROOF solution (user prefix ends with "## Problem\n"; problem is appended) ---
PROOF_USER_PREFIX             = "Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.\n\nYour final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:\n\n```txt\nHere is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.\n\nPlease evaluate the solution and score it according to the following criteria:\n- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1\n- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5\n- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0\n\nAdditionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1\n```\n\nYour final response should be in the following format:\n\n## Solution // Your final solution should start with this exact same markdown title\n... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.\n\n---\n\nHere is your task input:\n\n## Problem\n"
PROOF_USER_PREFIX_SELF_VERIFY = "Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.\n\nYour final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:\n\n```txt\nHere is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.\n\nPlease evaluate the solution and score it according to the following criteria:\n- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1\n- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5\n- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0\n\nAdditionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1\n```\n\nIn fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.\n\nIn your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.\n- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.\n- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.\n- Only when you fail to locate any issues in your solution should you score it with 1.\n- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.\n- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!\n\nYour final response should be in the following format:\n\n## Solution // Your final solution should start with this exact same markdown title\n... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.\n\n## Self Evaluation // Your evaluation of your own solution above should start with this exact same markdown title\n\nHere is my evaluation of the solution: // Your analysis should start with this exact same phrase\n... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.\n\nBased on my evaluation, the final overal score should be:\n\\boxed{...} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful RE-examination of your own solution above\n\n---\n\nHere is your task input:\n\n## Problem\n"

# --- EVALUATION (prefix ends with "## Problem\n"; then problem, sep, solution) ---
EVAL_USER_PREFIX   = "## Instruction\n\nYour task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.\n\nPlease evaluate the solution and score it according to the following criteria:\n- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1\n- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5\n- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0\n- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1\n\nPlease carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:\n\nHere is my evaluation of the solution:\n... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.\n\nBased on my evaluation, the final overal score should be:\n\\\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria\n\n---\n\nHere is your task input:\n\n## Problem\n"
EVAL_SOLUTION_SEP  = '\n\n## Solution\n'

# --- ANALYSIS (prefix ends with "Problem:\n"; then problem, sep, solution, suffix) ---
ANALYSIS_USER_PREFIX  = 'Carefully analyze the given problem statement and the proposed solution, and then write out your analysis regarding the correctness of the proposed solution. \n\nAfter the analysis, you must provide a score based on the following grading scale:\n\n- 0: Incorrect - The solution is completely incorrect or irrelevant.\n- 1: Partial - The solution is partially correct but has significant errors or omissions.\n- 6: Almost - The solution is almost correct but contains minor errors or inaccuracies.\n- 7: Correct - The solution is fully correct and complete.\n\n\nProblem:\n'
ANALYSIS_SOLUTION_SEP = '\n\nSolution:\n'
ANALYSIS_SUFFIX       = 'Analyze the solution carefully, then provide your grade as a single number (0, 1, 6, or 7) in \\boxed{{}}.'


def _sys(user_content):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}]


def build_proof_user(problem: str, self_verify: bool = False) -> str:
    prefix = PROOF_USER_PREFIX_SELF_VERIFY if self_verify else PROOF_USER_PREFIX
    return prefix + problem


def build_evaluation_user(problem: str, solution: str) -> str:
    return EVAL_USER_PREFIX + problem + EVAL_SOLUTION_SEP + solution


def build_analysis_user(problem: str, solution: str) -> str:
    return ANALYSIS_USER_PREFIX + problem + ANALYSIS_SOLUTION_SEP + solution + "\n\n" + ANALYSIS_SUFFIX


def build_proof_messages(problem: str, self_verify: bool = False) -> list:
    """Proof/solution. BOTH variants embed the 0/0.5/1 grading rubric and tell the model to optimize
    for it; self_verify=True ADDITIONALLY requires an explicit "## Self Evaluation" block + a
    \\boxed{{}} self-score in the output. self_verify=False outputs the "## Solution" block only."""
    return _sys(build_proof_user(problem, self_verify))


def build_evaluation_messages(problem: str, solution: str) -> list:
    """Evaluation task: score the solution 0 / 0.5 / 1 in \\boxed{{}}."""
    return _sys(build_evaluation_user(problem, solution))


def build_analysis_messages(problem: str, solution: str) -> list:
    """Analysis task: grade the solution 0 / 1 / 6 / 7 in \\boxed{{}}."""
    return _sys(build_analysis_user(problem, solution))


if __name__ == "__main__":
    P, S = "<PROBLEM>", "<SOLUTION>"
    # structural self-checks
    assert build_proof_user(P).endswith("## Problem\n" + P)
    assert build_proof_user(P, True) != build_proof_user(P, False)
    # BOTH variants embed the 0/0.5/1 rubric; only self_verify outputs an explicit self-eval + boxed score
    assert "score it according to the following criteria" in build_proof_user(P, False)  # rubric in BOTH
    assert "score it according to the following criteria" in build_proof_user(P, True)
    assert "## Self Evaluation" not in build_proof_user(P, False)  # plain: no self-eval output
    assert "## Self Evaluation" in build_proof_user(P, True)       # self_verify: writes self-eval + boxed score
    assert "\\boxed" not in build_proof_user(P, False)             # plain: NO boxed self-score
    assert "\\boxed{...}" in build_proof_user(P, True)
    assert build_evaluation_user(P, S).endswith("## Solution\n" + S)
    assert build_analysis_user(P, S).endswith("(0, 1, 6, or 7) in \\boxed{{}}.")
    assert "\\boxed{{}}" in build_analysis_user(P, S)           # double brace preserved
    for tag, fn in [("PROOF (plain)", lambda: build_proof_messages(P)),
                    ("PROOF (self-verify)", lambda: build_proof_messages(P, True)),
                    ("EVALUATION", lambda: build_evaluation_messages(P, S)),
                    ("ANALYSIS", lambda: build_analysis_messages(P, S))]:
        m = fn(); print(f"== {tag} ==  user-len={len(m[1]['content'])}")
    print("OK — all structural checks passed")
