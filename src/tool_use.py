"""
Python REPL tool integration for tool-integrated reasoning (TIR).

The model can emit a ```python ... ``` block mid-generation.
We execute it and inject the output back before continuing generation.
"""

import re
import subprocess


CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
TOOL_OUTPUT_TEMPLATE = "\n[TOOL OUTPUT]: {output}\n"


def execute_python(code: str, timeout: int = 10) -> str:
    """Run a Python snippet and return stdout (or stderr on failure)."""
    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout.strip()
    if not output and result.stderr.strip():
        output = f"ERROR: {result.stderr.strip()}"
    return output


def run_with_tools(model, tokenizer, problem: str, max_new_tokens: int = 1024) -> str:
    """
    Generate a solution, executing any Python code blocks that appear.
    Loops until no more code blocks are emitted or max iterations reached.
    """
    from src.dataset import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(problem=problem)
    context = prompt
    max_iterations = 5

    for _ in range(max_iterations):
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        context += generated

        match = CODE_BLOCK_RE.search(generated)
        if not match:
            break  # no tool call — generation is complete

        code = match.group(1)
        tool_output = execute_python(code)
        context += TOOL_OUTPUT_TEMPLATE.format(output=tool_output)

    return context[len(prompt):]
