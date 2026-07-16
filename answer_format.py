"""Output instructions shared by every engine that can return mathematics."""

import re

MATH_FORMAT_INSTRUCTIONS = """

Math formatting (required whenever the answer includes a formula, equation, matrix, or symbol):
- Write math as valid LaTeX.
- Put display equations and matrices on their own line inside $$...$$.
- Use $...$ only for short inline symbols.
- Never put a multi-part formula or matrix in plain text, brackets, or ordinary parentheses.
"""


_BARE_LATEX_COMMAND_RE = re.compile(
    r"\\(?:models|Leftrightarrow|Rightarrow|Leftarrow|llbracket|rrbracket|"
    r"subseteq|supseteq|mathbf|mathrm|frac|begin|end|sum|prod|ldots|cdots)\b"
)
_TRAILING_PROSE_RE = re.compile(
    r"\s+(?=(?:This|The|It|Here|Where|Such|These|Therefore|Thus|Formula|Equation)\b)"
)
_DELIMITED_MATH_RE = re.compile(
    r"\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)"
)


def _wrap_equation_and_keep_prose(expression: str) -> list[str]:
    parts = _TRAILING_PROSE_RE.split(expression.strip(), maxsplit=1)
    wrapped = [f"$${parts[0]}$$"]
    if len(parts) == 2:
        wrapped.append(parts[1])
    return wrapped


def _protect_delimited_math(line: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def stash(match: re.Match) -> str:
        blocks.append(match.group(0))
        return f"\x02MATH{len(blocks) - 1}\x03"

    return _DELIMITED_MATH_RE.sub(stash, line), blocks


def _restore_delimited_math(line: str, blocks: list[str]) -> str:
    return re.sub(r"\x02MATH(\d+)\x03", lambda m: blocks[int(m.group(1))], line)


def repair_bare_latex(answer: str) -> str:
    """Wrap bare LaTeX equation lines so KaTeX receives valid delimiters.

    Models occasionally emit a correct expression containing ``\\models`` or
    ``\\Leftrightarrow`` but omit ``$$``. Markdown then consumes the backslashes.
    Existing delimited math is left untouched.
    """
    repaired = []
    in_display_math = False
    for line in answer.splitlines():
        if line.count("$$") % 2:
            in_display_math = not in_display_math
            repaired.append(line)
            continue

        safe_line, math_blocks = _protect_delimited_math(line)
        bare_command = _BARE_LATEX_COMMAND_RE.search(safe_line)
        if not in_display_math and bare_command:
            command_start = bare_command.start()
            colon = safe_line.rfind(":", 0, command_start)
            if colon >= 0:
                prefix = safe_line[: colon + 1].rstrip()
                expression = safe_line[colon + 1 :].strip()
                pieces = [prefix, *_wrap_equation_and_keep_prose(expression)]
            else:
                pieces = _wrap_equation_and_keep_prose(safe_line)
            repaired.extend(_restore_delimited_math(piece, math_blocks) for piece in pieces)
        else:
            repaired.append(line)
    return "\n".join(repaired)
