"""Output instructions shared by every engine that can return mathematics."""

MATH_FORMAT_INSTRUCTIONS = """

Math formatting (required whenever the answer includes a formula, equation, matrix, or symbol):
- Write math as valid LaTeX.
- Put display equations and matrices on their own line inside $$...$$.
- Use $...$ only for short inline symbols.
- Never put a multi-part formula or matrix in plain text, brackets, or ordinary parentheses.
"""
