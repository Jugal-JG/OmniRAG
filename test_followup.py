import unittest

from followup import (
    extract_labelled_reference_numbers,
    explicit_engine_request,
    normalize_reference_typos,
    resolve_labelled_followup,
)
from answer_format import repair_bare_latex


class LabelledFollowupTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {
                "q": "what is formula 15?",
                "a": "Formula (15) is a Beta-CDF approximation.",
            }
        ]

    def test_adds_label_to_named_followup(self):
        self.assertEqual(
            resolve_labelled_followup("what is the name of the formula", self.history),
            "what is the name of the formula (15)",
        )

    def test_adds_label_to_pronoun_followup(self):
        resolved = resolve_labelled_followup("what does it mean?", self.history)
        self.assertIn("formula (15)", resolved)

    def test_keeps_explicit_new_reference(self):
        query = "what does formula 18 mean?"
        self.assertEqual(resolve_labelled_followup(query, self.history), query)

    def test_does_not_rewrite_unrelated_question(self):
        query = "summarize the conclusion"
        self.assertEqual(resolve_labelled_followup(query, self.history), query)

    def test_corrects_formula_typo(self):
        self.assertEqual(
            normalize_reference_typos("name of forumula 15"),
            "name of formula 15",
        )

    def test_extracts_multiple_formula_labels(self):
        self.assertEqual(
            extract_labelled_reference_numbers("what are formulas 15 and 16 about"),
            {"15", "16"},
        )

    def test_repairs_bare_latex_equation(self):
        answer = "Rule:\n(S, X) \\models p \\Leftrightarrow S \\models q"
        repaired = repair_bare_latex(answer)
        self.assertIn("$$(S, X) \\models p \\Leftrightarrow S \\models q$$", repaired)

    def test_repairs_equation_without_swallowing_trailing_prose(self):
        answer = "Rule: (S, X) \\models p \\Leftrightarrow S \\models q This rule compares values."
        repaired = repair_bare_latex(answer)
        self.assertIn("$$(S, X) \\models p \\Leftrightarrow S \\models q$$", repaired)
        self.assertIn("This rule compares values.", repaired)

    def test_repairs_bare_equation_on_line_that_also_has_inline_math(self):
        answer = (
            "For $p_i$, the formula states: (S, X) \\models p "
            "\\Leftrightarrow S \\models q This means the values agree."
        )
        repaired = repair_bare_latex(answer)
        self.assertIn("For $p_i$, the formula states:", repaired)
        self.assertIn("$$(S, X) \\models p \\Leftrightarrow S \\models q$$", repaired)
        self.assertIn("This means the values agree.", repaired)

    def test_detects_explicit_basic_rag_request(self):
        self.assertEqual(
            explicit_engine_request('use basic rag engine to answer "formula 15"'),
            "basic_rag",
        )

    def test_detects_other_explicit_engine_requests(self):
        cases = {
            "use the router query engine for this": "router_engine",
            "answer using the sub-question engine": "subquestion",
            "use ReAct agent for this question": "react",
            "route this query to multi-document agent": "multi_document",
            "use the multimodal engine": "multimodal",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(explicit_engine_request(query), expected)


if __name__ == "__main__":
    unittest.main()
