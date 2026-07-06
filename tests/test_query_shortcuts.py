"""
تست میان‌برهای query — گفت‌وگوی ساده و محاسبات نباید وارد پایپ‌لاین RAG شوند
اجرا:  python tests/test_query_shortcuts.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.query_shortcuts import is_conversational, extract_calc_expression


class TestConversational(unittest.TestCase):
    def test_conversational_queries(self):
        for q in [
            "hello", "Hi", "hey", "سلام", "سلام علیکم", "خوبی؟", "چطوری",
            "مرسی", "thanks", "how are you?", "who are you", "تست",
            "hello there", "سلام خوبی", "ok", "چه خبر؟", "good morning",
            "خداحافظ",
        ]:
            self.assertTrue(is_conversational(q), f"expected conversational: {q!r}")

    def test_informational_queries_pass_through(self):
        for q in [
            "what is this?",
            "I need new information about iran",
            "hello world program in python",
            "سلام هوش مصنوعی چیست؟",
            "explain the doc",
            "28% of 8795576",
            "what is the capital of France",
            "چرا آسمان آبی است",
            "iran",
        ]:
            self.assertFalse(is_conversational(q), f"expected NOT conversational: {q!r}")


class TestCalcExtraction(unittest.TestCase):
    def test_extractable(self):
        cases = {
            "28% of 8795576": "(28 / 100) * 8795576",
            "20 درصد 150": "(20 / 100) * 150",
            "2+2": "2+2",
            "calculate 3 * (4 + 5)": "3 * (4 + 5)",
            "محاسبه کن 12/4": "12/4",
            "2^10": "2**10",
            "1,000 * 3": "1000 * 3",
        }
        for q, expected in cases.items():
            self.assertEqual(extract_calc_expression(q), expected, q)

    def test_not_extractable(self):
        for q in ["hello", "what is 42", "iran", "explain the doc", ""]:
            self.assertIsNone(extract_calc_expression(q), q)

class TestThinkingSplit(unittest.TestCase):
    """جداسازی توکن‌های فکر از پاسخ مدل"""

    def test_split_thinking(self):
        from llm.thinking import split_thinking
        cases = [
            ("<think>step 1</think>answer", "step 1", "answer"),
            ("<thinking>hmm</thinking>hi", "hmm", "hi"),
            ("plain", "", "plain"),
            ("<think>unclosed thought", "unclosed thought", ""),
            ("REASONING: because X\nANSWER: 42", "because X", "42"),
        ]
        for raw, t_exp, a_exp in cases:
            t, a = split_thinking(raw)
            self.assertEqual(t, t_exp, raw)
            self.assertEqual(a, a_exp, raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
