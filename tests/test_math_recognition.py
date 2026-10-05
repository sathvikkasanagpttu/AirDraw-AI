"""Unit tests for handwritten math parser and safe AST evaluator."""

import pytest
from app.math_recognition.recognizer import MathRecognizer


def test_math_evaluator_basic_arithmetic():
    rec = MathRecognizer()
    # Addition
    val, _ = rec.evaluate("3 + 7")
    assert val == "10"

    # Subtraction
    val, _ = rec.evaluate("15 - 8")
    assert val == "7"

    # Multiplication
    val, _ = rec.evaluate("6 * 7")
    assert val == "42"

    # Division
    val, _ = rec.evaluate("10 / 2")
    assert val == "5"


def test_math_evaluator_decimals():
    rec = MathRecognizer()
    val, _ = rec.evaluate("2.5 + 4.5")
    assert val == "7"

    val, _ = rec.evaluate("7 / 2")
    assert val == "3.5"


def test_math_evaluator_division_by_zero():
    rec = MathRecognizer()
    val, _ = rec.evaluate("9 / 0")
    assert val == "Div/0 Error"


def test_math_evaluator_incomplete_expression():
    rec = MathRecognizer()
    val, _ = rec.evaluate("42 +")
    assert val is None

    val, _ = rec.evaluate("+")
    assert val is None

    val, _ = rec.evaluate("?")
    assert val is None


def test_math_evaluator_security_against_injection():
    rec = MathRecognizer()
    # Attempt code injection string
    malicious = "__import__('os').system('ls')"
    val, _ = rec.evaluate(malicious)
    assert val is None  # Safely rejected by AST whitelist
