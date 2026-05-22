"""
Test script to demonstrate math_verify usage for answer verification
"""

try:
    from math_verify import math_equal
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("math_verify not installed. Install with: pip install math-verify")
    exit(1)


def test_math_verify():
    """Test various mathematical equivalences"""

    print("Testing math_verify for mathematical equivalence checking\n")
    print("=" * 60)

    # Test cases: (predicted_answer, ground_truth, expected_result, description)
    test_cases = [
        # Basic arithmetic
        ("42", "42", True, "Exact integer match"),
        ("42.0", "42", True, "Float vs integer"),
        ("1/2", "0.5", True, "Fraction vs decimal"),

        # Expressions
        ("2 + 2", "4", True, "Simple addition"),
        ("10 - 3", "7", True, "Subtraction"),
        ("3 * 4", "12", True, "Multiplication"),
        ("15 / 3", "5", True, "Division"),

        # Algebraic equivalence
        ("x + x", "2x", True, "Algebraic simplification"),
        ("x^2 - 1", "(x-1)(x+1)", True, "Factorization"),
        ("2*x + 3*x", "5*x", True, "Like terms"),

        # Fractions
        ("1/2", "2/4", True, "Equivalent fractions"),
        ("3/4", "0.75", True, "Fraction to decimal"),
        ("6/8", "3/4", True, "Simplified fraction"),

        # Square roots
        ("sqrt(4)", "2", True, "Square root"),
        ("sqrt(9)", "3", True, "Square root"),
        ("2*sqrt(2)", "sqrt(8)", True, "Radical simplification"),

        # Trigonometry (if supported)
        ("sin^2(x) + cos^2(x)", "1", True, "Trig identity"),

        # Inequalities and wrong answers
        ("42", "43", False, "Different integers"),
        ("1/2", "1/3", False, "Different fractions"),
        ("x", "y", False, "Different variables"),

        # Complex expressions
        ("(x+1)^2", "x^2 + 2x + 1", True, "Expansion"),
        ("2^3", "8", True, "Exponentiation"),
        ("log(100)", "2", True, "Logarithm (base 10)"),

        # Scientific notation
        ("1e6", "1000000", True, "Scientific notation"),
        ("2.5e-3", "0.0025", True, "Scientific notation decimal"),
    ]

    print("\nRunning test cases:\n")

    passed = 0
    failed = 0
    errors = 0

    for pred, truth, expected, desc in test_cases:
        try:
            result = math_equal(pred, truth)
            status = "✓" if result == expected else "✗"

            if result == expected:
                passed += 1
                print(f"{status} PASS: {desc}")
            else:
                failed += 1
                print(f"{status} FAIL: {desc}")

            print(f"  Predicted: {pred}")
            print(f"  Ground Truth: {truth}")
            print(f"  Result: {result} (Expected: {expected})")
            print()

        except Exception as e:
            errors += 1
            print(f"✗ ERROR: {desc}")
            print(f"  Predicted: {pred}")
            print(f"  Ground Truth: {truth}")
            print(f"  Error: {e}")
            print()

    print("=" * 60)
    print(f"\nTest Summary:")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print()


def test_extraction_and_verification():
    """Test answer extraction followed by verification"""

    print("\n" + "=" * 60)
    print("Testing Answer Extraction + Verification")
    print("=" * 60 + "\n")

    # Simulated model outputs with various formats
    test_outputs = [
        {
            "output": "Let me solve this step by step.\n15 + 27 = 42\nTherefore, the final answer is 42.",
            "ground_truth": "42",
            "description": "Simple extraction"
        },
        {
            "output": "To solve this, we calculate:\n$\\boxed{3/4}$",
            "ground_truth": "0.75",
            "description": "Boxed LaTeX answer"
        },
        {
            "output": "Step 1: Simplify the expression\nStep 2: Calculate\nThe answer is: 16",
            "ground_truth": "2^4",
            "description": "Answer with colon"
        },
        {
            "output": "After solving, we get x = 5 #### 5",
            "ground_truth": "5",
            "description": "GSM8K format"
        },
    ]

    import re

    def extract_answer(text):
        """Simple answer extraction"""
        patterns = [
            r"\\boxed\{([^}]+)\}",
            r"####\s*(.+)",
            r"(?:final answer|answer)(?:\s+is)?:?\s*(.+?)(?:\.|$)",
            r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return text.strip()

    for i, test in enumerate(test_outputs, 1):
        print(f"Test {i}: {test['description']}")
        print(f"Output: {test['output'][:80]}...")

        extracted = extract_answer(test['output'])
        print(f"Extracted: {extracted}")
        print(f"Ground Truth: {test['ground_truth']}")

        try:
            is_correct = math_equal(extracted, test['ground_truth'])
            print(f"Verification: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        except Exception as e:
            print(f"Verification Error: {e}")

        print()


if __name__ == "__main__":
    if MATH_VERIFY_AVAILABLE:
        test_math_verify()
        test_extraction_and_verification()
    else:
        print("Please install math_verify first:")
        print("pip install math-verify")
