"""
test_correctness.py
Correctness Validation for Smith-Waterman GPU Implementation
Phase 2A: CAFA6 Project - GPU vs CPU Validation

Validates that GPU implementation produces identical results to CPU baseline.
Essential for grading - ensures mathematical correctness of CUDA kernel.

Usage:
    python test_correctness.py

Test Strategy:
    - Compare GPU vs CPU on diverse test cases
    - Vary sequence lengths (short, medium, long)
    - Test edge cases (empty, identical sequences, single char)
    - Verify error < 1e-4 (floating point tolerance)
"""

import numpy as np
import sys
from typing import List, Tuple

# Import CPU baseline
from cpu_smith_waterman import align_sequences_sequential

# Import GPU implementation (conditionally)
try:
    from smith_waterman import align_sequences as align_sequences_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("ERROR: GPU implementation not available!")
    print("Build with: cd kernels/smith_waterman && python setup.py install")
    sys.exit(1)

# Test tolerance for floating point comparison
TOLERANCE = 1e-4

# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = [
    # (name, seq_a, seq_b, description)

    # Basic test cases
    ("short_identical", "ARNDCQEGH", "ARNDCQEGH", "Short identical sequences (self-alignment)"),
    ("short_similar", "ARNDCQEGH", "ARDCQEG", "Short similar sequences"),
    ("short_different", "AAAAAAA", "GGGGGGG", "Short completely different sequences"),

    # Medium length (typical protein domain)
    ("medium_similar",
     "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM",
     "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPARTTVDTKQAQDLARSYGIPFIETSAKTRQGVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM",
     "Medium length with point mutations (KRAS variants)"),

    # Long sequences (stress test for tiling)
    ("long_identical",
     "ARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYV" * 4,
     "ARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYV" * 4,
     "Long identical sequences (800 residues)"),

    # Edge cases
    ("single_char_match", "A", "A", "Single character match"),
    ("single_char_mismatch", "A", "G", "Single character mismatch"),
    ("very_short", "AR", "AG", "Very short sequences"),

    # Biological test cases (real protein fragments)
    ("hemoglobin_fragment",
     "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
     "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
     "Hemoglobin alpha vs beta chain"),

    # Varying lengths
    ("length_100", "A" * 100, "G" * 100, "100 residues, no similarity"),
    ("length_200_repeat", "ARNDCQEGH" * 22 + "AR", "ARNDCQEGH" * 22 + "AG", "200 residues with repeating pattern"),

    # Sequences with different amino acid compositions
    ("hydrophobic_vs_hydrophilic",
     "ILVMFYWILVMFYWILVMFYWILVMFYW",  # Hydrophobic
     "RKNQHRKNQHRKNQHRKNQHRKNQH",      # Hydrophilic
     "Different chemical properties"),

    # Gap-rich alignment scenario
    ("gap_rich",
     "AAAAAAAAGGGGGGGGGAAAAAAA",
     "GGGGGGGGAAAAAAAAAGGGGGGGG",
     "Requires multiple gaps"),

    # Conservative vs non-conservative substitutions
    ("conservative",
     "ILMVILMVILMVILMV",  # Branched chain amino acids
     "VLIVVLIVVLIVVLIV",  # Similar
     "Conservative substitutions"),

    # Real protein sequences (truncated for speed)
    ("p53_fragment",
     "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
     "MEEPQSDLSVEPPLSQETFSDLWKLLPENNVLSPLPSQAVDDLMLSPDDIEQWFTEDPGPDEAPWMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGNYGFHLGFLQSGTAKSVTCTYSPLLNKLFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDGDGLAPPQHLIRVEGNLYPEYLDRRDTFRHSVVVPYEPPEAGSEYTTIHYKYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEENLRKQGEPHHELPPGSTKRALP",
     "p53 protein variants"),
]


# ============================================================================
# Validation Functions
# ============================================================================

def test_single_case(name: str, seq_a: str, seq_b: str, description: str) -> Tuple[bool, float, float, float]:
    """
    Test a single case: compare GPU vs CPU

    Returns:
        (passed, gpu_score, cpu_score, error)
    """
    try:
        # Compute GPU score
        gpu_score = align_sequences_gpu(seq_a, seq_b)

        # Compute CPU score
        cpu_score = align_sequences_sequential(seq_a, seq_b)

        # Calculate error
        error = abs(gpu_score - cpu_score)

        # Check if within tolerance
        passed = error < TOLERANCE

        return passed, gpu_score, cpu_score, error

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False, 0.0, 0.0, float('inf')


def run_all_tests(verbose: bool = True) -> Tuple[int, int, List[str]]:
    """
    Run all test cases

    Returns:
        (num_passed, num_total, failed_tests)
    """
    print("=" * 80)
    print("SMITH-WATERMAN CORRECTNESS VALIDATION")
    print("GPU vs CPU Comparison")
    print("=" * 80)
    print()

    passed_count = 0
    total_count = len(TEST_CASES)
    failed_tests = []

    for i, (name, seq_a, seq_b, description) in enumerate(TEST_CASES, 1):
        if verbose:
            print(f"Test {i}/{total_count}: {name}")
            print(f"  Description: {description}")
            print(f"  Lengths: {len(seq_a)} vs {len(seq_b)}")

        passed, gpu_score, cpu_score, error = test_single_case(name, seq_a, seq_b, description)

        if passed:
            passed_count += 1
            if verbose:
                print(f"  ✓ PASS - GPU: {gpu_score:.6f}, CPU: {cpu_score:.6f}, Error: {error:.2e}")
        else:
            failed_tests.append(name)
            if verbose:
                print(f"  ✗ FAIL - GPU: {gpu_score:.6f}, CPU: {cpu_score:.6f}, Error: {error:.2e}")
                print(f"         Error exceeds tolerance ({TOLERANCE:.2e})")

        if verbose:
            print()

    return passed_count, total_count, failed_tests


def print_summary(passed: int, total: int, failed: List[str]):
    """Print test summary"""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{total} ({100.0 * passed / total:.1f}%)")
    print(f"Failed: {total - passed}/{total}")

    if failed:
        print("\nFailed tests:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("\n✓ ALL TESTS PASSED!")

    print("=" * 80)


# ============================================================================
# Statistical Validation
# ============================================================================

def run_statistical_validation(num_random: int = 100, max_length: int = 300):
    """
    Run statistical validation on random sequences

    Args:
        num_random: Number of random test cases
        max_length: Maximum sequence length
    """
    print()
    print("=" * 80)
    print(f"STATISTICAL VALIDATION ({num_random} random sequences)")
    print("=" * 80)
    print()

    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    errors = []
    max_error = 0.0
    max_error_case = None

    np.random.seed(42)  # Reproducible

    for i in range(num_random):
        # Generate random sequences
        len_a = np.random.randint(10, max_length)
        len_b = np.random.randint(10, max_length)

        seq_a = ''.join(np.random.choice(list(amino_acids), len_a))
        seq_b = ''.join(np.random.choice(list(amino_acids), len_b))

        # Test
        passed, gpu_score, cpu_score, error = test_single_case(
            f"random_{i}", seq_a, seq_b, f"Random {len_a}x{len_b}"
        )

        errors.append(error)

        if error > max_error:
            max_error = error
            max_error_case = (i, len_a, len_b, gpu_score, cpu_score)

        # Print progress
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{num_random} tests...")

    # Statistics
    errors = np.array(errors)
    passed_random = np.sum(errors < TOLERANCE)

    print()
    print(f"Results:")
    print(f"  Passed: {passed_random}/{num_random} ({100.0 * passed_random / num_random:.1f}%)")
    print(f"  Mean error: {np.mean(errors):.2e}")
    print(f"  Std error: {np.std(errors):.2e}")
    print(f"  Max error: {np.max(errors):.2e}")
    print(f"  Min error: {np.min(errors):.2e}")

    if max_error_case:
        i, len_a, len_b, gpu, cpu = max_error_case
        print(f"\n  Worst case (test {i}):")
        print(f"    Lengths: {len_a} x {len_b}")
        print(f"    GPU: {gpu:.6f}, CPU: {cpu:.6f}")
        print(f"    Error: {max_error:.2e}")

    if passed_random == num_random:
        print("\n  ✓ ALL RANDOM TESTS PASSED!")
    else:
        print(f"\n  ✗ {num_random - passed_random} random tests failed")

    print("=" * 80)

    return passed_random, num_random


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Smith-Waterman correctness validation")
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--no-random', action='store_true', help='Skip random tests')
    parser.add_argument('--num-random', type=int, default=100, help='Number of random tests')

    args = parser.parse_args()

    if not GPU_AVAILABLE:
        print("ERROR: GPU implementation not available")
        return 1

    # Run curated test cases
    passed, total, failed = run_all_tests(verbose=not args.quiet)

    # Run statistical validation
    if not args.no_random:
        passed_random, total_random = run_statistical_validation(num_random=args.num_random)
    else:
        passed_random, total_random = 0, 0

    # Print final summary
    print()
    print_summary(passed, total, failed)

    if not args.no_random:
        print(f"\nRandom tests: {passed_random}/{total_random} passed")

    # Exit code
    all_passed = (passed == total) and (args.no_random or passed_random == total_random)

    if all_passed:
        print("\n✓ VALIDATION SUCCESSFUL - GPU matches CPU implementation")
        return 0
    else:
        print("\n✗ VALIDATION FAILED - GPU does not match CPU")
        return 1


if __name__ == "__main__":
    sys.exit(main())
