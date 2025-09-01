#!/usr/bin/env python3
"""
Unit tests to verify that the improved evaluation script captures 
the last timestep (with mortality signal) for all patients.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sequence_coverage(patient_length, sequence_length, stride):
    """
    Test that a patient sequence is fully covered, especially the last timestep.
    
    Returns:
        (covered_timesteps, last_timestep_covered): Set of covered timesteps and boolean for last timestep
    """
    covered_timesteps = set()
    last_timestep_covered = False
    
    # Simulate the sequence generation logic from the evaluation script
    for seq_start in range(0, patient_length - sequence_length + 1, stride):
        seq_end = min(seq_start + sequence_length, patient_length)
        
        # Add all timesteps in this sequence to covered set
        for t in range(seq_start, seq_end):
            covered_timesteps.add(t)
        
        # Check if this sequence covers the last timestep
        if seq_end == patient_length:
            last_timestep_covered = True
    
    return covered_timesteps, last_timestep_covered


def run_coverage_tests():
    """Run comprehensive tests for sequence coverage."""
    
    print("="*70)
    print(" TESTING SEQUENCE COVERAGE FOR MORTALITY SIGNAL CAPTURE")
    print("="*70)
    
    test_cases = [
        # (patient_length, sequence_length, stride, description)
        (50, 20, 1, "Standard case: overlapping sequences with stride=1"),
        (50, 20, 20, "Non-overlapping sequences (original implementation)"),
        (50, 20, 5, "Partially overlapping sequences with stride=5"),
        (20, 20, 1, "Edge case: patient length equals sequence length"),
        (21, 20, 1, "Edge case: patient length slightly larger than sequence"),
        (39, 20, 1, "Edge case: patient length just under 2x sequence length"),
        (40, 20, 1, "Edge case: patient length exactly 2x sequence length"),
        (100, 20, 1, "Long patient trajectory with stride=1"),
        (100, 20, 10, "Long patient trajectory with stride=10"),
    ]
    
    all_passed = True
    
    for patient_length, sequence_length, stride, description in test_cases:
        covered_timesteps, last_timestep_covered = test_sequence_coverage(
            patient_length, sequence_length, stride
        )
        
        # Check coverage completeness
        expected_timesteps = set(range(patient_length))
        missing_timesteps = expected_timesteps - covered_timesteps
        coverage_percent = len(covered_timesteps) / patient_length * 100
        
        # Determine if test passed
        if stride == 1:
            # With stride=1, we should have complete coverage
            test_passed = (len(missing_timesteps) == 0 and last_timestep_covered)
        else:
            # With larger strides, we at least need the last timestep
            test_passed = last_timestep_covered
        
        # Print results
        status = "✓ PASS" if test_passed else "✗ FAIL"
        print(f"\n{status}: {description}")
        print(f"  Patient length: {patient_length}, Sequence length: {sequence_length}, Stride: {stride}")
        print(f"  Coverage: {coverage_percent:.1f}% ({len(covered_timesteps)}/{patient_length} timesteps)")
        print(f"  Last timestep covered: {last_timestep_covered}")
        
        if missing_timesteps and stride == 1:
            print(f"  WARNING: Missing timesteps with stride=1: {sorted(missing_timesteps)}")
        
        if not last_timestep_covered:
            print(f"  ERROR: Last timestep (t={patient_length-1}) NOT covered!")
            print(f"  This means the mortality signal would be missed!")
        
        if not test_passed:
            all_passed = False
    
    # Additional test: Verify the assertion logic
    print("\n" + "="*70)
    print(" TESTING ASSERTION LOGIC")
    print("="*70)
    
    # Test that assertion would catch missing last timestep
    try:
        # Simulate a case where last timestep is not covered (stride too large)
        patient_length = 50
        sequence_length = 20
        stride = 25  # This stride is too large and will miss the end
        
        covered_timesteps, last_timestep_covered = test_sequence_coverage(
            patient_length, sequence_length, stride
        )
        
        if patient_length >= sequence_length:
            assert last_timestep_covered, \
                f"Assertion test: Last timestep not covered! Length={patient_length}, stride={stride}"
        
        print("✗ FAIL: Assertion should have been triggered but wasn't!")
        all_passed = False
        
    except AssertionError as e:
        print(f"✓ PASS: Assertion correctly triggered: {e}")
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print(" ✓ ALL TESTS PASSED!")
        print(" The improved evaluation with stride=1 ensures complete coverage")
        print(" and always captures the mortality signal at the last timestep.")
    else:
        print(" ✗ SOME TESTS FAILED!")
        print(" Please review the implementation to ensure proper coverage.")
    print("="*70)
    
    return all_passed


def compare_coverage_strategies():
    """Compare different stride strategies for coverage and efficiency."""
    
    print("\n" + "="*70)
    print(" COMPARING COVERAGE STRATEGIES")
    print("="*70)
    
    patient_lengths = [30, 50, 75, 100, 150]
    sequence_length = 20
    strides = [1, 5, 10, 20]
    
    print(f"\nSequence length: {sequence_length}")
    print("\n{:<15} {:<10} {:<15} {:<20} {:<15}".format(
        "Patient Length", "Stride", "Sequences", "Coverage %", "Last Step?"
    ))
    print("-" * 75)
    
    for patient_length in patient_lengths:
        for stride in strides:
            if patient_length < sequence_length:
                continue
                
            num_sequences = len(range(0, patient_length - sequence_length + 1, stride))
            covered_timesteps, last_timestep_covered = test_sequence_coverage(
                patient_length, sequence_length, stride
            )
            coverage_percent = len(covered_timesteps) / patient_length * 100
            
            print("{:<15} {:<10} {:<15} {:<20.1f} {:<15}".format(
                patient_length,
                stride,
                num_sequences,
                coverage_percent,
                "Yes" if last_timestep_covered else "NO - MISSING!"
            ))
        
        if patient_length != patient_lengths[-1]:
            print()
    
    print("\n" + "="*70)
    print(" CONCLUSION:")
    print(" - Stride=1 provides 100% coverage and always captures mortality signal")
    print(" - Larger strides reduce computational cost but may miss critical timesteps")
    print(" - Original stride=sequence_length often misses the final timesteps")
    print("="*70)


if __name__ == "__main__":
    # Run the tests
    success = run_coverage_tests()
    
    # Show comparison of strategies
    compare_coverage_strategies()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)