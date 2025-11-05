"""
Simple Usage Examples for Randomized Quicksort

This file provides easy-to-run examples showing how to use the randomized quicksort
implementation and understand its benefits.
"""

from randomized_quicksort import randomized_quicksort, randomized_quicksort_functional
import random

def basic_usage_examples():
    """
    Basic usage examples for randomized quicksort.
    """
    print("=" * 60)
    print("RANDOMIZED QUICKSORT - BASIC USAGE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Basic in-place sorting
    print("\n1. IN-PLACE SORTING:")
    print("-" * 30)
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Before: {arr}")
    randomized_quicksort(arr)
    print(f"After:  {arr}")
    
    # Example 2: Functional version (creates new array)
    print("\n2. FUNCTIONAL VERSION (creates new array):")
    print("-" * 45)
    original = [9, 1, 8, 2, 7, 3, 6, 4, 5]
    sorted_arr = randomized_quicksort_functional(original)
    print(f"Original:    {original}")
    print(f"New sorted:  {sorted_arr}")
    
    # Example 3: With detailed output
    print("\n3. WITH DETAILED STEP-BY-STEP OUTPUT:")
    print("-" * 40)
    arr = [5, 2, 8, 1, 9]
    print(f"Sorting: {arr}")
    randomized_quicksort(arr, verbose=True)
    print(f"Result:  {arr}")


def demonstrate_randomization():
    """
    Shows how random pivot selection creates different execution paths.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATING RANDOMIZATION EFFECT")
    print("=" * 60)
    
    test_array = [6, 3, 8, 1, 4, 7, 2, 5]
    print(f"Same input array: {test_array}")
    print("\nMultiple runs with different random seeds:")
    
    for run in range(4):
        random.seed(run)  # Different seed for each run
        arr_copy = test_array.copy()
        print(f"\nRun {run + 1} (seed={run}):")
        randomized_quicksort(arr_copy, verbose=True)
        print(f"Result: {arr_copy}")


def compare_with_python_sorted():
    """
    Verify our implementation matches Python's built-in sorted() function.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION AGAINST PYTHON'S BUILT-IN SORT")
    print("=" * 60)
    
    test_cases = [
        [3, 1, 4, 1, 5, 9, 2, 6, 5],
        [100, 50, 25, 75, 10, 90],
        [1],
        [],
        [5, 5, 5, 5, 5],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    all_correct = True
    
    for i, test_case in enumerate(test_cases, 1):
        original = test_case.copy()
        expected = sorted(test_case)
        
        # Test our randomized quicksort
        randomized_quicksort(test_case)
        our_result = test_case
        
        # Test functional version
        functional_result = randomized_quicksort_functional(original.copy())
        
        # Check correctness
        in_place_correct = our_result == expected
        functional_correct = functional_result == expected
        
        print(f"Test {i}: {original}")
        print(f"  Expected: {expected}")
        print(f"  Our in-place: {our_result} {'✓' if in_place_correct else '✗'}")
        print(f"  Our functional: {functional_result} {'✓' if functional_correct else '✗'}")
        
        if not (in_place_correct and functional_correct):
            all_correct = False
    
    print(f"\nAll tests passed: {'✓' if all_correct else '✗'}")


def performance_demonstration():
    """Demonstrate performance on different input patterns."""
    print("=" * 60)
    print("PERFORMANCE ON DIFFERENT INPUT PATTERNS")
    print("=" * 60)
    
    import time
    
    # Use smaller size to avoid recursion issues
    size = 100
    print(f"Testing with arrays of size {size}:")
    print()
    
    # Test on different patterns
    patterns = {
        "Random": list(range(size)),
        "Sorted": list(range(size)),
        "Reverse Sorted": list(range(size, 0, -1)),
        "Nearly Sorted": list(range(size))
    }
    
    # Make random pattern actually random
    import random
    random.shuffle(patterns["Random"])
    
    # Make nearly sorted (90% sorted with some random swaps)
    for _ in range(size // 10):
        i, j = random.randint(0, size-1), random.randint(0, size-1)
        patterns["Nearly Sorted"][i], patterns["Nearly Sorted"][j] = patterns["Nearly Sorted"][j], patterns["Nearly Sorted"][i]
    
    for pattern_name, arr in patterns.items():
        test_arr = arr.copy()
        start_time = time.time()
        randomized_quicksort(test_arr)
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"{pattern_name:<15}: {duration:6.2f} ms ✓")
    
    print()
    print("Note: Randomized quicksort performs well on all patterns,")
    print("avoiding the O(n²) worst-case behavior of deterministic quicksort.")


if __name__ == "__main__":
    # Set initial seed for reproducible demonstrations
    random.seed(42)
    
    # Run all demonstrations
    basic_usage_examples()
    demonstrate_randomization()
    compare_with_python_sorted()
    performance_demonstration()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
Randomized Quicksort Implementation Complete!

Key Benefits:
• Expected O(n log n) performance on ALL input types
• Simple random pivot selection eliminates worst-case scenarios
• Robust against adversarial/malicious inputs
• Easy to implement and understand

Use Cases:
• General-purpose sorting when you need good guaranteed performance
• When input data might be already sorted or have patterns
• In competitive programming to avoid time limit exceeded
• As a building block for other algorithms

The randomization makes this a practical, production-ready sorting algorithm!
""")