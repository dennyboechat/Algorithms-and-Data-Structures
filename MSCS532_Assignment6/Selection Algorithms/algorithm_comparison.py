"""
Comparison of Deterministic vs Randomized Selection Algorithms
MSCS532 Assignment 6

This module provides comprehensive comparison between the deterministic (Median of Medians)
and randomized (Quickselect) selection algorithms.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

import time
import random
import statistics
from typing import List, Tuple, Dict, Any

from median_of_medians import median_of_medians_select
from randomized_quickselect import (
    randomized_select,
    randomized_select_iterative,
    quickselect_with_median_of_three
)


def performance_comparison(sizes: List[int] = None, num_trials: int = 10) -> Dict[str, Any]:
    """
    Compare performance of deterministic vs randomized selection algorithms.
    
    Args:
        sizes: List of array sizes to test
        num_trials: Number of trials per size for averaging
    
    Returns:
        Dictionary containing performance results
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000, 10000]
    
    results = {
        'sizes': sizes,
        'median_of_medians': [],
        'randomized_recursive': [],
        'randomized_iterative': [],
        'randomized_median3': [],
        'sorting_baseline': []
    }
    
    print("=== Performance Comparison: Deterministic vs Randomized ===\n")
    print(f"Running {num_trials} trials per size...\n")
    
    for size in sizes:
        print(f"Testing array size: {size}")
        
        # Initialize timing lists for this size
        times_mom = []
        times_rand_rec = []
        times_rand_iter = []
        times_rand_med3 = []
        times_sort = []
        
        for trial in range(num_trials):
            # Generate random array
            arr = [random.randint(1, 10000) for _ in range(size)]
            k = size // 2  # Find median
            
            # Test Median of Medians (Deterministic)
            arr_copy = arr.copy()
            start = time.time()
            result_mom = median_of_medians_select(arr_copy, k)
            times_mom.append(time.time() - start)
            
            # Test Randomized Quickselect (Recursive)
            arr_copy = arr.copy()
            start = time.time()
            result_rand_rec = randomized_select(arr_copy, k)
            times_rand_rec.append(time.time() - start)
            
            # Test Randomized Quickselect (Iterative)
            arr_copy = arr.copy()
            start = time.time()
            result_rand_iter = randomized_select_iterative(arr_copy, k)
            times_rand_iter.append(time.time() - start)
            
            # Test Randomized with Median-of-Three
            arr_copy = arr.copy()
            start = time.time()
            result_rand_med3 = quickselect_with_median_of_three(arr_copy, k)
            times_rand_med3.append(time.time() - start)
            
            # Test Sorting Baseline
            arr_copy = arr.copy()
            start = time.time()
            result_sort = sorted(arr_copy)[k-1]
            times_sort.append(time.time() - start)
            
            # Verify all results match
            assert result_mom == result_rand_rec == result_rand_iter == result_rand_med3 == result_sort
        
        # Calculate statistics
        avg_mom = statistics.mean(times_mom)
        avg_rand_rec = statistics.mean(times_rand_rec)
        avg_rand_iter = statistics.mean(times_rand_iter)
        avg_rand_med3 = statistics.mean(times_rand_med3)
        avg_sort = statistics.mean(times_sort)
        
        results['median_of_medians'].append(avg_mom)
        results['randomized_recursive'].append(avg_rand_rec)
        results['randomized_iterative'].append(avg_rand_iter)
        results['randomized_median3'].append(avg_rand_med3)
        results['sorting_baseline'].append(avg_sort)
        
        # Display results
        print(f"  Median of Medians:      {avg_mom:.6f} ± {statistics.stdev(times_mom):.6f}s")
        print(f"  Randomized (Recursive): {avg_rand_rec:.6f} ± {statistics.stdev(times_rand_rec):.6f}s")
        print(f"  Randomized (Iterative): {avg_rand_iter:.6f} ± {statistics.stdev(times_rand_iter):.6f}s")
        print(f"  Randomized (Median-3):  {avg_rand_med3:.6f} ± {statistics.stdev(times_rand_med3):.6f}s")
        print(f"  Sorting Baseline:       {avg_sort:.6f} ± {statistics.stdev(times_sort):.6f}s")
        print()
    
    return results


def worst_case_analysis():
    """
    Analyze worst-case behavior of both approaches.
    """
    print("=== Worst-Case Behavior Analysis ===\n")
    
    # Test with potentially problematic inputs
    test_cases = {
        "Sorted Array": list(range(1, 1001)),
        "Reverse Sorted": list(range(1000, 0, -1)),
        "All Identical": [42] * 1000,
        "Two Values": [1, 2] * 500,
        "Nearly Sorted": list(range(1, 1001))
    }
    
    # Shuffle the "nearly sorted" case slightly
    nearly_sorted = test_cases["Nearly Sorted"]
    for i in range(0, len(nearly_sorted), 50):
        if i + 1 < len(nearly_sorted):
            nearly_sorted[i], nearly_sorted[i + 1] = nearly_sorted[i + 1], nearly_sorted[i]
    
    for case_name, arr in test_cases.items():
        print(f"Testing {case_name} (size: {len(arr)}):")
        k = len(arr) // 2
        
        # Run multiple trials to see variance in randomized algorithms
        trials = 10
        times_mom = []
        times_rand = []
        
        for _ in range(trials):
            # Test Deterministic
            start = time.time()
            result_mom = median_of_medians_select(arr.copy(), k)
            times_mom.append(time.time() - start)
            
            # Test Randomized
            start = time.time()
            result_rand = randomized_select(arr.copy(), k)
            times_rand.append(time.time() - start)
            
            # Verify results match
            assert result_mom == result_rand
        
        avg_mom = statistics.mean(times_mom)
        std_mom = statistics.stdev(times_mom) if len(times_mom) > 1 else 0
        avg_rand = statistics.mean(times_rand)
        std_rand = statistics.stdev(times_rand) if len(times_rand) > 1 else 0
        
        print(f"  Deterministic: {avg_mom:.6f} ± {std_mom:.6f}s (variance: {std_mom/avg_mom*100:.1f}%)")
        print(f"  Randomized:    {avg_rand:.6f} ± {std_rand:.6f}s (variance: {std_rand/avg_rand*100:.1f}%)")
        print(f"  Ratio (R/D):   {avg_rand/avg_mom:.2f}x")
        print()


def theoretical_comparison():
    """
    Display theoretical comparison between the algorithms.
    """
    print("=== Theoretical Comparison ===\n")
    
    comparison_table = [
        ["Aspect", "Median of Medians", "Randomized Quickselect"],
        ["Time Complexity (Best)", "O(n)", "O(n)"],
        ["Time Complexity (Average)", "O(n)", "O(n)"],
        ["Time Complexity (Worst)", "O(n)", "O(n²)"],
        ["Space Complexity", "O(log n)", "O(log n) expected, O(n) worst"],
        ["Deterministic", "Yes", "No"],
        ["Constant Factors", "Higher", "Lower"],
        ["Implementation Complexity", "More Complex", "Simpler"],
        ["Practical Performance", "Consistent", "Usually Faster"],
        ["Worst-Case Guarantees", "Strong", "Probabilistic"],
        ["Cache Performance", "Good", "Very Good"],
        ["Real-time Systems", "Suitable", "Risk of bad performance"],
    ]
    
    # Print table
    col_widths = [max(len(row[i]) for row in comparison_table) for i in range(3)]
    
    for i, row in enumerate(comparison_table):
        formatted_row = " | ".join(row[j].ljust(col_widths[j]) for j in range(3))
        print(formatted_row)
        if i == 0:  # After header
            print("-" * len(formatted_row))
    
    print()


def practical_recommendations():
    """
    Provide practical recommendations for algorithm selection.
    """
    print("=== Practical Recommendations ===\n")
    
    recommendations = [
        {
            "Use Case": "General Purpose Applications",
            "Recommendation": "Randomized Quickselect",
            "Reason": "Better average performance, simpler implementation"
        },
        {
            "Use Case": "Real-time Systems",
            "Recommendation": "Median of Medians",
            "Reason": "Guaranteed worst-case performance"
        },
        {
            "Use Case": "Safety-Critical Systems",
            "Recommendation": "Median of Medians",
            "Reason": "Deterministic behavior, no risk of degradation"
        },
        {
            "Use Case": "Interactive Applications",
            "Recommendation": "Randomized Quickselect",
            "Reason": "Lower constant factors, better user experience"
        },
        {
            "Use Case": "Batch Processing",
            "Recommendation": "Either (context dependent)",
            "Reason": "Both suitable, choose based on data characteristics"
        },
        {
            "Use Case": "Adversarial Inputs",
            "Recommendation": "Median of Medians",
            "Reason": "Resistant to worst-case input patterns"
        },
        {
            "Use Case": "Memory Constrained",
            "Recommendation": "Iterative Quickselect",
            "Reason": "O(1) space complexity"
        },
        {
            "Use Case": "Educational/Research",
            "Recommendation": "Both",
            "Reason": "Different algorithmic techniques worth understanding"
        }
    ]
    
    for rec in recommendations:
        print(f"🎯 {rec['Use Case']}:")
        print(f"   Algorithm: {rec['Recommendation']}")
        print(f"   Rationale: {rec['Reason']}")
        print()


def run_comprehensive_comparison():
    """
    Run all comparison analyses.
    """
    print("COMPREHENSIVE COMPARISON: DETERMINISTIC VS RANDOMIZED SELECTION")
    print("=" * 70)
    print()
    
    # Set seed for reproducible results
    random.seed(42)
    
    # Run performance comparison
    results = performance_comparison([100, 500, 1000, 2500, 5000])
    
    # Analyze worst-case behavior
    worst_case_analysis()
    
    # Show theoretical comparison
    theoretical_comparison()
    
    # Provide recommendations
    practical_recommendations()
    
    # Summary insights
    print("=== Key Insights ===\n")
    print("1. Randomized algorithms typically perform better in practice")
    print("2. Deterministic algorithms provide stronger guarantees")
    print("3. Choice depends on application requirements:")
    print("   - Performance-critical → Randomized")
    print("   - Safety-critical → Deterministic")
    print("4. Both achieve O(n) complexity for selection problems")
    print("5. Median-of-three pivot selection improves randomized performance")
    print()
    
    return results


def interactive_comparison():
    """
    Interactive comparison tool for users to test specific cases.
    """
    print("=== Interactive Comparison Tool ===\n")
    
    test_arrays = {
        "Small Random": [random.randint(1, 100) for _ in range(20)],
        "Medium Sorted": list(range(1, 101)),
        "Large Random": [random.randint(1, 1000) for _ in range(1000)],
        "All Identical": [42] * 100,
        "Nearly Sorted": list(range(1, 51)) + [random.randint(1, 50) for _ in range(50)]
    }
    
    for name, arr in test_arrays.items():
        print(f"Testing: {name} (size: {len(arr)})")
        k = len(arr) // 2
        
        # Time both approaches
        start = time.time()
        result_det = median_of_medians_select(arr, k)
        time_det = time.time() - start
        
        start = time.time()
        result_rand = randomized_select(arr, k)
        time_rand = time.time() - start
        
        print(f"  Deterministic: {time_det:.6f}s (result: {result_det})")
        print(f"  Randomized:    {time_rand:.6f}s (result: {result_rand})")
        print(f"  Speedup:       {time_det/time_rand:.2f}x {'(randomized faster)' if time_rand < time_det else '(deterministic faster)'}")
        print()


if __name__ == "__main__":
    # Run comprehensive comparison
    run_comprehensive_comparison()
    
    # Interactive comparison
    interactive_comparison()