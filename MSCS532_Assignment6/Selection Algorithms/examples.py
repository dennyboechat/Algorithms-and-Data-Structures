"""
Example Usage of Median of Medians Selection Algorithm
MSCS532 Assignment 6

This file demonstrates various use cases of the deterministic selection algorithm.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

from median_of_medians import (
    median_of_medians_select, 
    find_median, 
    find_kth_smallest, 
    find_kth_largest
)

def example_basic_usage():
    """Demonstrate basic usage of the selection algorithm."""
    print("=== Basic Usage Examples ===\n")
    
    # Example 1: Simple array
    arr1 = [64, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    print(f"Array: {arr1}")
    print(f"Sorted: {sorted(arr1)}")
    print(f"3rd smallest element: {find_kth_smallest(arr1, 3)}")
    print(f"Median (5th smallest): {find_median(arr1)}")
    print(f"8th smallest element: {find_kth_smallest(arr1, 8)}")
    print(f"2nd largest element: {find_kth_largest(arr1, 2)}")
    print()


def example_edge_cases():
    """Demonstrate handling of edge cases."""
    print("=== Edge Cases ===\n")
    
    # Single element
    single = [42]
    print(f"Single element array {single}:")
    print(f"  Median: {find_median(single)}")
    print()
    
    # Two elements
    pair = [10, 5]
    print(f"Two element array {pair}:")
    print(f"  1st smallest: {find_kth_smallest(pair, 1)}")
    print(f"  2nd smallest: {find_kth_smallest(pair, 2)}")
    print(f"  Median: {find_median(pair)}")
    print()
    
    # Array with duplicates
    duplicates = [7, 3, 7, 1, 7, 9, 3]
    print(f"Array with duplicates {duplicates}:")
    print(f"  Sorted: {sorted(duplicates)}")
    print(f"  3rd smallest: {find_kth_smallest(duplicates, 3)}")
    print(f"  Median: {find_median(duplicates)}")
    print()


def example_practical_applications():
    """Demonstrate practical applications of the algorithm."""
    print("=== Practical Applications ===\n")
    
    # Finding quartiles
    grades = [85, 92, 78, 96, 88, 76, 94, 82, 90, 87, 91, 79]
    n = len(grades)
    print(f"Student grades: {grades}")
    print(f"Total students: {n}")
    
    q1_pos = n // 4
    q2_pos = n // 2  # median
    q3_pos = 3 * n // 4
    
    print(f"First Quartile (25th percentile): {find_kth_smallest(grades, q1_pos)}")
    print(f"Second Quartile (50th percentile - Median): {find_kth_smallest(grades, q2_pos)}")
    print(f"Third Quartile (75th percentile): {find_kth_smallest(grades, q3_pos)}")
    print()
    
    # Finding top performers
    print("Top 3 performers:")
    for i in range(1, 4):
        score = find_kth_largest(grades, i)
        print(f"  {i}{'st' if i == 1 else 'nd' if i == 2 else 'rd'} highest: {score}")
    print()


def example_performance_analysis():
    """Demonstrate performance characteristics."""
    print("=== Performance Analysis ===\n")
    
    import time
    import random
    
    # Test different array sizes
    sizes = [50, 100, 500, 1000]
    
    for size in sizes:
        # Create worst-case scenario: reverse sorted
        arr = list(range(size, 0, -1))
        k = size // 2  # Find median
        
        start_time = time.time()
        result = median_of_medians_select(arr, k)
        end_time = time.time()
        
        print(f"Array size {size:4d}: {end_time - start_time:.6f} seconds (result: {result})")
    
    print("\nNote: Time complexity remains O(n) even for worst-case inputs!")
    print()


def example_comparison_with_sorting():
    """Compare with traditional sorting approach."""
    print("=== Comparison with Sorting ===\n")
    
    import time
    import random
    
    # Large array for comparison
    size = 10000
    arr = [random.randint(1, 1000) for _ in range(size)]
    k = size // 2  # Find median
    
    # Median of Medians approach
    arr_copy1 = arr.copy()
    start_time = time.time()
    result_mom = median_of_medians_select(arr_copy1, k)
    time_mom = time.time() - start_time
    
    # Sorting approach
    arr_copy2 = arr.copy()
    start_time = time.time()
    sorted_arr = sorted(arr_copy2)
    result_sort = sorted_arr[k-1]
    time_sort = time.time() - start_time
    
    print(f"Array size: {size}")
    print(f"Finding {k}th smallest element...")
    print(f"Median of Medians: {time_mom:.6f} seconds")
    print(f"Sorting approach:   {time_sort:.6f} seconds")
    print(f"Results match: {result_mom == result_sort}")
    print(f"MoM vs Sort ratio: {time_mom/time_sort:.2f}x")
    print()
    
    print("Advantages of Median of Medians:")
    print("- Guaranteed O(n) worst-case performance")
    print("- No degradation on adversarial inputs")
    print("- Deterministic behavior (no randomness)")
    print("- Better for real-time systems requiring guarantees")
    print()


def interactive_demo():
    """Provide an interactive demonstration."""
    print("=== Interactive Demo ===")
    print("Try these examples or create your own:\n")
    
    examples = [
        ([15, 20, 35, 40, 50], 3),
        ([1, 2, 3, 4, 5, 6, 7], 4),
        ([100, 80, 60, 40, 20], 2),
        ([5, 5, 5, 5, 5], 3),
    ]
    
    for i, (arr, k) in enumerate(examples, 1):
        result = find_kth_smallest(arr, k)
        print(f"Example {i}: Array {arr}")
        print(f"           {k}th smallest = {result}")
        print(f"           Verification: {sorted(arr)} -> position {k} = {result}")
        print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_edge_cases()
    example_practical_applications()
    example_performance_analysis()
    example_comparison_with_sorting()
    interactive_demo()
    
    print("=== Summary ===")
    print("The Median of Medians algorithm provides:")
    print("✓ O(n) worst-case time complexity")
    print("✓ Deterministic behavior")
    print("✓ Robust handling of edge cases")
    print("✓ Practical applications in statistics and data analysis")
    print("✓ Superior worst-case guarantees compared to randomized approaches")