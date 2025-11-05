"""
Test file for Quicksort implementations.

This file contains comprehensive tests and performance comparisons
for the quicksort algorithms.
"""

import time
import random
from quicksort import quicksort, quicksort_with_different_pivots
from simple_quicksort import simple_quicksort, quicksort_inplace


def test_correctness():
    """
    Tests the correctness of all quicksort implementations.
    """
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    test_cases = [
        [],  # Empty array
        [1],  # Single element
        [2, 1],  # Two elements
        [3, 1, 4, 1, 5, 9, 2, 6, 5],  # Random array
        [5, 4, 3, 2, 1],  # Reverse sorted
        [1, 2, 3, 4, 5],  # Already sorted
        [3, 3, 3, 3, 3],  # All duplicates
        [1, 3, 2, 3, 1, 2],  # Mixed duplicates
    ]
    
    for i, test_arr in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_arr}")
        expected = sorted(test_arr)
        
        # Test simple_quicksort
        result1 = simple_quicksort(test_arr.copy())
        
        # Test quicksort_inplace
        arr_copy = test_arr.copy()
        quicksort_inplace(arr_copy)
        result2 = arr_copy
        
        # Test original quicksort
        arr_copy2 = test_arr.copy()
        quicksort(arr_copy2)
        result3 = arr_copy2
        
        # Check results
        success1 = result1 == expected
        success2 = result2 == expected
        success3 = result3 == expected
        
        print(f"  Expected: {expected}")
        print(f"  Simple quicksort: {result1} {'✓' if success1 else '✗'}")
        print(f"  In-place quicksort: {result2} {'✓' if success2 else '✗'}")
        print(f"  Original quicksort: {result3} {'✓' if success3 else '✗'}")
        
        if not (success1 and success2 and success3):
            print("  ❌ FAILED!")
        else:
            print("  ✅ PASSED!")


def performance_comparison():
    """
    Compares performance of different implementations and pivot strategies.
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Generate test data
    sizes = [100, 1000, 5000]
    
    for size in sizes:
        print(f"\nArray size: {size}")
        print("-" * 30)
        
        # Create different types of test data
        random_data = [random.randint(1, 1000) for _ in range(size)]
        sorted_data = list(range(size))
        reverse_data = list(range(size, 0, -1))
        
        datasets = [
            ("Random", random_data),
            ("Sorted", sorted_data),
            ("Reverse", reverse_data)
        ]
        
        for data_type, data in datasets:
            print(f"\n  {data_type} data:")
            
            # Test in-place quicksort
            test_data = data.copy()
            start_time = time.time()
            quicksort(test_data)
            end_time = time.time()
            print(f"    In-place quicksort: {(end_time - start_time)*1000:.2f} ms")
            
            # Test different pivot strategies (only for smaller arrays to avoid timeout)
            if size <= 1000:
                strategies = ["first", "last", "middle", "random"]
                for strategy in strategies:
                    test_data = data.copy()
                    start_time = time.time()
                    quicksort_with_different_pivots(test_data, strategy)
                    end_time = time.time()
                    print(f"    {strategy.capitalize()} pivot: {(end_time - start_time)*1000:.2f} ms")


def educational_demonstration():
    """
    Educational demonstration showing the algorithm in action.
    """
    print("\n" + "=" * 60)
    print("EDUCATIONAL DEMONSTRATION")
    print("=" * 60)
    
    # Small array to clearly show the process
    demo_array = [8, 3, 5, 4, 7, 6, 1, 2]
    print(f"Demonstrating quicksort on: {demo_array}")
    print("\nUsing simple quicksort (creates new arrays):")
    print("-" * 50)
    
    result = simple_quicksort(demo_array)
    
    print(f"\nFinal result: {result}")
    
    print("\nNow using in-place quicksort:")
    print("-" * 50)
    demo_array_copy = demo_array.copy()
    quicksort_inplace(demo_array_copy)
    print(f"\nFinal result: {demo_array_copy}")


def worst_case_scenario():
    """
    Demonstrates the worst-case scenario for quicksort.
    """
    print("\n" + "=" * 60)
    print("WORST-CASE SCENARIO DEMONSTRATION")
    print("=" * 60)
    
    # Already sorted array is worst case for "last element" pivot strategy
    worst_case_array = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Worst-case array (sorted): {worst_case_array}")
    print("\nThis will cause O(n²) behavior with 'last' pivot strategy")
    print("because the pivot will always be the largest element,")
    print("creating unbalanced partitions.\n")
    
    # Show how different pivot strategies handle this
    strategies = ["last", "first", "middle", "random"]
    
    for strategy in strategies:
        print(f"Using '{strategy}' pivot strategy:")
        test_array = worst_case_array.copy()
        start_time = time.time()
        result = quicksort_with_different_pivots(test_array, strategy)
        end_time = time.time()
        print(f"  Time: {(end_time - start_time)*1000:.4f} ms")
        print(f"  Result: {result}")
        print()


if __name__ == "__main__":
    # Run all tests and demonstrations
    test_correctness()
    educational_demonstration()
    performance_comparison()
    worst_case_scenario()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. PIVOT SELECTION matters:
   - 'Last' or 'First' can be O(n²) on sorted data
   - 'Middle' or 'Random' generally perform better
   - 'Median-of-three' is often used in practice

2. PARTITIONING is the core operation:
   - Lomuto scheme: simple but does extra swaps
   - Hoare scheme: more efficient but complex
   - 3-way partitioning: handles duplicates well

3. RECURSION depth affects performance:
   - Best case: O(log n) depth
   - Worst case: O(n) depth (becomes like selection sort)
   - Tail recursion optimization can help

4. PRACTICAL CONSIDERATIONS:
   - Use hybrid approaches (switch to insertion sort for small arrays)
   - Consider introsort (switches to heapsort if recursion depth exceeds limit)
   - For stability, use merge sort instead
""")