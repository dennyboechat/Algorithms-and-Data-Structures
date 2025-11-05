"""
Randomized Quicksort Implementation

This module implements a randomized version of the Quicksort algorithm where
the pivot is chosen randomly from the subarray being sorted. This approach
provides better average-case performance and helps avoid worst-case scenarios
that can occur with deterministic pivot selection.

Key Benefits of Randomized Quicksort:
- Expected O(n log n) performance regardless of input order
- Eliminates worst-case behavior on already sorted data
- Better performance on adversarial inputs
- Simple modification to standard quicksort
"""

import random
import time
from typing import List, Optional


def randomized_quicksort(arr: List[int], low: int = 0, high: Optional[int] = None, 
                        verbose: bool = False) -> None:
    """
    Randomized quicksort implementation that sorts array in-place.
    
    The key difference from standard quicksort is that the pivot is chosen
    randomly from the current subarray, which provides better expected
    performance and avoids worst-case scenarios.
    
    Args:
        arr: List of integers to sort (modified in-place)
        low: Starting index of subarray (default: 0)
        high: Ending index of subarray (default: len(arr)-1)
        verbose: If True, prints detailed step information
    
    Time Complexity:
        - Expected: O(n log n)
        - Worst case: O(n²) but extremely unlikely with random pivots
    
    Space Complexity: O(log n) for recursion stack
    """
    if high is None:
        high = len(arr) - 1
    
    # Base case: subarray has 0 or 1 elements
    if low < high:
        if verbose:
            print(f"Sorting subarray[{low}:{high}]: {arr[low:high+1]}")
        
        # STEP 1: Random pivot selection and partitioning
        pivot_index = randomized_partition(arr, low, high, verbose)
        
        if verbose:
            print(f"After partitioning around pivot {arr[pivot_index]}: {arr[low:high+1]}")
            print(f"Pivot {arr[pivot_index]} is at final position {pivot_index}")
        
        # STEP 2: Recursively sort left and right subarrays
        randomized_quicksort(arr, low, pivot_index - 1, verbose)
        randomized_quicksort(arr, pivot_index + 1, high, verbose)


def randomized_partition(arr: List[int], low: int, high: int, verbose: bool = False) -> int:
    """
    Partition function with random pivot selection.
    
    This is the key function that makes quicksort randomized. Instead of
    always choosing the last element (or first, or middle), we randomly
    select any element from the current subarray as the pivot.
    
    Args:
        arr: Array to partition
        low: Start index of subarray
        high: End index of subarray
        verbose: If True, prints partition details
    
    Returns:
        Final position of the pivot element
    """
    # RANDOM PIVOT SELECTION: Choose random index between low and high
    random_index = random.randint(low, high)
    
    if verbose:
        print(f"  Randomly selected pivot: arr[{random_index}] = {arr[random_index]}")
    
    # Swap randomly chosen element with last element
    # This allows us to use the standard Lomuto partition scheme
    arr[random_index], arr[high] = arr[high], arr[random_index]
    
    if verbose and random_index != high:
        print(f"  Swapped pivot to end: {arr[low:high+1]}")
    
    # Now use standard Lomuto partition with the randomly chosen pivot
    return lomuto_partition(arr, low, high, verbose)


def lomuto_partition(arr: List[int], low: int, high: int, verbose: bool = False) -> int:
    """
    Standard Lomuto partition scheme.
    
    Partitions the array so that elements ≤ pivot are on the left,
    and elements > pivot are on the right.
    
    Args:
        arr: Array to partition
        low: Start index
        high: End index (contains the pivot)
        verbose: If True, prints partition steps
    
    Returns:
        Final position of pivot
    """
    pivot = arr[high]  # Last element is our pivot
    i = low - 1        # Index of smaller element
    
    if verbose:
        print(f"  Partitioning with pivot = {pivot}")
    
    # Traverse through array and move smaller elements to left
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            if verbose and i != j:
                print(f"    Moved {arr[i]} to position {i}")
    
    # Place pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    if verbose:
        print(f"  Placed pivot {pivot} at final position {i + 1}")
    
    return i + 1


def randomized_quicksort_functional(arr: List[int]) -> List[int]:
    """
    Functional (non-in-place) version of randomized quicksort.
    
    This version creates new arrays and is easier to understand,
    but uses more memory than the in-place version.
    
    Args:
        arr: List to sort
    
    Returns:
        New sorted list
    """
    # Base case
    if len(arr) <= 1:
        return arr
    
    # Random pivot selection
    pivot_index = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_index]
    
    # Partition into three parts
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort and combine
    return (randomized_quicksort_functional(left) + 
            middle + 
            randomized_quicksort_functional(right))


def compare_pivot_strategies():
    """
    Demonstrates the difference between deterministic and randomized pivot selection.
    """
    print("=" * 70)
    print("COMPARING DETERMINISTIC VS RANDOMIZED PIVOT SELECTION")
    print("=" * 70)
    
    # Test on worst-case data for deterministic quicksort
    worst_case_data = list(range(1, 11))  # Already sorted: [1, 2, 3, ..., 10]
    print(f"Test data (worst case for last-element pivot): {worst_case_data}")
    
    print("\n1. DETERMINISTIC QUICKSORT (last element pivot):")
    print("-" * 50)
    deterministic_arr = worst_case_data.copy()
    start_time = time.time()
    quicksort_deterministic(deterministic_arr, 0, len(deterministic_arr) - 1, verbose=True)
    deterministic_time = time.time() - start_time
    print(f"Result: {deterministic_arr}")
    print(f"Time: {deterministic_time:.6f} seconds")
    
    print("\n2. RANDOMIZED QUICKSORT:")
    print("-" * 50)
    randomized_arr = worst_case_data.copy()
    start_time = time.time()
    randomized_quicksort(randomized_arr, verbose=True)
    randomized_time = time.time() - start_time
    print(f"Result: {randomized_arr}")
    print(f"Time: {randomized_time:.6f} seconds")


def quicksort_deterministic(arr: List[int], low: int, high: int, verbose: bool = False):
    """
    Standard deterministic quicksort for comparison (always uses last element as pivot).
    """
    if low < high:
        if verbose:
            print(f"Sorting subarray[{low}:{high}]: {arr[low:high+1]}")
        
        pivot_index = lomuto_partition(arr, low, high, verbose)
        
        if verbose:
            print(f"Pivot {arr[pivot_index]} at position {pivot_index}")
        
        quicksort_deterministic(arr, low, pivot_index - 1, verbose)
        quicksort_deterministic(arr, pivot_index + 1, high, verbose)


def performance_analysis():
    """
    Analyzes performance of randomized quicksort on various input types.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS: RANDOMIZED QUICKSORT")
    print("=" * 70)
    
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\nArray size: {size}")
        print("-" * 30)
        
        # Test different input patterns
        test_cases = [
            ("Random", [random.randint(1, 1000) for _ in range(size)]),
            ("Sorted", list(range(1, size + 1))),
            ("Reverse", list(range(size, 0, -1))),
            ("All Same", [42] * size),
            ("Nearly Sorted", list(range(1, size + 1))),
        ]
        
        # Add some random swaps to "nearly sorted"
        for _ in range(size // 20):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            test_cases[4][1][i], test_cases[4][1][j] = test_cases[4][1][j], test_cases[4][1][i]
        
        for test_name, test_data in test_cases:
            arr_copy = test_data.copy()
            
            start_time = time.time()
            randomized_quicksort(arr_copy)
            end_time = time.time()
            
            # Verify correctness
            is_sorted = arr_copy == sorted(test_data)
            
            print(f"  {test_name:12}: {(end_time - start_time)*1000:6.2f} ms "
                  f"{'✓' if is_sorted else '✗'}")


def demonstrate_randomized_quicksort():
    """
    Main demonstration function showing randomized quicksort in action.
    """
    print("=" * 70)
    print("RANDOMIZED QUICKSORT DEMONSTRATION")
    print("=" * 70)
    
    print("\nKey Features:")
    print("• Random pivot selection from current subarray")
    print("• Expected O(n log n) performance on all inputs")
    print("• Eliminates worst-case behavior on sorted data")
    print("• Simple modification to standard quicksort")
    
    # Example 1: Small array with detailed output
    print("\n" + "=" * 50)
    print("EXAMPLE 1: Step-by-step execution")
    print("=" * 50)
    
    example_arr = [64, 34, 25, 12, 22, 11, 90, 5]
    print(f"Original array: {example_arr}")
    print("\nStep-by-step execution:")
    
    # Set seed for reproducible example
    random.seed(42)
    randomized_quicksort(example_arr, verbose=True)
    
    print(f"\nFinal sorted array: {example_arr}")
    
    # Example 2: Multiple runs showing different pivot choices
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Multiple runs with different random choices")
    print("=" * 50)
    
    original = [7, 2, 1, 6, 8, 5, 3, 4]
    print(f"Original array: {original}")
    
    for run in range(3):
        arr_copy = original.copy()
        print(f"\nRun {run + 1}:")
        randomized_quicksort(arr_copy, verbose=True)
        print(f"Result: {arr_copy}")
    
    # Example 3: Functional version
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Functional (non-in-place) version")
    print("=" * 50)
    
    test_arr = [9, 1, 8, 2, 7, 3, 6, 4, 5]
    print(f"Original: {test_arr}")
    sorted_arr = randomized_quicksort_functional(test_arr)
    print(f"Sorted:   {sorted_arr}")
    print(f"Original unchanged: {test_arr}")


if __name__ == "__main__":
    # Set seed for reproducible demonstrations
    random.seed(42)
    
    # Run all demonstrations
    demonstrate_randomized_quicksort()
    compare_pivot_strategies()
    performance_analysis()
    
    print("\n" + "=" * 70)
    print("SUMMARY: RANDOMIZED QUICKSORT")
    print("=" * 70)
    print("""
Key Advantages of Randomized Quicksort:

1. EXPECTED PERFORMANCE: O(n log n) regardless of input order
2. ROBUSTNESS: Eliminates worst-case behavior on sorted/reverse sorted data
3. SIMPLICITY: Simple modification to standard quicksort
4. PRACTICAL: Used in many real-world implementations

Algorithm Steps:
1. Choose random pivot from current subarray
2. Partition array around the chosen pivot
3. Recursively sort left and right subarrays

The randomization ensures that even adversarial inputs will have
good expected performance, making this a robust sorting algorithm
suitable for production use.
""")