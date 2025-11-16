"""
Randomized Selection Algorithm - Randomized Quickselect
MSCS532 Assignment 6

This module implements the randomized selection algorithm (Randomized Quickselect) 
that finds the k-th smallest element in an unsorted array in expected linear time O(n).

The algorithm uses random pivot selection to achieve good average-case performance,
though worst-case can be O(n²) in rare cases.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

import random
from typing import List, Union


def randomized_select(arr: List[Union[int, float]], k: int) -> Union[int, float]:
    """
    Find the k-th smallest element using Randomized Quickselect.
    
    This is a randomized selection algorithm with O(n) expected time complexity.
    
    Args:
        arr (List): The input array of comparable elements
        k (int): The position of the desired element (1-based index)
                1 <= k <= len(arr)
    
    Returns:
        The k-th smallest element in the array
    
    Raises:
        ValueError: If k is out of valid range
        TypeError: If arr is not iterable or elements are not comparable
    
    Time Complexity: O(n) expected, O(n²) worst-case
    Space Complexity: O(log n) expected due to recursion depth
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    if k < 1 or k > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}, got {k}")
    
    return _quickselect_recursive(arr.copy(), 0, len(arr) - 1, k - 1)


def _quickselect_recursive(arr: List, left: int, right: int, k: int) -> Union[int, float]:
    """
    Recursive helper function for randomized quickselect.
    
    Args:
        arr (List): Array to search in (will be modified in-place)
        left (int): Left boundary of current subarray
        right (int): Right boundary of current subarray
        k (int): Target index (0-based)
    
    Returns:
        The k-th smallest element
    """
    if left == right:
        return arr[left]
    
    # Randomly select pivot and partition
    pivot_index = _randomized_partition(arr, left, right)
    
    if k == pivot_index:
        return arr[k]
    elif k < pivot_index:
        return _quickselect_recursive(arr, left, pivot_index - 1, k)
    else:
        return _quickselect_recursive(arr, pivot_index + 1, right, k)


def _randomized_partition(arr: List, left: int, right: int) -> int:
    """
    Randomly select a pivot and partition the array around it.
    
    Args:
        arr (List): Array to partition
        left (int): Left boundary
        right (int): Right boundary
    
    Returns:
        Final position of the pivot element
    """
    # Randomly choose pivot
    random_index = random.randint(left, right)
    
    # Swap random element with the rightmost element
    arr[random_index], arr[right] = arr[right], arr[random_index]
    
    # Use the standard partition algorithm
    return _partition(arr, left, right)


def _partition(arr: List, left: int, right: int) -> int:
    """
    Standard partition algorithm used in quicksort.
    
    Args:
        arr (List): Array to partition
        left (int): Left boundary
        right (int): Right boundary (contains pivot)
    
    Returns:
        Final position of pivot after partitioning
    """
    pivot = arr[right]
    i = left - 1
    
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


def randomized_select_iterative(arr: List[Union[int, float]], k: int) -> Union[int, float]:
    """
    Iterative version of randomized select to avoid recursion overhead.
    
    Args:
        arr (List): The input array of comparable elements
        k (int): The position of the desired element (1-based index)
    
    Returns:
        The k-th smallest element
    
    Time Complexity: O(n) expected, O(n²) worst-case
    Space Complexity: O(1)
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    if k < 1 or k > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}, got {k}")
    
    arr_copy = arr.copy()
    left, right = 0, len(arr_copy) - 1
    k = k - 1  # Convert to 0-based index
    
    while left <= right:
        if left == right:
            return arr_copy[left]
        
        pivot_index = _randomized_partition(arr_copy, left, right)
        
        if k == pivot_index:
            return arr_copy[k]
        elif k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1
    
    return arr_copy[k]


def find_median_random(arr: List[Union[int, float]]) -> Union[int, float]:
    """
    Find the median using randomized selection.
    
    Args:
        arr (List): The input array
    
    Returns:
        The median element (for odd length) or lower median (for even length)
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    n = len(arr)
    median_position = (n + 1) // 2
    return randomized_select(arr, median_position)


def find_kth_smallest_random(arr: List[Union[int, float]], k: int) -> Union[int, float]:
    """
    Wrapper function to find the k-th smallest element using randomization.
    
    Args:
        arr (List): The input array
        k (int): The position (1-based index)
    
    Returns:
        The k-th smallest element
    """
    return randomized_select(arr, k)


def find_kth_largest_random(arr: List[Union[int, float]], k: int) -> Union[int, float]:
    """
    Find the k-th largest element using randomized selection.
    
    Args:
        arr (List): The input array
        k (int): The position from the largest (1-based index)
    
    Returns:
        The k-th largest element
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    if k < 1 or k > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}, got {k}")
    
    # Convert k-th largest to equivalent smallest position
    equivalent_smallest = len(arr) - k + 1
    return randomized_select(arr, equivalent_smallest)


def quickselect_with_median_of_three(arr: List[Union[int, float]], k: int) -> Union[int, float]:
    """
    Randomized quickselect using median-of-three pivot selection for better performance.
    
    Args:
        arr (List): The input array
        k (int): The position (1-based index)
    
    Returns:
        The k-th smallest element
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    if k < 1 or k > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}, got {k}")
    
    return _quickselect_median_of_three(arr.copy(), 0, len(arr) - 1, k - 1)


def _quickselect_median_of_three(arr: List, left: int, right: int, k: int) -> Union[int, float]:
    """
    Quickselect with median-of-three pivot selection.
    """
    if left == right:
        return arr[left]
    
    # Use median-of-three for pivot selection
    pivot_index = _median_of_three_partition(arr, left, right)
    
    if k == pivot_index:
        return arr[k]
    elif k < pivot_index:
        return _quickselect_median_of_three(arr, left, pivot_index - 1, k)
    else:
        return _quickselect_median_of_three(arr, pivot_index + 1, right, k)


def _median_of_three_partition(arr: List, left: int, right: int) -> int:
    """
    Partition using median-of-three pivot selection.
    """
    mid = (left + right) // 2
    
    # Find median of three elements
    if arr[mid] < arr[left]:
        arr[left], arr[mid] = arr[mid], arr[left]
    if arr[right] < arr[left]:
        arr[left], arr[right] = arr[right], arr[left]
    if arr[right] < arr[mid]:
        arr[mid], arr[right] = arr[right], arr[mid]
    
    # Place median at the end for partitioning
    arr[mid], arr[right] = arr[right], arr[mid]
    
    return _partition(arr, left, right)


# Demonstration and performance analysis functions
def demonstrate_randomized_algorithms():
    """
    Demonstrate various randomized selection algorithms.
    """
    print("=== Randomized Selection Algorithms Demo ===\n")
    
    # Set seed for reproducible results in demo
    random.seed(42)
    
    # Test Case 1: Basic example
    arr1 = [12, 3, 5, 7, 4, 19, 26, 1, 9]
    print(f"Array: {arr1}")
    print(f"Sorted: {sorted(arr1)}")
    print(f"3rd smallest (recursive): {find_kth_smallest_random(arr1, 3)}")
    print(f"3rd smallest (iterative): {randomized_select_iterative(arr1, 3)}")
    print(f"3rd smallest (median-of-3): {quickselect_with_median_of_three(arr1, 3)}")
    print(f"Median: {find_median_random(arr1)}")
    print(f"2nd largest: {find_kth_largest_random(arr1, 2)}")
    print()
    
    # Test Case 2: Larger random array
    arr2 = [random.randint(1, 100) for _ in range(20)]
    k = 10
    print(f"Random array (first 10): {arr2[:10]}...")
    print(f"{k}th smallest: {find_kth_smallest_random(arr2, k)}")
    print(f"Verification: {sorted(arr2)[k-1]}")
    print()


def performance_comparison_randomized():
    """
    Compare performance of different randomized approaches.
    """
    import time
    
    print("=== Performance Comparison - Randomized Algorithms ===\n")
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        # Create test array
        arr = [random.randint(1, 1000) for _ in range(size)]
        k = size // 2  # Find median
        
        # Test recursive version
        start_time = time.time()
        result_recursive = randomized_select(arr, k)
        time_recursive = time.time() - start_time
        
        # Test iterative version
        start_time = time.time()
        result_iterative = randomized_select_iterative(arr, k)
        time_iterative = time.time() - start_time
        
        # Test median-of-three version
        start_time = time.time()
        result_median3 = quickselect_with_median_of_three(arr, k)
        time_median3 = time.time() - start_time
        
        print(f"Array size: {size}")
        print(f"  Recursive:     {time_recursive:.6f} seconds")
        print(f"  Iterative:     {time_iterative:.6f} seconds")
        print(f"  Median-of-3:   {time_median3:.6f} seconds")
        print(f"  Results match: {result_recursive == result_iterative == result_median3}")
        print()


def worst_case_analysis():
    """
    Demonstrate potential worst-case behavior and mitigation strategies.
    """
    print("=== Worst-Case Analysis ===\n")
    
    # Create worst-case scenario: already sorted array
    # This can lead to O(n²) behavior with bad luck in pivot selection
    worst_case_arr = list(range(1, 101))  # [1, 2, 3, ..., 100]
    
    print("Worst-case scenario: sorted array")
    print(f"Array: [1, 2, 3, ..., 100]")
    
    # Multiple runs to show variance in performance
    times = []
    for i in range(5):
        start_time = time.time()
        result = randomized_select(worst_case_arr, 50)  # Find median
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.6f} seconds (result: {result})")
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Time variation: {max(times) - min(times):.6f} seconds")
    print()
    
    print("Note: Performance can vary due to random pivot selection.")
    print("Median-of-three helps reduce worst-case probability.")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_randomized_algorithms()
    
    # Performance comparisons
    performance_comparison_randomized()
    
    # Worst-case analysis
    import time
    worst_case_analysis()
    
    # Interactive example
    print("=== Interactive Example ===")
    test_arr = [64, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    print(f"Test array: {test_arr}")
    print("Try different approaches:")
    print(f"5th smallest (recursive): {randomized_select(test_arr, 5)}")
    print(f"5th smallest (iterative): {randomized_select_iterative(test_arr, 5)}")
    print(f"5th smallest (median-of-3): {quickselect_with_median_of_three(test_arr, 5)}")
    print(f"Verification (sorted): {sorted(test_arr)[4]}")