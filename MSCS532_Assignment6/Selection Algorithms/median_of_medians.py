"""
Deterministic Selection Algorithm - Median of Medians
MSCS532 Assignment 6

This module implements the deterministic selection algorithm that finds the k-th smallest
element in an unsorted array in worst-case linear time O(n).

The algorithm uses the "Median of Medians" approach to guarantee good pivot selection,
ensuring worst-case linear time complexity.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

def median_of_medians_select(arr, k):
    """
    Find the k-th smallest element using the Median of Medians algorithm.
    
    This is a deterministic selection algorithm that guarantees O(n) worst-case time complexity.
    
    Args:
        arr (list): The input array of comparable elements
        k (int): The position of the desired element (1-based index)
                1 <= k <= len(arr)
    
    Returns:
        The k-th smallest element in the array
    
    Raises:
        ValueError: If k is out of valid range
        TypeError: If arr is not iterable or elements are not comparable
    
    Time Complexity: O(n) worst-case
    Space Complexity: O(log n) due to recursion depth
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    if k < 1 or k > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}, got {k}")
    
    return _select_recursive(arr.copy(), k)


def _select_recursive(arr, k):
    """
    Recursive helper function for the selection algorithm.
    
    Args:
        arr (list): Array to search in (will be modified)
        k (int): Position of desired element (1-based)
    
    Returns:
        The k-th smallest element
    """
    n = len(arr)
    
    # Base case: small arrays
    if n <= 5:
        arr.sort()
        return arr[k-1]
    
    # Step 1: Divide into groups of 5 and find median of each group
    medians = []
    for i in range(0, n, 5):
        group = arr[i:i+5]
        group.sort()
        median_idx = len(group) // 2
        medians.append(group[median_idx])
    
    # Step 2: Recursively find the median of medians
    pivot = _select_recursive(medians, (len(medians) + 1) // 2)
    
    # Step 3: Partition around the pivot
    less_than_pivot = []
    equal_to_pivot = []
    greater_than_pivot = []
    
    for element in arr:
        if element < pivot:
            less_than_pivot.append(element)
        elif element == pivot:
            equal_to_pivot.append(element)
        else:
            greater_than_pivot.append(element)
    
    # Step 4: Recursively search the appropriate partition
    less_count = len(less_than_pivot)
    equal_count = len(equal_to_pivot)
    
    if k <= less_count:
        # k-th smallest is in the 'less than pivot' partition
        return _select_recursive(less_than_pivot, k)
    elif k <= less_count + equal_count:
        # k-th smallest is the pivot itself
        return pivot
    else:
        # k-th smallest is in the 'greater than pivot' partition
        return _select_recursive(greater_than_pivot, k - less_count - equal_count)


def find_median(arr):
    """
    Find the median of an array using the deterministic selection algorithm.
    
    Args:
        arr (list): The input array
    
    Returns:
        The median element (for odd length) or the lower median (for even length)
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    n = len(arr)
    median_position = (n + 1) // 2
    return median_of_medians_select(arr, median_position)


def find_kth_smallest(arr, k):
    """
    Wrapper function to find the k-th smallest element.
    
    Args:
        arr (list): The input array
        k (int): The position (1-based index)
    
    Returns:
        The k-th smallest element
    """
    return median_of_medians_select(arr, k)


def find_kth_largest(arr, k):
    """
    Find the k-th largest element using the selection algorithm.
    
    Args:
        arr (list): The input array
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
    return median_of_medians_select(arr, equivalent_smallest)


# Demonstration and testing functions
def demonstrate_algorithm():
    """
    Demonstrate the algorithm with various test cases.
    """
    print("=== Median of Medians Selection Algorithm Demo ===\n")
    
    # Test Case 1: Basic example
    arr1 = [12, 3, 5, 7, 4, 19, 26, 1, 9]
    print(f"Array: {arr1}")
    print(f"3rd smallest: {find_kth_smallest(arr1, 3)}")
    print(f"5th smallest (median): {find_median(arr1)}")
    print(f"2nd largest: {find_kth_largest(arr1, 2)}")
    print()
    
    # Test Case 2: Array with duplicates
    arr2 = [5, 2, 8, 2, 9, 1, 5, 5]
    print(f"Array with duplicates: {arr2}")
    print(f"4th smallest: {find_kth_smallest(arr2, 4)}")
    print(f"Median: {find_median(arr2)}")
    print()
    
    # Test Case 3: Already sorted array
    arr3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Sorted array: {arr3}")
    print(f"6th smallest: {find_kth_smallest(arr3, 6)}")
    print(f"Median: {find_median(arr3)}")
    print()
    
    # Test Case 4: Reverse sorted array
    arr4 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    print(f"Reverse sorted array: {arr4}")
    print(f"3rd smallest: {find_kth_smallest(arr4, 3)}")
    print(f"Median: {find_median(arr4)}")
    print()


def performance_comparison():
    """
    Compare the performance of different selection approaches.
    """
    import time
    import random
    
    print("=== Performance Comparison ===\n")
    
    # Generate test data
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        arr = [random.randint(1, 1000) for _ in range(size)]
        k = size // 2  # Find median
        
        # Test Median of Medians
        start_time = time.time()
        result_mom = median_of_medians_select(arr, k)
        mom_time = time.time() - start_time
        
        # Test built-in sorted approach
        start_time = time.time()
        result_sorted = sorted(arr)[k-1]
        sorted_time = time.time() - start_time
        
        print(f"Array size: {size}")
        print(f"Median of Medians: {mom_time:.6f} seconds")
        print(f"Sorted approach: {sorted_time:.6f} seconds")
        print(f"Results match: {result_mom == result_sorted}")
        print()


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_algorithm()
    
    # Run performance comparison
    performance_comparison()
    
    # Interactive testing
    print("=== Interactive Testing ===")
    print("You can test the algorithm with custom inputs:")
    print("Example: find_kth_smallest([3, 1, 4, 1, 5, 9, 2, 6], 4)")
    print("Result:", find_kth_smallest([3, 1, 4, 1, 5, 9, 2, 6], 4))