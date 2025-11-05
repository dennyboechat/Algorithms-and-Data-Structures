"""
Quicksort Algorithm Implementation in Python

This module implements the Quicksort algorithm with detailed explanations
of each step: pivot selection, partitioning, and recursive sorting.

Time Complexity:
- Best/Average Case: O(n log n)
- Worst Case: O(n²)

Space Complexity: O(log n) for recursion stack
"""

def quicksort(arr, low=0, high=None):
    """
    Main quicksort function that recursively sorts the array.
    
    Args:
        arr: List of comparable elements to sort
        low: Starting index of the subarray (default: 0)
        high: Ending index of the subarray (default: len(arr)-1)
    
    Returns:
        None (sorts in-place)
    """
    if high is None:
        high = len(arr) - 1
    
    # Base case: if low >= high, subarray has 0 or 1 element
    if low < high:
        # Step 1: Partition the array and get pivot index
        pivot_index = partition(arr, low, high)
        
        # Step 2: Recursively sort left subarray (elements < pivot)
        quicksort(arr, low, pivot_index - 1)
        
        # Step 3: Recursively sort right subarray (elements > pivot)
        quicksort(arr, pivot_index + 1, high)


def partition(arr, low, high):
    """
    Partitions the array around a pivot element.
    
    Strategy: Lomuto partition scheme
    - Choose last element as pivot
    - Rearrange array so elements ≤ pivot are on left, elements > pivot on right
    - Return final position of pivot
    
    Args:
        arr: Array to partition
        low: Starting index
        high: Ending index
    
    Returns:
        Final index position of the pivot element
    """
    # Step 1: Select pivot (last element in this implementation)
    pivot = arr[high]
    print(f"Partitioning range [{low}:{high}], pivot = {pivot}")
    
    # Step 2: Initialize pointer for smaller element
    # i tracks the boundary between elements ≤ pivot and elements > pivot
    i = low - 1
    
    # Step 3: Traverse through array from low to high-1
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1  # Increment index of smaller element
            arr[i], arr[j] = arr[j], arr[i]  # Swap elements
            print(f"  Swapped {arr[j]} and {arr[i]} at positions {j} and {i}")
    
    # Step 4: Place pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    print(f"  Placed pivot {pivot} at position {i + 1}")
    print(f"  Array after partition: {arr[low:high+1]}")
    
    return i + 1  # Return position of pivot


def quicksort_with_different_pivots(arr, pivot_strategy="last"):
    """
    Quicksort with different pivot selection strategies.
    
    Args:
        arr: Array to sort
        pivot_strategy: "first", "last", "middle", or "random"
    
    Returns:
        Sorted array (creates new copy)
    """
    import random
    
    def partition_flexible(arr, low, high, pivot_strategy):
        # Select pivot based on strategy
        if pivot_strategy == "first":
            pivot_idx = low
        elif pivot_strategy == "middle":
            pivot_idx = (low + high) // 2
        elif pivot_strategy == "random":
            pivot_idx = random.randint(low, high)
        else:  # "last" (default)
            pivot_idx = high
        
        # Move chosen pivot to end for standard partition
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        return partition(arr, low, high)
    
    def quicksort_flexible(arr, low, high, pivot_strategy):
        if low < high:
            pivot_index = partition_flexible(arr, low, high, pivot_strategy)
            quicksort_flexible(arr, low, pivot_index - 1, pivot_strategy)
            quicksort_flexible(arr, pivot_index + 1, high, pivot_strategy)
    
    # Create copy to avoid modifying original
    result = arr.copy()
    if len(result) > 1:
        quicksort_flexible(result, 0, len(result) - 1, pivot_strategy)
    return result


def demonstrate_quicksort():
    """
    Demonstrates the quicksort algorithm with various examples.
    """
    print("=" * 60)
    print("QUICKSORT ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    # Test cases
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1],  # Single element
        [],   # Empty array
        [3, 3, 3, 3],  # Duplicate elements
        [5, 4, 3, 2, 1],  # Reverse sorted
        [1, 2, 3, 4, 5]   # Already sorted
    ]
    
    for i, original_arr in enumerate(test_arrays, 1):
        print(f"\nTest Case {i}:")
        print(f"Original array: {original_arr}")
        
        if len(original_arr) <= 1:
            print("Array has ≤ 1 element, no sorting needed")
            continue
        
        # Make a copy for sorting
        arr_to_sort = original_arr.copy()
        
        print("\nStep-by-step sorting process:")
        quicksort(arr_to_sort)
        
        print(f"Final sorted array: {arr_to_sort}")
        print("-" * 40)


def compare_pivot_strategies():
    """
    Compares different pivot selection strategies.
    """
    print("\n" + "=" * 60)
    print("COMPARING PIVOT SELECTION STRATEGIES")
    print("=" * 60)
    
    test_array = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    strategies = ["first", "last", "middle", "random"]
    
    print(f"Original array: {test_array}")
    
    for strategy in strategies:
        print(f"\nUsing '{strategy}' pivot strategy:")
        sorted_arr = quicksort_with_different_pivots(test_array, strategy)
        print(f"Result: {sorted_arr}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_quicksort()
    compare_pivot_strategies()
    
    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
    Quicksort Algorithm Steps:
    
    1. PIVOT SELECTION:
       - Choose an element as pivot (first, last, middle, or random)
       - Common choice: last element (Lomuto partition)
    
    2. PARTITIONING:
       - Rearrange array so elements ≤ pivot are on left
       - Elements > pivot are on right
       - Pivot ends up in its final sorted position
    
    3. RECURSIVE SORTING:
       - Recursively apply quicksort to left subarray
       - Recursively apply quicksort to right subarray
       - Base case: subarrays with ≤ 1 element
    
    Key Properties:
    - In-place sorting (O(1) extra space, excluding recursion)
    - Divide-and-conquer approach
    - Average case: O(n log n), Worst case: O(n²)
    - Worst case occurs when pivot is always smallest/largest
    """)