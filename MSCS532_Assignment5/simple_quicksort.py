"""
Simple Quicksort Implementation - Educational Version

This version clearly demonstrates the three main steps of Quicksort:
1. Pivot Selection
2. Partitioning
3. Recursive Sorting
"""

def simple_quicksort(arr):
    """
    Simple quicksort implementation that's easy to understand.
    
    Args:
        arr: List of elements to sort
    
    Returns:
        New sorted list (not in-place)
    """
    # Base case: arrays with 0 or 1 element are already sorted
    if len(arr) <= 1:
        return arr
    
    # STEP 1: PIVOT SELECTION
    # Choose the middle element as pivot (good for partially sorted arrays)
    pivot_index = len(arr) // 2
    pivot = arr[pivot_index]
    print(f"Selected pivot: {pivot} (at index {pivot_index})")
    
    # STEP 2: PARTITIONING
    # Create three lists: elements less than, equal to, and greater than pivot
    left = []    # Elements < pivot
    middle = []  # Elements = pivot (handles duplicates)
    right = []   # Elements > pivot
    
    for element in arr:
        if element < pivot:
            left.append(element)
        elif element == pivot:
            middle.append(element)
        else:
            right.append(element)
    
    print(f"Partitioned into: left={left}, middle={middle}, right={right}")
    
    # STEP 3: RECURSIVE SORTING
    # Recursively sort left and right subarrays, then combine
    sorted_left = simple_quicksort(left)
    sorted_right = simple_quicksort(right)
    
    # Combine sorted subarrays with pivot(s) in the middle
    result = sorted_left + middle + sorted_right
    print(f"Combined result: {result}")
    
    return result


def quicksort_inplace(arr, low=0, high=None):
    """
    In-place quicksort implementation for memory efficiency.
    
    Args:
        arr: List to sort (modified in-place)
        low: Starting index
        high: Ending index
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        print(f"\nSorting subarray from index {low} to {high}: {arr[low:high+1]}")
        
        # STEP 1 & 2: Partition and get pivot position
        pivot_pos = partition_inplace(arr, low, high)
        
        # STEP 3: Recursively sort subarrays
        print(f"Pivot {arr[pivot_pos]} is now at correct position {pivot_pos}")
        
        # Sort left side (elements smaller than pivot)
        quicksort_inplace(arr, low, pivot_pos - 1)
        
        # Sort right side (elements larger than pivot)
        quicksort_inplace(arr, pivot_pos + 1, high)


def partition_inplace(arr, low, high):
    """
    Partitions array in-place using last element as pivot.
    
    Args:
        arr: Array to partition
        low: Start index
        high: End index
    
    Returns:
        Final position of pivot
    """
    # STEP 1: PIVOT SELECTION - choose last element
    pivot = arr[high]
    print(f"  Partitioning with pivot: {pivot}")
    
    # STEP 2: PARTITIONING
    # i keeps track of position where next small element should go
    i = low - 1
    
    # Compare each element with pivot
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            # Swap current element with element at position i
            arr[i], arr[j] = arr[j], arr[i]
            print(f"    Swapped {arr[j]} and {arr[i]}: {arr[low:high+1]}")
    
    # Place pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    print(f"    Placed pivot {pivot} at position {i + 1}: {arr[low:high+1]}")
    
    return i + 1


def demonstrate_steps():
    """
    Demonstrates each step of the quicksort algorithm clearly.
    """
    print("=" * 50)
    print("QUICKSORT STEP-BY-STEP DEMONSTRATION")
    print("=" * 50)
    
    # Example 1: Simple quicksort (not in-place)
    print("\n1. SIMPLE QUICKSORT (creates new arrays):")
    print("-" * 40)
    test_arr1 = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original array: {test_arr1}")
    sorted_arr1 = simple_quicksort(test_arr1)
    print(f"Final sorted array: {sorted_arr1}\n")
    
    # Example 2: In-place quicksort
    print("2. IN-PLACE QUICKSORT (memory efficient):")
    print("-" * 40)
    test_arr2 = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original array: {test_arr2}")
    quicksort_inplace(test_arr2)
    print(f"Final sorted array: {test_arr2}")


def algorithm_explanation():
    """
    Provides detailed explanation of the quicksort algorithm.
    """
    print("\n" + "=" * 50)
    print("QUICKSORT ALGORITHM EXPLANATION")
    print("=" * 50)
    
    explanation = """
STEP 1: PIVOT SELECTION
• Choose an element from the array as the 'pivot'
• Common strategies:
  - First element: Simple but poor for sorted arrays
  - Last element: Most common in implementations  
  - Middle element: Good for partially sorted arrays
  - Random element: Helps avoid worst-case scenarios
  - Median-of-three: Choose median of first, middle, last

STEP 2: PARTITIONING
• Rearrange the array so that:
  - All elements smaller than pivot are on the left
  - All elements greater than pivot are on the right
  - Elements equal to pivot can be on either side
• After partitioning, the pivot is in its final sorted position
• This is the key step that makes quicksort work

STEP 3: RECURSIVE SORTING
• Apply quicksort recursively to the left subarray
• Apply quicksort recursively to the right subarray
• Base case: arrays with 0 or 1 elements are already sorted
• The recursion naturally handles the divide-and-conquer approach

COMPLEXITY ANALYSIS:
• Best Case: O(n log n) - when pivot divides array evenly
• Average Case: O(n log n) - random pivot selection
• Worst Case: O(n²) - when pivot is always min or max
• Space: O(log n) for recursion stack in average case

ADVANTAGES:
• In-place sorting (minimal extra memory)
• Generally faster than other O(n log n) algorithms
• Cache-friendly due to locality of reference

DISADVANTAGES:
• Worst-case O(n²) performance
• Not stable (doesn't preserve relative order of equal elements)
• Performance depends heavily on pivot selection
"""
    print(explanation)


if __name__ == "__main__":
    demonstrate_steps()
    algorithm_explanation()