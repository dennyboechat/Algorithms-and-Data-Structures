"""
Heapsort Algorithm Implementation

This module implements the Heapsort algorithm, which is a comparison-based sorting
algorithm that uses a max-heap data structure to sort elements in ascending order.

Time Complexity: O(n log n) in all cases (best, average, worst)
Space Complexity: O(1) - in-place sorting algorithm
"""

from typing import List


def heapify(arr: List[int], n: int, i: int) -> None:
    """
    Maintain the max-heap property for a subtree rooted at index i.
    
    This function assumes that the binary trees rooted at left and right 
    children of i are max-heaps, but arr[i] might be smaller than its children,
    thus violating the max-heap property.
    
    Args:
        arr: The array representing the heap
        n: Size of the heap
        i: Index of the root of the subtree to heapify
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1  # Left child index
    right = 2 * i + 2  # Right child index
    
    # Check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # Check if right child exists and is greater than current largest
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # If largest is not root, swap and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        # Recursively heapify the affected subtree
        heapify(arr, n, largest)


def build_max_heap(arr: List[int]) -> None:
    """
    Build a max-heap from an unsorted array.
    
    This function converts an array into a max-heap by calling heapify
    on all non-leaf nodes in bottom-up manner.
    
    Args:
        arr: The array to be converted into a max-heap
    """
    n = len(arr)
    
    # Start from the last non-leaf node and heapify each node
    # Last non-leaf node is at index (n//2 - 1)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)


def heapsort(arr: List[int]) -> List[int]:
    """
    Sort an array using the Heapsort algorithm.
    
    The algorithm works as follows:
    1. Build a max-heap from the input array
    2. Repeatedly extract the maximum element (root) and place it at the end
    3. Restore the heap property for the remaining elements
    
    Args:
        arr: The array to be sorted
        
    Returns:
        A new sorted array (original array is not modified)
    """
    # Create a copy to avoid modifying the original array
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # Build max heap
    build_max_heap(arr_copy)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        # Move current root (maximum element) to end
        arr_copy[0], arr_copy[i] = arr_copy[i], arr_copy[0]
        
        # Call heapify on the reduced heap
        heapify(arr_copy, i, 0)
    
    return arr_copy


def heapsort_inplace(arr: List[int]) -> None:
    """
    Sort an array in-place using the Heapsort algorithm.
    
    This version modifies the original array instead of creating a copy.
    
    Args:
        arr: The array to be sorted (modified in-place)
    """
    n = len(arr)
    
    # Build max heap
    build_max_heap(arr)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        # Move current root (maximum element) to end
        arr[0], arr[i] = arr[i], arr[0]
        
        # Call heapify on the reduced heap
        heapify(arr, i, 0)


def is_max_heap(arr: List[int]) -> bool:
    """
    Check if an array satisfies the max-heap property.
    
    Args:
        arr: The array to check
        
    Returns:
        True if the array is a valid max-heap, False otherwise
    """
    n = len(arr)
    
    # Check all non-leaf nodes
    for i in range(n // 2):
        left = 2 * i + 1
        right = 2 * i + 2
        
        # Check left child
        if left < n and arr[i] < arr[left]:
            return False
        
        # Check right child
        if right < n and arr[i] < arr[right]:
            return False
    
    return True


# Example usage and demonstration
if __name__ == "__main__":
    # Test with various arrays
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1],
        [],
        [3, 3, 3, 3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse sorted
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Already sorted
    ]
    
    for i, arr in enumerate(test_arrays, 1):
        print(f"Test {i}:")
        print(f"Original array: {arr}")
        
        if arr:  # Only sort non-empty arrays
            # Test heap building
            heap_arr = arr.copy()
            build_max_heap(heap_arr)
            print(f"Max heap:       {heap_arr}")
            print(f"Is valid heap:  {is_max_heap(heap_arr)}")
            
            # Test sorting
            sorted_arr = heapsort(arr)
            print(f"Sorted array:   {sorted_arr}")
            print(f"Is sorted:      {sorted_arr == sorted(arr)}")
        else:
            print("Empty array - no sorting needed")
        
        print("-" * 50)