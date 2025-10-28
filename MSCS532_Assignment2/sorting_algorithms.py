"""
Sorting Algorithms Implementation and Performance Analysis
Implements Quick Sort and Merge Sort with detailed performance tracking.
"""

import time
import random
import copy
import sys
from typing import List, Tuple

# Increase recursion limit for large datasets
sys.setrecursionlimit(15000)


class QuickSort:
    """Quick Sort implementation with performance tracking."""
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.recursive_calls = 0
    
    def reset_counters(self):
        """Reset performance counters."""
        self.comparisons = 0
        self.swaps = 0
        self.recursive_calls = 0
    
    def partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Partition function using random pivot to avoid worst-case performance.
        Returns the partition index.
        """
        # Choose random pivot and swap with last element
        random_idx = random.randint(low, high)
        arr[random_idx], arr[high] = arr[high], arr[random_idx]
        
        pivot = arr[high]
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            self.comparisons += 1
            if arr[j] <= pivot:
                i += 1
                if i != j:  # Only count actual swaps
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swaps += 1
        
        if i + 1 != high:  # Only swap if needed
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swaps += 1
        return i + 1
    
    def _quick_sort_recursive(self, arr: List[int], low: int, high: int):
        """Recursive quick sort implementation."""
        if low < high:
            self.recursive_calls += 1
            # Partition the array
            pi = self.partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            self._quick_sort_recursive(arr, low, pi - 1)
            self._quick_sort_recursive(arr, pi + 1, high)
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort the array using Quick Sort algorithm.
        Returns a new sorted array.
        """
        self.reset_counters()
        arr_copy = copy.deepcopy(arr)
        if len(arr_copy) <= 1:
            return arr_copy
        
        self._quick_sort_recursive(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
    
    def get_performance_stats(self) -> dict:
        """Return performance statistics."""
        return {
            'comparisons': self.comparisons,
            'swaps': self.swaps,
            'recursive_calls': self.recursive_calls
        }


class MergeSort:
    """Merge Sort implementation with performance tracking."""
    
    def __init__(self):
        self.comparisons = 0
        self.merges = 0
        self.recursive_calls = 0
    
    def reset_counters(self):
        """Reset performance counters."""
        self.comparisons = 0
        self.merges = 0
        self.recursive_calls = 0
    
    def merge(self, arr: List[int], left: int, mid: int, right: int):
        """
        Merge two sorted subarrays arr[left...mid] and arr[mid+1...right].
        """
        # Create temp arrays for the two subarrays
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        
        # Initial indices for left_arr, right_arr, and merged array
        i = j = 0
        k = left
        
        # Merge the temp arrays back into arr[left...right]
        while i < len(left_arr) and j < len(right_arr):
            self.comparisons += 1
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
            self.merges += 1
        
        # Copy remaining elements of left_arr, if any
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
            self.merges += 1
        
        # Copy remaining elements of right_arr, if any
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
            self.merges += 1
    
    def _merge_sort_recursive(self, arr: List[int], left: int, right: int):
        """Recursive merge sort implementation."""
        if left < right:
            self.recursive_calls += 1
            mid = (left + right) // 2
            
            # Sort first and second halves
            self._merge_sort_recursive(arr, left, mid)
            self._merge_sort_recursive(arr, mid + 1, right)
            
            # Merge the sorted halves
            self.merge(arr, left, mid, right)
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort the array using Merge Sort algorithm.
        Returns a new sorted array.
        """
        self.reset_counters()
        arr_copy = copy.deepcopy(arr)
        if len(arr_copy) <= 1:
            return arr_copy
        
        self._merge_sort_recursive(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
    
    def get_performance_stats(self) -> dict:
        """Return performance statistics."""
        return {
            'comparisons': self.comparisons,
            'merges': self.merges,
            'recursive_calls': self.recursive_calls
        }


def verify_sorted(arr: List[int]) -> bool:
    """Verify if an array is sorted in ascending order."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


if __name__ == "__main__":
    # Simple test
    test_arr = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50]
    print(f"Original array: {test_arr}")
    
    # Test Quick Sort
    quick_sort = QuickSort()
    sorted_arr_quick = quick_sort.sort(test_arr)
    print(f"Quick Sort result: {sorted_arr_quick}")
    print(f"Quick Sort verified: {verify_sorted(sorted_arr_quick)}")
    print(f"Quick Sort stats: {quick_sort.get_performance_stats()}")
    
    print()
    
    # Test Merge Sort
    merge_sort = MergeSort()
    sorted_arr_merge = merge_sort.sort(test_arr)
    print(f"Merge Sort result: {sorted_arr_merge}")
    print(f"Merge Sort verified: {verify_sorted(sorted_arr_merge)}")
    print(f"Merge Sort stats: {merge_sort.get_performance_stats()}")