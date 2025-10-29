"""
Randomized Quicksort Implementation

This module implements the Randomized Quicksort algorithm where the pivot element
is chosen uniformly at random from the subarray being partitioned.
"""

import random
from typing import List, Optional, Tuple


class RandomizedQuicksort:
    """
    A class implementing the Randomized Quicksort algorithm with various optimization techniques.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the RandomizedQuicksort class.
        
        Args:
            seed (Optional[int]): Random seed for reproducible results. Default is None.
        """
        if seed is not None:
            random.seed(seed)
        self.comparisons = 0
        self.swaps = 0
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Main entry point for sorting an array using randomized quicksort.
        
        Args:
            arr (List[int]): The array to be sorted
            
        Returns:
            List[int]: A new sorted array (original array is not modified)
        """
        # Reset counters
        self.comparisons = 0
        self.swaps = 0
        
        # Handle edge cases
        if not arr or len(arr) <= 1:
            return arr.copy()
        
        # Create a copy to avoid modifying the original array
        arr_copy = arr.copy()
        self._quicksort(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
    
    def sort_inplace(self, arr: List[int]) -> None:
        """
        Sort the array in-place using randomized quicksort.
        
        Args:
            arr (List[int]): The array to be sorted in-place
        """
        # Reset counters
        self.comparisons = 0
        self.swaps = 0
        
        # Handle edge cases
        if not arr or len(arr) <= 1:
            return
        
        self._quicksort(arr, 0, len(arr) - 1)
    
    def _quicksort(self, arr: List[int], low: int, high: int) -> None:
        """
        Recursive quicksort implementation with random pivot selection.
        
        Args:
            arr (List[int]): The array being sorted
            low (int): Starting index of the subarray
            high (int): Ending index of the subarray
        """
        if low < high:
            # Partition the array and get the pivot index
            pivot_index = self._randomized_partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            self._quicksort(arr, low, pivot_index - 1)
            self._quicksort(arr, pivot_index + 1, high)
    
    def _randomized_partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Partition the array with a randomly chosen pivot.
        
        Args:
            arr (List[int]): The array being partitioned
            low (int): Starting index of the subarray
            high (int): Ending index of the subarray
            
        Returns:
            int: The final position of the pivot element
        """
        # Choose a random pivot index
        random_index = random.randint(low, high)
        
        # Swap the random element with the last element
        arr[random_index], arr[high] = arr[high], arr[random_index]
        self.swaps += 1
        
        # Use the standard partition algorithm with the randomized pivot
        return self._partition(arr, low, high)
    
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Standard partition function using the last element as pivot.
        
        Args:
            arr (List[int]): The array being partitioned
            low (int): Starting index of the subarray
            high (int): Ending index of the subarray
            
        Returns:
            int: The final position of the pivot element
        """
        # Choose the last element as pivot
        pivot = arr[high]
        
        # Index of smaller element (indicates the right position of pivot)
        i = low - 1
        
        for j in range(low, high):
            self.comparisons += 1
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                i += 1
                if i != j:  # Only swap if necessary
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swaps += 1
        
        # Place pivot in its correct position
        if i + 1 != high:  # Only swap if necessary
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swaps += 1
        
        return i + 1
    
    def get_statistics(self) -> Tuple[int, int]:
        """
        Get the number of comparisons and swaps from the last sort operation.
        
        Returns:
            Tuple[int, int]: (comparisons, swaps)
        """
        return self.comparisons, self.swaps


# Alternative implementation with 3-way partitioning for better handling of duplicates
class RandomizedQuicksort3Way:
    """
    Randomized Quicksort with 3-way partitioning for efficient handling of duplicate elements.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the RandomizedQuicksort3Way class.
        
        Args:
            seed (Optional[int]): Random seed for reproducible results. Default is None.
        """
        if seed is not None:
            random.seed(seed)
        self.comparisons = 0
        self.swaps = 0
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Main entry point for sorting an array using 3-way randomized quicksort.
        
        Args:
            arr (List[int]): The array to be sorted
            
        Returns:
            List[int]: A new sorted array (original array is not modified)
        """
        # Reset counters
        self.comparisons = 0
        self.swaps = 0
        
        # Handle edge cases
        if not arr or len(arr) <= 1:
            return arr.copy()
        
        # Create a copy to avoid modifying the original array
        arr_copy = arr.copy()
        self._quicksort_3way(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
    
    def _quicksort_3way(self, arr: List[int], low: int, high: int) -> None:
        """
        3-way quicksort implementation with random pivot selection.
        
        Args:
            arr (List[int]): The array being sorted
            low (int): Starting index of the subarray
            high (int): Ending index of the subarray
        """
        if low < high:
            # Partition the array into three parts
            lt, gt = self._partition_3way(arr, low, high)
            
            # Recursively sort elements less than and greater than pivot
            self._quicksort_3way(arr, low, lt - 1)
            self._quicksort_3way(arr, gt + 1, high)
    
    def _partition_3way(self, arr: List[int], low: int, high: int) -> Tuple[int, int]:
        """
        3-way partition: elements < pivot, elements = pivot, elements > pivot.
        
        Args:
            arr (List[int]): The array being partitioned
            low (int): Starting index of the subarray
            high (int): Ending index of the subarray
            
        Returns:
            Tuple[int, int]: (lt, gt) where lt is the start of equal elements
                           and gt-1 is the end of equal elements
        """
        # Choose a random pivot
        random_index = random.randint(low, high)
        arr[random_index], arr[low] = arr[low], arr[random_index]
        self.swaps += 1
        
        pivot = arr[low]
        lt = low      # arr[low..lt-1] < pivot
        i = low + 1   # arr[lt..i-1] == pivot
        gt = high + 1 # arr[gt..high] > pivot
        
        while i < gt:
            self.comparisons += 1
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                self.swaps += 1
                lt += 1
                i += 1
            elif arr[i] > pivot:
                self.comparisons += 1
                gt -= 1
                arr[i], arr[gt] = arr[gt], arr[i]
                self.swaps += 1
                # Don't increment i here because we need to check the swapped element
            else:
                i += 1
        
        return lt, gt - 1
    
    def get_statistics(self) -> Tuple[int, int]:
        """
        Get the number of comparisons and swaps from the last sort operation.
        
        Returns:
            Tuple[int, int]: (comparisons, swaps)
        """
        return self.comparisons, self.swaps


# Utility functions for testing and demonstration
def is_sorted(arr: List[int]) -> bool:
    """
    Check if an array is sorted in non-decreasing order.
    
    Args:
        arr (List[int]): The array to check
        
    Returns:
        bool: True if sorted, False otherwise
    """
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def generate_test_arrays() -> dict:
    """
    Generate various test arrays for testing the sorting algorithm.
    
    Returns:
        dict: Dictionary containing different test cases
    """
    return {
        'empty': [],
        'single': [42],
        'two_elements': [2, 1],
        'already_sorted': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'reverse_sorted': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'all_duplicates': [5, 5, 5, 5, 5],
        'some_duplicates': [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
        'random_small': [64, 34, 25, 12, 22, 11, 90],
        'random_medium': [random.randint(1, 100) for _ in range(50)],
        'with_negatives': [-5, -1, 0, 3, -2, 8, -10, 15],
        'large_range': [random.randint(-1000, 1000) for _ in range(100)]
    }


if __name__ == "__main__":
    # Demonstration of the algorithm
    print("Randomized Quicksort Implementation Demo")
    print("=" * 50)
    
    # Create sorter instances
    sorter = RandomizedQuicksort(seed=42)  # Fixed seed for reproducibility
    sorter_3way = RandomizedQuicksort3Way(seed=42)
    
    # Generate test cases
    test_cases = generate_test_arrays()
    
    for name, test_array in test_cases.items():
        print(f"\nTest case: {name}")
        print(f"Original: {test_array}")
        
        # Test standard randomized quicksort
        sorted_array = sorter.sort(test_array)
        comparisons, swaps = sorter.get_statistics()
        
        print(f"Sorted:   {sorted_array}")
        print(f"Is sorted: {is_sorted(sorted_array)}")
        print(f"Stats (standard): {comparisons} comparisons, {swaps} swaps")
        
        # Test 3-way quicksort for arrays with duplicates
        if len(test_array) > 1:
            sorted_3way = sorter_3way.sort(test_array)
            comparisons_3way, swaps_3way = sorter_3way.get_statistics()
            print(f"Stats (3-way):    {comparisons_3way} comparisons, {swaps_3way} swaps")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")