"""
Test Suite for Heapsort Algorithm

This module contains comprehensive test cases to validate the correctness
and robustness of the heapsort implementation.
"""

import unittest
import random
import time
from typing import List
from heapsort import heapsort, heapsort_inplace, build_max_heap, heapify, is_max_heap


class TestHeapsort(unittest.TestCase):
    """Test cases for the Heapsort algorithm implementation."""

    def test_empty_array(self):
        """Test heapsort with an empty array."""
        arr = []
        result = heapsort(arr)
        self.assertEqual(result, [])

    def test_single_element(self):
        """Test heapsort with a single element."""
        arr = [42]
        result = heapsort(arr)
        self.assertEqual(result, [42])

    def test_two_elements_sorted(self):
        """Test heapsort with two elements already sorted."""
        arr = [1, 2]
        result = heapsort(arr)
        self.assertEqual(result, [1, 2])

    def test_two_elements_reverse(self):
        """Test heapsort with two elements in reverse order."""
        arr = [2, 1]
        result = heapsort(arr)
        self.assertEqual(result, [1, 2])

    def test_basic_unsorted_array(self):
        """Test heapsort with a basic unsorted array."""
        arr = [64, 34, 25, 12, 22, 11, 90]
        expected = [11, 12, 22, 25, 34, 64, 90]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_already_sorted_array(self):
        """Test heapsort with an already sorted array."""
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_reverse_sorted_array(self):
        """Test heapsort with a reverse sorted array."""
        arr = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_duplicate_elements(self):
        """Test heapsort with duplicate elements."""
        arr = [5, 2, 8, 2, 9, 1, 5, 5]
        expected = [1, 2, 2, 5, 5, 5, 8, 9]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_all_same_elements(self):
        """Test heapsort with all identical elements."""
        arr = [7, 7, 7, 7, 7]
        expected = [7, 7, 7, 7, 7]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_negative_numbers(self):
        """Test heapsort with negative numbers."""
        arr = [-5, -1, -9, -3, -7]
        expected = [-9, -7, -5, -3, -1]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_mixed_positive_negative(self):
        """Test heapsort with mixed positive and negative numbers."""
        arr = [3, -1, 4, -5, 2, -3, 0]
        expected = [-5, -3, -1, 0, 2, 3, 4]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_large_numbers(self):
        """Test heapsort with large numbers."""
        arr = [1000000, 999999, 1000001, 500000]
        expected = [500000, 999999, 1000000, 1000001]
        result = heapsort(arr)
        self.assertEqual(result, expected)

    def test_random_array(self):
        """Test heapsort with a randomly generated array."""
        random.seed(42)  # For reproducible results
        arr = [random.randint(1, 100) for _ in range(20)]
        result = heapsort(arr)
        expected = sorted(arr)
        self.assertEqual(result, expected)

    def test_large_random_array(self):
        """Test heapsort with a large randomly generated array."""
        random.seed(123)
        arr = [random.randint(-1000, 1000) for _ in range(1000)]
        result = heapsort(arr)
        expected = sorted(arr)
        self.assertEqual(result, expected)

    def test_original_array_unchanged(self):
        """Test that the original array is not modified by heapsort."""
        original = [5, 2, 8, 1, 9]
        arr_copy = original.copy()
        heapsort(arr_copy)
        self.assertEqual(original, [5, 2, 8, 1, 9])

    def test_heapsort_inplace(self):
        """Test the in-place version of heapsort."""
        arr = [64, 34, 25, 12, 22, 11, 90]
        expected = [11, 12, 22, 25, 34, 64, 90]
        heapsort_inplace(arr)
        self.assertEqual(arr, expected)

    def test_build_max_heap(self):
        """Test the build_max_heap function."""
        arr = [4, 10, 3, 5, 1]
        build_max_heap(arr)
        self.assertTrue(is_max_heap(arr))

    def test_is_max_heap_valid(self):
        """Test is_max_heap with a valid max heap."""
        arr = [10, 8, 9, 4, 7, 5, 3, 2, 1, 6]
        self.assertTrue(is_max_heap(arr))

    def test_is_max_heap_invalid(self):
        """Test is_max_heap with an invalid heap."""
        arr = [1, 8, 9, 4, 7, 5, 3, 2, 10, 6]  # 1 is at root but 10 is in the tree
        self.assertFalse(is_max_heap(arr))

    def test_heapify_maintains_property(self):
        """Test that heapify maintains the max-heap property."""
        # Create an array where only the root violates heap property
        # but its subtrees are valid max-heaps
        arr = [1, 9, 8, 4, 7, 5, 3, 2, 1, 6]  # Root is 1, but children are 9 and 8
        heapify(arr, len(arr), 0)
        # After heapifying from root, it should be a valid heap
        self.assertTrue(is_max_heap(arr))


class TestHeapsortPerformance(unittest.TestCase):
    """Performance tests for the Heapsort algorithm."""

    def test_performance_comparison(self):
        """Compare heapsort performance with Python's built-in sort."""
        # Generate a large random array
        random.seed(42)
        size = 10000
        arr = [random.randint(1, 10000) for _ in range(size)]
        
        # Test heapsort
        start_time = time.time()
        heapsort_result = heapsort(arr)
        heapsort_time = time.time() - start_time
        
        # Test built-in sort
        start_time = time.time()
        builtin_result = sorted(arr)
        builtin_time = time.time() - start_time
        
        # Verify correctness
        self.assertEqual(heapsort_result, builtin_result)
        
        # Print performance comparison
        print(f"\nPerformance comparison for {size} elements:")
        print(f"Heapsort time: {heapsort_time:.6f} seconds")
        print(f"Built-in sort time: {builtin_time:.6f} seconds")
        print(f"Ratio (heapsort/builtin): {heapsort_time/builtin_time:.2f}")

    def test_worst_case_performance(self):
        """Test heapsort performance on worst-case input."""
        # For heapsort, worst case is typically already sorted array
        size = 5000
        arr = list(range(size))  # Already sorted array
        
        start_time = time.time()
        result = heapsort(arr)
        end_time = time.time()
        
        self.assertEqual(result, arr)  # Should still be sorted
        
        elapsed_time = end_time - start_time
        print(f"\nWorst case performance for {size} elements: {elapsed_time:.6f} seconds")


def run_stress_tests():
    """Run additional stress tests with various array configurations."""
    print("\n" + "="*60)
    print("RUNNING STRESS TESTS")
    print("="*60)
    
    test_configs = [
        ("Random small arrays", lambda: [random.randint(1, 100) for _ in range(10)]),
        ("Random medium arrays", lambda: [random.randint(1, 1000) for _ in range(100)]),
        ("Nearly sorted arrays", lambda: sorted([random.randint(1, 100) for _ in range(50)]) + [random.randint(1, 100)]),
        ("Many duplicates", lambda: [random.choice([1, 2, 3, 4, 5]) for _ in range(50)]),
        ("Single element repeated", lambda: [42] * 100),
    ]
    
    for test_name, generator in test_configs:
        print(f"\nTesting: {test_name}")
        for i in range(10):  # Run each test 10 times
            arr = generator()
            result = heapsort(arr)
            expected = sorted(arr)
            assert result == expected, f"Failed on iteration {i+1} with array: {arr}"
        print(f"✓ Passed all 10 iterations")
    
    print("\n✓ All stress tests passed!")


if __name__ == "__main__":
    # Run the standard unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional stress tests
    run_stress_tests()