"""
Test Suite for Randomized Selection Algorithms
MSCS532 Assignment 6

This module contains comprehensive tests for the randomized quickselect algorithms.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

import unittest
import random
from randomized_quickselect import (
    randomized_select,
    randomized_select_iterative,
    quickselect_with_median_of_three,
    find_median_random,
    find_kth_smallest_random,
    find_kth_largest_random
)


class TestRandomizedQuickselect(unittest.TestCase):
    """Test cases for randomized quickselect algorithms."""
    
    def setUp(self):
        """Set up test fixtures with fixed random seed for reproducibility."""
        random.seed(42)
    
    def test_basic_selection_recursive(self):
        """Test basic k-th element selection with recursive approach."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        sorted_arr = sorted(arr)
        
        for k in range(1, len(arr) + 1):
            expected = sorted_arr[k-1]
            result = randomized_select(arr, k)
            self.assertEqual(result, expected, f"Failed for k={k}")
    
    def test_basic_selection_iterative(self):
        """Test basic k-th element selection with iterative approach."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        sorted_arr = sorted(arr)
        
        for k in range(1, len(arr) + 1):
            expected = sorted_arr[k-1]
            result = randomized_select_iterative(arr, k)
            self.assertEqual(result, expected, f"Failed for k={k}")
    
    def test_median_of_three_selection(self):
        """Test k-th element selection with median-of-three pivot."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        sorted_arr = sorted(arr)
        
        for k in range(1, len(arr) + 1):
            expected = sorted_arr[k-1]
            result = quickselect_with_median_of_three(arr, k)
            self.assertEqual(result, expected, f"Failed for k={k}")
    
    def test_single_element(self):
        """Test with single element array."""
        arr = [42]
        
        self.assertEqual(randomized_select(arr, 1), 42)
        self.assertEqual(randomized_select_iterative(arr, 1), 42)
        self.assertEqual(quickselect_with_median_of_three(arr, 1), 42)
        self.assertEqual(find_median_random(arr), 42)
    
    def test_two_elements(self):
        """Test with two element array."""
        arr = [10, 5]
        
        # Test all approaches
        self.assertEqual(randomized_select(arr, 1), 5)
        self.assertEqual(randomized_select(arr, 2), 10)
        self.assertEqual(randomized_select_iterative(arr, 1), 5)
        self.assertEqual(randomized_select_iterative(arr, 2), 10)
        self.assertEqual(quickselect_with_median_of_three(arr, 1), 5)
        self.assertEqual(quickselect_with_median_of_three(arr, 2), 10)
    
    def test_duplicates(self):
        """Test with arrays containing duplicates."""
        arr = [5, 2, 8, 2, 9, 1, 5, 5]
        sorted_arr = sorted(arr)
        
        for k in range(1, len(arr) + 1):
            expected = sorted_arr[k-1]
            
            # Test all three approaches
            result1 = randomized_select(arr, k)
            result2 = randomized_select_iterative(arr, k)
            result3 = quickselect_with_median_of_three(arr, k)
            
            self.assertEqual(result1, expected, f"Recursive failed for k={k}")
            self.assertEqual(result2, expected, f"Iterative failed for k={k}")
            self.assertEqual(result3, expected, f"Median-of-3 failed for k={k}")
    
    def test_already_sorted(self):
        """Test with already sorted array."""
        arr = list(range(1, 11))  # [1, 2, 3, ..., 10]
        
        for k in range(1, len(arr) + 1):
            self.assertEqual(randomized_select(arr, k), k)
            self.assertEqual(randomized_select_iterative(arr, k), k)
            self.assertEqual(quickselect_with_median_of_three(arr, k), k)
    
    def test_reverse_sorted(self):
        """Test with reverse sorted array."""
        arr = list(range(10, 0, -1))  # [10, 9, 8, ..., 1]
        
        for k in range(1, len(arr) + 1):
            self.assertEqual(randomized_select(arr, k), k)
            self.assertEqual(randomized_select_iterative(arr, k), k)
            self.assertEqual(quickselect_with_median_of_three(arr, k), k)
    
    def test_median_functions(self):
        """Test median finding functions."""
        # Odd length array
        arr_odd = [3, 1, 4, 5, 9]
        expected_median_odd = 4  # Middle element when sorted: [1, 3, 4, 5, 9]
        self.assertEqual(find_median_random(arr_odd), expected_median_odd)
        
        # Even length array
        arr_even = [3, 1, 4, 5]
        expected_median_even = 3  # Lower middle when sorted: [1, 3, 4, 5]
        self.assertEqual(find_median_random(arr_even), expected_median_even)
    
    def test_kth_largest(self):
        """Test k-th largest element selection."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        
        self.assertEqual(find_kth_largest_random(arr, 1), 9)  # largest
        self.assertEqual(find_kth_largest_random(arr, 2), 6)  # 2nd largest
        self.assertEqual(find_kth_largest_random(arr, 8), 1)  # smallest (8th largest)
    
    def test_large_arrays(self):
        """Test with larger arrays."""
        for size in [50, 100, 200]:
            arr = [random.randint(1, 1000) for _ in range(size)]
            sorted_arr = sorted(arr)
            
            # Test a few random positions
            test_positions = [1, size//4, size//2, 3*size//4, size]
            
            for k in test_positions:
                expected = sorted_arr[k-1]
                
                # Test all approaches
                result1 = randomized_select(arr, k)
                result2 = randomized_select_iterative(arr, k)
                result3 = quickselect_with_median_of_three(arr, k)
                
                self.assertEqual(result1, expected, f"Recursive failed for size={size}, k={k}")
                self.assertEqual(result2, expected, f"Iterative failed for size={size}, k={k}")
                self.assertEqual(result3, expected, f"Median-of-3 failed for size={size}, k={k}")
    
    def test_random_arrays_comprehensive(self):
        """Comprehensive test with multiple random arrays."""
        for test_run in range(20):  # Multiple test runs
            size = random.randint(10, 100)
            arr = [random.randint(1, 1000) for _ in range(size)]
            k = random.randint(1, size)
            
            expected = sorted(arr)[k-1]
            
            # Test all three approaches
            result1 = randomized_select(arr, k)
            result2 = randomized_select_iterative(arr, k)
            result3 = quickselect_with_median_of_three(arr, k)
            
            self.assertEqual(result1, expected, f"Recursive failed on run {test_run}")
            self.assertEqual(result2, expected, f"Iterative failed on run {test_run}")
            self.assertEqual(result3, expected, f"Median-of-3 failed on run {test_run}")
    
    def test_error_cases(self):
        """Test error handling for all algorithms."""
        arr = [1, 2, 3, 4, 5]
        
        # Test k out of bounds for all functions
        algorithms = [
            randomized_select,
            randomized_select_iterative,
            quickselect_with_median_of_three,
            find_kth_smallest_random,
            find_kth_largest_random
        ]
        
        for algo in algorithms:
            with self.assertRaises(ValueError):
                algo(arr, 0)
            with self.assertRaises(ValueError):
                algo(arr, 6)
        
        # Test empty array
        for algo in algorithms:
            with self.assertRaises(ValueError):
                algo([], 1)
        
        with self.assertRaises(ValueError):
            find_median_random([])
    
    def test_array_not_modified(self):
        """Test that original arrays are not modified."""
        original = [3, 1, 4, 1, 5, 9, 2, 6]
        
        # Test all algorithms
        arr_copy1 = original.copy()
        randomized_select(arr_copy1, 4)
        self.assertEqual(original, arr_copy1)
        
        arr_copy2 = original.copy()
        randomized_select_iterative(arr_copy2, 4)
        self.assertEqual(original, arr_copy2)
        
        arr_copy3 = original.copy()
        quickselect_with_median_of_three(arr_copy3, 4)
        self.assertEqual(original, arr_copy3)
    
    def test_consistency_across_algorithms(self):
        """Test that all algorithms produce consistent results."""
        for _ in range(10):  # Multiple test runs
            arr = [random.randint(1, 100) for _ in range(20)]
            k = random.randint(1, 20)
            
            result1 = randomized_select(arr, k)
            result2 = randomized_select_iterative(arr, k)
            result3 = quickselect_with_median_of_three(arr, k)
            
            # All should produce the same result
            self.assertEqual(result1, result2)
            self.assertEqual(result2, result3)
    
    def test_extreme_cases(self):
        """Test extreme cases that might cause issues."""
        # All identical elements
        arr_identical = [7] * 20
        for k in [1, 10, 20]:
            self.assertEqual(randomized_select(arr_identical, k), 7)
            self.assertEqual(randomized_select_iterative(arr_identical, k), 7)
            self.assertEqual(quickselect_with_median_of_three(arr_identical, k), 7)
        
        # Two distinct values only
        arr_binary = [1, 2] * 10  # [1, 2, 1, 2, ...]
        random.shuffle(arr_binary)
        sorted_binary = sorted(arr_binary)
        
        for k in [1, 5, 10, 15, 20]:
            expected = sorted_binary[k-1]
            self.assertEqual(randomized_select(arr_binary, k), expected)
            self.assertEqual(randomized_select_iterative(arr_binary, k), expected)
            self.assertEqual(quickselect_with_median_of_three(arr_binary, k), expected)
    
    def test_performance_characteristics(self):
        """Test that algorithms perform reasonably on various input patterns."""
        import time
        
        # Test different patterns
        patterns = {
            "random": lambda n: [random.randint(1, 1000) for _ in range(n)],
            "sorted": lambda n: list(range(1, n+1)),
            "reverse": lambda n: list(range(n, 0, -1)),
            "identical": lambda n: [42] * n
        }
        
        size = 1000
        k = size // 2
        
        for pattern_name, pattern_func in patterns.items():
            arr = pattern_func(size)
            
            # Test that all algorithms complete in reasonable time
            start_time = time.time()
            result1 = randomized_select(arr, k)
            time1 = time.time() - start_time
            
            start_time = time.time()
            result2 = randomized_select_iterative(arr, k)
            time2 = time.time() - start_time
            
            start_time = time.time()
            result3 = quickselect_with_median_of_three(arr, k)
            time3 = time.time() - start_time
            
            # All should complete within reasonable time (< 1 second for size 1000)
            self.assertLess(time1, 1.0, f"Recursive too slow on {pattern_name}")
            self.assertLess(time2, 1.0, f"Iterative too slow on {pattern_name}")
            self.assertLess(time3, 1.0, f"Median-of-3 too slow on {pattern_name}")
            
            # Results should be consistent
            self.assertEqual(result1, result2)
            self.assertEqual(result2, result3)


def run_randomized_tests():
    """Run all tests for randomized algorithms with detailed output."""
    print("Running comprehensive test suite for Randomized Quickselect algorithms...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRandomizedQuickselect)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All randomized algorithm tests passed!")
    else:
        print("❌ Some tests failed.")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    run_randomized_tests()