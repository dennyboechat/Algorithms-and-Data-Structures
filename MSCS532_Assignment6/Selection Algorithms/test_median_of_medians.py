"""
Test Suite for Median of Medians Selection Algorithm
MSCS532 Assignment 6

This module contains comprehensive tests for the deterministic selection algorithm.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

import unittest
import random
from median_of_medians import (
    median_of_medians_select, 
    find_median, 
    find_kth_smallest, 
    find_kth_largest
)


class TestMedianOfMedians(unittest.TestCase):
    """Test cases for the Median of Medians selection algorithm."""
    
    def test_basic_selection(self):
        """Test basic k-th element selection."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        
        # Test various k values
        self.assertEqual(find_kth_smallest(arr, 1), 1)  # minimum
        self.assertEqual(find_kth_smallest(arr, 5), 3)  # 5th smallest
        self.assertEqual(find_kth_smallest(arr, 10), 9)  # maximum
    
    def test_single_element(self):
        """Test with single element array."""
        arr = [42]
        self.assertEqual(find_kth_smallest(arr, 1), 42)
        self.assertEqual(find_median(arr), 42)
    
    def test_two_elements(self):
        """Test with two element array."""
        arr = [10, 5]
        self.assertEqual(find_kth_smallest(arr, 1), 5)
        self.assertEqual(find_kth_smallest(arr, 2), 10)
        self.assertEqual(find_median(arr), 5)  # Lower median for even length
    
    def test_duplicates(self):
        """Test with duplicate elements."""
        arr = [5, 2, 8, 2, 9, 1, 5, 5]
        sorted_arr = sorted(arr)
        
        for k in range(1, len(arr) + 1):
            expected = sorted_arr[k-1]
            result = find_kth_smallest(arr, k)
            self.assertEqual(result, expected, f"Failed for k={k}")
    
    def test_already_sorted(self):
        """Test with already sorted array."""
        arr = list(range(1, 11))  # [1, 2, 3, ..., 10]
        
        for k in range(1, len(arr) + 1):
            self.assertEqual(find_kth_smallest(arr, k), k)
    
    def test_reverse_sorted(self):
        """Test with reverse sorted array."""
        arr = list(range(10, 0, -1))  # [10, 9, 8, ..., 1]
        
        for k in range(1, len(arr) + 1):
            self.assertEqual(find_kth_smallest(arr, k), k)
    
    def test_median_odd_length(self):
        """Test median for odd length arrays."""
        arr = [3, 1, 4, 5, 9]
        expected_median = 4  # Middle element when sorted: [1, 3, 4, 5, 9]
        self.assertEqual(find_median(arr), expected_median)
    
    def test_median_even_length(self):
        """Test median for even length arrays."""
        arr = [3, 1, 4, 5]
        expected_median = 3  # Lower middle when sorted: [1, 3, 4, 5]
        self.assertEqual(find_median(arr), expected_median)
    
    def test_kth_largest(self):
        """Test k-th largest element selection."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        
        self.assertEqual(find_kth_largest(arr, 1), 9)  # largest
        self.assertEqual(find_kth_largest(arr, 2), 6)  # 2nd largest
        self.assertEqual(find_kth_largest(arr, 8), 1)  # smallest (8th largest)
    
    def test_large_array(self):
        """Test with larger arrays."""
        # Test with array of size 100
        arr = list(range(100, 0, -1))  # [100, 99, 98, ..., 1]
        
        # Test a few positions
        self.assertEqual(find_kth_smallest(arr, 1), 1)
        self.assertEqual(find_kth_smallest(arr, 50), 50)
        self.assertEqual(find_kth_smallest(arr, 100), 100)
    
    def test_random_arrays(self):
        """Test with random arrays and compare against sorting."""
        for _ in range(10):  # Run multiple random tests
            size = random.randint(10, 100)
            arr = [random.randint(1, 1000) for _ in range(size)]
            k = random.randint(1, size)
            
            # Compare with sorting-based approach
            expected = sorted(arr)[k-1]
            result = find_kth_smallest(arr, k)
            self.assertEqual(result, expected, f"Failed for array size {size}, k={k}")
    
    def test_error_cases(self):
        """Test error handling."""
        arr = [1, 2, 3, 4, 5]
        
        # Test k out of bounds
        with self.assertRaises(ValueError):
            find_kth_smallest(arr, 0)
        
        with self.assertRaises(ValueError):
            find_kth_smallest(arr, 6)
        
        # Test empty array
        with self.assertRaises(ValueError):
            find_kth_smallest([], 1)
        
        with self.assertRaises(ValueError):
            find_median([])
    
    def test_array_not_modified(self):
        """Test that the original array is not modified."""
        original = [3, 1, 4, 1, 5, 9, 2, 6]
        arr_copy = original.copy()
        
        find_kth_smallest(arr_copy, 4)
        
        self.assertEqual(original, arr_copy, "Original array was modified")
    
    def test_worst_case_input(self):
        """Test with inputs that could trigger worst-case behavior."""
        # Create an array that might challenge the pivot selection
        arr = []
        for i in range(25):  # Create groups that might have bad medians
            arr.extend([i, i, i, i, i])  # 5 identical elements per group
        
        random.shuffle(arr)
        
        # Test that it still works correctly
        for k in [1, 25, 62, 125]:
            expected = sorted(arr)[k-1]
            result = find_kth_smallest(arr, k)
            self.assertEqual(result, expected, f"Failed for worst-case k={k}")


def run_comprehensive_tests():
    """Run all tests and provide detailed output."""
    print("Running comprehensive test suite for Median of Medians algorithm...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMedianOfMedians)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    run_comprehensive_tests()