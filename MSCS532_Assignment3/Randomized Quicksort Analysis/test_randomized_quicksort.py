"""
Test Suite for Randomized Quicksort Implementation

This module contains comprehensive tests for the randomized quicksort implementation,
including edge cases, performance tests, and correctness validation.
"""

import unittest
import random
import time
from typing import List
from randomized_quicksort import RandomizedQuicksort, RandomizedQuicksort3Way, is_sorted, generate_test_arrays


class TestRandomizedQuicksort(unittest.TestCase):
    """Test cases for the RandomizedQuicksort class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sorter = RandomizedQuicksort(seed=42)  # Fixed seed for reproducibility
        self.sorter_3way = RandomizedQuicksort3Way(seed=42)
    
    def test_empty_array(self):
        """Test sorting an empty array."""
        result = self.sorter.sort([])
        self.assertEqual(result, [])
        self.assertTrue(is_sorted(result))
    
    def test_single_element(self):
        """Test sorting a single-element array."""
        test_array = [42]
        result = self.sorter.sort(test_array)
        self.assertEqual(result, [42])
        self.assertTrue(is_sorted(result))
    
    def test_two_elements(self):
        """Test sorting a two-element array."""
        test_array = [2, 1]
        result = self.sorter.sort(test_array)
        self.assertEqual(result, [1, 2])
        self.assertTrue(is_sorted(result))
    
    def test_already_sorted(self):
        """Test sorting an already sorted array."""
        test_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.sorter.sort(test_array)
        self.assertEqual(result, test_array)
        self.assertTrue(is_sorted(result))
    
    def test_reverse_sorted(self):
        """Test sorting a reverse-sorted array."""
        test_array = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.sorter.sort(test_array)
        self.assertEqual(result, expected)
        self.assertTrue(is_sorted(result))
    
    def test_all_duplicates(self):
        """Test sorting an array with all duplicate elements."""
        test_array = [5, 5, 5, 5, 5]
        result = self.sorter.sort(test_array)
        self.assertEqual(result, [5, 5, 5, 5, 5])
        self.assertTrue(is_sorted(result))
    
    def test_some_duplicates(self):
        """Test sorting an array with some duplicate elements."""
        test_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = self.sorter.sort(test_array)
        expected = sorted(test_array)
        self.assertEqual(result, expected)
        self.assertTrue(is_sorted(result))
    
    def test_negative_numbers(self):
        """Test sorting an array with negative numbers."""
        test_array = [-5, -1, 0, 3, -2, 8, -10, 15]
        result = self.sorter.sort(test_array)
        expected = sorted(test_array)
        self.assertEqual(result, expected)
        self.assertTrue(is_sorted(result))
    
    def test_large_range(self):
        """Test sorting an array with a large range of values."""
        random.seed(42)
        test_array = [random.randint(-1000, 1000) for _ in range(100)]
        result = self.sorter.sort(test_array)
        expected = sorted(test_array)
        self.assertEqual(result, expected)
        self.assertTrue(is_sorted(result))
    
    def test_inplace_sorting(self):
        """Test in-place sorting functionality."""
        test_array = [64, 34, 25, 12, 22, 11, 90]
        original = test_array.copy()
        self.sorter.sort_inplace(test_array)
        expected = sorted(original)
        self.assertEqual(test_array, expected)
        self.assertTrue(is_sorted(test_array))
    
    def test_original_array_unchanged(self):
        """Test that the original array is not modified by sort()."""
        test_array = [64, 34, 25, 12, 22, 11, 90]
        original = test_array.copy()
        result = self.sorter.sort(test_array)
        self.assertEqual(test_array, original)  # Original should be unchanged
        self.assertTrue(is_sorted(result))
    
    def test_statistics_collection(self):
        """Test that statistics are properly collected."""
        test_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        self.sorter.sort(test_array)
        comparisons, swaps = self.sorter.get_statistics()
        self.assertGreater(comparisons, 0)
        self.assertGreater(swaps, 0)


class TestRandomizedQuicksort3Way(unittest.TestCase):
    """Test cases for the RandomizedQuicksort3Way class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sorter_3way = RandomizedQuicksort3Way(seed=42)
    
    def test_empty_array_3way(self):
        """Test 3-way sorting an empty array."""
        result = self.sorter_3way.sort([])
        self.assertEqual(result, [])
        self.assertTrue(is_sorted(result))
    
    def test_single_element_3way(self):
        """Test 3-way sorting a single-element array."""
        test_array = [42]
        result = self.sorter_3way.sort(test_array)
        self.assertEqual(result, [42])
        self.assertTrue(is_sorted(result))
    
    def test_many_duplicates_3way(self):
        """Test 3-way sorting with many duplicates (should be more efficient)."""
        test_array = [5, 5, 5, 5, 5, 1, 1, 1, 9, 9, 9, 9]
        result = self.sorter_3way.sort(test_array)
        expected = sorted(test_array)
        self.assertEqual(result, expected)
        self.assertTrue(is_sorted(result))
    
    def test_random_with_duplicates_3way(self):
        """Test 3-way sorting with random array containing duplicates."""
        random.seed(42)
        test_array = [random.randint(1, 10) for _ in range(50)]  # Small range to ensure duplicates
        result = self.sorter_3way.sort(test_array)
        expected = sorted(test_array)
        self.assertEqual(result, expected)
        self.assertTrue(is_sorted(result))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_is_sorted_empty(self):
        """Test is_sorted with empty array."""
        self.assertTrue(is_sorted([]))
    
    def test_is_sorted_single(self):
        """Test is_sorted with single element."""
        self.assertTrue(is_sorted([5]))
    
    def test_is_sorted_true(self):
        """Test is_sorted with sorted array."""
        self.assertTrue(is_sorted([1, 2, 3, 4, 5]))
        self.assertTrue(is_sorted([1, 1, 2, 2, 3]))  # With duplicates
    
    def test_is_sorted_false(self):
        """Test is_sorted with unsorted array."""
        self.assertFalse(is_sorted([5, 4, 3, 2, 1]))
        self.assertFalse(is_sorted([1, 3, 2, 4, 5]))
    
    def test_generate_test_arrays(self):
        """Test that generate_test_arrays returns expected structure."""
        test_arrays = generate_test_arrays()
        expected_keys = {
            'empty', 'single', 'two_elements', 'already_sorted', 
            'reverse_sorted', 'all_duplicates', 'some_duplicates',
            'random_small', 'random_medium', 'with_negatives', 'large_range'
        }
        self.assertEqual(set(test_arrays.keys()), expected_keys)
        self.assertEqual(test_arrays['empty'], [])
        self.assertEqual(len(test_arrays['single']), 1)
        self.assertEqual(len(test_arrays['two_elements']), 2)


class TestPerformanceAndEdgeCases(unittest.TestCase):
    """Test performance characteristics and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sorter = RandomizedQuicksort(seed=42)
        self.sorter_3way = RandomizedQuicksort3Way(seed=42)
    
    def test_large_array_performance(self):
        """Test performance with larger arrays."""
        random.seed(42)
        test_array = [random.randint(1, 10000) for _ in range(1000)]
        
        start_time = time.time()
        result = self.sorter.sort(test_array)
        end_time = time.time()
        
        self.assertTrue(is_sorted(result))
        self.assertEqual(len(result), len(test_array))
        
        # Should complete reasonably quickly
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5.0)  # Should take less than 5 seconds
    
    def test_worst_case_scenario(self):
        """Test with potentially worst-case scenarios."""
        # Test with many duplicates
        test_array = [1] * 100 + [2] * 100 + [3] * 100
        result = self.sorter.sort(test_array)
        self.assertTrue(is_sorted(result))
        self.assertEqual(len(result), 300)
        
        # Compare with 3-way partitioning
        result_3way = self.sorter_3way.sort(test_array)
        self.assertEqual(result, result_3way)
    
    def test_randomness_different_seeds(self):
        """Test that different seeds produce different intermediate behavior."""
        test_array = [random.randint(1, 100) for _ in range(20)]
        
        sorter1 = RandomizedQuicksort(seed=1)
        sorter2 = RandomizedQuicksort(seed=2)
        
        result1 = sorter1.sort(test_array)
        stats1 = sorter1.get_statistics()
        
        result2 = sorter2.sort(test_array)
        stats2 = sorter2.get_statistics()
        
        # Results should be the same (both sorted)
        self.assertEqual(result1, result2)
        
        # But statistics might be different due to different pivot choices
        # (though this is not guaranteed, it's likely for different seeds)
        self.assertTrue(is_sorted(result1))
        self.assertTrue(is_sorted(result2))


def run_comprehensive_tests():
    """Run all tests and print results."""
    print("Running Comprehensive Test Suite for Randomized Quicksort")
    print("=" * 60)
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRandomizedQuicksort))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomizedQuicksort3Way))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndEdgeCases))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_comprehensive_tests()