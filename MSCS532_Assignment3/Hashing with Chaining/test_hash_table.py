"""
Comprehensive test suite for Hash Table with Chaining implementation.

This module contains unit tests for all hash table operations including
edge cases, collision scenarios, and performance validation.

"""

import unittest
import random
import string
from hash_table import HashTableChaining, HashNode, UniversalHashFunction


class TestHashNode(unittest.TestCase):
    """Test cases for HashNode class."""
    
    def test_node_creation(self):
        """Test hash node creation and initialization."""
        node = HashNode("key1", "value1")
        self.assertEqual(node.key, "key1")
        self.assertEqual(node.value, "value1")
        self.assertIsNone(node.next)
    
    def test_node_linking(self):
        """Test linking nodes together."""
        node1 = HashNode("key1", "value1")
        node2 = HashNode("key2", "value2")
        node1.next = node2
        
        self.assertEqual(node1.next, node2)
        self.assertEqual(node1.next.key, "key2")
        self.assertIsNone(node2.next)


class TestUniversalHashFunction(unittest.TestCase):
    """Test cases for UniversalHashFunction class."""
    
    def test_hash_function_creation(self):
        """Test hash function initialization."""
        table_size = 10
        hash_func = UniversalHashFunction(table_size)
        
        self.assertEqual(hash_func.table_size, table_size)
        self.assertGreater(hash_func.a, 0)
        self.assertGreaterEqual(hash_func.b, 0)
        self.assertLess(hash_func.a, hash_func.prime)
        self.assertLess(hash_func.b, hash_func.prime)
    
    def test_hash_range(self):
        """Test that hash function returns values in correct range."""
        table_size = 10
        hash_func = UniversalHashFunction(table_size)
        
        test_keys = ["apple", "banana", "cherry", 123, 456, "test", ""]
        for key in test_keys:
            hash_value = hash_func.hash(key)
            self.assertGreaterEqual(hash_value, 0)
            self.assertLess(hash_value, table_size)
    
    def test_hash_consistency(self):
        """Test that same key always produces same hash value."""
        hash_func = UniversalHashFunction(10)
        key = "test_key"
        
        hash1 = hash_func.hash(key)
        hash2 = hash_func.hash(key)
        hash3 = hash_func.hash(key)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)


class TestHashTableChaining(unittest.TestCase):
    """Test cases for HashTableChaining class."""
    
    def setUp(self):
        """Set up test hash table before each test."""
        self.ht = HashTableChaining(initial_size=7, load_factor_threshold=0.75)
    
    def test_initialization(self):
        """Test hash table initialization."""
        self.assertEqual(self.ht.size, 7)
        self.assertEqual(self.ht.count, 0)
        self.assertEqual(len(self.ht.table), 7)
        self.assertEqual(self.ht.load_factor_threshold, 0.75)
        self.assertEqual(self.ht.collision_count, 0)
        
        # Check that all slots are initially None
        for slot in self.ht.table:
            self.assertIsNone(slot)
    
    def test_insert_single_item(self):
        """Test inserting a single key-value pair."""
        self.ht.insert("key1", "value1")
        
        self.assertEqual(self.ht.count, 1)
        self.assertEqual(self.ht.search("key1"), "value1")
        self.assertTrue(self.ht.contains("key1"))
    
    def test_insert_multiple_items(self):
        """Test inserting multiple key-value pairs."""
        test_data = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        
        for key, value in test_data:
            self.ht.insert(key, value)
        
        self.assertEqual(self.ht.count, len(test_data))
        
        for key, value in test_data:
            self.assertEqual(self.ht.search(key), value)
    
    def test_insert_duplicate_key(self):
        """Test inserting duplicate keys (should update value)."""
        self.ht.insert("key1", "value1")
        self.assertEqual(self.ht.search("key1"), "value1")
        self.assertEqual(self.ht.count, 1)
        
        # Insert same key with different value
        self.ht.insert("key1", "new_value")
        self.assertEqual(self.ht.search("key1"), "new_value")
        self.assertEqual(self.ht.count, 1)  # Count should remain same
    
    def test_search_existing_keys(self):
        """Test searching for existing keys."""
        test_data = {"apple": 5, "banana": 3, "cherry": 8}
        
        for key, value in test_data.items():
            self.ht.insert(key, value)
        
        for key, expected_value in test_data.items():
            found_value = self.ht.search(key)
            self.assertEqual(found_value, expected_value)
    
    def test_search_nonexistent_key(self):
        """Test searching for non-existent keys."""
        self.ht.insert("key1", "value1")
        
        result = self.ht.search("nonexistent")
        self.assertIsNone(result)
    
    def test_search_empty_table(self):
        """Test searching in empty table."""
        result = self.ht.search("any_key")
        self.assertIsNone(result)
    
    def test_delete_existing_key(self):
        """Test deleting existing keys."""
        test_data = {"a": 1, "b": 2, "c": 3}
        
        for key, value in test_data.items():
            self.ht.insert(key, value)
        
        # Delete middle key
        success = self.ht.delete("b")
        self.assertTrue(success)
        self.assertEqual(self.ht.count, 2)
        self.assertIsNone(self.ht.search("b"))
        
        # Other keys should still exist
        self.assertEqual(self.ht.search("a"), 1)
        self.assertEqual(self.ht.search("c"), 3)
    
    def test_delete_nonexistent_key(self):
        """Test deleting non-existent keys."""
        self.ht.insert("key1", "value1")
        
        success = self.ht.delete("nonexistent")
        self.assertFalse(success)
        self.assertEqual(self.ht.count, 1)
    
    def test_delete_from_empty_table(self):
        """Test deleting from empty table."""
        success = self.ht.delete("any_key")
        self.assertFalse(success)
        self.assertEqual(self.ht.count, 0)
    
    def test_delete_first_node_in_chain(self):
        """Test deleting the first node in a collision chain."""
        # Force collision by using keys that hash to same value
        # We'll insert multiple items and hope for collision, then verify
        keys = ["key1", "key2", "key3", "key4", "key5"]
        for key in keys:
            self.ht.insert(key, f"value_{key}")
        
        # Delete first key and verify others remain
        success = self.ht.delete("key1")
        self.assertTrue(success)
        self.assertIsNone(self.ht.search("key1"))
        
        # Check other keys still exist
        for key in keys[1:]:
            self.assertIsNotNone(self.ht.search(key))
    
    def test_contains_method(self):
        """Test the contains method."""
        self.ht.insert("test_key", "test_value")
        
        self.assertTrue(self.ht.contains("test_key"))
        self.assertFalse(self.ht.contains("nonexistent"))
    
    def test_magic_methods(self):
        """Test magic methods (__getitem__, __setitem__, etc.)."""
        # Test __setitem__ and __getitem__
        self.ht["key1"] = "value1"
        self.assertEqual(self.ht["key1"], "value1")
        
        # Test __contains__
        self.assertTrue("key1" in self.ht)
        self.assertFalse("nonexistent" in self.ht)
        
        # Test __len__
        self.assertEqual(len(self.ht), 1)
        
        # Test __delitem__
        del self.ht["key1"]
        self.assertEqual(len(self.ht), 0)
        
        # Test KeyError for non-existent key
        with self.assertRaises(KeyError):
            _ = self.ht["nonexistent"]
        
        with self.assertRaises(KeyError):
            del self.ht["nonexistent"]
    
    def test_iterators(self):
        """Test iterator methods (keys, values, items)."""
        test_data = {"a": 1, "b": 2, "c": 3}
        
        for key, value in test_data.items():
            self.ht.insert(key, value)
        
        # Test keys()
        keys = list(self.ht.keys())
        self.assertEqual(set(keys), set(test_data.keys()))
        
        # Test values()
        values = list(self.ht.values())
        self.assertEqual(set(values), set(test_data.values()))
        
        # Test items()
        items = list(self.ht.items())
        self.assertEqual(set(items), set(test_data.items()))
    
    def test_load_factor_calculation(self):
        """Test load factor calculation."""
        self.assertEqual(self.ht.get_load_factor(), 0.0)
        
        self.ht.insert("key1", "value1")
        self.assertAlmostEqual(self.ht.get_load_factor(), 1/7, places=2)
        
        self.ht.insert("key2", "value2")
        self.assertAlmostEqual(self.ht.get_load_factor(), 2/7, places=2)
    
    def test_resize_functionality(self):
        """Test that hash table resizes when load factor exceeds threshold."""
        initial_size = self.ht.size
        
        # Insert enough items to trigger resize
        num_items = int(self.ht.size * self.ht.load_factor_threshold) + 1
        for i in range(num_items):
            self.ht.insert(f"key{i}", f"value{i}")
        
        # Should have resized
        self.assertGreater(self.ht.size, initial_size)
        self.assertEqual(self.ht.resize_count, 1)
        
        # All items should still be accessible
        for i in range(num_items):
            self.assertEqual(self.ht.search(f"key{i}"), f"value{i}")
    
    def test_collision_handling(self):
        """Test collision handling with many items."""
        # Create small table to force collisions
        small_ht = HashTableChaining(initial_size=3, load_factor_threshold=2.0)
        
        # Insert many items to guarantee collisions
        for i in range(10):
            small_ht.insert(f"key{i}", f"value{i}")
        
        # Verify all items are still accessible
        for i in range(10):
            self.assertEqual(small_ht.search(f"key{i}"), f"value{i}")
        
        # Should have had collisions
        self.assertGreater(small_ht.collision_count, 0)
    
    def test_statistics(self):
        """Test statistics generation."""
        # Insert some data
        for i in range(5):
            self.ht.insert(f"key{i}", f"value{i}")
        
        stats = self.ht.get_statistics()
        
        # Check required statistics
        self.assertIn('size', stats)
        self.assertIn('count', stats)
        self.assertIn('load_factor', stats)
        self.assertIn('collision_count', stats)
        self.assertIn('max_chain_length', stats)
        self.assertIn('avg_chain_length', stats)
        
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['size'], self.ht.size)
        self.assertGreaterEqual(stats['max_chain_length'], 1)
    
    def test_different_key_types(self):
        """Test hash table with different key types."""
        test_cases = [
            ("string_key", "string_value"),
            (123, "int_key_value"),
            ((1, 2), "tuple_key_value"),
            (frozenset([1, 2, 3]), "frozenset_value")
        ]
        
        for key, value in test_cases:
            self.ht.insert(key, value)
        
        for key, expected_value in test_cases:
            self.assertEqual(self.ht.search(key), expected_value)
    
    def test_stress_operations(self):
        """Stress test with many operations."""
        num_operations = 1000
        keys = [f"key{i}" for i in range(num_operations)]
        
        # Insert many items
        for i, key in enumerate(keys):
            self.ht.insert(key, i)
        
        # Verify all insertions
        for i, key in enumerate(keys):
            self.assertEqual(self.ht.search(key), i)
        
        # Delete half the items
        for i in range(0, num_operations, 2):
            success = self.ht.delete(keys[i])
            self.assertTrue(success)
        
        # Verify deletions
        for i in range(num_operations):
            if i % 2 == 0:
                self.assertIsNone(self.ht.search(keys[i]))
            else:
                self.assertEqual(self.ht.search(keys[i]), i)
    
    def test_next_prime_function(self):
        """Test the next prime finding function."""
        test_cases = [
            (10, 11),
            (11, 11),
            (12, 13),
            (20, 23),
            (100, 101)
        ]
        
        for input_val, expected in test_cases:
            result = self.ht._next_prime(input_val)
            self.assertEqual(result, expected)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_empty_string_key(self):
        """Test using empty string as key."""
        ht = HashTableChaining()
        ht.insert("", "empty_key_value")
        self.assertEqual(ht.search(""), "empty_key_value")
    
    def test_none_value(self):
        """Test storing None as a value."""
        ht = HashTableChaining()
        ht.insert("key", None)
        self.assertIsNone(ht.search("key"))
        self.assertTrue(ht.contains("key"))  # Key exists even though value is None
    
    def test_large_keys(self):
        """Test with very large keys."""
        ht = HashTableChaining()
        large_key = "x" * 10000
        ht.insert(large_key, "large_key_value")
        self.assertEqual(ht.search(large_key), "large_key_value")
    
    def test_unicode_keys(self):
        """Test with Unicode keys."""
        ht = HashTableChaining()
        unicode_keys = ["café", "naïve", "résumé", "🚀", "测试"]
        
        for i, key in enumerate(unicode_keys):
            ht.insert(key, f"value{i}")
        
        for i, key in enumerate(unicode_keys):
            self.assertEqual(ht.search(key), f"value{i}")


class TestPerformance(unittest.TestCase):
    """Performance-related tests."""
    
    def test_performance_with_good_distribution(self):
        """Test performance with well-distributed keys."""
        ht = HashTableChaining(initial_size=100)
        
        # Generate random keys
        keys = [''.join(random.choices(string.ascii_letters, k=10)) 
                for _ in range(500)]
        
        # Insert all keys
        for i, key in enumerate(keys):
            ht.insert(key, i)
        
        # Get statistics
        stats = ht.get_statistics()
        
        # With good hash function, max chain length should be reasonable
        # This is probabilistic, so we use a generous upper bound
        self.assertLess(stats['max_chain_length'], 20)
        
        # Most slots should be utilized reasonably well
        self.assertGreater(stats['utilization'], 0.3)


if __name__ == '__main__':
    # Configure test output
    unittest.TestCase.maxDiff = None
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestHashNode,
        TestUniversalHashFunction, 
        TestHashTableChaining,
        TestEdgeCases,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")