"""
Demonstration script for Hash Table with Chaining implementation.

This script showcases the hash table functionality with various examples,
performance analysis, and collision scenarios.

"""

import time
import random
import string
from collections import defaultdict
from hash_table import HashTableChaining


def demo_basic_operations():
    """Demonstrate basic hash table operations."""
    print("=" * 60)
    print("BASIC OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create a hash table
    ht = HashTableChaining(initial_size=7)
    
    print("1. Creating hash table with initial size 7")
    print(f"   Initial state: size={ht.size}, count={ht.count}")
    
    # Insert operations
    print("\n2. Insert Operations:")
    fruits = [
        ("apple", 5), ("banana", 3), ("orange", 8), 
        ("grape", 12), ("kiwi", 6), ("mango", 9), ("pear", 4)
    ]
    
    for fruit, quantity in fruits:
        ht.insert(fruit, quantity)
        print(f"   Inserted: {fruit} -> {quantity}")
    
    print(f"\n   After insertions: size={ht.size}, count={ht.count}")
    print(f"   Load factor: {ht.get_load_factor():.3f}")
    
    # Display table structure
    print("\n3. Hash Table Structure:")
    ht.display()
    
    # Search operations
    print("\n4. Search Operations:")
    search_keys = ["apple", "banana", "coconut", "grape"]
    for key in search_keys:
        result = ht.search(key)
        status = f"Found: {result}" if result is not None else "Not found"
        print(f"   Search '{key}': {status}")
    
    # Update operation
    print("\n5. Update Operation:")
    print(f"   Before update - apple: {ht.search('apple')}")
    ht.insert("apple", 10)  # Update existing key
    print(f"   After update - apple: {ht.search('apple')}")
    
    # Delete operations
    print("\n6. Delete Operations:")
    delete_keys = ["banana", "coconut", "grape"]
    for key in delete_keys:
        success = ht.delete(key)
        status = "Success" if success else "Failed (key not found)"
        print(f"   Delete '{key}': {status}")
    
    print(f"\n   After deletions: count={ht.count}")
    ht.display()
    
    # Magic methods demonstration
    print("\n7. Magic Methods (Pythonic Interface):")
    ht["watermelon"] = 15  # __setitem__
    print(f"   ht['watermelon'] = 15")
    print(f"   ht['watermelon'] = {ht['watermelon']}")  # __getitem__
    print(f"   'watermelon' in ht: {'watermelon' in ht}")  # __contains__
    print(f"   len(ht): {len(ht)}")  # __len__
    
    del ht["watermelon"]  # __delitem__
    print(f"   After del ht['watermelon']: len(ht) = {len(ht)}")


def demo_collision_scenarios():
    """Demonstrate collision handling."""
    print("\n" + "=" * 60)
    print("COLLISION HANDLING DEMONSTRATION")
    print("=" * 60)
    
    # Create small table to force collisions
    ht = HashTableChaining(initial_size=5, load_factor_threshold=2.0)
    
    print("1. Creating small hash table (size=5) to force collisions")
    
    # Insert many items to guarantee collisions
    items = [(f"item{i}", f"value{i}") for i in range(15)]
    
    print("\n2. Inserting 15 items into 5-slot table:")
    for key, value in items:
        ht.insert(key, value)
        print(f"   Inserted: {key} -> {value}")
    
    print(f"\n   Collision count: {ht.collision_count}")
    print(f"   Load factor: {ht.get_load_factor():.3f}")
    
    # Show table structure with chains
    print("\n3. Table structure showing collision chains:")
    ht.display()
    
    # Verify all items are accessible
    print("\n4. Verifying all items are accessible:")
    all_found = True
    for key, expected_value in items:
        found_value = ht.search(key)
        if found_value != expected_value:
            print(f"   ERROR: {key} expected {expected_value}, got {found_value}")
            all_found = False
    
    if all_found:
        print("   ✓ All items successfully retrieved despite collisions")
    
    # Show statistics
    stats = ht.get_statistics()
    print(f"\n5. Collision Statistics:")
    print(f"   Max chain length: {stats['max_chain_length']}")
    print(f"   Average chain length: {stats['avg_chain_length']:.2f}")
    print(f"   Utilization (non-empty slots): {stats['utilization']:.2f}")


def demo_resize_behavior():
    """Demonstrate automatic resizing."""
    print("\n" + "=" * 60)
    print("AUTOMATIC RESIZING DEMONSTRATION")
    print("=" * 60)
    
    ht = HashTableChaining(initial_size=7, load_factor_threshold=0.75)
    
    print(f"1. Initial table: size={ht.size}, threshold={ht.load_factor_threshold}")
    
    # Insert items one by one and show when resize occurs
    print("\n2. Inserting items and monitoring resize:")
    for i in range(15):
        key = f"key{i:02d}"
        value = f"value{i:02d}"
        
        old_size = ht.size
        ht.insert(key, value)
        
        if ht.size != old_size:
            print(f"   Item {i+1}: RESIZE occurred! {old_size} -> {ht.size}")
        else:
            print(f"   Item {i+1}: Inserted {key}, load factor: {ht.get_load_factor():.3f}")
    
    print(f"\n3. Final state:")
    print(f"   Size: {ht.size}")
    print(f"   Count: {ht.count}")
    print(f"   Load factor: {ht.get_load_factor():.3f}")
    print(f"   Resize count: {ht.resize_count}")
    
    # Verify all items still accessible after resize
    print(f"\n4. Verifying all items accessible after resize:")
    for i in range(15):
        key = f"key{i:02d}"
        expected_value = f"value{i:02d}"
        found_value = ht.search(key)
        if found_value != expected_value:
            print(f"   ERROR: {key} not found correctly after resize")
            break
    else:
        print("   ✓ All items correctly accessible after resize")


def demo_performance_analysis():
    """Analyze performance characteristics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Test with different table sizes
    sizes = [50, 100, 500, 1000]
    num_operations = 1000
    
    print(f"Testing performance with {num_operations} operations:")
    print(f"{'Table Size':<12} {'Insert Time':<12} {'Search Time':<12} {'Delete Time':<12} {'Collisions':<12}")
    print("-" * 60)
    
    for size in sizes:
        ht = HashTableChaining(initial_size=size, load_factor_threshold=0.75)
        
        # Generate test data
        keys = [f"key{i:06d}" for i in range(num_operations)]
        values = [f"value{i:06d}" for i in range(num_operations)]
        
        # Time insert operations
        start_time = time.time()
        for key, value in zip(keys, values):
            ht.insert(key, value)
        insert_time = time.time() - start_time
        
        # Time search operations
        start_time = time.time()
        for key in keys:
            ht.search(key)
        search_time = time.time() - start_time
        
        # Time delete operations (half the keys)
        delete_keys = keys[:num_operations//2]
        start_time = time.time()
        for key in delete_keys:
            ht.delete(key)
        delete_time = time.time() - start_time
        
        print(f"{size:<12} {insert_time*1000:<11.2f}ms {search_time*1000:<11.2f}ms "
              f"{delete_time*1000:<11.2f}ms {ht.collision_count:<12}")


def demo_hash_distribution():
    """Analyze hash function distribution quality."""
    print("\n" + "=" * 60)
    print("HASH FUNCTION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    table_size = 20
    ht = HashTableChaining(initial_size=table_size, load_factor_threshold=2.0)
    
    # Generate different types of keys
    key_sets = {
        "Sequential": [f"key{i:03d}" for i in range(100)],
        "Random": [''.join(random.choices(string.ascii_letters, k=8)) for _ in range(100)],
        "Patterns": [f"user_{i%10}_{i//10}" for i in range(100)]
    }
    
    print("Analyzing hash distribution for different key patterns:\n")
    
    for pattern_name, keys in key_sets.items():
        # Reset table
        ht = HashTableChaining(initial_size=table_size, load_factor_threshold=2.0)
        
        # Count distributions - do this BEFORE inserting to avoid resize issues
        distribution = [0] * table_size
        
        for key in keys:
            hash_value = ht._hash(key)
            distribution[hash_value] += 1
        
        # Now insert the items (this may cause resize, but we already have distribution)
        for key in keys:
            ht.insert(key, f"value_{key}")
        
        # Calculate statistics
        min_count = min(distribution)
        max_count = max(distribution)
        avg_count = sum(distribution) / len(distribution)
        empty_slots = distribution.count(0)
        
        print(f"{pattern_name} Keys:")
        print(f"  Distribution: {distribution}")
        print(f"  Min/Max/Avg per slot: {min_count}/{max_count}/{avg_count:.1f}")
        print(f"  Empty slots: {empty_slots}/{table_size}")
        print(f"  Collisions: {ht.collision_count}")
        
        stats = ht.get_statistics()
        print(f"  Max chain length: {stats['max_chain_length']}")
        print()


def demo_different_data_types():
    """Demonstrate hash table with various data types."""
    print("\n" + "=" * 60)
    print("DIFFERENT DATA TYPES DEMONSTRATION")
    print("=" * 60)
    
    ht = HashTableChaining()
    
    # Test various key types
    test_data = [
        # (key, value, description)
        ("string_key", "String value", "String key"),
        (42, "Integer value", "Integer key"),
        ((1, 2, 3), "Tuple value", "Tuple key"),
        (frozenset([1, 2, 3]), "Frozenset value", "Frozenset key"),
        ("", "Empty string value", "Empty string key"),
        ("unicode_🚀", "Unicode value", "Unicode key"),
        ("very_long_key_" + "x" * 100, "Long key value", "Very long key"),
    ]
    
    print("Testing various key types:")
    for key, value, description in test_data:
        ht.insert(key, value)
        retrieved = ht.search(key)
        status = "✓" if retrieved == value else "✗"
        print(f"  {status} {description}: {repr(key)[:50]}...")
    
    print(f"\nFinal table count: {len(ht)}")
    
    # Test iterators
    print("\nTesting iterators:")
    keys = list(ht.keys())
    values = list(ht.values())
    items = list(ht.items())
    
    print(f"  Keys count: {len(keys)}")
    print(f"  Values count: {len(values)}")
    print(f"  Items count: {len(items)}")


def demo_stress_test():
    """Perform stress testing."""
    print("\n" + "=" * 60)
    print("STRESS TEST")
    print("=" * 60)
    
    num_items = 10000
    ht = HashTableChaining(initial_size=100)
    
    print(f"Stress testing with {num_items} items...")
    
    # Generate random data
    test_data = {}
    for i in range(num_items):
        key = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        value = random.randint(1, 1000000)
        test_data[key] = value
    
    # Insert all items
    start_time = time.time()
    for key, value in test_data.items():
        ht.insert(key, value)
    insert_time = time.time() - start_time
    
    print(f"✓ Inserted {len(test_data)} items in {insert_time:.3f} seconds")
    print(f"  Average insert time: {(insert_time/len(test_data))*1000000:.2f} microseconds/item")
    
    # Verify all items
    start_time = time.time()
    errors = 0
    for key, expected_value in test_data.items():
        found_value = ht.search(key)
        if found_value != expected_value:
            errors += 1
    search_time = time.time() - start_time
    
    print(f"✓ Searched {len(test_data)} items in {search_time:.3f} seconds")
    print(f"  Average search time: {(search_time/len(test_data))*1000000:.2f} microseconds/item")
    print(f"  Search errors: {errors}")
    
    # Delete random subset
    delete_keys = list(test_data.keys())[:num_items//3]
    start_time = time.time()
    deleted_count = 0
    for key in delete_keys:
        if ht.delete(key):
            deleted_count += 1
    delete_time = time.time() - start_time
    
    print(f"✓ Deleted {deleted_count} items in {delete_time:.3f} seconds")
    print(f"  Average delete time: {(delete_time/deleted_count)*1000000:.2f} microseconds/item")
    
    # Final statistics
    stats = ht.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Table size: {stats['size']}")
    print(f"  Item count: {stats['count']}")
    print(f"  Load factor: {stats['load_factor']:.3f}")
    print(f"  Collisions: {stats['collision_count']}")
    print(f"  Max chain length: {stats['max_chain_length']}")
    print(f"  Average chain length: {stats['avg_chain_length']:.2f}")
    print(f"  Resize count: {stats['resize_count']}")


def main():
    """Run all demonstrations."""
    print("HASH TABLE WITH CHAINING - COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases a hash table implementation using chaining")
    print("for collision resolution with a universal hash function family.")
    print()
    
    # Run all demonstrations
    demo_basic_operations()
    demo_collision_scenarios()
    demo_resize_behavior()
    demo_performance_analysis()
    demo_hash_distribution()
    demo_different_data_types()
    demo_stress_test()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key features demonstrated:")
    print("✓ Universal hash function family")
    print("✓ Chaining collision resolution")
    print("✓ Dynamic resizing")
    print("✓ Insert, search, delete operations")
    print("✓ Multiple data types support")
    print("✓ Performance characteristics")
    print("✓ Stress testing capabilities")


if __name__ == "__main__":
    main()