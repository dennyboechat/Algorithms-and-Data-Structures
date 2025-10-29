"""
Demonstration Script for Randomized Quicksort

This script demonstrates the randomized quicksort implementation with
various test cases and provides detailed output showing the algorithm's behavior.
"""

import random
import time
from randomized_quicksort import (
    RandomizedQuicksort, 
    RandomizedQuicksort3Way, 
    is_sorted, 
    generate_test_arrays
)


def demonstrate_basic_functionality():
    """Demonstrate basic sorting functionality."""
    print("🚀 Basic Functionality Demonstration")
    print("=" * 50)
    
    sorter = RandomizedQuicksort(seed=42)
    
    # Test various array types
    test_cases = {
        "Random Array": [64, 34, 25, 12, 22, 11, 90, 5, 77, 30],
        "Already Sorted": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Reverse Sorted": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "With Duplicates": [3, 7, 3, 1, 7, 9, 3, 7, 1, 9],
        "All Same": [5, 5, 5, 5, 5],
        "Negative Numbers": [-5, 3, -1, 0, 8, -3, 2],
        "Single Element": [42],
        "Empty Array": []
    }
    
    for name, arr in test_cases.items():
        print(f"\n📋 Test Case: {name}")
        print(f"   Input:  {arr}")
        
        if arr:  # Non-empty array
            result = sorter.sort(arr)
            comparisons, swaps = sorter.get_statistics()
            
            print(f"   Output: {result}")
            print(f"   ✓ Sorted: {is_sorted(result)}")
            print(f"   📊 Stats: {comparisons} comparisons, {swaps} swaps")
        else:
            result = sorter.sort(arr)
            print(f"   Output: {result}")
            print(f"   ✓ Sorted: {is_sorted(result)}")


def demonstrate_randomization_effect():
    """Demonstrate the effect of randomization on performance."""
    print("\n\n🎲 Randomization Effect Demonstration")
    print("=" * 50)
    
    # Test array that could be problematic for deterministic quicksort
    test_array = list(range(1, 21))  # Sorted array: worst case for basic quicksort
    
    print(f"Test Array: {test_array}")
    print("\nComparing different random seeds:")
    
    seeds = [1, 42, 100, 2023, 12345]
    
    for seed in seeds:
        sorter = RandomizedQuicksort(seed=seed)
        start_time = time.perf_counter()
        result = sorter.sort(test_array)
        end_time = time.perf_counter()
        
        comparisons, swaps = sorter.get_statistics()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"  Seed {seed:5d}: {comparisons:2d} comparisons, {swaps:2d} swaps, "
              f"{execution_time:.3f}ms")
        
        # Verify correctness
        assert is_sorted(result), f"Sorting failed for seed {seed}"
    
    print("\n💡 Notice how different seeds lead to different numbers of operations!")
    print("   This demonstrates that randomization helps avoid worst-case scenarios.")


def demonstrate_3way_partitioning():
    """Demonstrate the benefits of 3-way partitioning."""
    print("\n\n🔀 3-Way Partitioning Demonstration")
    print("=" * 50)
    
    # Create arrays with many duplicates
    test_cases = [
        {
            "name": "Many Duplicates",
            "array": [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
        },
        {
            "name": "Few Unique Values", 
            "array": [random.choice([1, 2, 3, 4, 5]) for _ in range(20)]
        },
        {
            "name": "Mostly Same Value",
            "array": [7] * 15 + [1, 2, 3, 8, 9]
        }
    ]
    
    standard_sorter = RandomizedQuicksort(seed=42)
    threeway_sorter = RandomizedQuicksort3Way(seed=42)
    
    for test_case in test_cases:
        name = test_case["name"]
        arr = test_case["array"]
        
        print(f"\n📊 Test Case: {name}")
        print(f"   Array: {arr}")
        
        # Test standard quicksort
        result_std = standard_sorter.sort(arr)
        comp_std, swap_std = standard_sorter.get_statistics()
        
        # Test 3-way quicksort
        result_3way = threeway_sorter.sort(arr)
        comp_3way, swap_3way = threeway_sorter.get_statistics()
        
        print(f"   Standard: {comp_std:2d} comparisons, {swap_std:2d} swaps")
        print(f"   3-way:    {comp_3way:2d} comparisons, {swap_3way:2d} swaps")
        
        # Calculate improvement
        if comp_std > 0:
            improvement = (comp_std - comp_3way) / comp_std * 100
            print(f"   💚 Improvement: {improvement:.1f}% fewer comparisons with 3-way")
        
        # Verify both produce same result
        assert result_std == result_3way, "Results don't match!"
        assert is_sorted(result_std), "Standard result not sorted!"
        assert is_sorted(result_3way), "3-way result not sorted!"


def demonstrate_edge_cases():
    """Demonstrate handling of edge cases."""
    print("\n\n⚠️  Edge Cases Demonstration")
    print("=" * 50)
    
    sorter = RandomizedQuicksort()
    
    edge_cases = [
        ("Empty array", []),
        ("Single element", [42]),
        ("Two elements (sorted)", [1, 2]),
        ("Two elements (reverse)", [2, 1]),
        ("All zeros", [0, 0, 0, 0]),
        ("Large numbers", [999999, 1000000, 999998]),
        ("Negative numbers", [-10, -5, -1, -100]),
        ("Mixed signs", [-3, 0, 5, -1, 2]),
        ("Very large range", [-1000000, 1000000, 0])
    ]
    
    for name, arr in edge_cases:
        print(f"\n🧪 {name}:")
        print(f"   Input:  {arr}")
        
        result = sorter.sort(arr)
        print(f"   Output: {result}")
        print(f"   ✓ Sorted: {is_sorted(result)}")
        
        # Additional verification
        if arr:
            assert len(result) == len(arr), "Length mismatch!"
            assert sorted(arr) == result, "Incorrect sorting!"


def demonstrate_performance_scaling():
    """Demonstrate performance scaling with different array sizes."""
    print("\n\n📈 Performance Scaling Demonstration")
    print("=" * 50)
    
    sorter = RandomizedQuicksort(seed=42)
    sizes = [10, 50, 100, 500, 1000, 2000]
    
    print("Array Size | Time (ms) | Comparisons | Swaps | Operations/Element")
    print("-" * 65)
    
    for size in sizes:
        # Generate random array
        test_array = [random.randint(1, size) for _ in range(size)]
        
        # Measure performance
        start_time = time.perf_counter()
        result = sorter.sort(test_array)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000  # milliseconds
        comparisons, swaps = sorter.get_statistics()
        operations_per_element = (comparisons + swaps) / size
        
        print(f"{size:10d} | {execution_time:8.3f} | {comparisons:11d} | "
              f"{swaps:5d} | {operations_per_element:17.1f}")
        
        # Verify correctness
        assert is_sorted(result), f"Sorting failed for size {size}"
    
    print("\n💡 Notice how the algorithm scales efficiently!")
    print("   The operations per element grows logarithmically, not linearly.")


def demonstrate_stability_and_properties():
    """Demonstrate algorithm properties."""
    print("\n\n🔍 Algorithm Properties Demonstration")
    print("=" * 50)
    
    sorter = RandomizedQuicksort()
    
    # Test in-place vs copy behavior
    print("🔄 In-place vs Copy Behavior:")
    original = [5, 2, 8, 1, 9]
    original_copy = original.copy()
    
    # Test copy version
    sorted_copy = sorter.sort(original)
    print(f"   Original after sort(): {original}")
    print(f"   Returned array:         {sorted_copy}")
    print(f"   Original unchanged: {original == original_copy}")
    
    # Test in-place version
    sorter.sort_inplace(original)
    print(f"   Original after sort_inplace(): {original}")
    print(f"   In-place sorting works: {is_sorted(original)}")
    
    print("\n🎯 Determinism with Fixed Seed:")
    test_array = [random.randint(1, 20) for _ in range(10)]
    
    # Sort with same seed multiple times
    results = []
    for i in range(3):
        sorter_fixed = RandomizedQuicksort(seed=42)
        result = sorter_fixed.sort(test_array)
        comparisons, swaps = sorter_fixed.get_statistics()
        results.append((result, comparisons, swaps))
        print(f"   Run {i+1}: {comparisons} comparisons, {swaps} swaps")
    
    # Check if all runs produced same statistics (with fixed seed, they should)
    all_same = all(r[1:] == results[0][1:] for r in results)
    print(f"   Reproducible with fixed seed: {all_same}")


def main():
    """Main demonstration function."""
    print("🎯 RANDOMIZED QUICKSORT COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("This demonstration showcases the randomized quicksort implementation")
    print("with various test cases, edge cases, and performance characteristics.")
    print("=" * 60)
    
    # Set global seed for reproducible demonstrations
    random.seed(2023)
    
    # Run all demonstrations
    demonstrate_basic_functionality()
    demonstrate_randomization_effect()
    demonstrate_3way_partitioning()
    demonstrate_edge_cases()
    demonstrate_performance_scaling()
    demonstrate_stability_and_properties()
    
    print("\n\n✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print("• Randomized pivot selection prevents worst-case scenarios")
    print("• 3-way partitioning improves performance with duplicates")
    print("• Algorithm handles all edge cases correctly")
    print("• Performance scales well: O(n log n) average case")
    print("• Implementation is robust and efficient")
    print("• Both in-place and copy variants are available")


if __name__ == "__main__":
    main()