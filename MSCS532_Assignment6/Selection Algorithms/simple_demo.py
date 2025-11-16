"""
Simple Example: Deterministic vs Randomized Selection
MSCS532 Assignment 6

This script provides a simple demonstration of both selection algorithms
for easy comparison and understanding.

Author: Assignment 6 Implementation
Date: November 16, 2025
"""

import time
import random
from median_of_medians import median_of_medians_select
from randomized_quickselect import randomized_select

def simple_comparison():
    """Simple side-by-side comparison of both algorithms."""
    print("=" * 60)
    print("SIMPLE COMPARISON: DETERMINISTIC vs RANDOMIZED SELECTION")
    print("=" * 60)
    
    # Example array
    arr = [64, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    k = 5  # Find 5th smallest
    
    print(f"\nArray: {arr}")
    print(f"Sorted: {sorted(arr)}")
    print(f"Looking for the {k}th smallest element...\n")
    
    # Deterministic approach
    print("🔒 DETERMINISTIC (Median of Medians):")
    start = time.time()
    result_det = median_of_medians_select(arr, k)
    time_det = time.time() - start
    print(f"   Result: {result_det}")
    print(f"   Time: {time_det:.6f} seconds")
    print(f"   Guarantee: O(n) worst-case")
    print(f"   Behavior: Always consistent")
    
    print()
    
    # Randomized approach
    print("🎲 RANDOMIZED (Quickselect):")
    start = time.time()
    result_rand = randomized_select(arr, k)
    time_rand = time.time() - start
    print(f"   Result: {result_rand}")
    print(f"   Time: {time_rand:.6f} seconds")
    print(f"   Guarantee: O(n) expected, O(n²) worst-case")
    print(f"   Behavior: Usually faster, small variance")
    
    print()
    
    # Verification
    print("✅ VERIFICATION:")
    print(f"   Both algorithms agree: {result_det == result_rand}")
    print(f"   Correct answer: {sorted(arr)[k-1]}")
    if time_rand < time_det:
        print(f"   Randomized was {time_det/time_rand:.2f}x faster")
    else:
        print(f"   Deterministic was {time_rand/time_det:.2f}x faster")


def multiple_test_cases():
    """Test both algorithms on multiple scenarios."""
    print("\n" + "=" * 60)
    print("MULTIPLE TEST CASES")
    print("=" * 60)
    
    test_cases = [
        ("Small array", [3, 1, 4, 1, 5, 9, 2], 3),
        ("With duplicates", [5, 5, 5, 3, 3, 1, 7], 4),
        ("Already sorted", [1, 2, 3, 4, 5, 6, 7, 8], 6),
        ("Random medium", [random.randint(1, 100) for _ in range(20)], 10)
    ]
    
    for name, arr, k in test_cases:
        print(f"\n📊 Test Case: {name}")
        print(f"   Array size: {len(arr)}, Finding {k}th smallest")
        
        # Run both algorithms
        result_det = median_of_medians_select(arr, k)
        result_rand = randomized_select(arr, k)
        
        print(f"   Deterministic result: {result_det}")
        print(f"   Randomized result: {result_rand}")
        print(f"   Results match: {'✅' if result_det == result_rand else '❌'}")


def performance_showcase():
    """Showcase performance characteristics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SHOWCASE")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 2000]
    
    print(f"{'Size':<8} {'Deterministic':<15} {'Randomized':<15} {'Winner':<15}")
    print("-" * 60)
    
    for size in sizes:
        arr = [random.randint(1, 1000) for _ in range(size)]
        k = size // 2
        
        # Time deterministic
        start = time.time()
        median_of_medians_select(arr, k)
        time_det = time.time() - start
        
        # Time randomized
        start = time.time()
        randomized_select(arr, k)
        time_rand = time.time() - start
        
        winner = "Randomized" if time_rand < time_det else "Deterministic"
        ratio = max(time_det, time_rand) / min(time_det, time_rand)
        
        print(f"{size:<8} {time_det*1000:<15.3f} {time_rand*1000:<15.3f} {winner} ({ratio:.1f}x)")
    
    print("\nNote: Times in milliseconds")


def practical_advice():
    """Provide practical advice for choosing algorithms."""
    print("\n" + "=" * 60)
    print("PRACTICAL ADVICE")
    print("=" * 60)
    
    print("\n🎯 CHOOSE DETERMINISTIC WHEN:")
    print("   • You need guaranteed worst-case performance")
    print("   • Working with real-time or safety-critical systems")
    print("   • Facing potentially adversarial inputs")
    print("   • Consistency is more important than speed")
    
    print("\n🎯 CHOOSE RANDOMIZED WHEN:")
    print("   • Average performance is most important")
    print("   • Working with general-purpose applications")
    print("   • Implementation simplicity is valued")
    print("   • Memory usage should be minimized (iterative version)")
    
    print("\n📊 PERFORMANCE SUMMARY:")
    print("   • Randomized: Typically 1.5-2x faster in practice")
    print("   • Deterministic: Consistent performance, no surprises")
    print("   • Both: O(n) complexity for selection problems")
    print("   • Space: Both use O(log n) space due to recursion")


if __name__ == "__main__":
    # Set seed for reproducible randomized results in demo
    random.seed(42)
    
    # Run all demonstrations
    simple_comparison()
    multiple_test_cases()
    performance_showcase()
    practical_advice()
    
    print("\n" + "=" * 60)
    print("🎉 CONCLUSION")
    print("=" * 60)
    print("Both algorithms solve the selection problem efficiently!")
    print("Choose based on your specific requirements:")
    print("• Deterministic for guarantees")
    print("• Randomized for performance")
    print("=" * 60)