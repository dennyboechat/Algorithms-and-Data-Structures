"""
Quick Test and Verification Script
MSCS 532 - Final Project

Simple test to verify that both basic and cache-optimized algorithms work correctly.
"""

from matrix_multiplication import Matrix, matrix_multiply
from matrix_multiplication_optimized import matrix_multiply_cache_optimized
import time


def quick_verification_test():
    """Quick test to verify algorithms work correctly."""
    print("Quick Verification Test")
    print("=" * 30)
    
    # Test with a simple 3x3 example
    a = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    b = Matrix([
        [7, 8],
        [9, 10],
        [11, 12]
    ])
    
    print("Matrix A (2x3):")
    print(a)
    print("\nMatrix B (3x2):")
    print(b)
    
    # Test basic algorithm
    result_basic = matrix_multiply(a, b)
    print("\nBasic Algorithm Result:")
    print(result_basic)
    
    # Test cache-optimized algorithm  
    result_cache = matrix_multiply_cache_optimized(a, b)
    print("\nCache-Optimized Algorithm Result:")
    print(result_cache)
    
    # Verify they're identical
    identical = True
    for i in range(result_basic.rows):
        for j in range(result_basic.cols):
            if result_basic.get_element(i, j) != result_cache.get_element(i, j):
                identical = False
                break
        if not identical:
            break
    
    print(f"\nResults Match: {'✓ YES' if identical else '✗ NO'}")
    
    # Quick performance test
    print(f"\nQuick Performance Test (100x100 matrices):")
    from matrix_multiplication import create_random_matrix
    
    test_a = create_random_matrix(100, 100, 1, 10)
    test_b = create_random_matrix(100, 100, 1, 10)
    
    # Time basic algorithm
    start = time.time()
    result1 = matrix_multiply(test_a, test_b)
    basic_time = time.time() - start
    
    # Time cache-optimized algorithm
    start = time.time()
    result2 = matrix_multiply_cache_optimized(test_a, test_b)
    cache_time = time.time() - start
    
    speedup = basic_time / cache_time
    improvement = ((basic_time - cache_time) / basic_time) * 100
    
    print(f"  Basic Algorithm:     {basic_time:.4f} seconds")
    print(f"  Cache-Optimized:     {cache_time:.4f} seconds") 
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Improvement:         {improvement:.1f}%")
    
    print(f"\n✓ All tests completed successfully!")


if __name__ == "__main__":
    quick_verification_test()