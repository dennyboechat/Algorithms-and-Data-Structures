"""
Heapsort Algorithm Demonstration

This script provides an interactive demonstration of the Heapsort algorithm,
showing step-by-step execution, visualizations, and performance analysis.
"""

import random
import time
from typing import List, Tuple
from heapsort import heapsort, heapsort_inplace, build_max_heap, heapify, is_max_heap

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def visualize_array(arr: List[int], title: str = "Array") -> None:
    """
    Create a simple text-based visualization of an array.
    
    Args:
        arr: The array to visualize
        title: Title for the visualization
    """
    print(f"\n{title}:")
    if not arr:
        print("  (empty)")
        return
    
    print(f"  Values: {arr}")
    print(f"  Indices:", end="")
    for i in range(len(arr)):
        print(f" {i:2d}", end="")
    print()


def demonstrate_heap_building(arr: List[int]) -> None:
    """
    Demonstrate the heap building process step by step.
    
    Args:
        arr: The array to build a heap from
    """
    print("\n" + "="*60)
    print("HEAP BUILDING DEMONSTRATION")
    print("="*60)
    
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    visualize_array(arr_copy, "Original Array")
    
    print(f"\nBuilding max-heap from the bottom up...")
    print(f"Starting from last non-leaf node at index {n//2 - 1}")
    
    # Show each step of heap building
    for i in range(n // 2 - 1, -1, -1):
        print(f"\nStep: Heapifying subtree rooted at index {i}")
        print(f"Before heapify: {arr_copy}")
        
        # Apply heapify and show the change
        heapify(arr_copy, n, i)
        print(f"After heapify:  {arr_copy}")
        print(f"Is valid heap so far: {is_max_heap(arr_copy)}")
    
    visualize_array(arr_copy, "Final Max-Heap")
    print(f"Heap property satisfied: {is_max_heap(arr_copy)}")


def demonstrate_sorting_process(arr: List[int]) -> None:
    """
    Demonstrate the complete heapsort process step by step.
    
    Args:
        arr: The array to sort
    """
    print("\n" + "="*60)
    print("HEAPSORT PROCESS DEMONSTRATION")
    print("="*60)
    
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    visualize_array(arr_copy, "Original Array")
    
    # Build the heap first
    print("\nStep 1: Building max-heap...")
    build_max_heap(arr_copy)
    visualize_array(arr_copy, "Max-Heap Built")
    
    print("\nStep 2: Extracting maximum elements...")
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        print(f"\nIteration {n - i}:")
        print(f"  Current heap: {arr_copy[:i+1]} | Sorted: {arr_copy[i+1:]}")
        print(f"  Extracting maximum: {arr_copy[0]}")
        
        # Move current root to end
        arr_copy[0], arr_copy[i] = arr_copy[i], arr_copy[0]
        print(f"  After extraction: {arr_copy[:i]} | Sorted: {arr_copy[i:]}")
        
        # Restore heap property
        heapify(arr_copy, i, 0)
        print(f"  After heapify: {arr_copy[:i]} | Sorted: {arr_copy[i:]}")
    
    visualize_array(arr_copy, "Final Sorted Array")


def performance_analysis(sizes: List[int]) -> None:
    """
    Analyze heapsort performance across different input sizes.
    
    Args:
        sizes: List of array sizes to test
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    heapsort_times = []
    builtin_times = []
    
    for size in sizes:
        print(f"\nTesting with {size} elements...")
        
        # Generate random array
        random.seed(42)
        arr = [random.randint(1, 1000) for _ in range(size)]
        
        # Test heapsort
        start_time = time.time()
        heapsort(arr)
        heapsort_time = time.time() - start_time
        heapsort_times.append(heapsort_time)
        
        # Test built-in sort
        start_time = time.time()
        sorted(arr)
        builtin_time = time.time() - start_time
        builtin_times.append(builtin_time)
        
        print(f"  Heapsort: {heapsort_time:.6f}s")
        print(f"  Built-in: {builtin_time:.6f}s")
        print(f"  Ratio: {heapsort_time/builtin_time:.2f}x")
    
    # Create performance plot if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, heapsort_times, 'o-', label='Heapsort', linewidth=2, markersize=6)
        plt.plot(sizes, builtin_times, 's-', label='Built-in sort', linewidth=2, markersize=6)
        plt.xlabel('Array Size')
        plt.ylabel('Time (seconds)')
        plt.title('Heapsort vs Built-in Sort Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('/Users/dennyboechat/Applications/Algorithms-and-Data-Structures/MSCS532_Assignment4/Heapsort/performance_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print("\nPerformance plot saved as 'performance_comparison.png'")
    else:
        print("\nMatplotlib not available - skipping performance plot")


def test_different_input_types():
    """Test heapsort with different types of input arrays."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT INPUT TYPES")
    print("="*60)
    
    test_cases = [
        ("Random array", [64, 34, 25, 12, 22, 11, 90]),
        ("Already sorted", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("Reverse sorted", [9, 8, 7, 6, 5, 4, 3, 2, 1]),
        ("Many duplicates", [5, 5, 5, 2, 2, 8, 8, 8, 8]),
        ("All same", [3, 3, 3, 3, 3]),
        ("Single element", [42]),
        ("Empty array", []),
        ("Negative numbers", [-5, -1, -9, -3, -7]),
        ("Mixed pos/neg", [3, -1, 4, -5, 2, -3, 0]),
    ]
    
    for test_name, arr in test_cases:
        print(f"\n{test_name}:")
        print(f"  Original: {arr}")
        
        if arr:  # Only process non-empty arrays
            start_time = time.time()
            sorted_arr = heapsort(arr)
            end_time = time.time()
            
            print(f"  Sorted:   {sorted_arr}")
            print(f"  Time:     {(end_time - start_time)*1000:.3f} ms")
            print(f"  Correct:  {sorted_arr == sorted(arr)}")
        else:
            print("  (Empty array - no sorting needed)")


def interactive_demo():
    """Run an interactive demonstration of the heapsort algorithm."""
    print("="*80)
    print("WELCOME TO THE HEAPSORT ALGORITHM DEMONSTRATION")
    print("="*80)
    
    while True:
        print("\nChoose an option:")
        print("1. Demonstrate heap building process")
        print("2. Demonstrate complete sorting process")
        print("3. Test different input types")
        print("4. Performance analysis")
        print("5. Enter custom array")
        print("6. Run all tests")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '0':
            print("Thank you for exploring the Heapsort algorithm!")
            break
        elif choice == '1':
            arr = [64, 34, 25, 12, 22, 11, 90]
            demonstrate_heap_building(arr)
        elif choice == '2':
            arr = [64, 34, 25, 12, 22, 11, 90]
            demonstrate_sorting_process(arr)
        elif choice == '3':
            test_different_input_types()
        elif choice == '4':
            sizes = [100, 500, 1000, 2000, 5000]
            performance_analysis(sizes)
        elif choice == '5':
            try:
                user_input = input("Enter array elements separated by spaces: ")
                arr = [int(x) for x in user_input.split()]
                print(f"\nYou entered: {arr}")
                
                sub_choice = input("Choose: (1) Show heap building, (2) Show sorting process, (3) Both: ")
                if sub_choice in ['1', '3']:
                    demonstrate_heap_building(arr)
                if sub_choice in ['2', '3']:
                    demonstrate_sorting_process(arr)
            except ValueError:
                print("Invalid input! Please enter integers separated by spaces.")
        elif choice == '6':
            arr = [64, 34, 25, 12, 22, 11, 90]
            demonstrate_heap_building(arr)
            demonstrate_sorting_process(arr)
            test_different_input_types()
            sizes = [100, 500, 1000, 2000]
            performance_analysis(sizes)
        else:
            print("Invalid choice! Please enter a number between 0 and 6.")


if __name__ == "__main__":
    # Check if running interactively or as a script
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Run automatic demonstration
        print("Running automatic demonstration...")
        arr = [64, 34, 25, 12, 22, 11, 90]
        demonstrate_heap_building(arr)
        demonstrate_sorting_process(arr)
        test_different_input_types()
        
        # Quick performance test
        sizes = [100, 500, 1000]
        performance_analysis(sizes)
    else:
        # Run interactive demonstration
        interactive_demo()