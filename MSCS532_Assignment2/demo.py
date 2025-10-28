"""
Quick Demonstration of Sorting Algorithm Performance Analysis
This script provides a quick overview of the implemented algorithms and their performance.
"""

from sorting_algorithms import QuickSort, MergeSort, verify_sorted
from performance_analysis import DatasetGenerator, PerformanceMeasurer
import time


def demonstrate_basic_sorting():
    """Demonstrate basic sorting functionality."""
    print("="*60)
    print("BASIC SORTING DEMONSTRATION")
    print("="*60)
    
    # Test data
    test_data = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 1, 42]
    print(f"Original array: {test_data}")
    print(f"Array size: {len(test_data)} elements")
    print()
    
    # Test Quick Sort
    print("Quick Sort Results:")
    print("-" * 20)
    quick_sort = QuickSort()
    start_time = time.perf_counter()
    sorted_quick = quick_sort.sort(test_data)
    end_time = time.perf_counter()
    
    print(f"Sorted array: {sorted_quick}")
    print(f"Execution time: {(end_time - start_time)*1000:.3f} ms")
    print(f"Verification: {verify_sorted(sorted_quick)}")
    stats = quick_sort.get_performance_stats()
    print(f"Comparisons: {stats['comparisons']}")
    print(f"Swaps: {stats['swaps']}")
    print(f"Recursive calls: {stats['recursive_calls']}")
    print()
    
    # Test Merge Sort
    print("Merge Sort Results:")
    print("-" * 20)
    merge_sort = MergeSort()
    start_time = time.perf_counter()
    sorted_merge = merge_sort.sort(test_data)
    end_time = time.perf_counter()
    
    print(f"Sorted array: {sorted_merge}")
    print(f"Execution time: {(end_time - start_time)*1000:.3f} ms")
    print(f"Verification: {verify_sorted(sorted_merge)}")
    stats = merge_sort.get_performance_stats()
    print(f"Comparisons: {stats['comparisons']}")
    print(f"Merges: {stats['merges']}")
    print(f"Recursive calls: {stats['recursive_calls']}")


def demonstrate_dataset_generation():
    """Demonstrate different dataset types."""
    print("\n" + "="*60)
    print("DATASET GENERATION DEMONSTRATION")
    print("="*60)
    
    generator = DatasetGenerator()
    size = 15
    
    datasets = {
        'Random': generator.generate_random(size, 1, 50),
        'Sorted': generator.generate_sorted(size, 1, 50),
        'Reverse Sorted': generator.generate_reverse_sorted(size, 1, 50),
        'Nearly Sorted': generator.generate_nearly_sorted(size, 0.2),  # 20% swaps
        'Duplicate Heavy': generator.generate_duplicate_heavy(size, 5)  # 5 unique values
    }
    
    for dataset_name, dataset in datasets.items():
        print(f"{dataset_name:15}: {dataset}")


def demonstrate_performance_comparison():
    """Demonstrate performance comparison on different dataset types."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON DEMONSTRATION")
    print("="*60)
    
    generator = DatasetGenerator()
    algorithms = {
        'QuickSort': QuickSort(),
        'MergeSort': MergeSort()
    }
    
    # Test on different dataset types with size 1000
    size = 1000
    test_cases = [
        ('Random Data', generator.generate_random(size)),
        ('Sorted Data', generator.generate_sorted(size)),
        ('Reverse Sorted', generator.generate_reverse_sorted(size)),
        ('Nearly Sorted', generator.generate_nearly_sorted(size, 0.05))
    ]
    
    print(f"Testing with {size} elements...")
    print(f"{'Dataset Type':<15} {'QuickSort (ms)':<15} {'MergeSort (ms)':<15} {'Winner':<10}")
    print("-" * 65)
    
    for test_name, dataset in test_cases:
        quick_time = 0
        merge_time = 0
        
        # Test QuickSort
        quick_sort = QuickSort()
        metrics = PerformanceMeasurer.measure_algorithm_performance(
            quick_sort, dataset, test_name
        )
        quick_time = metrics.execution_time * 1000  # Convert to ms
        
        # Test MergeSort
        merge_sort = MergeSort()
        metrics = PerformanceMeasurer.measure_algorithm_performance(
            merge_sort, dataset, test_name
        )
        merge_time = metrics.execution_time * 1000  # Convert to ms
        
        winner = "QuickSort" if quick_time < merge_time else "MergeSort"
        
        print(f"{test_name:<15} {quick_time:<15.3f} {merge_time:<15.3f} {winner:<10}")


def demonstrate_memory_analysis():
    """Demonstrate memory usage analysis."""
    print("\n" + "="*60)
    print("MEMORY USAGE DEMONSTRATION")
    print("="*60)
    
    generator = DatasetGenerator()
    size = 5000
    dataset = generator.generate_random(size)
    
    print(f"Testing memory usage with {size} elements...")
    print(f"{'Algorithm':<12} {'Peak Memory (MB)':<18} {'Memory Delta (MB)':<18}")
    print("-" * 50)
    
    # Test QuickSort memory
    quick_sort = QuickSort()
    metrics = PerformanceMeasurer.measure_algorithm_performance(
        quick_sort, dataset, "memory_test"
    )
    print(f"{'QuickSort':<12} {metrics.memory_peak:<18.3f} {metrics.memory_current:<18.3f}")
    
    # Test MergeSort memory
    merge_sort = MergeSort()
    metrics = PerformanceMeasurer.measure_algorithm_performance(
        merge_sort, dataset, "memory_test"
    )
    print(f"{'MergeSort':<12} {metrics.memory_peak:<18.3f} {metrics.memory_current:<18.3f}")


def demonstrate_scalability():
    """Demonstrate algorithm scalability with different sizes."""
    print("\n" + "="*60)
    print("SCALABILITY DEMONSTRATION")
    print("="*60)
    
    generator = DatasetGenerator()
    sizes = [100, 500, 1000, 2500, 5000]
    
    print("Quick Sort Performance Scaling:")
    print(f"{'Size':<8} {'Time (ms)':<12} {'Comparisons':<12} {'Swaps':<8}")
    print("-" * 40)
    
    for size in sizes:
        dataset = generator.generate_random(size)
        quick_sort = QuickSort()
        metrics = PerformanceMeasurer.measure_algorithm_performance(
            quick_sort, dataset, f"scale_test_{size}"
        )
        
        time_ms = metrics.execution_time * 1000
        stats = metrics.algorithm_stats
        print(f"{size:<8} {time_ms:<12.3f} {stats['comparisons']:<12} {stats['swaps']:<8}")
    
    print("\nMerge Sort Performance Scaling:")
    print(f"{'Size':<8} {'Time (ms)':<12} {'Comparisons':<12} {'Merges':<8}")
    print("-" * 40)
    
    for size in sizes:
        dataset = generator.generate_random(size)
        merge_sort = MergeSort()
        metrics = PerformanceMeasurer.measure_algorithm_performance(
            merge_sort, dataset, f"scale_test_{size}"
        )
        
        time_ms = metrics.execution_time * 1000
        stats = metrics.algorithm_stats
        print(f"{size:<8} {time_ms:<12.3f} {stats['comparisons']:<12} {stats['merges']:<8}")


def main():
    """Run all demonstrations."""
    print("SORTING ALGORITHM PERFORMANCE ANALYSIS - DEMONSTRATION")
    print("="*60)
    print("This demonstration showcases the implemented Quick Sort and Merge Sort")
    print("algorithms with comprehensive performance analysis capabilities.")
    print()
    
    try:
        demonstrate_basic_sorting()
        demonstrate_dataset_generation()
        demonstrate_performance_comparison()
        demonstrate_memory_analysis()
        demonstrate_scalability()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("For comprehensive analysis, run: python main_analysis.py")
        print("For visualizations, run: python visualization.py")
        print("See README.md for detailed documentation.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()