"""
Comprehensive Sorting Algorithm Performance Comparison
Runs Quick Sort and Merge Sort on various datasets and analyzes performance.
"""

import sys
import json
from datetime import datetime
from typing import Dict, List
from sorting_algorithms import QuickSort, MergeSort
from performance_analysis import (
    DatasetGenerator, 
    PerformanceMeasurer, 
    PerformanceMetrics,
    print_performance_summary
)


def save_results_to_json(results: Dict, filename: str = None):
    """Save performance results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_results_{timestamp}.json"
    
    # Convert PerformanceMetrics objects to dictionaries for JSON serialization
    json_results = {}
    for alg_name, alg_results in results.items():
        json_results[alg_name] = {}
        for dataset_key, metrics in alg_results.items():
            json_results[alg_name][dataset_key] = {
                'execution_time': metrics.execution_time,
                'memory_peak': metrics.memory_peak,
                'memory_current': metrics.memory_current,
                'algorithm_stats': metrics.algorithm_stats,
                'dataset_size': metrics.dataset_size,
                'dataset_type': metrics.dataset_type
            }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def run_comprehensive_analysis():
    """Run a comprehensive performance analysis."""
    print("Starting Comprehensive Sorting Algorithm Analysis")
    print("=" * 60)
    
    # Initialize algorithms
    algorithms = {
        'QuickSort': QuickSort(),
        'MergeSort': MergeSort()
    }
    
    # Initialize dataset generator
    generator = DatasetGenerator()
    
    # Define test configurations - using smaller sizes for demonstration
    test_sizes = [100, 500, 1000, 2500, 5000]
    dataset_types = ['random', 'sorted', 'reverse_sorted', 'nearly_sorted', 'duplicate_heavy']
    
    all_results = {}
    
    print(f"Testing with dataset sizes: {test_sizes}")
    print(f"Dataset types: {dataset_types}")
    print()
    
    # Run tests for each algorithm
    for alg_name, alg_instance in algorithms.items():
        print(f"\nTesting {alg_name}...")
        print("-" * 40)
        
        all_results[alg_name] = {}
        
        for size in test_sizes:
            for dataset_type in dataset_types:
                # Generate appropriate dataset
                if dataset_type == 'random':
                    dataset = generator.generate_random(size)
                elif dataset_type == 'sorted':
                    dataset = generator.generate_sorted(size)
                elif dataset_type == 'reverse_sorted':
                    dataset = generator.generate_reverse_sorted(size)
                elif dataset_type == 'nearly_sorted':
                    dataset = generator.generate_nearly_sorted(size, 0.05)  # 5% swaps
                elif dataset_type == 'duplicate_heavy':
                    dataset = generator.generate_duplicate_heavy(size, max(1, size // 20))  # 5% unique
                
                dataset_key = f"{dataset_type}_{size}"
                
                try:
                    print(f"  Processing {dataset_key}...")
                    
                    # Measure performance
                    metrics = PerformanceMeasurer.measure_algorithm_performance(
                        alg_instance, dataset, dataset_key
                    )
                    
                    all_results[alg_name][dataset_key] = metrics
                    
                    # Quick verification
                    if not metrics.algorithm_stats.get('is_correctly_sorted', False):
                        print(f"    WARNING: {alg_name} failed to sort {dataset_key} correctly!")
                    
                except Exception as e:
                    print(f"    ERROR: Failed to process {dataset_key} with {alg_name}: {e}")
                    continue
    
    return all_results


def analyze_and_compare_results(results: Dict):
    """Analyze and compare the results."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Print comprehensive summary
    print_performance_summary(results)
    
    # Performance comparison analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Compare algorithms on each dataset type
    dataset_keys = set()
    for alg_results in results.values():
        dataset_keys.update(alg_results.keys())
    
    dataset_keys = sorted(list(dataset_keys))
    
    print("\nExecution Time Comparison (seconds):")
    print("-" * 50)
    print(f"{'Dataset':<20} {'QuickSort':<12} {'MergeSort':<12} {'Winner':<10}")
    print("-" * 54)
    
    quicksort_wins = 0
    mergesort_wins = 0
    
    for dataset_key in dataset_keys:
        if dataset_key in results.get('QuickSort', {}) and dataset_key in results.get('MergeSort', {}):
            qs_time = results['QuickSort'][dataset_key].execution_time
            ms_time = results['MergeSort'][dataset_key].execution_time
            
            winner = "QuickSort" if qs_time < ms_time else "MergeSort"
            if qs_time < ms_time:
                quicksort_wins += 1
            else:
                mergesort_wins += 1
            
            print(f"{dataset_key:<20} {qs_time:<12.6f} {ms_time:<12.6f} {winner:<10}")
    
    print("-" * 54)
    print(f"Overall wins: QuickSort={quicksort_wins}, MergeSort={mergesort_wins}")
    
    # Memory usage comparison
    print("\nPeak Memory Usage Comparison (MB):")
    print("-" * 50)
    print(f"{'Dataset':<20} {'QuickSort':<12} {'MergeSort':<12} {'Winner':<10}")
    print("-" * 54)
    
    for dataset_key in dataset_keys:
        if dataset_key in results.get('QuickSort', {}) and dataset_key in results.get('MergeSort', {}):
            qs_mem = results['QuickSort'][dataset_key].memory_peak
            ms_mem = results['MergeSort'][dataset_key].memory_peak
            
            winner = "QuickSort" if qs_mem < ms_mem else "MergeSort"
            
            print(f"{dataset_key:<20} {qs_mem:<12.2f} {ms_mem:<12.2f} {winner:<10}")
    
    # Algorithm-specific insights
    print("\n" + "="*80)
    print("ALGORITHM-SPECIFIC INSIGHTS")
    print("="*80)
    
    # Analyze QuickSort performance
    if 'QuickSort' in results:
        print("\nQuickSort Analysis:")
        print("-" * 30)
        worst_case = None
        best_case = None
        
        for dataset_key, metrics in results['QuickSort'].items():
            if worst_case is None or metrics.execution_time > worst_case[1].execution_time:
                worst_case = (dataset_key, metrics)
            if best_case is None or metrics.execution_time < best_case[1].execution_time:
                best_case = (dataset_key, metrics)
        
        if worst_case and best_case:
            print(f"Best performance: {best_case[0]} ({best_case[1].execution_time:.6f}s)")
            print(f"Worst performance: {worst_case[0]} ({worst_case[1].execution_time:.6f}s)")
            ratio = worst_case[1].execution_time / best_case[1].execution_time
            print(f"Performance ratio (worst/best): {ratio:.2f}x")
    
    # Analyze MergeSort performance
    if 'MergeSort' in results:
        print("\nMergeSort Analysis:")
        print("-" * 30)
        times = [metrics.execution_time for metrics in results['MergeSort'].values()]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"Average execution time: {avg_time:.6f}s")
            print(f"Min execution time: {min_time:.6f}s")
            print(f"Max execution time: {max_time:.6f}s")
            print(f"Consistency ratio (max/min): {max_time/min_time:.2f}x")


def main():
    """Main function to run the complete analysis."""
    print("Sorting Algorithm Performance Analysis")
    print("=====================================")
    print("This script will test Quick Sort and Merge Sort algorithms")
    print("on various datasets and provide comprehensive performance analysis.\n")
    
    try:
        # Run comprehensive analysis
        results = run_comprehensive_analysis()
        
        # Save results to JSON
        json_filename = save_results_to_json(results)
        
        # Analyze and compare results
        analyze_and_compare_results(results)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results have been saved to: {json_filename}")
        print("You can use this data for further analysis or visualization.")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()