"""
Performance Analysis Utilities for Sorting Algorithms
Provides comprehensive performance measurement and dataset generation capabilities.
"""

import time
import psutil
import random
import tracemalloc
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import gc


@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics."""
    execution_time: float
    memory_peak: float  # Peak memory usage in MB
    memory_current: float  # Current memory usage in MB
    algorithm_stats: Dict[str, Any]  # Algorithm-specific statistics
    dataset_size: int
    dataset_type: str


class DatasetGenerator:
    """Generate various types of datasets for testing sorting algorithms."""
    
    @staticmethod
    def generate_random(size: int, min_val: int = 1, max_val: int = 10000) -> List[int]:
        """Generate a random dataset."""
        return [random.randint(min_val, max_val) for _ in range(size)]
    
    @staticmethod
    def generate_sorted(size: int, min_val: int = 1, max_val: int = 10000) -> List[int]:
        """Generate a sorted dataset."""
        if size <= 0:
            return []
        if size == 1:
            return [min_val]
        
        step = max(1, (max_val - min_val) // size)
        result = list(range(min_val, min_val + size * step, step))
        return result[:size]
    
    @staticmethod
    def generate_reverse_sorted(size: int, min_val: int = 1, max_val: int = 10000) -> List[int]:
        """Generate a reverse sorted dataset."""
        if size <= 0:
            return []
        if size == 1:
            return [max_val]
        
        step = max(1, (max_val - min_val) // size)
        result = list(range(max_val, max_val - size * step, -step))
        return result[:size]
    
    @staticmethod
    def generate_nearly_sorted(size: int, swap_percentage: float = 0.1) -> List[int]:
        """Generate a nearly sorted dataset with some random swaps."""
        arr = DatasetGenerator.generate_sorted(size)
        num_swaps = int(size * swap_percentage)
        
        for _ in range(num_swaps):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        
        return arr
    
    @staticmethod
    def generate_duplicate_heavy(size: int, num_unique: int = None) -> List[int]:
        """Generate a dataset with many duplicate values."""
        if num_unique is None:
            num_unique = max(1, size // 10)  # 10% unique values by default
        
        unique_values = random.sample(range(1, 10000), min(num_unique, 9999))
        return [random.choice(unique_values) for _ in range(size)]


class PerformanceMeasurer:
    """Measure performance metrics for sorting algorithms."""
    
    @staticmethod
    def measure_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def measure_algorithm_performance(
        algorithm_instance,
        dataset: List[int],
        dataset_type: str
    ) -> PerformanceMetrics:
        """
        Measure comprehensive performance metrics for a sorting algorithm.
        
        Args:
            algorithm_instance: Instance of QuickSort or MergeSort
            dataset: The data to sort
            dataset_type: Description of the dataset type
        
        Returns:
            PerformanceMetrics object with all measured data
        """
        # Force garbage collection before measurement
        gc.collect()
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = PerformanceMeasurer.measure_memory_usage()
        
        # Measure execution time
        start_time = time.perf_counter()
        sorted_result = algorithm_instance.sort(dataset)
        end_time = time.perf_counter()
        
        # Get peak memory usage
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert bytes to MB
        peak_memory_mb = peak_memory / 1024 / 1024
        current_memory_mb = PerformanceMeasurer.measure_memory_usage()
        
        # Get algorithm-specific statistics
        algorithm_stats = algorithm_instance.get_performance_stats()
        
        # Verify the result is correctly sorted
        is_sorted = all(sorted_result[i] <= sorted_result[i + 1] 
                       for i in range(len(sorted_result) - 1))
        algorithm_stats['is_correctly_sorted'] = is_sorted
        
        return PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_peak=peak_memory_mb,
            memory_current=current_memory_mb - initial_memory,
            algorithm_stats=algorithm_stats,
            dataset_size=len(dataset),
            dataset_type=dataset_type
        )
    
    @staticmethod
    def compare_algorithms(
        algorithms: Dict[str, Any],
        datasets: Dict[str, List[int]],
        sizes: List[int] = None
    ) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """
        Compare multiple algorithms on multiple datasets.
        
        Args:
            algorithms: Dictionary of algorithm name -> algorithm instance
            datasets: Dictionary of dataset type -> dataset
            sizes: List of dataset sizes to test (if None, uses provided datasets)
        
        Returns:
            Nested dictionary: algorithm_name -> dataset_type -> PerformanceMetrics
        """
        if sizes is None:
            # Use provided datasets as-is
            results = {}
            for alg_name, alg_instance in algorithms.items():
                results[alg_name] = {}
                for dataset_type, dataset in datasets.items():
                    print(f"Testing {alg_name} on {dataset_type} dataset (size: {len(dataset)})...")
                    metrics = PerformanceMeasurer.measure_algorithm_performance(
                        alg_instance, dataset, dataset_type
                    )
                    results[alg_name][dataset_type] = metrics
            return results
        
        # Generate datasets of specified sizes
        generator = DatasetGenerator()
        results = {}
        
        for alg_name, alg_instance in algorithms.items():
            results[alg_name] = {}
            for size in sizes:
                for dataset_type in ['random', 'sorted', 'reverse_sorted', 'nearly_sorted']:
                    # Generate dataset
                    if dataset_type == 'random':
                        dataset = generator.generate_random(size)
                    elif dataset_type == 'sorted':
                        dataset = generator.generate_sorted(size)
                    elif dataset_type == 'reverse_sorted':
                        dataset = generator.generate_reverse_sorted(size)
                    elif dataset_type == 'nearly_sorted':
                        dataset = generator.generate_nearly_sorted(size)
                    
                    dataset_key = f"{dataset_type}_{size}"
                    print(f"Testing {alg_name} on {dataset_key}...")
                    
                    metrics = PerformanceMeasurer.measure_algorithm_performance(
                        alg_instance, dataset, dataset_key
                    )
                    results[alg_name][dataset_key] = metrics
        
        return results


def print_performance_summary(results: Dict[str, Dict[str, PerformanceMetrics]]):
    """Print a formatted summary of performance results."""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    for alg_name, alg_results in results.items():
        print(f"\n{alg_name.upper()} ALGORITHM:")
        print("-" * 50)
        
        for dataset_key, metrics in alg_results.items():
            print(f"\nDataset: {dataset_key}")
            print(f"  Dataset Size: {metrics.dataset_size:,} elements")
            print(f"  Execution Time: {metrics.execution_time:.6f} seconds")
            print(f"  Peak Memory: {metrics.memory_peak:.2f} MB")
            print(f"  Memory Delta: {metrics.memory_current:.2f} MB")
            print(f"  Correctly Sorted: {metrics.algorithm_stats.get('is_correctly_sorted', 'Unknown')}")
            
            # Print algorithm-specific stats
            for stat_name, stat_value in metrics.algorithm_stats.items():
                if stat_name != 'is_correctly_sorted':
                    print(f"  {stat_name.replace('_', ' ').title()}: {stat_value:,}")


if __name__ == "__main__":
    # Test the utilities with smaller datasets
    from sorting_algorithms import QuickSort, MergeSort
    
    # Generate test datasets
    generator = DatasetGenerator()
    test_datasets = {
        'random_100': generator.generate_random(100),
        'sorted_100': generator.generate_sorted(100),
        'reverse_100': generator.generate_reverse_sorted(100),
        'nearly_sorted_100': generator.generate_nearly_sorted(100),
    }
    
    # Initialize algorithms
    algorithms = {
        'QuickSort': QuickSort(),
        'MergeSort': MergeSort()
    }
    
    # Run comparison
    results = PerformanceMeasurer.compare_algorithms(algorithms, test_datasets)
    
    # Print results
    print_performance_summary(results)