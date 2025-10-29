"""
Performance Analysis for Randomized Quicksort

This module analyzes the performance characteristics of the randomized quicksort
implementation across different input types and sizes.
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from randomized_quicksort import RandomizedQuicksort, RandomizedQuicksort3Way


class PerformanceAnalyzer:
    """
    A class for analyzing the performance of randomized quicksort algorithms.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.results = {}
    
    def generate_test_data(self, size: int, data_type: str) -> List[int]:
        """
        Generate test data of different types.
        
        Args:
            size (int): Size of the array to generate
            data_type (str): Type of data ('random', 'sorted', 'reverse', 'duplicates')
            
        Returns:
            List[int]: Generated test array
        """
        if data_type == 'random':
            return [random.randint(1, size) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(1, size + 1))
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'duplicates':
            # Array with many duplicates
            unique_values = max(1, size // 10)
            return [random.randint(1, unique_values) for _ in range(size)]
        elif data_type == 'few_unique':
            # Array with very few unique values
            return [random.randint(1, 5) for _ in range(size)]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def measure_performance(self, algorithm, test_array: List[int], runs: int = 5) -> Dict:
        """
        Measure the performance of a sorting algorithm.
        
        Args:
            algorithm: The sorting algorithm instance
            test_array (List[int]): The array to sort
            runs (int): Number of runs to average over
            
        Returns:
            Dict: Performance metrics
        """
        times = []
        comparisons_list = []
        swaps_list = []
        
        for _ in range(runs):
            # Create a fresh copy for each run
            array_copy = test_array.copy()
            
            start_time = time.perf_counter()
            algorithm.sort(array_copy)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            comparisons, swaps = algorithm.get_statistics()
            comparisons_list.append(comparisons)
            swaps_list.append(swaps)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_comparisons': np.mean(comparisons_list),
            'std_comparisons': np.std(comparisons_list),
            'avg_swaps': np.mean(swaps_list),
            'std_swaps': np.std(swaps_list),
            'min_time': min(times),
            'max_time': max(times)
        }
    
    def analyze_scalability(self, sizes: List[int], data_type: str = 'random', runs: int = 3):
        """
        Analyze how the algorithm scales with input size.
        
        Args:
            sizes (List[int]): List of array sizes to test
            data_type (str): Type of test data
            runs (int): Number of runs per size
        """
        print(f"\nScalability Analysis - Data Type: {data_type}")
        print("=" * 60)
        
        standard_sorter = RandomizedQuicksort(seed=42)
        three_way_sorter = RandomizedQuicksort3Way(seed=42)
        
        standard_results = {'sizes': [], 'times': [], 'comparisons': [], 'swaps': []}
        three_way_results = {'sizes': [], 'times': [], 'comparisons': [], 'swaps': []}
        
        for size in sizes:
            print(f"Testing size: {size:,}")
            
            # Generate test data
            test_array = self.generate_test_data(size, data_type)
            
            # Test standard randomized quicksort
            standard_metrics = self.measure_performance(standard_sorter, test_array, runs)
            standard_results['sizes'].append(size)
            standard_results['times'].append(standard_metrics['avg_time'])
            standard_results['comparisons'].append(standard_metrics['avg_comparisons'])
            standard_results['swaps'].append(standard_metrics['avg_swaps'])
            
            # Test 3-way quicksort
            three_way_metrics = self.measure_performance(three_way_sorter, test_array, runs)
            three_way_results['sizes'].append(size)
            three_way_results['times'].append(three_way_metrics['avg_time'])
            three_way_results['comparisons'].append(three_way_metrics['avg_comparisons'])
            three_way_results['swaps'].append(three_way_metrics['avg_swaps'])
            
            print(f"  Standard: {standard_metrics['avg_time']:.4f}s, "
                  f"{standard_metrics['avg_comparisons']:.0f} comparisons")
            print(f"  3-way:    {three_way_metrics['avg_time']:.4f}s, "
                  f"{three_way_metrics['avg_comparisons']:.0f} comparisons")
        
        self.results[data_type] = {
            'standard': standard_results,
            'three_way': three_way_results
        }
    
    def compare_data_types(self, size: int = 1000, runs: int = 5):
        """
        Compare performance across different data types.
        
        Args:
            size (int): Size of arrays to test
            runs (int): Number of runs per test
        """
        print(f"\nData Type Comparison Analysis - Array Size: {size:,}")
        print("=" * 60)
        
        data_types = ['random', 'sorted', 'reverse', 'duplicates', 'few_unique']
        standard_sorter = RandomizedQuicksort(seed=42)
        three_way_sorter = RandomizedQuicksort3Way(seed=42)
        
        results = {}
        
        for data_type in data_types:
            print(f"\nTesting {data_type} data:")
            
            test_array = self.generate_test_data(size, data_type)
            
            standard_metrics = self.measure_performance(standard_sorter, test_array, runs)
            three_way_metrics = self.measure_performance(three_way_sorter, test_array, runs)
            
            results[data_type] = {
                'standard': standard_metrics,
                'three_way': three_way_metrics
            }
            
            print(f"  Standard: {standard_metrics['avg_time']:.4f}s ± {standard_metrics['std_time']:.4f}s")
            print(f"            {standard_metrics['avg_comparisons']:.0f} ± {standard_metrics['std_comparisons']:.0f} comparisons")
            print(f"  3-way:    {three_way_metrics['avg_time']:.4f}s ± {three_way_metrics['std_time']:.4f}s")
            print(f"            {three_way_metrics['avg_comparisons']:.0f} ± {three_way_metrics['std_comparisons']:.0f} comparisons")
            
            # Calculate speedup for 3-way when beneficial
            if data_type in ['duplicates', 'few_unique']:
                speedup = standard_metrics['avg_time'] / three_way_metrics['avg_time']
                print(f"  3-way speedup: {speedup:.2f}x")
        
        return results
    
    def plot_results(self, save_plots: bool = True):
        """
        Create plots showing the performance analysis results.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        if not self.results:
            print("No results to plot. Run analysis first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Randomized Quicksort Performance Analysis', fontsize=16)
        
        # Plot 1: Time complexity
        ax1 = axes[0, 0]
        for data_type, results in self.results.items():
            sizes = results['standard']['sizes']
            times = results['standard']['times']
            ax1.loglog(sizes, times, 'o-', label=f'{data_type} (standard)', alpha=0.7)
            
            times_3way = results['three_way']['times']
            ax1.loglog(sizes, times_3way, 's--', label=f'{data_type} (3-way)', alpha=0.7)
        
        ax1.set_xlabel('Array Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Time Complexity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Comparisons
        ax2 = axes[0, 1]
        for data_type, results in self.results.items():
            sizes = results['standard']['sizes']
            comparisons = results['standard']['comparisons']
            ax2.loglog(sizes, comparisons, 'o-', label=f'{data_type} (standard)', alpha=0.7)
            
            comparisons_3way = results['three_way']['comparisons']
            ax2.loglog(sizes, comparisons_3way, 's--', label=f'{data_type} (3-way)', alpha=0.7)
        
        ax2.set_xlabel('Array Size')
        ax2.set_ylabel('Number of Comparisons')
        ax2.set_title('Comparison Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Swaps
        ax3 = axes[1, 0]
        for data_type, results in self.results.items():
            sizes = results['standard']['sizes']
            swaps = results['standard']['swaps']
            ax3.loglog(sizes, swaps, 'o-', label=f'{data_type} (standard)', alpha=0.7)
            
            swaps_3way = results['three_way']['swaps']
            ax3.loglog(sizes, swaps_3way, 's--', label=f'{data_type} (3-way)', alpha=0.7)
        
        ax3.set_xlabel('Array Size')
        ax3.set_ylabel('Number of Swaps')
        ax3.set_title('Swap Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency comparison
        ax4 = axes[1, 1]
        if 'duplicates' in self.results:
            results = self.results['duplicates']
            sizes = results['standard']['sizes']
            
            # Calculate relative performance
            time_ratio = np.array(results['standard']['times']) / np.array(results['three_way']['times'])
            comp_ratio = np.array(results['standard']['comparisons']) / np.array(results['three_way']['comparisons'])
            
            ax4.semilogx(sizes, time_ratio, 'o-', label='Time Ratio (Standard/3-way)', linewidth=2)
            ax4.semilogx(sizes, comp_ratio, 's-', label='Comparison Ratio (Standard/3-way)', linewidth=2)
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
            
            ax4.set_xlabel('Array Size')
            ax4.set_ylabel('Performance Ratio')
            ax4.set_title('3-way vs Standard (Duplicate-heavy data)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('quicksort_performance_analysis.png', dpi=300, bbox_inches='tight')
            print("\nPlot saved as 'quicksort_performance_analysis.png'")
        
        plt.show()
    
    def theoretical_analysis(self):
        """
        Provide theoretical analysis of the algorithm.
        """
        print("\nTheoretical Analysis of Randomized Quicksort")
        print("=" * 50)
        print("""
Time Complexity:
- Best Case:    O(n log n) - When pivot consistently divides array in half
- Average Case: O(n log n) - Expected performance with random pivots
- Worst Case:   O(n²) - When pivot is always smallest/largest (very rare with randomization)

Space Complexity:
- Average Case: O(log n) - Due to recursion stack depth
- Worst Case:   O(n) - In highly unbalanced partitions

Key Properties:
1. Randomization makes worst-case extremely unlikely
2. Expected number of comparisons: ~1.39 n log n
3. In-place sorting (O(1) extra space excluding recursion)
4. Not stable (relative order of equal elements may change)

3-way Partitioning Benefits:
- Optimal for arrays with many duplicates
- Reduces comparisons when duplicate values are present
- Degrades gracefully to standard quicksort behavior
- Time complexity improves to O(n) for arrays with few unique values

Edge Cases Handled:
- Empty arrays
- Single-element arrays
- Arrays with all identical elements
- Already sorted arrays
- Reverse sorted arrays
- Arrays with negative numbers
- Arrays with mixed positive/negative values
        """)


def main():
    """
    Main function to run the complete performance analysis.
    """
    print("Randomized Quicksort Performance Analysis")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    analyzer = PerformanceAnalyzer()
    
    # Theoretical analysis
    analyzer.theoretical_analysis()
    
    # Scalability analysis for different data types
    sizes = [100, 250, 500, 1000, 2500, 5000]
    
    print("\nRunning scalability analysis...")
    analyzer.analyze_scalability(sizes, 'random', runs=3)
    analyzer.analyze_scalability(sizes, 'duplicates', runs=3)
    
    # Data type comparison
    print("\nRunning data type comparison...")
    analyzer.compare_data_types(size=2000, runs=5)
    
    # Create performance plots
    print("\nGenerating performance plots...")
    try:
        analyzer.plot_results(save_plots=True)
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
        print("Install matplotlib with: pip install matplotlib")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()