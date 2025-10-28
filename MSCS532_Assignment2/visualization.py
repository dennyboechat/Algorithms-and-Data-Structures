"""
Visualization Module for Sorting Algorithm Performance Analysis
Creates charts and graphs to visualize performance comparisons.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns
from pathlib import Path


class PerformanceVisualizer:
    """Create visualizations for sorting algorithm performance analysis."""
    
    def __init__(self, results_data: Dict = None, json_file: str = None):
        """
        Initialize with either results data or JSON file.
        
        Args:
            results_data: Dictionary of performance results
            json_file: Path to JSON file containing results
        """
        if json_file:
            self.load_from_json(json_file)
        elif results_data:
            self.data = results_data
        else:
            raise ValueError("Either results_data or json_file must be provided")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_from_json(self, json_file: str):
        """Load performance data from JSON file."""
        with open(json_file, 'r') as f:
            self.data = json.load(f)
    
    def _extract_data_for_plotting(self) -> pd.DataFrame:
        """Extract and organize data for plotting."""
        rows = []
        
        for algorithm, datasets in self.data.items():
            for dataset_key, metrics in datasets.items():
                # Parse dataset key to extract type and size
                parts = dataset_key.split('_')
                size = int(parts[-1])
                dataset_type = '_'.join(parts[:-1])
                
                row = {
                    'Algorithm': algorithm,
                    'Dataset_Type': dataset_type,
                    'Size': size,
                    'Execution_Time': metrics['execution_time'],
                    'Memory_Peak': metrics['memory_peak'],
                    'Memory_Current': metrics['memory_current'],
                    'Dataset_Key': dataset_key
                }
                
                # Add algorithm-specific metrics
                for stat_name, stat_value in metrics.get('algorithm_stats', {}).items():
                    if isinstance(stat_value, (int, float)):
                        row[f'Stat_{stat_name}'] = stat_value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_execution_time_comparison(self, save_path: str = None) -> str:
        """Create execution time comparison charts."""
        df = self._extract_data_for_plotting()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Execution Time Comparison: Quick Sort vs Merge Sort', fontsize=16)
        
        # 1. Line plot: Execution time vs dataset size for each dataset type
        ax1 = axes[0, 0]
        dataset_types = df['Dataset_Type'].unique()
        
        for dtype in dataset_types:
            for alg in df['Algorithm'].unique():
                subset = df[(df['Dataset_Type'] == dtype) & (df['Algorithm'] == alg)]
                if not subset.empty:
                    ax1.plot(subset['Size'], subset['Execution_Time'], 
                            marker='o', label=f'{alg} ({dtype})', linewidth=2)
        
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Dataset Size')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar plot: Average execution time by dataset type
        ax2 = axes[0, 1]
        avg_times = df.groupby(['Algorithm', 'Dataset_Type'])['Execution_Time'].mean().reset_index()
        
        bar_width = 0.35
        algorithms = avg_times['Algorithm'].unique()
        dataset_types = avg_times['Dataset_Type'].unique()
        x = np.arange(len(dataset_types))
        
        for i, alg in enumerate(algorithms):
            alg_data = avg_times[avg_times['Algorithm'] == alg]
            values = [alg_data[alg_data['Dataset_Type'] == dt]['Execution_Time'].iloc[0] 
                     if not alg_data[alg_data['Dataset_Type'] == dt].empty else 0 
                     for dt in dataset_types]
            ax2.bar(x + i * bar_width, values, bar_width, label=alg)
        
        ax2.set_xlabel('Dataset Type')
        ax2.set_ylabel('Average Execution Time (seconds)')
        ax2.set_title('Average Execution Time by Dataset Type')
        ax2.set_xticks(x + bar_width / 2)
        ax2.set_xticklabels(dataset_types, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot: Execution time vs dataset size (colored by algorithm)
        ax3 = axes[1, 0]
        for alg in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == alg]
            ax3.scatter(alg_data['Size'], alg_data['Execution_Time'], 
                       label=alg, alpha=0.7, s=50)
        
        ax3.set_xlabel('Dataset Size')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Execution Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot: Execution time distribution by algorithm
        ax4 = axes[1, 1]
        execution_times_by_alg = [df[df['Algorithm'] == alg]['Execution_Time'].values 
                                 for alg in df['Algorithm'].unique()]
        ax4.boxplot(execution_times_by_alg, labels=df['Algorithm'].unique())
        ax4.set_ylabel('Execution Time (seconds)')
        ax4.set_title('Execution Time Distribution by Algorithm')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'execution_time_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def create_memory_usage_comparison(self, save_path: str = None) -> str:
        """Create memory usage comparison charts."""
        df = self._extract_data_for_plotting()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Memory Usage Comparison: Quick Sort vs Merge Sort', fontsize=16)
        
        # 1. Peak memory usage vs dataset size
        ax1 = axes[0, 0]
        for alg in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == alg]
            ax1.scatter(alg_data['Size'], alg_data['Memory_Peak'], 
                       label=f'{alg} (Peak)', alpha=0.7, s=50)
        
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Peak Memory Usage (MB)')
        ax1.set_title('Peak Memory Usage vs Dataset Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory usage by dataset type
        ax2 = axes[0, 1]
        avg_memory = df.groupby(['Algorithm', 'Dataset_Type'])['Memory_Peak'].mean().reset_index()
        
        bar_width = 0.35
        algorithms = avg_memory['Algorithm'].unique()
        dataset_types = avg_memory['Dataset_Type'].unique()
        x = np.arange(len(dataset_types))
        
        for i, alg in enumerate(algorithms):
            alg_data = avg_memory[avg_memory['Algorithm'] == alg]
            values = [alg_data[alg_data['Dataset_Type'] == dt]['Memory_Peak'].iloc[0] 
                     if not alg_data[alg_data['Dataset_Type'] == dt].empty else 0 
                     for dt in dataset_types]
            ax2.bar(x + i * bar_width, values, bar_width, label=alg)
        
        ax2.set_xlabel('Dataset Type')
        ax2.set_ylabel('Average Peak Memory (MB)')
        ax2.set_title('Average Peak Memory by Dataset Type')
        ax2.set_xticks(x + bar_width / 2)
        ax2.set_xticklabels(dataset_types, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory efficiency (Memory per element)
        ax3 = axes[1, 0]
        df['Memory_Per_Element'] = df['Memory_Peak'] / df['Size'] * 1024  # Convert to KB per element
        
        for alg in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == alg]
            ax3.scatter(alg_data['Size'], alg_data['Memory_Per_Element'], 
                       label=alg, alpha=0.7, s=50)
        
        ax3.set_xlabel('Dataset Size')
        ax3.set_ylabel('Memory per Element (KB)')
        ax3.set_title('Memory Efficiency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot: Memory usage distribution
        ax4 = axes[1, 1]
        memory_by_alg = [df[df['Algorithm'] == alg]['Memory_Peak'].values 
                        for alg in df['Algorithm'].unique()]
        ax4.boxplot(memory_by_alg, labels=df['Algorithm'].unique())
        ax4.set_ylabel('Peak Memory Usage (MB)')
        ax4.set_title('Memory Usage Distribution by Algorithm')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'memory_usage_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def create_algorithm_specific_analysis(self, save_path: str = None) -> str:
        """Create algorithm-specific performance analysis charts."""
        df = self._extract_data_for_plotting()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithm-Specific Performance Analysis', fontsize=16)
        
        # Quick Sort specific metrics
        if 'QuickSort' in df['Algorithm'].values:
            qs_data = df[df['Algorithm'] == 'QuickSort']
            
            # Comparisons vs dataset size
            if 'Stat_comparisons' in qs_data.columns:
                ax1 = axes[0, 0]
                ax1.scatter(qs_data['Size'], qs_data['Stat_comparisons'], 
                           color='red', alpha=0.7, s=50)
                ax1.set_xlabel('Dataset Size')
                ax1.set_ylabel('Number of Comparisons')
                ax1.set_title('QuickSort: Comparisons vs Size')
                ax1.grid(True, alpha=0.3)
            
            # Swaps vs dataset size
            if 'Stat_swaps' in qs_data.columns:
                ax2 = axes[0, 1]
                ax2.scatter(qs_data['Size'], qs_data['Stat_swaps'], 
                           color='blue', alpha=0.7, s=50)
                ax2.set_xlabel('Dataset Size')
                ax2.set_ylabel('Number of Swaps')
                ax2.set_title('QuickSort: Swaps vs Size')
                ax2.grid(True, alpha=0.3)
            
            # Recursive calls vs dataset size
            if 'Stat_recursive_calls' in qs_data.columns:
                ax3 = axes[0, 2]
                ax3.scatter(qs_data['Size'], qs_data['Stat_recursive_calls'], 
                           color='green', alpha=0.7, s=50)
                ax3.set_xlabel('Dataset Size')
                ax3.set_ylabel('Recursive Calls')
                ax3.set_title('QuickSort: Recursive Calls vs Size')
                ax3.grid(True, alpha=0.3)
        
        # Merge Sort specific metrics
        if 'MergeSort' in df['Algorithm'].values:
            ms_data = df[df['Algorithm'] == 'MergeSort']
            
            # Comparisons vs dataset size
            if 'Stat_comparisons' in ms_data.columns:
                ax4 = axes[1, 0]
                ax4.scatter(ms_data['Size'], ms_data['Stat_comparisons'], 
                           color='orange', alpha=0.7, s=50)
                ax4.set_xlabel('Dataset Size')
                ax4.set_ylabel('Number of Comparisons')
                ax4.set_title('MergeSort: Comparisons vs Size')
                ax4.grid(True, alpha=0.3)
            
            # Merges vs dataset size
            if 'Stat_merges' in ms_data.columns:
                ax5 = axes[1, 1]
                ax5.scatter(ms_data['Size'], ms_data['Stat_merges'], 
                           color='purple', alpha=0.7, s=50)
                ax5.set_xlabel('Dataset Size')
                ax5.set_ylabel('Number of Merges')
                ax5.set_title('MergeSort: Merges vs Size')
                ax5.grid(True, alpha=0.3)
            
            # Recursive calls vs dataset size
            if 'Stat_recursive_calls' in ms_data.columns:
                ax6 = axes[1, 2]
                ax6.scatter(ms_data['Size'], ms_data['Stat_recursive_calls'], 
                           color='brown', alpha=0.7, s=50)
                ax6.set_xlabel('Dataset Size')
                ax6.set_ylabel('Recursive Calls')
                ax6.set_title('MergeSort: Recursive Calls vs Size')
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'algorithm_specific_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def create_comprehensive_report(self, output_dir: str = "performance_report"):
        """Create a comprehensive visual report."""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print("Generating comprehensive performance visualization report...")
        
        # Generate all charts
        exec_time_chart = self.create_execution_time_comparison(
            f"{output_dir}/execution_time_comparison.png"
        )
        memory_chart = self.create_memory_usage_comparison(
            f"{output_dir}/memory_usage_comparison.png"
        )
        alg_specific_chart = self.create_algorithm_specific_analysis(
            f"{output_dir}/algorithm_specific_analysis.png"
        )
        
        # Create summary statistics
        df = self._extract_data_for_plotting()
        summary_stats = self._generate_summary_statistics(df)
        
        # Save summary to text file
        with open(f"{output_dir}/summary_statistics.txt", 'w') as f:
            f.write("SORTING ALGORITHM PERFORMANCE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary_stats)
        
        print(f"\nComprehensive report generated in '{output_dir}/' directory:")
        print(f"  - {exec_time_chart}")
        print(f"  - {memory_chart}")
        print(f"  - {alg_specific_chart}")
        print(f"  - {output_dir}/summary_statistics.txt")
        
        return output_dir
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> str:
        """Generate summary statistics text."""
        summary = []
        
        # Overall statistics
        summary.append("OVERALL PERFORMANCE SUMMARY")
        summary.append("-" * 30)
        
        for alg in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == alg]
            summary.append(f"\n{alg}:")
            summary.append(f"  Average execution time: {alg_data['Execution_Time'].mean():.6f} seconds")
            summary.append(f"  Min execution time: {alg_data['Execution_Time'].min():.6f} seconds")
            summary.append(f"  Max execution time: {alg_data['Execution_Time'].max():.6f} seconds")
            summary.append(f"  Average peak memory: {alg_data['Memory_Peak'].mean():.2f} MB")
            summary.append(f"  Total datasets tested: {len(alg_data)}")
        
        # Best/worst performers
        summary.append(f"\nBEST/WORST PERFORMERS")
        summary.append("-" * 30)
        
        fastest_overall = df.loc[df['Execution_Time'].idxmin()]
        slowest_overall = df.loc[df['Execution_Time'].idxmax()]
        
        summary.append(f"Fastest execution: {fastest_overall['Algorithm']} on {fastest_overall['Dataset_Key']} ({fastest_overall['Execution_Time']:.6f}s)")
        summary.append(f"Slowest execution: {slowest_overall['Algorithm']} on {slowest_overall['Dataset_Key']} ({slowest_overall['Execution_Time']:.6f}s)")
        
        return "\n".join(summary)


def create_visualizations_from_json(json_file: str):
    """Standalone function to create visualizations from a JSON results file."""
    visualizer = PerformanceVisualizer(json_file=json_file)
    return visualizer.create_comprehensive_report()


if __name__ == "__main__":
    # Example usage - this will look for a results JSON file
    import glob
    
    # Find the most recent results file
    json_files = glob.glob("performance_results_*.json")
    if json_files:
        latest_file = max(json_files)
        print(f"Creating visualizations from: {latest_file}")
        create_visualizations_from_json(latest_file)
    else:
        print("No performance results JSON file found.")
        print("Please run main_analysis.py first to generate performance data.")