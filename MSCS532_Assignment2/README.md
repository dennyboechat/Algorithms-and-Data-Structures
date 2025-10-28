# Sorting Algorithm Performance Analysis

This project implements and compares the performance of Quick Sort and Merge Sort algorithms on various datasets. It provides comprehensive analysis including execution time, memory usage, and algorithm-specific metrics.

## Features

### Sorting Algorithms
- **Quick Sort**: Efficient divide-and-conquer algorithm with last-element pivot
- **Merge Sort**: Stable divide-and-conquer algorithm with guaranteed O(n log n) performance

### Performance Metrics
- Execution time measurement
- Memory usage tracking (peak and current)
- Algorithm-specific counters:
  - Comparisons
  - Swaps (Quick Sort) / Merges (Merge Sort)
  - Recursive calls

### Dataset Types
- **Random**: Randomly generated integers
- **Sorted**: Already sorted in ascending order
- **Reverse Sorted**: Sorted in descending order
- **Nearly Sorted**: Mostly sorted with some random swaps
- **Duplicate Heavy**: Contains many duplicate values

### Test Sizes
- 100, 500, 1,000, 2,500, 5,000, 10,000 elements

## Installation

1. Ensure Python 3.7+ is installed
2. Install required packages:
   ```bash
   pip install psutil matplotlib numpy pandas seaborn
   ```

## Usage

### Run Complete Analysis
```bash
python main_analysis.py
```

This will:
1. Test both algorithms on all dataset types and sizes
2. Generate detailed performance metrics
3. Save results to a JSON file
4. Display comprehensive analysis in the terminal

### Generate Visualizations
After running the main analysis:
```bash
python visualization.py
```

This creates:
- Execution time comparison charts
- Memory usage analysis
- Algorithm-specific performance metrics
- Comprehensive report in `performance_report/` directory

### Use Individual Components

#### Test Basic Functionality
```python
from sorting_algorithms import QuickSort, MergeSort

# Test data
data = [64, 34, 25, 12, 22, 11, 90]

# Quick Sort
qs = QuickSort()
sorted_data = qs.sort(data)
print(f"Sorted: {sorted_data}")
print(f"Stats: {qs.get_performance_stats()}")

# Merge Sort
ms = MergeSort()
sorted_data = ms.sort(data)
print(f"Sorted: {sorted_data}")
print(f"Stats: {ms.get_performance_stats()}")
```

#### Generate Custom Datasets
```python
from performance_analysis import DatasetGenerator

gen = DatasetGenerator()
random_data = gen.generate_random(1000)
sorted_data = gen.generate_sorted(1000)
reverse_data = gen.generate_reverse_sorted(1000)
```

#### Measure Performance
```python
from performance_analysis import PerformanceMeasurer
from sorting_algorithms import QuickSort

algorithm = QuickSort()
data = [3, 1, 4, 1, 5, 9, 2, 6]

metrics = PerformanceMeasurer.measure_algorithm_performance(
    algorithm, data, "test_dataset"
)
print(f"Execution time: {metrics.execution_time:.6f} seconds")
print(f"Memory usage: {metrics.memory_peak:.2f} MB")
```

## Algorithm Implementations

### Quick Sort
- Uses last element as pivot
- In-place partitioning
- Recursive implementation
- Time Complexity: O(n log n) average, O(n²) worst case
- Space Complexity: O(log n) average

### Merge Sort
- Divide-and-conquer approach
- Stable sorting (preserves relative order of equal elements)
- Not in-place (requires additional memory)
- Time Complexity: O(n log n) always
- Space Complexity: O(n)

## Performance Analysis Results

The analysis typically shows:

### Execution Time
- **Merge Sort**: More consistent performance across different dataset types
- **Quick Sort**: Faster on average but sensitive to input order
- **Best Case**: Quick Sort on random data
- **Worst Case**: Quick Sort on sorted/reverse sorted data

### Memory Usage
- **Merge Sort**: Higher memory usage due to temporary arrays
- **Quick Sort**: More memory efficient with in-place sorting

### Dataset Sensitivity
- **Sorted Data**: Merge Sort significantly outperforms Quick Sort
- **Random Data**: Quick Sort often faster than Merge Sort
- **Reverse Sorted**: Similar to sorted data patterns
- **Nearly Sorted**: Quick Sort performance degrades
- **Duplicate Heavy**: Both algorithms handle well, slight Quick Sort advantage

## Output Files

### JSON Results
```json
{
  "QuickSort": {
    "random_1000": {
      "execution_time": 0.001234,
      "memory_peak": 0.15,
      "algorithm_stats": {
        "comparisons": 8743,
        "swaps": 654,
        "recursive_calls": 31
      }
    }
  }
}
```

### Visualizations
- `execution_time_comparison.png`: Time performance charts
- `memory_usage_comparison.png`: Memory analysis charts
- `algorithm_specific_analysis.png`: Algorithm internals
- `summary_statistics.txt`: Text summary

## Technical Notes

### Memory Measurement
- Uses `tracemalloc` for precise memory tracking
- Measures peak memory during sorting operation
- Accounts for Python object overhead

### Timing Precision
- Uses `time.perf_counter()` for high-precision timing
- Excludes data generation from timing measurements
- Includes garbage collection to ensure clean measurements

### Dataset Generation
- Configurable value ranges
- Reproducible with seed setting
- Various distribution patterns supported

## Extending the Analysis

### Adding New Algorithms
1. Implement the algorithm class with `sort()` method
2. Add performance counters similar to existing implementations
3. Include in `main_analysis.py` algorithms dictionary

### Custom Dataset Types
1. Add generator method to `DatasetGenerator` class
2. Update dataset types list in `main_analysis.py`
3. Modify visualization code if needed

### Additional Metrics
1. Add measurement in `PerformanceMeasurer.measure_algorithm_performance()`
2. Update `PerformanceMetrics` dataclass
3. Include in visualization and analysis code

## Performance Optimization Tips

### For Quick Sort
- Consider randomized pivot selection for worst-case avoidance
- Use iterative implementation for large datasets to avoid stack overflow
- Switch to insertion sort for small subarrays

### For Merge Sort
- Use in-place merging to reduce memory usage
- Optimize for nearly sorted data with natural merge variants
- Consider iterative bottom-up approach

## References

- Cormen, T. H., et al. "Introduction to Algorithms" (CLRS)
- Sedgewick, R. "Algorithms in Python"
- Python Performance Tips: https://wiki.python.org/moin/PythonSpeed

---