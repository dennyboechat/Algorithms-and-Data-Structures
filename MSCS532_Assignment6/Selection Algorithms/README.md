# Selection Algorithms - Deterministic and Randomized Approaches

## Overview

This repository contains comprehensive Python implementations of both deterministic and randomized selection algorithms for finding the k-th smallest element in an unsorted array. It includes detailed comparisons, performance analysis, and practical examples.

## Files

### Core Implementations
- `median_of_medians.py` - Deterministic selection (Median of Medians) with O(n) worst-case guarantee
- `randomized_quickselect.py` - Randomized selection (Quickselect) with O(n) expected time
- `algorithm_comparison.py` - Comprehensive comparison between both approaches

### Testing and Examples
- `test_median_of_medians.py` - Test suite for deterministic algorithm
- `test_randomized_quickselect.py` - Test suite for randomized algorithms
- `examples.py` - Practical usage examples for deterministic algorithm

### Documentation
- `README.md` - This comprehensive documentation

## Algorithms Implemented

### 1. Deterministic Selection (Median of Medians)
A deterministic selection algorithm that guarantees linear time performance in the worst case:

1. **Grouping**: Divide the input array into groups of 5 elements
2. **Local Medians**: Find the median of each group (constant time since groups have ≤5 elements)
3. **Pivot Selection**: Recursively find the median of the medians to use as a pivot
4. **Partitioning**: Partition the original array around this pivot
5. **Recursion**: Recursively search the appropriate partition

### 2. Randomized Selection (Quickselect)
A randomized selection algorithm with excellent expected performance:

1. **Random Pivot**: Randomly select a pivot element
2. **Partitioning**: Partition array around the pivot
3. **Recursive Search**: Recursively search the appropriate partition
4. **Optimizations**: Multiple variants including median-of-three pivot selection

## Key Features

### Deterministic Algorithm
- **Worst-case O(n) time complexity**: Guaranteed linear time performance
- **Deterministic**: No randomization, suitable for real-time systems
- **Consistent performance**: No variance in execution time
- **Robust against adversarial inputs**: Immune to worst-case input patterns

### Randomized Algorithm
- **Expected O(n) time complexity**: Excellent average-case performance
- **Lower constant factors**: Typically faster in practice
- **Simpler implementation**: More straightforward to understand and code
- **Multiple variants**: Recursive, iterative, and median-of-three implementations

## Time and Space Complexity Comparison

| Algorithm | Best Case | Average Case | Worst Case | Space Complexity |
|-----------|-----------|--------------|------------|------------------|
| Median of Medians | O(n) | O(n) | O(n) | O(log n) |
| Randomized Quickselect | O(n) | O(n) | O(n²) | O(log n) expected |
| Sorting | O(n log n) | O(n log n) | O(n log n) | O(1) or O(n) |
| Heap-based | O(n + k log n) | O(n + k log n) | O(n + k log n) | O(n) |

**Key Differences:**
- **Deterministic**: Guaranteed O(n) worst-case, higher constant factors
- **Randomized**: Better practical performance, small probability of O(n²) degradation

## Usage

### Deterministic Selection (Median of Medians)

```python
from median_of_medians import find_kth_smallest, find_median, find_kth_largest

# Find the k-th smallest element (1-based indexing)
arr = [12, 3, 5, 7, 4, 19, 26, 1, 9]
third_smallest = find_kth_smallest(arr, 3)  # Returns 4

# Find the median
median = find_median(arr)  # Returns 7

# Find the k-th largest element
second_largest = find_kth_largest(arr, 2)  # Returns 12
```

### Randomized Selection (Quickselect)

```python
from randomized_quickselect import (
    randomized_select, 
    randomized_select_iterative,
    quickselect_with_median_of_three
)

# Basic randomized selection
arr = [12, 3, 5, 7, 4, 19, 26, 1, 9]
third_smallest = randomized_select(arr, 3)  # Returns 4

# Iterative version (O(1) space)
result = randomized_select_iterative(arr, 5)  # Returns 7

# Median-of-three optimization
result = quickselect_with_median_of_three(arr, 3)  # Returns 4
```

### Algorithm Comparison

```python
from algorithm_comparison import run_comprehensive_comparison

# Run detailed performance comparison
results = run_comprehensive_comparison()
```

## API Reference

### `median_of_medians_select(arr, k)`

Main algorithm function that finds the k-th smallest element.

**Parameters:**
- `arr` (list): Input array of comparable elements
- `k` (int): Position of desired element (1-based index, 1 ≤ k ≤ len(arr))

**Returns:**
- The k-th smallest element

**Raises:**
- `ValueError`: If k is out of range or array is empty
- `TypeError`: If array elements are not comparable

### `find_kth_smallest(arr, k)`

Wrapper function for finding k-th smallest element.

### `find_median(arr)`

Find the median element using the selection algorithm.

### `find_kth_largest(arr, k)`

Find the k-th largest element by converting to equivalent smallest position.

## Running the Code

### Execute the main file to see demonstrations:

```bash
python median_of_medians.py
```

This will run:
- Basic algorithm demonstrations
- Performance comparisons
- Interactive examples

### Run the test suite:

```bash
python test_median_of_medians.py
```

Or use unittest:

```bash
python -m unittest test_median_of_medians.py -v
```

## Algorithm Analysis

### Why Median of Medians Works

The key insight is that by choosing the median of medians as a pivot, we guarantee that:

1. At least 30% of elements are smaller than the pivot
2. At least 30% of elements are larger than the pivot
3. This ensures that each recursive call reduces the problem size by at least 30%
4. The recurrence relation T(n) = T(n/5) + T(7n/10) + O(n) solves to O(n)

### Detailed Steps

1. **Group Formation**: ⌈n/5⌉ groups of at most 5 elements each
2. **Median Extraction**: Find median of each group in O(1) time
3. **Recursive Median**: Find median of the ⌈n/5⌉ medians recursively
4. **Partitioning**: Partition original array using this median as pivot
5. **Recursive Selection**: Recurse on the appropriate partition

## Test Coverage

The test suite includes:

- Basic functionality tests
- Edge cases (empty arrays, single elements)
- Arrays with duplicates
- Large arrays (up to 100+ elements)
- Random array testing
- Error condition testing
- Performance verification
- Worst-case scenario testing

## Example Output

```
=== Median of Medians Selection Algorithm Demo ===

Array: [12, 3, 5, 7, 4, 19, 26, 1, 9]
3rd smallest: 4
5th smallest (median): 7
2nd largest: 19

Array with duplicates: [5, 2, 8, 2, 9, 1, 5, 5]
4th smallest: 5
Median: 5

=== Performance Comparison ===

Array size: 1000
Median of Medians: 0.002341 seconds
Sorted approach: 0.000123 seconds
Results match: True
```

## When to Use Each Algorithm

### Use Deterministic (Median of Medians) When:
- **Real-time systems** requiring guaranteed response times
- **Safety-critical applications** where worst-case behavior matters
- **Adversarial environments** with potentially malicious inputs
- **Formal verification** requirements
- **Consistent performance** is more important than optimal average performance

### Use Randomized (Quickselect) When:
- **General-purpose applications** where average performance matters most
- **Interactive systems** where user experience is prioritized
- **Large datasets** where the probability of worst-case is negligible
- **Memory-constrained environments** (using iterative version)
- **Simple implementation** is preferred

## Applications

These algorithms are particularly useful for:

- **Database query optimization** (finding percentiles, quartiles)
- **Statistical analysis** (median, quantiles, outlier detection)
- **Computer graphics** (median filtering, image processing)
- **Data science** (exploratory data analysis)
- **Competitive programming** where time limits are strict
- **Operating systems** (scheduling, resource allocation)
- **Network algorithms** (load balancing, QoS)

## Implementation Notes

- The algorithm modifies a copy of the input array, leaving the original unchanged
- Groups of 5 are used because they provide the optimal constant factors
- The implementation handles edge cases gracefully
- Error messages provide clear guidance for incorrect usage

## Educational Value

This implementation demonstrates:

- Advanced algorithm design techniques
- Divide-and-conquer strategies
- Worst-case analysis importance
- Trade-offs between average and worst-case performance
- Deterministic vs. randomized algorithms