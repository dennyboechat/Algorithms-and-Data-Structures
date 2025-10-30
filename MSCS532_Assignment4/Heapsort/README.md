# Heapsort Algorithm Implementation

A comprehensive implementation of the Heapsort algorithm in Python with detailed documentation, testing, and demonstrations.

## Analysis and Comparison

The files analysis.md and comparison.md contain detailed reports about Heapsort algorithms.

## Algorithm Overview

Heapsort is a comparison-based sorting algorithm that uses a **max-heap** data structure to sort elements in ascending order. It combines the best features of merge sort and insertion sort:

- **Time Complexity**: O(n log n) in all cases (best, average, worst)
- **Space Complexity**: O(1) - in-place sorting algorithm
- **Stability**: Not stable (relative order of equal elements may change)
- **In-place**: Yes (constant extra memory usage)

## How Heapsort Works

The algorithm operates in two main phases:

### Phase 1: Build Max-Heap
1. Convert the input array into a max-heap where each parent node is greater than its children
2. Start from the last non-leaf node and apply heapify in bottom-up manner
3. Time complexity: O(n)

### Phase 2: Extract Maximum Elements
1. Swap the root (maximum element) with the last element
2. Reduce heap size by 1
3. Restore the heap property by calling heapify on the new root
4. Repeat until the heap is empty
5. Time complexity: O(n log n)

## Heap Data Structure Properties

A **max-heap** is a complete binary tree where:
- Each parent node is greater than or equal to its children
- The tree is filled level by level from left to right
- For array representation with 0-based indexing:
  - Parent of node at index `i`: `(i-1)//2`
  - Left child of node at index `i`: `2*i + 1`
  - Right child of node at index `i`: `2*i + 2`

## Files Description

### `heapsort.py`
The main implementation containing:
- `heapify(arr, n, i)`: Maintains heap property for subtree rooted at index i
- `build_max_heap(arr)`: Converts array into max-heap
- `heapsort(arr)`: Sorts array and returns new sorted copy
- `heapsort_inplace(arr)`: Sorts array in-place
- `is_max_heap(arr)`: Validates if array satisfies max-heap property

### `test_heapsort.py`
Comprehensive test suite including:
- Edge cases (empty, single element, duplicates)
- Different input types (sorted, reverse sorted, random)
- Performance tests and stress tests
- Validation of heap properties

### `demo.py`
Interactive demonstration script featuring:
- Step-by-step heap building visualization
- Complete sorting process demonstration
- Performance analysis and comparison
- Support for custom input arrays

## Usage Examples

### Basic Usage

```python
from heapsort import heapsort, heapsort_inplace

# Sort a copy of the array
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = heapsort(arr)
print(f"Original: {arr}")
print(f"Sorted: {sorted_arr}")

# Sort in-place
arr = [64, 34, 25, 12, 22, 11, 90]
heapsort_inplace(arr)
print(f"Sorted in-place: {arr}")
```

### Building and Validating Heaps

```python
from heapsort import build_max_heap, is_max_heap

# Build a max-heap
arr = [4, 10, 3, 5, 1]
build_max_heap(arr)
print(f"Max-heap: {arr}")
print(f"Is valid heap: {is_max_heap(arr)}")
```

## Running the Code

### Execute Main Implementation
```bash
python heapsort.py
```

### Run Test Suite
```bash
python test_heapsort.py
```

### Interactive Demonstration
```bash
python demo.py
```

### Automatic Demonstration
```bash
python demo.py --auto
```

## Algorithm Visualization

Here's how heapsort works on the array `[64, 34, 25, 12, 22, 11, 90]`:

### Step 1: Build Max-Heap
```
Original: [64, 34, 25, 12, 22, 11, 90]
                64
               /  \
              34   25
             / |   | \
            12 22 11 90

After heapify: [90, 34, 64, 12, 22, 11, 25]
                90
               /  \
              34   64
             / |   | \
            12 22 11 25
```

### Step 2: Extract Elements
```
Extract 90: [64, 34, 25, 12, 22, 11] | [90]
Extract 64: [34, 22, 25, 12, 11]     | [64, 90]
Extract 34: [25, 22, 11, 12]         | [34, 64, 90]
...continue until sorted...
Final:      []                       | [11, 12, 22, 25, 34, 64, 90]
```

## Performance Characteristics

### Time Complexity Analysis
- **Best Case**: O(n log n) - Even with sorted input
- **Average Case**: O(n log n) - Typical random input
- **Worst Case**: O(n log n) - Consistent performance

### Space Complexity
- **O(1)** auxiliary space (excluding input array)
- In-place sorting algorithm

### Comparison with Other Algorithms

| Algorithm    | Best Case  | Average Case | Worst Case | Space    | Stable |
|-------------|------------|--------------|------------|----------|--------|
| Heapsort    | O(n log n) | O(n log n)   | O(n log n) | O(1)     | No     |
| Quicksort   | O(n log n) | O(n log n)   | O(n²)      | O(log n) | No     |
| Mergesort   | O(n log n) | O(n log n)   | O(n log n) | O(n)     | Yes    |
| Bubble Sort | O(n)       | O(n²)        | O(n²)      | O(1)     | Yes    |

## Advantages and Disadvantages

### Advantages
- **Guaranteed O(n log n)** performance in all cases
- **In-place sorting** with O(1) space complexity
- **Simple implementation** with clear algorithmic steps
- **No worst-case degradation** unlike quicksort

### Disadvantages
- **Not stable** - relative order of equal elements may change
- **Poor cache performance** due to non-sequential memory access
- **Slower than quicksort** in practice due to constant factors
- **Not adaptive** - doesn't perform better on partially sorted data

## Applications

Heapsort is particularly useful when:
- **Memory is limited** (O(1) space requirement)
- **Predictable performance** is required (guaranteed O(n log n))
- **Worst-case performance** matters more than average case
- **Implementing priority queues** or heap-based data structures

## Implementation Details

### Key Functions

1. **`heapify(arr, n, i)`**
   - Maintains max-heap property for subtree rooted at index i
   - Assumes left and right subtrees are already max-heaps
   - Recursively fixes violations by swapping with larger child

2. **`build_max_heap(arr)`**
   - Converts arbitrary array into max-heap
   - Processes nodes from last non-leaf to root
   - Calls heapify on each node

3. **`heapsort(arr)`**
   - Main sorting function
   - Builds max-heap then repeatedly extracts maximum
   - Returns new sorted array without modifying original

### Error Handling
- Handles empty arrays gracefully
- Works with single-element arrays
- Supports negative numbers and duplicates
- Type hints for better code documentation

## Testing

The test suite covers:
- **Edge cases**: empty arrays, single elements, duplicates
- **Input variations**: sorted, reverse sorted, random arrays
- **Data types**: positive, negative, mixed numbers
- **Performance tests**: comparison with built-in sort
- **Stress tests**: multiple iterations with random data

Run tests with detailed output:
```bash
python -m unittest test_heapsort.py -v
```

## Dependencies

- **Python 3.6+** (type hints support)
- **matplotlib** (optional, for performance visualization)
- **unittest** (standard library, for testing)