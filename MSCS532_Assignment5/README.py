"""
README: Quicksort Implementation Guide

This directory contains comprehensive implementations of the Quicksort algorithm
with detailed explanations and examples.

Note: Report.docx provides an in-depth analysis of the algorithm, its complexity,
and practical considerations.

Running the Code:
================

To see the algorithm in action:

    python simple_quicksort.py

To run comprehensive tests:

    python test_quicksort.py

Core Algorithm Steps:
====================

STEP 1: PIVOT SELECTION
• Choose an element from the array as the 'pivot'
• Common strategies:
  - Last element (most common in implementations)
  - First element (simple but poor for sorted arrays)
  - Middle element (good for partially sorted arrays)
  - Random element (helps avoid worst-case scenarios)
  - Median-of-three (robust for various inputs)

STEP 2: PARTITIONING
• Rearrange the array so that:
  - All elements smaller than pivot are on the left
  - All elements greater than pivot are on the right
  - Elements equal to pivot can be on either side
• After partitioning, the pivot is in its final sorted position
• This is accomplished using the Lomuto or Hoare partition schemes

STEP 3: RECURSIVE SORTING
• Apply quicksort recursively to the left subarray (elements < pivot)
• Apply quicksort recursively to the right subarray (elements > pivot)
• Base case: arrays with 0 or 1 elements are already sorted
• The recursion naturally handles the divide-and-conquer approach

Implementation Highlights:
=========================

1. SIMPLE VERSION (simple_quicksort.py):
   - Uses middle element as pivot for good performance on partially sorted data
   - Creates new arrays for easy understanding
   - Shows clear separation of the three steps
   - Best for learning and understanding the algorithm

2. IN-PLACE VERSION (quicksort.py):
   - Memory efficient - sorts without creating new arrays
   - Uses Lomuto partition scheme
   - Supports different pivot selection strategies
   - Production-ready implementation

3. EDUCATIONAL FEATURES:
   - Step-by-step output showing pivot selection and partitioning
   - Clear variable names and extensive comments
   - Multiple test cases including edge cases
   - Performance comparisons between strategies

Complexity Analysis:
===================

Time Complexity:
• Best Case: O(n log n) - when pivot divides array evenly
• Average Case: O(n log n) - with random pivot selection
• Worst Case: O(n²) - when pivot is always min or max element

Space Complexity:
• O(log n) for recursion stack in average case
• O(n) in worst case due to unbalanced recursion

When to Use Quicksort:
=====================

ADVANTAGES:
✓ Generally faster than other O(n log n) algorithms
✓ In-place sorting (minimal extra memory)
✓ Cache-friendly due to locality of reference
✓ Simple to implement and understand

DISADVANTAGES:
✗ Worst-case O(n²) performance
✗ Not stable (doesn't preserve relative order of equal elements)
✗ Performance depends heavily on pivot selection
✗ Poor performance on already sorted data with basic pivot strategies

Practical Tips:
==============

1. For production use, consider:
   - Hybrid approaches (switch to insertion sort for small arrays)
   - Introsort (switches to heapsort if recursion depth exceeds limit)
   - Three-way partitioning for arrays with many duplicates

2. For educational purposes:
   - Start with the simple version to understand the concept
   - Then study the in-place version for efficiency
   - Experiment with different pivot strategies

3. Common optimizations:
   - Use insertion sort for small subarrays (< 10-15 elements)
   - Implement iterative version to avoid stack overflow
   - Use median-of-three pivot selection

Example Usage:
=============

```python
# Simple usage
from simple_quicksort import simple_quicksort
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = simple_quicksort(arr)
print(sorted_arr)  # [11, 12, 22, 25, 34, 64, 90]

# In-place sorting
from simple_quicksort import quicksort_inplace
arr = [64, 34, 25, 12, 22, 11, 90]
quicksort_inplace(arr)
print(arr)  # [11, 12, 22, 25, 34, 64, 90]

# With different pivot strategies
from quicksort import quicksort_with_different_pivots
arr = [64, 34, 25, 12, 22, 11, 90]
result = quicksort_with_different_pivots(arr, "middle")
print(result)  # [11, 12, 22, 25, 34, 64, 90]
```

This implementation provides a solid foundation for understanding one of the
most important sorting algorithms in computer science!
"""

print("README file created successfully!")
print("\nQuicksort implementation is complete with:")
print("✓ Two different implementations (simple and in-place)")
print("✓ Detailed step-by-step explanations")
print("✓ Multiple pivot selection strategies")
print("✓ Comprehensive test suite")
print("✓ Educational demonstrations")
print("✓ Performance analysis")
print("\nAll files are ready to run and demonstrate the three main steps:")
print("1. Pivot Selection")
print("2. Partitioning")
print("3. Recursive Sorting")