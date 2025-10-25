# Insertion Sort Algorithm (Descending Order)

## Overview

This repository contains an implementation of the **Insertion Sort algorithm** that sorts an array in **monotonically decreasing order** (from largest to smallest).

## Algorithm Description

Insertion Sort is a simple sorting algorithm that builds the final sorted array one item at a time. It works by taking elements from the unsorted portion and inserting them into their correct position in the sorted portion of the array.

### Key Characteristics:
- **Time Complexity**: O(n²) in the worst case, O(n) in the best case (already sorted)
- **Space Complexity**: O(1) - sorts in-place
- **Stable**: Yes - maintains relative order of equal elements
- **Adaptive**: Yes - performs better on partially sorted arrays

### How it Works (Descending Order):
1. Start with the second element (index 1) as the first element is considered sorted
2. Compare the current element with elements in the sorted portion
3. Shift elements that are **smaller** than the current element to the right
4. Insert the current element in its correct position
5. Repeat until all elements are processed

This implementation is based on the algorithm described in:
> Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.).

## Files

- `insertion_sort_descending.py` - Main implementation of the insertion sort algorithm
- `hello_world.py` - Alternative file with the same implementation

## Usage

### Prerequisites
- Python 3.x installed on your system

### Running the Program

1. **Clone or download the repository**
2. **Navigate to the project directory**:
   ```bash
   cd MSCS532_Assignment1
   ```

3. **Run the program**:
   ```bash
   python insertion_sort_descending.py
   ```
   
   or
   
   ```bash
   python hello_world.py
   ```

### Example Output
```
Sorted array (decreasing): [9, 6, 5, 5, 2, 1]
```

## Code Example

```python
def insertion_sort_descending(arr):
    """
    Sorts the input list 'arr' in monotonically decreasing order
    using the Insertion Sort algorithm.
    """
    for j in range(1, len(arr)):
        key = arr[j]
        i = j - 1

        # Move elements that are smaller than key to one position ahead
        # of their current position to achieve decreasing order
        while i >= 0 and arr[i] < key:
            arr[i + 1] = arr[i]
            i -= 1

        arr[i + 1] = key

    return arr

# Example usage
numbers = [5, 2, 9, 1, 5, 6]
sorted_numbers = insertion_sort_descending(numbers)
print("Sorted array (decreasing):", sorted_numbers)
```

## Algorithm Walkthrough

Given the input array `[5, 2, 9, 1, 5, 6]`:

1. **Initial**: `[5, 2, 9, 1, 5, 6]`
2. **Step 1**: Compare 2 with 5 → `[5, 2, 9, 1, 5, 6]` (no change, 5 > 2)
3. **Step 2**: Compare 9 with 5, 2 → `[9, 5, 2, 1, 5, 6]` (9 is largest)
4. **Step 3**: Compare 1 with 9, 5, 2 → `[9, 5, 2, 1, 5, 6]` (no change, all > 1)
5. **Step 4**: Compare 5 with 9, 5, 2, 1 → `[9, 5, 5, 2, 1, 6]` (insert between 5s)
6. **Step 5**: Compare 6 with 9, 5, 5, 2, 1 → `[9, 6, 5, 5, 2, 1]` (insert after 9)

**Final Result**: `[9, 6, 5, 5, 2, 1]`

## Learning Objectives

This implementation demonstrates:
- Understanding of the insertion sort algorithm
- Modification of standard algorithms for different sorting orders
- In-place sorting techniques
- Algorithm analysis and complexity understanding

## License

This project is for educational purposes as part of computer science coursework.