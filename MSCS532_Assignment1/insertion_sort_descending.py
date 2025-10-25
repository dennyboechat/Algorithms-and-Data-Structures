def insertion_sort_descending(arr):
    for j in range(1, len(arr)):
        key = arr[j]
        i = j - 1

        while i >= 0 and arr[i] < key:
            arr[i + 1] = arr[i]
            i -= 1

        arr[i + 1] = key

    return arr

# Example usage
numbers = [5, 2, 9, 1, 5, 6]
sorted_numbers = insertion_sort_descending(numbers)
print("Sorted array (decreasing):", sorted_numbers)