"""
Matrix Multiplication Implementation using 2D Arrays
MSCS 532 - Final Project

This module implements matrix multiplication using 2D arrays (lists of lists) in Python.
It includes various approaches and optimizations for matrix multiplication.
"""

class Matrix:
    """
    A class to represent a matrix using 2D arrays (lists of lists).
    Provides methods for matrix operations including multiplication.
    """
    
    def __init__(self, data):
        """
        Initialize a matrix from a 2D list.
        
        Args:
            data: 2D list representing the matrix
        
        Raises:
            ValueError: If the input is not a valid matrix (rows of different lengths)
        """
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError("Matrix must be a non-empty list of lists")
        
        row_length = len(data[0])
        if not all(len(row) == row_length for row in data):
            raise ValueError("All rows must have the same length")
        
        self.data = [row[:] for row in data]  # Deep copy
        self.rows = len(data)
        self.cols = len(data[0])
    
    def __str__(self):
        """String representation of the matrix."""
        return '\n'.join(['\t'.join(map(str, row)) for row in self.data])
    
    def __repr__(self):
        """Detailed string representation of the matrix."""
        return f"Matrix({self.rows}x{self.cols}):\n{self.__str__()}"
    
    def get_element(self, i, j):
        """Get element at position (i, j)."""
        if 0 <= i < self.rows and 0 <= j < self.cols:
            return self.data[i][j]
        raise IndexError("Matrix index out of range")
    
    def set_element(self, i, j, value):
        """Set element at position (i, j)."""
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.data[i][j] = value
        else:
            raise IndexError("Matrix index out of range")
    
    def get_dimensions(self):
        """Return matrix dimensions as (rows, columns)."""
        return (self.rows, self.cols)


def matrix_multiply(matrix_a, matrix_b):
    """
    Matrix multiplication using the standard algorithm.
    Time Complexity: O(n^3)
    Space Complexity: O(n^2)
    
    Args:
        matrix_a: Matrix object or 2D list
        matrix_b: Matrix object or 2D list
    
    Returns:
        Matrix object containing the result
    
    Raises:
        ValueError: If matrices cannot be multiplied (incompatible dimensions)
    """
    # Convert to Matrix objects if needed
    if not isinstance(matrix_a, Matrix):
        matrix_a = Matrix(matrix_a)
    if not isinstance(matrix_b, Matrix):
        matrix_b = Matrix(matrix_b)
    
    # Check if multiplication is possible
    if matrix_a.cols != matrix_b.rows:
        raise ValueError(f"Cannot multiply {matrix_a.rows}x{matrix_a.cols} matrix "
                        f"with {matrix_b.rows}x{matrix_b.cols} matrix")
    
    # Initialize result matrix with zeros
    result_data = [[0 for _ in range(matrix_b.cols)] for _ in range(matrix_a.rows)]
    
    # Perform matrix multiplication
    for i in range(matrix_a.rows):
        for j in range(matrix_b.cols):
            for k in range(matrix_a.cols):
                result_data[i][j] += matrix_a.data[i][k] * matrix_b.data[k][j]
    
    return Matrix(result_data)


def create_zero_matrix(rows, cols):
    """Create a matrix filled with zeros."""
    return Matrix([[0 for _ in range(cols)] for _ in range(rows)])


def create_identity_matrix(size):
    """Create an identity matrix of given size."""
    data = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    return Matrix(data)


def create_random_matrix(rows, cols, min_val=0, max_val=10):
    """Create a matrix with random integers."""
    import random
    data = [[random.randint(min_val, max_val) for _ in range(cols)] 
            for _ in range(rows)]
    return Matrix(data)


def matrix_add(matrix_a, matrix_b):
    """
    Add two matrices.
    
    Args:
        matrix_a: Matrix object or 2D list
        matrix_b: Matrix object or 2D list
    
    Returns:
        Matrix object containing the result
    """
    if not isinstance(matrix_a, Matrix):
        matrix_a = Matrix(matrix_a)
    if not isinstance(matrix_b, Matrix):
        matrix_b = Matrix(matrix_b)
    
    if matrix_a.rows != matrix_b.rows or matrix_a.cols != matrix_b.cols:
        raise ValueError("Matrices must have the same dimensions for addition")
    
    result_data = []
    for i in range(matrix_a.rows):
        row = []
        for j in range(matrix_a.cols):
            row.append(matrix_a.data[i][j] + matrix_b.data[i][j])
        result_data.append(row)
    
    return Matrix(result_data)


def matrix_transpose(matrix):
    """
    Transpose a matrix.
    
    Args:
        matrix: Matrix object or 2D list
    
    Returns:
        Matrix object containing the transposed matrix
    """
    if not isinstance(matrix, Matrix):
        matrix = Matrix(matrix)
    
    result_data = []
    for j in range(matrix.cols):
        row = []
        for i in range(matrix.rows):
            row.append(matrix.data[i][j])
        result_data.append(row)
    
    return Matrix(result_data)





def demonstrate_matrix_operations():
    """
    Demonstrate various matrix operations with examples.
    """
    print("Matrix Multiplication Demonstration")
    print("=" * 40)
    
    # Example 1: Basic 3x3 multiplication
    print("\nExample 1: 3x3 Matrix Multiplication")
    a = Matrix([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    b = Matrix([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ])
    
    print("Matrix A:")
    print(a)
    print("Matrix B:")
    print(b)
    
    result = matrix_multiply(a, b)
    print("\nA × B =")
    print(result)
    
    # Example 2: Non-square matrices
    print("\n" + "="*40)
    print("Example 2: Non-square Matrix Multiplication")
    
    c = Matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])
    
    d = Matrix([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    
    print("Matrix C (2x4):")
    print(c)
    print("\nMatrix D (4x2):")
    print(d)
    
    result = matrix_multiply(c, d)
    print("\nC × D =")
    print(result)
    
    # Example 3: Identity matrix multiplication
    print("\n" + "="*40)
    print("Example 3: Identity Matrix Multiplication")
    
    identity = create_identity_matrix(3)
    print("Identity Matrix (3x3):")
    print(identity)
    
    result = matrix_multiply(a, identity)
    print("\nA × I =")
    print(result)
    
    # Example 4: Matrix addition and transpose
    print("\n" + "="*40)
    print("Example 4: Matrix Addition and Transpose")
    
    e = Matrix([
        [1, 2],
        [3, 4]
    ])
    
    f = Matrix([
        [5, 6],
        [7, 8]
    ])
    
    print("Matrix E:")
    print(e)
    print("\nMatrix F:")
    print(f)
    
    sum_result = matrix_add(e, f)
    print("\nE + F =")
    print(sum_result)
    
    transpose_e = matrix_transpose(e)
    print("\nTranspose of E:")
    print(transpose_e)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_matrix_operations()