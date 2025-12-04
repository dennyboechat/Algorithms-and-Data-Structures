"""
Simple Examples of Matrix Multiplication
MSCS 532 - Final Project

This script demonstrates basic usage of the matrix multiplication implementation.
"""

from matrix_multiplication import (
    Matrix, matrix_multiply,
    create_identity_matrix, create_zero_matrix, matrix_add, matrix_transpose
)


def example_1_basic_multiplication():
    """Example 1: Basic 2x2 matrix multiplication."""
    print("Example 1: Basic Matrix Multiplication")
    print("-" * 40)
    
    # Create two 2x2 matrices
    a = Matrix([
        [2, 3],
        [1, 4]
    ])
    
    b = Matrix([
        [5, 1],
        [2, 6]
    ])
    
    print("Matrix A:")
    print(a)
    print("\nMatrix B:")
    print(b)
    
    # Multiply the matrices
    result = matrix_multiply(a, b)
    
    print("\nA × B =")
    print(result)
    
    # Show the calculation step by step for the first element
    print(f"\nCalculation for result[0][0]: {a.get_element(0,0)} × {b.get_element(0,0)} + {a.get_element(0,1)} × {b.get_element(1,0)} = {result.get_element(0,0)}")
    print()


def example_2_rectangular_matrices():
    """Example 2: Multiplication of rectangular matrices."""
    print("Example 2: Rectangular Matrix Multiplication")
    print("-" * 40)
    
    # Create a 2x3 matrix
    a = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    # Create a 3x2 matrix
    b = Matrix([
        [7, 8],
        [9, 10],
        [11, 12]
    ])
    
    print("Matrix A (2×3):")
    print(a)
    print("\nMatrix B (3×2):")
    print(b)
    
    result = matrix_multiply(a, b)
    
    print("\nA × B (2×2) =")
    print(result)
    print(f"Result dimensions: {result.get_dimensions()}")
    print()


def example_3_identity_and_zero():
    """Example 3: Special matrices (identity and zero)."""
    print("Example 3: Special Matrices")
    print("-" * 40)
    
    # Create a test matrix
    a = Matrix([
        [1, 2],
        [3, 4]
    ])
    
    # Create identity matrix
    identity = create_identity_matrix(2)
    
    # Create zero matrix
    zero = create_zero_matrix(2, 2)
    
    print("Original Matrix A:")
    print(a)
    
    print("\nIdentity Matrix:")
    print(identity)
    
    print("\nZero Matrix:")
    print(zero)
    
    # Multiply with identity (should give original matrix)
    result_identity = matrix_multiply(a, identity)
    print("\nA × I =")
    print(result_identity)
    
    # Multiply with zero (should give zero matrix)
    result_zero = matrix_multiply(a, zero)
    print("\nA × 0 =")
    print(result_zero)
    print()


def example_4_additional_operations():
    """Example 4: Additional matrix operations."""
    print("Example 4: Additional Operations")
    print("-" * 40)
    
    # Create matrices for demonstration
    a = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    b = Matrix([
        [2, 1, 0],
        [1, 3, 2]
    ])
    
    print("Matrix A:")
    print(a)
    print("\nMatrix B:")
    print(b)
    
    # Matrix addition
    sum_result = matrix_add(a, b)
    print("\nA + B =")
    print(sum_result)
    
    # Matrix transpose
    transpose_a = matrix_transpose(a)
    print("\nTranspose of A:")
    print(transpose_a)
    
    # Show that we can multiply A by its transpose
    product = matrix_multiply(a, transpose_a)
    print("\nA × A^T =")
    print(product)
    print(f"Result dimensions: {product.get_dimensions()}")
    print()


def interactive_example():
    """Interactive example where user can input matrix values."""
    print("Interactive Example")
    print("-" * 40)
    print("Let's create a simple 2×2 matrix multiplication example.")
    
    try:
        # Get user input for first matrix
        print("\nEnter values for Matrix A (2×2):")
        a_data = []
        for i in range(2):
            row = []
            for j in range(2):
                value = float(input(f"Enter A[{i}][{j}]: "))
                row.append(value)
            a_data.append(row)
        
        # Get user input for second matrix
        print("\nEnter values for Matrix B (2×2):")
        b_data = []
        for i in range(2):
            row = []
            for j in range(2):
                value = float(input(f"Enter B[{i}][{j}]: "))
                row.append(value)
            b_data.append(row)
        
        # Create matrices
        a = Matrix(a_data)
        b = Matrix(b_data)
        
        print("\nYour matrices:")
        print("Matrix A:")
        print(a)
        print("\nMatrix B:")
        print(b)
        
        # Multiply
        result = matrix_multiply(a, b)
        print("\nA × B =")
        print(result)
        
    except (ValueError, KeyboardInterrupt):
        print("\nSkipping interactive example...")


def main():
    """Run all examples."""
    print("Matrix Multiplication Examples")
    print("=" * 50)
    print()
    
    example_1_basic_multiplication()
    example_2_rectangular_matrices()
    example_3_identity_and_zero()
    example_4_additional_operations()
    
    # Uncomment the line below if you want to try the interactive example
    # interactive_example()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()