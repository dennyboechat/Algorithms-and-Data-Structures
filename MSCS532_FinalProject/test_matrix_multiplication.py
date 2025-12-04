"""
Test Suite for Matrix Multiplication Implementation
MSCS 532 - Final Project

This module contains comprehensive tests for the matrix multiplication implementation.
"""

import unittest
import sys
import os

# Add the current directory to the path to import our matrix module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from matrix_multiplication import (
    Matrix, matrix_multiply,
    create_zero_matrix, create_identity_matrix, create_random_matrix,
    matrix_add, matrix_transpose
)


class TestMatrix(unittest.TestCase):
    """Test cases for the Matrix class."""
    
    def setUp(self):
        """Set up test matrices."""
        self.matrix_2x3 = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        self.matrix_3x2 = Matrix([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        
        self.matrix_3x3 = Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
    
    def test_matrix_creation(self):
        """Test matrix creation and basic properties."""
        self.assertEqual(self.matrix_2x3.rows, 2)
        self.assertEqual(self.matrix_2x3.cols, 3)
        self.assertEqual(self.matrix_2x3.get_element(0, 0), 1)
        self.assertEqual(self.matrix_2x3.get_element(1, 2), 6)
    
    def test_invalid_matrix_creation(self):
        """Test that invalid matrices raise appropriate errors."""
        with self.assertRaises(ValueError):
            Matrix([])  # Empty matrix
        
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])  # Inconsistent row lengths
    
    def test_element_access(self):
        """Test getting and setting matrix elements."""
        matrix = Matrix([[1, 2], [3, 4]])
        
        # Test valid access
        self.assertEqual(matrix.get_element(0, 1), 2)
        
        # Test setting element
        matrix.set_element(1, 0, 10)
        self.assertEqual(matrix.get_element(1, 0), 10)
        
        # Test invalid access
        with self.assertRaises(IndexError):
            matrix.get_element(2, 0)
        
        with self.assertRaises(IndexError):
            matrix.set_element(-1, 0, 5)
    
    def test_matrix_dimensions(self):
        """Test getting matrix dimensions."""
        self.assertEqual(self.matrix_2x3.get_dimensions(), (2, 3))
        self.assertEqual(self.matrix_3x2.get_dimensions(), (3, 2))


class TestMatrixMultiplication(unittest.TestCase):
    """Test cases for matrix multiplication algorithms."""
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        
        result = matrix_multiply(a, b)
        expected = Matrix([[19, 22], [43, 50]])
        
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.cols, 2)
        
        for i in range(2):
            for j in range(2):
                self.assertEqual(result.get_element(i, j), expected.get_element(i, j))
    
    def test_non_square_multiplication(self):
        """Test multiplication of non-square matrices."""
        a = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        b = Matrix([[1, 2], [3, 4], [5, 6]])  # 3x2
        
        result = matrix_multiply(a, b)
        expected = Matrix([[22, 28], [49, 64]])
        
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.cols, 2)
        
        for i in range(2):
            for j in range(2):
                self.assertEqual(result.get_element(i, j), expected.get_element(i, j))
    
    def test_incompatible_dimensions(self):
        """Test that incompatible matrices raise ValueError."""
        a = Matrix([[1, 2]])  # 1x2
        b = Matrix([[1], [2], [3]])  # 3x1
        
        with self.assertRaises(ValueError):
            matrix_multiply(a, b)
    
    def test_identity_multiplication(self):
        """Test multiplication with identity matrix."""
        a = Matrix([[1, 2], [3, 4]])
        identity = create_identity_matrix(2)
        
        result = matrix_multiply(a, identity)
        
        # Result should be the same as original matrix
        for i in range(2):
            for j in range(2):
                self.assertEqual(result.get_element(i, j), a.get_element(i, j))
    
    def test_zero_multiplication(self):
        """Test multiplication with zero matrix."""
        a = Matrix([[1, 2], [3, 4]])
        zero = create_zero_matrix(2, 2)
        
        result = matrix_multiply(a, zero)
        
        # Result should be all zeros
        for i in range(2):
            for j in range(2):
                self.assertEqual(result.get_element(i, j), 0)
    
    def test_large_matrix_multiplication(self):
        """Test multiplication of larger matrices."""
        # Create 10x10 matrices
        a = create_random_matrix(10, 10, 1, 5)
        b = create_random_matrix(10, 10, 1, 5)
        
        result = matrix_multiply(a, b)
        
        self.assertEqual(result.rows, 10)
        self.assertEqual(result.cols, 10)
        
        # Verify a specific calculation manually for first element
        expected_00 = sum(a.get_element(0, k) * b.get_element(k, 0) for k in range(10))
        self.assertEqual(result.get_element(0, 0), expected_00)


class TestMatrixOperations(unittest.TestCase):
    """Test cases for additional matrix operations."""
    
    def test_matrix_addition(self):
        """Test matrix addition."""
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        
        result = matrix_add(a, b)
        expected = Matrix([[6, 8], [10, 12]])
        
        for i in range(2):
            for j in range(2):
                self.assertEqual(result.get_element(i, j), expected.get_element(i, j))
    
    def test_incompatible_addition(self):
        """Test that incompatible matrices cannot be added."""
        a = Matrix([[1, 2]])
        b = Matrix([[1], [2]])
        
        with self.assertRaises(ValueError):
            matrix_add(a, b)
    
    def test_matrix_transpose(self):
        """Test matrix transpose operation."""
        a = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        
        result = matrix_transpose(a)
        expected = Matrix([[1, 4], [2, 5], [3, 6]])  # 3x2
        
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.cols, 2)
        
        for i in range(3):
            for j in range(2):
                self.assertEqual(result.get_element(i, j), expected.get_element(i, j))
    
    def test_identity_matrix_creation(self):
        """Test identity matrix creation."""
        identity = create_identity_matrix(3)
        
        self.assertEqual(identity.rows, 3)
        self.assertEqual(identity.cols, 3)
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertEqual(identity.get_element(i, j), 1)
                else:
                    self.assertEqual(identity.get_element(i, j), 0)
    
    def test_zero_matrix_creation(self):
        """Test zero matrix creation."""
        zero = create_zero_matrix(2, 3)
        
        self.assertEqual(zero.rows, 2)
        self.assertEqual(zero.cols, 3)
        
        for i in range(2):
            for j in range(3):
                self.assertEqual(zero.get_element(i, j), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_single_element_matrices(self):
        """Test multiplication of 1x1 matrices."""
        a = Matrix([[5]])
        b = Matrix([[3]])
        
        result = matrix_multiply(a, b)
        
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.cols, 1)
        self.assertEqual(result.get_element(0, 0), 15)
    
    def test_row_vector_column_vector(self):
        """Test row vector × column vector multiplication."""
        row = Matrix([[1, 2, 3]])  # 1x3
        col = Matrix([[4], [5], [6]])  # 3x1
        
        result = matrix_multiply(row, col)
        
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.cols, 1)
        self.assertEqual(result.get_element(0, 0), 32)  # 1*4 + 2*5 + 3*6 = 32
    
    def test_column_vector_row_vector(self):
        """Test column vector × row vector multiplication."""
        col = Matrix([[1], [2], [3]])  # 3x1
        row = Matrix([[4, 5, 6]])  # 1x3
        
        result = matrix_multiply(col, row)
        expected = Matrix([
            [4, 5, 6],
            [8, 10, 12],
            [12, 15, 18]
        ])
        
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.cols, 3)
        
        for i in range(3):
            for j in range(3):
                self.assertEqual(result.get_element(i, j), expected.get_element(i, j))


class TestPerformance(unittest.TestCase):
    """Test performance-related aspects."""
    
    def test_algorithm_performance(self):
        """Test that algorithm doesn't crash on moderate-sized matrices."""
        # This is more of a smoke test to ensure algorithm works
        a = create_random_matrix(50, 30, 1, 10)
        b = create_random_matrix(30, 40, 1, 10)
        
        # Should not raise any exceptions
        result = matrix_multiply(a, b)
        
        # Check result dimensions
        self.assertEqual(result.rows, 50)
        self.assertEqual(result.cols, 40)


def run_tests():
    """Run all tests and display results."""
    print("Running Matrix Multiplication Tests")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMatrix,
        TestMatrixMultiplication,
        TestMatrixOperations,
        TestEdgeCases,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)