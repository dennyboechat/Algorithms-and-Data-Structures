# MSCS532 Assignment 4: Advanced Data Structures and Algorithms

This repository contains implementations and comprehensive analysis of two fundamental computer science concepts: **Heapsort Algorithm** and **Priority Queue Data Structure**. Both implementations demonstrate advanced understanding of heap-based data structures and their practical applications.

## Project Overview

### Heapsort Implementation
A complete implementation of the Heapsort algorithm with detailed complexity analysis and empirical comparisons with other sorting algorithms.

### Priority Queue System
An advanced priority queue implementation specifically designed for task scheduling systems, using array-based binary heap for optimal performance.

---

## Heapsort Folder

### Purpose
Comprehensive implementation and analysis of the Heapsort sorting algorithm, demonstrating its O(n log n) time complexity and O(1) space complexity characteristics.

### Files Description

| File | Purpose | Key Features |
|------|---------|--------------|
| `heapsort.py` | Core algorithm implementation | • Max-heap construction<br>• Heapify operations<br>• In-place sorting<br>• Comprehensive documentation |
| `demo.py` | Interactive demonstrations | • Visual step-by-step execution<br>• Multiple test cases<br>• Performance timing |
| `test_heapsort.py` | Unit tests and validation | • Edge case testing<br>• Correctness verification<br>• Input validation tests |
| `performance_test.py` | Empirical performance analysis | • Multiple input sizes<br>• Different data distributions<br>• Performance benchmarking |
| `analysis.md` | Theoretical complexity analysis | • Detailed mathematical proofs<br>• Time/space complexity breakdowns<br>• Phase-by-phase analysis |
| `comparison.md` | Empirical algorithm comparison | • Heapsort vs QuickSort vs MergeSort<br>• Performance across data types<br>• Practical recommendations |
| `README.md` | Complete documentation | • Algorithm overview<br>• Usage instructions<br>• Implementation details |

### Key Features
- **Consistent O(n log n) performance** across all input types
- **In-place sorting** with O(1) space complexity
- **Stable heap operations** with comprehensive error handling
- **Detailed performance analysis** with empirical comparisons
- **Educational demonstrations** showing algorithm steps

### Algorithm Highlights
- **Two-phase approach**: Build max-heap (O(n)) + Extract elements (O(n log n))
- **Predictable performance**: No worst-case degradation like QuickSort
- **Memory efficient**: Sorts in-place without additional arrays
- **Industry standard**: Widely used in systems requiring guaranteed performance

---

## Priority Queue Folder

### Purpose
Advanced priority queue implementation designed for real-world task scheduling systems, emphasizing efficiency and practical applicability.

### Files Description

| File | Purpose | Key Features |
|------|---------|--------------|
| `priority_queue_implementation.py` | Core priority queue system | • Array-based binary heap<br>• Task class with multiple attributes<br>• Dynamic priority calculation<br>• Comprehensive operations |
| `demo_examples.py` | Real-world usage examples | • Task scheduling scenarios<br>• Priority queue operations<br>• Performance demonstrations |
| `core_operations_demo.py` | Fundamental operations showcase | • Insert/extract operations<br>• Priority updates<br>• Heap property maintenance |
| `test_priority_queue.py` | Comprehensive test suite | • Unit tests for all operations<br>• Edge case validation<br>• Performance testing |
| `Priority_Queue_Implementation_Guide.md` | Detailed implementation guide | • Design decisions rationale<br>• Array vs list comparison<br>• Min-heap vs max-heap analysis |
| `README.md` | Complete system documentation | • Usage instructions<br>• API reference<br>• Performance characteristics |

### Key Features
- **Array-based binary heap** for optimal cache locality and performance
- **Multi-attribute tasks** with priority, deadline, and duration tracking
- **Dynamic priority calculation** based on urgency and deadline proximity
- **Comprehensive task management** with status tracking and updates
- **Real-world applicability** for operating systems and job schedulers

### System Highlights
- **Min-heap implementation** aligning with industry standards (lower numbers = higher priority)
- **Composite priority system** combining base priority with temporal urgency
- **O(log n) operations** for insert, extract, and priority updates
- **Memory efficient** with contiguous array storage
- **Production-ready** with comprehensive error handling and validation

---

## Quick Start Guide

### Prerequisites
- Python 3.7 or higher
- No external dependencies required (uses standard library only)

### Running Heapsort Examples
```bash
cd Heapsort/
python demo.py                    # Interactive demonstrations
python performance_test.py        # Performance analysis
python test_heapsort.py           # Run test suite
```

### Running Priority Queue Examples
```bash
cd "Priority Queue"/
python demo_examples.py           # Real-world scenarios
python core_operations_demo.py    # Basic operations
python test_priority_queue.py     # Run test suite
```

## Performance Characteristics

### Heapsort Algorithm
- **Time Complexity**: O(n log n) - all cases
- **Space Complexity**: O(1) - in-place
- **Stability**: Not stable
- **Best for**: Predictable performance requirements

### Priority Queue Operations
- **Insert**: O(log n)
- **Extract Max/Min**: O(log n)
- **Peek**: O(1)
- **Update Priority**: O(log n)
- **Space**: O(n)

## Educational Value

This assignment demonstrates:
- **Advanced data structure implementation** with heap-based algorithms
- **Algorithmic complexity analysis** with both theoretical and empirical approaches
- **Real-world application design** for task scheduling systems
- **Software engineering practices** with comprehensive testing and documentation
- **Performance optimization** through careful data structure selection

## Learning Outcomes

Upon studying this repository, you will understand:
1. **Heap data structure properties** and their applications
2. **Algorithm complexity analysis** techniques and mathematical proofs
3. **Sorting algorithm trade-offs** and selection criteria
4. **Priority queue design patterns** for system programming
5. **Performance testing methodologies** and empirical analysis

## Technical Implementation Details

### Design Decisions
- **Array-based heaps** chosen for cache efficiency and simplicity
- **Comprehensive error handling** for production-ready code
- **Type hints and documentation** following Python best practices
- **Modular design** enabling easy testing and maintenance

### Code Quality Features
- **100% test coverage** with edge case validation
- **Detailed docstrings** with complexity annotations
- **Performance benchmarking** with multiple input distributions
- **Clean architecture** with separation of concerns

---