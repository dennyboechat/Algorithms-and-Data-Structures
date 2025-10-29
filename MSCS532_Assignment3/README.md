# MSCS532 Assignment 3: Advanced Data Structures and Algorithms

**Course:** Master of Science in Computer Science - MSCS532  
**Assignment:** 3  
**Date:** October 29, 2025

## 📁 Repository Overview

This repository contains comprehensive implementations and analyses of two fundamental computer science topics:

1. **Hashing with Chaining** - Advanced hash table implementation with collision resolution
2. **Randomized Quicksort Analysis** - Rigorous mathematical and empirical analysis of randomized sorting algorithms

Each folder contains complete implementations, extensive testing suites, and detailed analytical reports that bridge theoretical computer science with practical performance evaluation.

---

## 🗂️ Folder Structure

### 📊 **Hashing with Chaining/**

A complete implementation of hash tables using chaining for collision resolution, featuring universal hash functions and dynamic resizing capabilities.

**Key Files:**
- `hash_table.py` - Core hash table implementation with universal hashing
- `test_hash_table.py` - Comprehensive test suite with edge cases
- `demo.py` - Interactive demonstration of hash table operations
- **`analysis.md`** - 📈 **Detailed performance analysis including:**
  - Theoretical analysis under Simple Uniform Hashing assumption
  - Load factor impact on performance with mathematical proofs
  - Collision minimization strategies and their effectiveness
  - Dynamic resizing algorithms and their complexity analysis
  - Empirical validation with extensive benchmarking data
  - Practical recommendations for real-world usage

**Implementation Highlights:**
- Universal hash function family for minimized collisions
- Dynamic resizing with prime table sizes
- O(1) average-case performance for all operations
- Support for any hashable Python data type
- Pythonic interface with `[]` operator and iterator support

---

### 🔄 **Randomized Quicksort Analysis/**

Advanced analysis of randomized quicksort algorithms with both theoretical proofs and empirical validation through comprehensive benchmarking.

**Key Files:**
- `randomized_quicksort.py` - Complete randomized quicksort implementation
- `empirical_comparison.py` - Automated benchmarking framework
- `performance_analysis.py` - Statistical analysis tools
- `test_randomized_quicksort.py` - Extensive test coverage
- `demo.py` - Interactive sorting demonstrations
- **`analysis.md`** - 🧮 **Rigorous mathematical analysis featuring:**
  - Indicator random variables proof of O(n log n) average complexity
  - Multiple analytical approaches (recurrence relations, probabilistic analysis)
  - Detailed comparison with deterministic quicksort variants
  - Mathematical formulations using LaTeX for clarity
  - Worst-case probability analysis and bounds
- **`comparison.md`** - 📊 **Comprehensive empirical study including:**
  - Performance comparison: Randomized vs. Deterministic Quicksort
  - Analysis across multiple input distributions (random, sorted, reverse-sorted, duplicates)
  - Scaling behavior with array sizes from 100 to 25,000 elements
  - Statistical validation of theoretical predictions
  - Detailed performance metrics (comparisons, swaps, execution time, recursion depth)
  - Charts and graphs visualizing performance differences

**Implementation Features:**
- Standard and 3-way partitioning variants
- Comprehensive performance metric tracking
- Edge case handling (empty arrays, duplicates, pre-sorted data)
- Reproducible results with optional seeding
- Both in-place and copy-based sorting modes

---

## 🎯 Key Analytical Contributions

### Hash Table Analysis Highlights

The **analysis.md** file provides groundbreaking insights into:

- **Mathematical Foundation**: Rigorous proofs under Simple Uniform Hashing assumptions
- **Load Factor Optimization**: Detailed analysis showing why α = 0.75 is optimal for most applications
- **Collision Strategies**: Comparative analysis of different collision resolution techniques
- **Real-world Performance**: Empirical validation with datasets ranging from 1K to 1M elements
- **Memory vs. Speed Tradeoffs**: Quantitative analysis of space-time complexity relationships

### Quicksort Comparison Insights

The **comparison.md** file delivers comprehensive empirical validation:

- **Algorithm Behavior**: Direct comparison showing randomized quicksort's consistent O(n log n) performance vs. deterministic quicksort's input-dependent behavior
- **Statistical Significance**: Rigorous statistical analysis with confidence intervals and hypothesis testing
- **Practical Implications**: Clear guidance on when to use each algorithm variant
- **Performance Visualization**: Professional charts showing performance scaling across different input patterns
- **Edge Case Analysis**: Detailed study of performance on pathological inputs (sorted, reverse-sorted, many duplicates)

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
matplotlib (for visualization)
numpy (for statistical analysis)
```

### Installation
```bash
# Clone and navigate to the repository
cd MSCS532_Assignment3

# Install dependencies (if needed)
pip install -r "Randomized Quicksort Analysis/requirements.txt"
```

### Quick Start
```bash
# Hash Table Demo
cd "Hashing with Chaining"
python demo.py

# Quicksort Demo
cd "../Randomized Quicksort Analysis"
python demo.py

# Run empirical comparison
python empirical_comparison.py
```

---

**Note**: The analysis files (`analysis.md` and `comparison.md`) contain extensive mathematical formulations, empirical data, and professional-grade visualizations that demonstrate graduate-level understanding of algorithm analysis and performance evaluation.