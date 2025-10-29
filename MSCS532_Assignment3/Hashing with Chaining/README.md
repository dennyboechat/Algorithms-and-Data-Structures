# Hash Table with Chaining Implementation

A comprehensive Python implementation of a hash table using chaining for collision resolution, featuring a universal hash function family to minimize collisions.

## Analysis

The file analysis.md contains a details analysis of results.

## Features

### Core Operations
- **Insert**: Add key-value pairs with O(1) average time complexity
- **Search**: Retrieve values by key with O(1) average time complexity  
- **Delete**: Remove key-value pairs with O(1) average time complexity

### Advanced Features
- **Universal Hash Function Family**: Minimizes collisions using randomized hash functions
- **Dynamic Resizing**: Automatically expands when load factor exceeds threshold
- **Collision Resolution**: Efficient chaining with linked lists
- **Multiple Data Types**: Supports any hashable key type (strings, integers, tuples, etc.)
- **Pythonic Interface**: Supports `[]` operator, `in` keyword, `len()`, iterators

### Performance Optimizations
- Smart load factor management (default threshold: 0.75)
- Prime number table sizes for better distribution
- Collision tracking and statistics
- Memory-efficient node structure

## Implementation Details

### Hash Function
The implementation uses a **universal hash function family** with the formula:
```
h(k) = ((a * k + b) mod p) mod m
```
Where:
- `p` is a large prime number (2³¹ - 1)
- `a` and `b` are random coefficients
- `m` is the table size
- `k` is the integer representation of the key

### Collision Resolution
**Chaining** is used to handle collisions:
- Each table slot contains a linked list of nodes
- New colliding elements are appended to the chain
- Search traverses the chain to find the target key

### Dynamic Resizing
The table automatically resizes when:
- Load factor exceeds the threshold (default: 0.75)
- New size is the next prime ≥ 2 × current size
- All elements are rehashed with a new hash function

## Usage Examples

### Basic Operations

```python
from hash_table import HashTableChaining

# Create hash table
ht = HashTableChaining()

# Insert key-value pairs
ht.insert("apple", 5)
ht.insert("banana", 3)
ht.insert("orange", 8)

# Search for values
value = ht.search("apple")  # Returns 5
missing = ht.search("grape")  # Returns None

# Delete keys
success = ht.delete("banana")  # Returns True
failed = ht.delete("grape")   # Returns False

# Check existence
if ht.contains("apple"):
    print("Apple found!")
```

### Pythonic Interface

```python
# Dictionary-like syntax
ht["key1"] = "value1"        # Insert
value = ht["key1"]           # Search
del ht["key1"]               # Delete

# Membership testing
if "key1" in ht:
    print("Key exists")

# Length and iteration
print(f"Size: {len(ht)}")

for key in ht.keys():
    print(key)

for value in ht.values():
    print(value)

for key, value in ht.items():
    print(f"{key}: {value}")
```

### Advanced Usage

```python
# Custom configuration
ht = HashTableChaining(
    initial_size=100,           # Start with 100 slots
    load_factor_threshold=0.8   # Resize at 80% load
)

# Performance statistics
stats = ht.get_statistics()
print(f"Load factor: {stats['load_factor']:.2f}")
print(f"Collisions: {stats['collision_count']}")
print(f"Max chain length: {stats['max_chain_length']}")

# Display table structure
ht.display()
```

## Performance Analysis

### Time Complexity

| Operation | Average Case | Worst Case | Best Case |
|-----------|-------------|------------|-----------|
| Insert    | O(1)        | O(n)       | O(1)      |
| Search    | O(1)        | O(n)       | O(1)      |
| Delete    | O(1)        | O(n)       | O(1)      |

### Space Complexity
- **Storage**: O(n) where n is the number of key-value pairs
- **Overhead**: O(m) where m is the table size

### Load Factor Impact
- **< 0.5**: Excellent performance, low collision rate
- **0.5 - 0.75**: Good performance, acceptable collision rate  
- **0.75 - 1.0**: Degraded performance, higher collision rate
- **> 1.0**: Poor performance, automatic resize triggered

### Benchmark Results
Example performance with 10,000 operations:

```
Table Size    Insert Time    Search Time    Delete Time    Collisions
50            12.45ms        8.32ms         6.78ms         1,234
100           11.23ms        7.89ms         5.91ms         856
500           10.87ms        7.12ms         5.45ms         234
1000          10.92ms        7.08ms         5.42ms         123
```

## Testing

### Running Tests

```bash
# Run all tests
python test_hash_table.py

# Run specific test class
python -m unittest test_hash_table.TestHashTableChaining

# Run with verbose output
python test_hash_table.py -v
```

### Test Coverage

The test suite includes:

#### Core Functionality Tests
- ✅ Hash table initialization
- ✅ Insert operations (single, multiple, duplicates)
- ✅ Search operations (existing, non-existent keys)
- ✅ Delete operations (various scenarios)
- ✅ Magic methods (`__getitem__`, `__setitem__`, etc.)

#### Edge Cases
- ✅ Empty string keys
- ✅ None values
- ✅ Large keys (10,000+ characters)
- ✅ Unicode keys
- ✅ Various data types

#### Performance Tests
- ✅ Collision handling
- ✅ Resize behavior
- ✅ Stress testing (10,000+ operations)
- ✅ Hash distribution quality

#### Error Conditions
- ✅ KeyError handling
- ✅ Invalid operations
- ✅ Memory constraints

### File Descriptions

- **`hash_table.py`**: Core implementation containing:
  - `HashNode`: Linked list node for chaining
  - `UniversalHashFunction`: Hash function implementation
  - `HashTableChaining`: Main hash table class

- **`test_hash_table.py`**: Complete test suite with:
  - Unit tests for all components
  - Edge case testing
  - Performance validation
  - Error condition testing

- **`demo.py`**: Interactive demonstration featuring:
  - Basic operations walkthrough
  - Collision scenario analysis
  - Performance benchmarking
  - Stress testing

## Running the Demo

```bash
# Run complete demonstration
python demo.py

# Run basic hash table example
python hash_table.py
```

The demo script provides:
- Step-by-step operation examples
- Collision handling visualization
- Performance analysis
- Hash distribution testing
- Stress testing with 10,000+ items

## Requirements

- **Python**: 3.6+ (uses type hints)
- **Standard Library**: No external dependencies
- **Optional**: `matplotlib` for visualization (not required)

## Implementation Highlights

### Universal Hash Function Benefits
- **Theoretical Guarantee**: Expected O(1) performance
- **Collision Minimization**: Random coefficients reduce clustering
- **Adaptability**: New function generated on resize

### Chaining Advantages
- **Simplicity**: Easy to implement and understand
- **Flexibility**: Handles any number of collisions
- **Cache Locality**: Better than open addressing for large keys

### Dynamic Resizing Strategy
- **Prime Sizes**: Reduces clustering and improves distribution
- **Load Factor Control**: Maintains performance characteristics
- **Rehashing**: Ensures optimal hash function parameters

## Complexity Analysis

### Theoretical Analysis
With a good hash function and proper load factor:
- **Average case**: O(1) for all operations
- **Load factor α = n/m**: Expected chain length is α
- **Universal hashing**: Guarantees expected O(1) performance

### Practical Performance
Real-world performance depends on:
- **Key distribution**: Well-distributed keys perform better
- **Load factor**: Keep below 0.75 for optimal performance
- **Hash function quality**: Universal family ensures good behavior

## Customization Options

### Hash Function Tuning
```python
# Modify in UniversalHashFunction.__init__()
self.prime = 2**31 - 1  # Change prime for different universe
```

### Resize Strategy
```python
# Modify in HashTableChaining.__init__()
load_factor_threshold = 0.75  # Adjust resize trigger
```

### Performance Monitoring
```python
# Enable detailed statistics
stats = ht.get_statistics()
print(f"Collisions: {stats['collision_count']}")
print(f"Chain lengths: max={stats['max_chain_length']}, avg={stats['avg_chain_length']:.2f}")
```

---