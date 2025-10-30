# Priority Queue Implementation for Task Scheduling

This repository contains a comprehensive implementation of a priority queue using array-based binary heap data structure, specifically designed for efficient task scheduling systems.

## Key Features

### Data Structure Choice: Array-Based Binary Heap

**Why Array over List?**
- ✅ **Better cache locality** - contiguous memory allocation
- ✅ **No pointer overhead** - more memory efficient
- ✅ **Simpler implementation** - easy parent-child calculations
- ✅ **O(1) random access** - better performance for heap operations
- ✅ **Industry standard** - widely adopted approach

### Task Design

The `Task` class includes:
- **Task ID** - unique identifier
- **Priority levels** - CRITICAL, HIGH, MEDIUM, LOW, BACKGROUND
- **Temporal attributes** - arrival time, deadline, estimated duration
- **Composite priority** - combines base priority with urgency
- **Dynamic updates** - recalculates priority based on deadline proximity

### Heap Type: Min-Heap

**Why Min-Heap?**
- ✅ **Natural priority representation** - lower numbers = higher priority
- ✅ **Aligns with common systems** - Priority 1 > Priority 2
- ✅ **Library compatibility** - works with Python's heapq
- ✅ **Scheduling algorithm support** - optimal for EDF, SJF, Priority Scheduling

## Quick Start

### Installation

No external dependencies required - uses only Python standard library.

```bash
# Clone or download the files
# No pip install needed!
```

### Core Operations Example

```python
from priority_queue_implementation import Task, TaskPriority, ArrayBasedPriorityQueue
from datetime import datetime, timedelta

# Create priority queue
pq = ArrayBasedPriorityQueue()

# Create tasks
critical_task = Task("URGENT", TaskPriority.CRITICAL, datetime.now(),
                    datetime.now() + timedelta(minutes=5), 300)
normal_task = Task("NORMAL", TaskPriority.MEDIUM, datetime.now(),
                  datetime.now() + timedelta(hours=1), 600)

# 1. INSERT: Add tasks to queue - O(log n)
pq.insert(critical_task)
pq.insert(normal_task)

# 2. IS_EMPTY: Check if queue is empty - O(1)
if not pq.is_empty():
    print(f"Queue has {pq.get_size()} tasks")

# 3. EXTRACT_MIN: Get highest priority task - O(log n)
highest_priority = pq.extract_min()
print(f"Processing: {highest_priority}")

# 4. DECREASE_KEY: Increase task priority - O(log n)
pq.decrease_key("NORMAL", TaskPriority.HIGH)

# 5. INCREASE_KEY: Decrease task priority - O(log n)  
pq.increase_key("NORMAL", TaskPriority.LOW)
```

### Task Scheduler Usage

```python
from priority_queue_implementation import TaskScheduler

# Create scheduler
scheduler = TaskScheduler(use_custom_heap=True)

# Add tasks
scheduler.add_task(critical_task)
scheduler.add_task(normal_task)

# Execute scheduling cycle
scheduler.run_scheduling_cycle(max_tasks=5)

# Check status
print(scheduler.get_status())
```

## Running Tests

Execute the comprehensive test suite:

```bash
python test_priority_queue.py
```

Test coverage includes:
- ✅ Task creation and comparison
- ✅ **insert()** operation with heap property maintenance
- ✅ **extract_min()** operation with heap restoration
- ✅ **decrease_key()** and **increase_key()** operations
- ✅ **is_empty()** and boundary conditions
- ✅ Index mapping consistency
- ✅ Performance benchmarks
- ✅ Scheduler functionality

## 🎮 Running Demonstrations

### Core Operations Demo
See the four fundamental heap operations in action:

```bash
python core_operations_demo.py
```

Demonstrates:
1. **insert()** - Step-by-step heap building with analysis
2. **extract_min()** - Priority-ordered task processing  
3. **decrease_key()** - Dynamic priority increases
4. **increase_key()** - Dynamic priority decreases
5. **is_empty()** - Empty queue detection
6. **Performance Scaling** - Time complexity validation

### Application Examples
See real-world usage scenarios:

```bash
python demo_examples.py
```

Demonstrations include:
1. **Basic Usage** - Simple priority queue operations
2. **OS Scheduler** - Operating system task scheduling
3. **Web Server** - HTTP request prioritization
4. **Hospital ED** - Emergency department triage
5. **Dynamic Updates** - Runtime priority changes
6. **Batch Operations** - Efficient bulk insertions

## Performance Characteristics

### Core Heap Operations

| Operation | Best Case | Average Case | Worst Case | Space | Description |
|-----------|-----------|--------------|------------|-------|-------------|
| **insert()** | O(1) | O(log n) | O(log n) | O(1) | Add task maintaining heap property |
| **extract_min()** | O(1) | O(log n) | O(log n) | O(1) | Remove highest priority task |
| **decrease_key()** | O(1) | O(log n) | O(log n) | O(1) | Increase task priority (bubble up) |
| **increase_key()** | O(1) | O(log n) | O(log n) | O(1) | Decrease task priority (bubble down) |
| **is_empty()** | O(1) | O(1) | O(1) | O(1) | Check if queue is empty |
| **peek_min()** | O(1) | O(1) | O(1) | O(1) | View highest priority task |

**Overall Space Complexity:** O(n) for storing n tasks

### Detailed Operation Analysis

#### 1. insert() - Insert New Task
```python
# Algorithm:
# 1. Add task at end of array
# 2. Update index mapping
# 3. Bubble up to restore heap property

pq.insert(critical_task)  # O(log n)
```

#### 2. extract_min() - Remove Highest Priority Task  
```python
# Algorithm:
# 1. Save root (highest priority)
# 2. Move last element to root
# 3. Update index mapping
# 4. Bubble down to restore heap property

highest_priority = pq.extract_min()  # O(log n)
```

#### 3. decrease_key() - Increase Task Priority
```python
# Algorithm:
# 1. Find task using O(1) index mapping
# 2. Update priority (decrease value = increase actual priority)
# 3. Bubble up to restore heap property

pq.decrease_key("TASK_001", TaskPriority.CRITICAL)  # O(log n)
```

#### 4. increase_key() - Decrease Task Priority
```python
# Algorithm:
# 1. Find task using O(1) index mapping
# 2. Update priority (increase value = decrease actual priority)  
# 3. Bubble down to restore heap property

pq.increase_key("TASK_001", TaskPriority.LOW)  # O(log n)
```

#### 5. is_empty() - Check Empty Status
```python
# Algorithm:
# 1. Check if size equals zero
# 2. Return boolean result

if pq.is_empty():  # O(1)
    print("Queue is empty")
```

## 🏗️ Implementation Details

### Array-Based Heap Structure

```
Parent-Child Relationships:
- Parent of node i: (i-1)/2
- Left child of i: 2*i + 1  
- Right child of i: 2*i + 2
```

### Heapify Operations

- **Heapify Up:** Used after insertion to maintain heap property
- **Heapify Down:** Used after extraction to restore heap property
- **Bulk Heapify:** O(n) construction from unordered array

### Automatic Resizing

The array doubles in size when capacity is reached, ensuring:
- Amortized O(1) insertion
- No memory waste for small queues
- Efficient growth for large datasets

## Use Cases

### 1. Operating System Scheduling
- Process prioritization
- Interrupt handling
- Resource allocation

### 2. Network QoS
- Packet scheduling
- Bandwidth allocation
- Latency optimization

### 3. Web Server Request Handling
- API endpoint prioritization
- Load balancing
- Resource management

### 4. Medical Systems
- Emergency triage
- Treatment prioritization
- Resource allocation

### 5. Algorithmic Applications
- Dijkstra's shortest path
- A* search algorithm
- Huffman coding

## Advanced Features

### Dynamic Priority Updates

```python
# Update task priority at runtime
pq.update_task_priority("TASK_001", TaskPriority.CRITICAL)
```

### Batch Operations

```python
# Efficient bulk insertion
pq.enqueue_batch(task_list)

# Bulk heap construction
pq.build_heap(task_list)
```

### Composite Priority Calculation

Combines multiple factors:
- Base priority level
- Deadline urgency
- Arrival time
- Estimated duration

## Performance Optimization Tips

1. **Use batch operations** for multiple insertions
2. **Pre-allocate capacity** if maximum size is known
3. **Update priorities sparingly** - operation is O(log n)
4. **Use build_heap()** for initial construction from existing data
5. **Monitor queue size** for memory management

## Common Pitfalls and Solutions

### Memory Management
```python
# Good: Pre-allocate known capacity
pq = ArrayBasedPriorityQueue(capacity=1000)

# Avoid: Default small capacity for large datasets
```

### Priority Updates
```python
# Good: Batch priority updates
for task_id, priority in updates:
    pq.update_task_priority(task_id, priority)

# Avoid: Frequent individual updates in tight loops
```

### Task Comparison
```python
# Good: Implement __lt__ for heap operations
def __lt__(self, other):
    return self.composite_priority < other.composite_priority

# Avoid: Inconsistent comparison methods
```

## Algorithm Analysis

### Insertion Analysis
1. Add element at end of array: O(1)
2. Heapify up to maintain property: O(log n)
3. **Total: O(log n)**

### Extraction Analysis
1. Save root element: O(1)
2. Move last element to root: O(1)
3. Heapify down to restore property: O(log n)
4. **Total: O(log n)**

### Space Analysis
- Array storage: n elements
- No pointer overhead
- **Total: O(n)**