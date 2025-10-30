"""
Core Heap Operations Demonstration
Detailed demonstration of the four fundamental heap operations with time complexity analysis.
"""

from datetime import datetime, timedelta
import time
import random
from priority_queue_implementation import (
    Task, TaskPriority, ArrayBasedPriorityQueue
)


def demonstrate_insert_operation():
    """Demonstrate insert() operation with time complexity analysis"""
    print("=" * 60)
    print("OPERATION 1: insert() - Insert Task with Heap Property Maintenance")
    print("=" * 60)
    
    pq = ArrayBasedPriorityQueue(capacity=10)
    base_time = datetime.now()
    
    print("Algorithm Steps:")
    print("1. Add task at end of array (next available position)")
    print("2. Update index mapping for O(1) future lookups")
    print("3. Bubble up to restore heap property")
    print()
    
    # Create tasks with different priorities
    tasks = [
        Task("MEDIUM_1", TaskPriority.MEDIUM, base_time, base_time + timedelta(hours=1), 300),
        Task("HIGH_1", TaskPriority.HIGH, base_time, base_time + timedelta(minutes=30), 200),
        Task("LOW_1", TaskPriority.LOW, base_time, base_time + timedelta(hours=2), 400),
        Task("CRITICAL_1", TaskPriority.CRITICAL, base_time, base_time + timedelta(minutes=5), 100),
        Task("BACKGROUND_1", TaskPriority.BACKGROUND, base_time, base_time + timedelta(hours=4), 500),
    ]
    
    print("Inserting tasks and showing heap structure after each insertion:")
    print()
    
    for i, task in enumerate(tasks):
        print(f"Step {i+1}: Inserting {task.task_id} (Priority: {task.priority.name})")
        
        # Measure insertion time
        start_time = time.time()
        success = pq.insert(task)
        end_time = time.time()
        
        print(f"  Insertion successful: {success}")
        print(f"  Time taken: {(end_time - start_time) * 1000:.4f} ms")
        print(f"  Heap size: {pq.get_size()}")
        print(f"  Current root (highest priority): {pq.peek_min()}")
        
        # Show heap structure
        print("  Current heap order:")
        for j in range(pq.get_size()):
            task_at_j = pq.heap[j]
            print(f"    [{j}] {task_at_j.task_id} (score: {task_at_j.composite_priority:.2f})")
        print()
    
    print("Time Complexity Analysis:")
    print("  Best Case: O(1) - Task inserted at correct position immediately")
    print("  Average Case: O(log n) - Task bubbles up ~log n levels")
    print("  Worst Case: O(log n) - Task bubbles up to root")
    print("  Space Complexity: O(1) - Constant extra space")
    print()


def demonstrate_extract_min_operation():
    """Demonstrate extract_min() operation with time complexity analysis"""
    print("=" * 60)
    print("OPERATION 2: extract_min() - Remove Highest Priority Task")
    print("=" * 60)
    
    # Use the same priority queue from previous demo or create new one
    pq = ArrayBasedPriorityQueue(capacity=10)
    base_time = datetime.now()
    
    # Insert several tasks
    tasks = [
        Task("TASK_A", TaskPriority.MEDIUM, base_time, base_time + timedelta(hours=1), 300),
        Task("TASK_B", TaskPriority.HIGH, base_time, base_time + timedelta(minutes=30), 200),
        Task("TASK_C", TaskPriority.LOW, base_time, base_time + timedelta(hours=2), 400),
        Task("TASK_D", TaskPriority.CRITICAL, base_time, base_time + timedelta(minutes=5), 100),
        Task("TASK_E", TaskPriority.BACKGROUND, base_time, base_time + timedelta(hours=4), 500),
    ]
    
    for task in tasks:
        pq.insert(task)
    
    print("Algorithm Steps:")
    print("1. Save the root element (minimum priority value = highest actual priority)")
    print("2. Move last element to root position")
    print("3. Remove last element and decrease size")
    print("4. Update index mapping")
    print("5. Restore heap property by bubbling down")
    print("6. Return saved root element")
    print()
    
    print(f"Initial heap size: {pq.get_size()}")
    print("Initial heap structure:")
    for i in range(pq.get_size()):
        task = pq.heap[i]
        print(f"  [{i}] {task.task_id} (score: {task.composite_priority:.2f})")
    print()
    
    print("Extracting all tasks in priority order:")
    print()
    
    step = 1
    while not pq.is_empty():
        print(f"Step {step}: Extracting highest priority task")
        
        # Measure extraction time
        start_time = time.time()
        extracted_task = pq.extract_min()
        end_time = time.time()
        
        print(f"  Extracted: {extracted_task.task_id} (Priority: {extracted_task.priority.name})")
        print(f"  Time taken: {(end_time - start_time) * 1000:.4f} ms")
        print(f"  Remaining heap size: {pq.get_size()}")
        
        if not pq.is_empty():
            print(f"  New root: {pq.peek_min().task_id}")
            print("  Remaining heap structure:")
            for i in range(pq.get_size()):
                task = pq.heap[i]
                print(f"    [{i}] {task.task_id} (score: {task.composite_priority:.2f})")
        else:
            print("  Heap is now empty")
        
        print()
        step += 1
    
    print("Time Complexity Analysis:")
    print("  Best Case: O(1) - Replacement element already in correct position")
    print("  Average Case: O(log n) - Element sinks down ~log n levels")
    print("  Worst Case: O(log n) - Element sinks down to leaf level")
    print("  Space Complexity: O(1) - Constant extra space")
    print()


def demonstrate_decrease_key_operation():
    """Demonstrate decrease_key() operation (increase actual priority)"""
    print("=" * 60)
    print("OPERATION 3: decrease_key() - Increase Task Priority (Decrease Priority Value)")
    print("=" * 60)
    
    pq = ArrayBasedPriorityQueue(capacity=10)
    base_time = datetime.now()
    
    # Insert tasks
    tasks = [
        Task("ROUTINE_TASK", TaskPriority.LOW, base_time, base_time + timedelta(hours=2), 400),
        Task("NORMAL_TASK", TaskPriority.MEDIUM, base_time, base_time + timedelta(hours=1), 300),
        Task("URGENT_TASK", TaskPriority.HIGH, base_time, base_time + timedelta(minutes=30), 200),
        Task("SYSTEM_TASK", TaskPriority.CRITICAL, base_time, base_time + timedelta(minutes=5), 100),
    ]
    
    for task in tasks:
        pq.insert(task)
    
    print("Algorithm Steps:")
    print("1. Find task using index mapping O(1)")
    print("2. Update priority value (decrease value = increase actual priority)")
    print("3. Bubble up to restore heap property")
    print()
    
    print("Initial heap state:")
    print(f"Heap size: {pq.get_size()}")
    print("Current priority order:")
    for i in range(pq.get_size()):
        task = pq.heap[i]
        print(f"  [{i}] {task.task_id}: {task.priority.name} (score: {task.composite_priority:.2f})")
    
    print(f"\nCurrent highest priority task: {pq.peek_min().task_id}")
    print()
    
    # Scenario: A routine task becomes critical
    print("SCENARIO: ROUTINE_TASK discovered to be critical for system stability!")
    print("Decreasing key (LOW -> CRITICAL): priority value 4 -> 1")
    
    # Measure operation time
    start_time = time.time()
    success = pq.decrease_key("ROUTINE_TASK", TaskPriority.CRITICAL)
    end_time = time.time()
    
    print(f"Operation successful: {success}")
    print(f"Time taken: {(end_time - start_time) * 1000:.4f} ms")
    print()
    
    print("Updated heap state:")
    print("New priority order:")
    for i in range(pq.get_size()):
        task = pq.heap[i]
        print(f"  [{i}] {task.task_id}: {task.priority.name} (score: {task.composite_priority:.2f})")
    
    print(f"\nNew highest priority task: {pq.peek_min().task_id}")
    print()
    
    print("Time Complexity Analysis:")
    print("  Finding Task: O(1) with index mapping")
    print("  Priority Update: O(1)")
    print("  Heap Restoration: O(log n) - bubbling up")
    print("  Total: O(log n)")
    print()


def demonstrate_increase_key_operation():
    """Demonstrate increase_key() operation (decrease actual priority)"""
    print("=" * 60)
    print("OPERATION 4: increase_key() - Decrease Task Priority (Increase Priority Value)")
    print("=" * 60)
    
    pq = ArrayBasedPriorityQueue(capacity=10)
    base_time = datetime.now()
    
    # Insert tasks
    tasks = [
        Task("URGENT_REQUEST", TaskPriority.HIGH, base_time, base_time + timedelta(minutes=30), 200),
        Task("NORMAL_PROCESS", TaskPriority.MEDIUM, base_time, base_time + timedelta(hours=1), 300),
        Task("BACKUP_TASK", TaskPriority.LOW, base_time, base_time + timedelta(hours=2), 400),
        Task("CRITICAL_ALERT", TaskPriority.CRITICAL, base_time, base_time + timedelta(minutes=5), 100),
    ]
    
    for task in tasks:
        pq.insert(task)
    
    print("Algorithm Steps:")
    print("1. Find task using index mapping O(1)")
    print("2. Update priority value (increase value = decrease actual priority)")
    print("3. Bubble down to restore heap property")
    print()
    
    print("Initial heap state:")
    print(f"Heap size: {pq.get_size()}")
    print("Current priority order:")
    for i in range(pq.get_size()):
        task = pq.heap[i]
        print(f"  [{i}] {task.task_id}: {task.priority.name} (score: {task.composite_priority:.2f})")
    
    print(f"\nCurrent highest priority task: {pq.peek_min().task_id}")
    print()
    
    # Scenario: An urgent request is found to be non-critical
    print("SCENARIO: URGENT_REQUEST determined to be less important than initially thought")
    print("Increasing key (HIGH -> LOW): priority value 2 -> 4")
    
    # Measure operation time
    start_time = time.time()
    success = pq.increase_key("URGENT_REQUEST", TaskPriority.LOW)
    end_time = time.time()
    
    print(f"Operation successful: {success}")
    print(f"Time taken: {(end_time - start_time) * 1000:.4f} ms")
    print()
    
    print("Updated heap state:")
    print("New priority order:")
    for i in range(pq.get_size()):
        task = pq.heap[i]
        print(f"  [{i}] {task.task_id}: {task.priority.name} (score: {task.composite_priority:.2f})")
    
    print(f"\nCurrent highest priority task: {pq.peek_min().task_id}")
    print()
    
    print("Time Complexity Analysis:")
    print("  Finding Task: O(1) with index mapping")
    print("  Priority Update: O(1)")
    print("  Heap Restoration: O(log n) - bubbling down")
    print("  Total: O(log n)")
    print()


def demonstrate_is_empty_operation():
    """Demonstrate is_empty() operation"""
    print("=" * 60)
    print("OPERATION 5: is_empty() - Check if Priority Queue is Empty")
    print("=" * 60)
    
    pq = ArrayBasedPriorityQueue(capacity=5)
    
    print("Algorithm Steps:")
    print("1. Check if heap size equals zero")
    print("2. Return boolean result")
    print()
    
    # Test empty queue
    print("Test 1: New empty queue")
    start_time = time.time()
    is_empty = pq.is_empty()
    end_time = time.time()
    
    print(f"  Queue size: {pq.get_size()}")
    print(f"  is_empty() result: {is_empty}")
    print(f"  Time taken: {(end_time - start_time) * 1000:.4f} ms")
    print()
    
    # Add some tasks
    base_time = datetime.now()
    tasks = [
        Task("TASK_1", TaskPriority.HIGH, base_time, base_time + timedelta(minutes=30), 200),
        Task("TASK_2", TaskPriority.MEDIUM, base_time, base_time + timedelta(hours=1), 300),
    ]
    
    for task in tasks:
        pq.insert(task)
    
    print("Test 2: Queue with tasks")
    start_time = time.time()
    is_empty = pq.is_empty()
    end_time = time.time()
    
    print(f"  Queue size: {pq.get_size()}")
    print(f"  is_empty() result: {is_empty}")
    print(f"  Time taken: {(end_time - start_time) * 1000:.4f} ms")
    print()
    
    # Extract all tasks
    while not pq.is_empty():
        pq.extract_min()
    
    print("Test 3: Queue after extracting all tasks")
    start_time = time.time()
    is_empty = pq.is_empty()
    end_time = time.time()
    
    print(f"  Queue size: {pq.get_size()}")
    print(f"  is_empty() result: {is_empty}")
    print(f"  Time taken: {(end_time - start_time) * 1000:.4f} ms")
    print()
    
    print("Time Complexity Analysis:")
    print("  All Cases: O(1) - Simple size comparison")
    print("  Space Complexity: O(1) - No additional space required")
    print()


def demonstrate_performance_scaling():
    """Demonstrate how time complexity scales with input size"""
    print("=" * 60)
    print("PERFORMANCE SCALING ANALYSIS")
    print("=" * 60)
    
    print("Testing operations with different heap sizes to verify O(log n) complexity")
    print()
    
    sizes = [100, 500, 1000, 2000, 5000]
    
    for size in sizes:
        print(f"Testing with {size} tasks:")
        
        pq = ArrayBasedPriorityQueue(capacity=size + 100)
        base_time = datetime.now()
        
        # Fill heap
        fill_start = time.time()
        for i in range(size):
            priority = TaskPriority(random.randint(1, 5))
            task = Task(f"TASK_{i}", priority, base_time, 
                       base_time + timedelta(hours=1), 100)
            pq.insert(task)
        fill_end = time.time()
        
        # Test extract_min
        extract_start = time.time()
        for _ in range(min(10, size)):  # Extract 10 tasks or all if less
            pq.extract_min()
        extract_end = time.time()
        
        # Test decrease_key (if any tasks remain)
        decrease_start = time.time()
        if not pq.is_empty():
            # Try to update first task
            for i in range(min(5, pq.get_size())):
                if pq.heap[i]:
                    pq.decrease_key(pq.heap[i].task_id, TaskPriority.CRITICAL)
                    break
        decrease_end = time.time()
        
        print(f"  Insert {size} tasks: {(fill_end - fill_start)*1000:.2f} ms")
        print(f"  Extract 10 tasks: {(extract_end - extract_start)*1000:.2f} ms")
        print(f"  Decrease key: {(decrease_end - decrease_start)*1000:.4f} ms")
        print(f"  Final heap size: {pq.get_size()}")
        print()
    
    print("Expected: Time should grow logarithmically with heap size")
    print("O(log n) means doubling input size should not double execution time")
    print()


def run_all_operation_demonstrations():
    """Run all operation demonstrations"""
    print("🚀 Core Heap Operations - Comprehensive Analysis")
    print("=" * 70)
    print("This demonstration covers the four fundamental heap operations")
    print("with detailed time complexity analysis and practical examples.")
    print("=" * 70)
    print()
    
    # Run each demonstration
    demonstrate_insert_operation()
    input("Press Enter to continue to extract_min() demonstration...")
    
    demonstrate_extract_min_operation()
    input("Press Enter to continue to decrease_key() demonstration...")
    
    demonstrate_decrease_key_operation()
    input("Press Enter to continue to increase_key() demonstration...")
    
    demonstrate_increase_key_operation()
    input("Press Enter to continue to is_empty() demonstration...")
    
    demonstrate_is_empty_operation()
    input("Press Enter to continue to performance scaling analysis...")
    
    demonstrate_performance_scaling()
    
    print("=" * 70)
    print("🎉 All operation demonstrations completed!")
    print("=" * 70)
    print()
    print("Summary of Time Complexities:")
    print("  insert(): O(log n) - Bubble up after insertion")
    print("  extract_min(): O(log n) - Bubble down after removal")
    print("  decrease_key(): O(log n) - Bubble up after priority increase")
    print("  increase_key(): O(log n) - Bubble down after priority decrease")
    print("  is_empty(): O(1) - Simple size check")
    print()
    print("Key Optimizations:")
    print("  ✅ Index mapping for O(1) task lookup")
    print("  ✅ Array-based structure for cache efficiency")
    print("  ✅ Automatic resizing for dynamic capacity")
    print("  ✅ Composite priority for complex scheduling")


if __name__ == "__main__":
    run_all_operation_demonstrations()