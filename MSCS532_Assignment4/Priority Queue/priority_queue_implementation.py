"""
Priority Queue Implementation for Task Scheduling
This module implements a priority queue using array-based binary heap
for efficient task scheduling with multiple priority factors.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Set
import heapq


class TaskPriority(Enum):
    """Enumeration for task priority levels (lower number = higher priority)"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Enumeration for task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """
    Task class representing individual tasks with scheduling attributes.
    Supports composite priority calculation based on multiple factors.
    """
    
    def __init__(self, 
                 task_id: str,
                 priority: TaskPriority,
                 arrival_time: datetime,
                 deadline: datetime,
                 estimated_duration: int,  # in seconds
                 description: str = "",
                 dependencies: Optional[List[str]] = None):
        
        self.task_id = task_id
        self.priority = priority
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.estimated_duration = estimated_duration
        self.description = description
        self.dependencies = dependencies or []
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        
        # Calculated fields for scheduling
        self.urgency_score = self._calculate_urgency()
        self.composite_priority = self._calculate_composite_priority()
    
    def _calculate_urgency(self) -> float:
        """Calculate urgency based on deadline proximity"""
        time_until_deadline = (self.deadline - datetime.now()).total_seconds()
        if time_until_deadline <= 0:
            return float('inf')  # Overdue tasks have infinite urgency
        return self.estimated_duration / time_until_deadline
    
    def _calculate_composite_priority(self) -> float:
        """
        Calculate composite priority considering multiple factors.
        Lower values indicate higher priority (for min-heap).
        """
        base_priority = self.priority.value
        urgency_factor = min(self.urgency_score, 10)  # Cap urgency influence
        
        # Combine base priority with urgency (lower = higher priority)
        return base_priority + (1 / (urgency_factor + 1))
    
    def update_priority(self):
        """Recalculate priority scores (useful for dynamic scheduling)"""
        self.urgency_score = self._calculate_urgency()
        self.composite_priority = self._calculate_composite_priority()
    
    def __lt__(self, other):
        """Less than comparison for heap operations (min-heap)"""
        return self.composite_priority < other.composite_priority
    
    def __le__(self, other):
        """Less than or equal comparison for heap operations"""
        return self.composite_priority <= other.composite_priority
    
    def __gt__(self, other):
        """Greater than comparison for heap operations"""
        return self.composite_priority > other.composite_priority
    
    def __ge__(self, other):
        """Greater than or equal comparison for heap operations"""
        return self.composite_priority >= other.composite_priority
    
    def __eq__(self, other):
        """Equality comparison based on task ID"""
        return self.task_id == other.task_id
    
    def __hash__(self):
        """Hash function for using Task in sets/dictionaries"""
        return hash(self.task_id)
    
    def __repr__(self):
        return (f"Task(id={self.task_id}, priority={self.priority.name}, "
                f"score={self.composite_priority:.2f}, deadline={self.deadline.strftime('%H:%M:%S')})")


class ArrayBasedPriorityQueue:
    """
    Array-based priority queue implementation using binary min-heap.
    Optimized for task scheduling with O(log n) insertion and extraction.
    
    Core Operations:
    - insert(): O(log n) - Add task while maintaining heap property
    - extract_min(): O(log n) - Remove highest priority task
    - increase/decrease_key(): O(log n) - Modify task priority
    - is_empty(): O(1) - Check if queue is empty
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.heap = [None] * capacity
        self.size = 0
        # Index mapping for O(log n) priority updates
        self.task_index_map = {}  # task_id -> heap_index
    
    def _parent(self, index: int) -> int:
        """Get parent index. O(1) time complexity."""
        return (index - 1) // 2
    
    def _left_child(self, index: int) -> int:
        """Get left child index. O(1) time complexity."""
        return 2 * index + 1
    
    def _right_child(self, index: int) -> int:
        """Get right child index. O(1) time complexity."""
        return 2 * index + 2
    
    def _swap(self, i: int, j: int) -> None:
        """
        Swap elements at indices i and j and update index mapping.
        Time Complexity: O(1)
        """
        # Update index mapping before swap
        if self.heap[i]:
            self.task_index_map[self.heap[i].task_id] = j
        if self.heap[j]:
            self.task_index_map[self.heap[j].task_id] = i
            
        # Perform the swap
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def _heapify_up(self, index: int) -> None:
        """
        Restore heap property upward (after insertion or priority decrease).
        
        Time Complexity Analysis:
        - Best Case: O(1) - Element is already in correct position
        - Average Case: O(log n) - Element bubbles up log n levels on average
        - Worst Case: O(log n) - Element bubbles up to root (height of tree)
        
        Space Complexity: O(1) - Only uses constant extra space
        """
        while index > 0:
            parent_index = self._parent(index)
            if self.heap[index] < self.heap[parent_index]:
                self._swap(index, parent_index)
                index = parent_index
            else:
                break
    
    def _heapify_down(self, index: int) -> None:
        """
        Restore heap property downward (after extraction or priority increase).
        
        Time Complexity Analysis:
        - Best Case: O(1) - Element is already in correct position
        - Average Case: O(log n) - Element sinks down log n levels on average  
        - Worst Case: O(log n) - Element sinks down to leaf (height of tree)
        
        Space Complexity: O(1) - Only uses constant extra space
        """
        while self._left_child(index) < self.size:
            min_child_index = self._left_child(index)
            right_child_index = self._right_child(index)
            
            # Find the smaller child
            if (right_child_index < self.size and 
                self.heap[right_child_index] < self.heap[min_child_index]):
                min_child_index = right_child_index
            
            # If heap property is satisfied, break
            if self.heap[index] <= self.heap[min_child_index]:
                break
                
            self._swap(index, min_child_index)
            index = min_child_index
    
    def _resize(self) -> None:
        """
        Double the capacity when the heap is full.
        Time Complexity: O(n) - Must copy all elements
        Space Complexity: O(n) - New array allocation
        """
        old_capacity = self.capacity
        self.capacity *= 2
        new_heap = [None] * self.capacity
        
        # Copy existing elements
        for i in range(self.size):
            new_heap[i] = self.heap[i]
        
        self.heap = new_heap
        print(f"Resized heap from {old_capacity} to {self.capacity}")
    
    def insert(self, task: Task) -> bool:
        """
        Insert a task into the priority queue while maintaining heap property.
        
        Algorithm:
        1. Add task at the end of the heap (next available position)
        2. Update index mapping for O(log n) future operations
        3. Bubble up to restore heap property
        
        Time Complexity Analysis:
        - Best Case: O(1) - Task inserted at correct position immediately
        - Average Case: O(log n) - Task bubbles up approximately log n levels
        - Worst Case: O(log n) - Task bubbles up to root
        
        Space Complexity: O(1) - Only uses constant extra space
        
        Args:
            task: Task object to insert
            
        Returns:
            bool: True if insertion successful
        """
        # Resize if necessary
        if self.size >= self.capacity:
            self._resize()
        
        # Add task at the end
        self.heap[self.size] = task
        
        # Update index mapping
        self.task_index_map[task.task_id] = self.size
        
        # Restore heap property by bubbling up
        self._heapify_up(self.size)
        self.size += 1
        
        return True
    
    def extract_min(self) -> Optional[Task]:
        """
        Remove and return the task with the highest priority (lowest value in min-heap).
        
        Algorithm:
        1. Save the root element (minimum priority value = highest actual priority)
        2. Move the last element to root position
        3. Remove the last element and decrease size
        4. Update index mapping
        5. Restore heap property by bubbling down
        6. Return the saved root element
        
        Time Complexity Analysis:
        - Best Case: O(1) - Replacement element is already in correct position
        - Average Case: O(log n) - Element sinks down approximately log n levels
        - Worst Case: O(log n) - Element sinks down to leaf level
        
        Space Complexity: O(1) - Only uses constant extra space
        
        Returns:
            Optional[Task]: Highest priority task or None if empty
        """
        if self.size == 0:
            return None
        
        # Save the minimum element (root)
        min_task = self.heap[0]
        
        # Remove from index mapping
        if min_task.task_id in self.task_index_map:
            del self.task_index_map[min_task.task_id]
        
        # Move last element to root and decrease size
        self.heap[0] = self.heap[self.size - 1]
        self.size -= 1
        
        # Update index mapping for moved element
        if self.size > 0 and self.heap[0]:
            self.task_index_map[self.heap[0].task_id] = 0
            # Restore heap property by bubbling down
            self._heapify_down(0)
        
        return min_task
    
    def peek_min(self) -> Optional[Task]:
        """
        Return the highest priority task without removing it.
        
        Time Complexity: O(1) - Direct array access
        Space Complexity: O(1) - No additional space
        
        Returns:
            Optional[Task]: Highest priority task or None if empty
        """
        return self.heap[0] if self.size > 0 else None
    
    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty.
        
        Time Complexity: O(1) - Simple size comparison
        Space Complexity: O(1) - No additional space required
        
        Returns:
            bool: True if queue is empty, False otherwise
        """
        return self.size == 0
    
    def is_full(self) -> bool:
        """
        Check if the queue is at capacity (will auto-resize if needed).
        
        Time Complexity: O(1) - Simple size comparison
        Space Complexity: O(1) - No additional space
        
        Returns:
            bool: True if at capacity, False otherwise
        """
        return self.size >= self.capacity
    
    def get_size(self) -> int:
        """
        Return the number of tasks in the queue.
        
        Time Complexity: O(1) - Direct variable access
        Space Complexity: O(1) - No additional space
        
        Returns:
            int: Number of tasks currently in queue
        """
        return self.size
    
    def increase_key(self, task_id: str, new_priority: TaskPriority) -> bool:
        """
        Increase the priority value (decrease actual priority) of an existing task.
        
        Algorithm:
        1. Find task using index mapping O(1)
        2. Update priority value
        3. Bubble down to restore heap property (since priority value increased)
        
        Time Complexity Analysis:
        - Finding Task: O(1) with index mapping
        - Priority Update: O(1)
        - Heap Restoration: O(log n) - bubbling down
        - Total: O(log n)
        
        Args:
            task_id: Unique identifier of task to update
            new_priority: New priority level (should be lower priority than current)
            
        Returns:
            bool: True if update successful, False if task not found
        """
        if task_id not in self.task_index_map:
            return False
        
        index = self.task_index_map[task_id]
        old_priority = self.heap[index].priority
        
        # Verify this is actually an increase in priority value
        if new_priority.value <= old_priority.value:
            return False  # Use decrease_key for this operation
        
        # Update priority
        self.heap[index].priority = new_priority
        self.heap[index].update_priority()
        
        # Restore heap property by bubbling down (priority value increased)
        self._heapify_down(index)
        
        return True
    
    def decrease_key(self, task_id: str, new_priority: TaskPriority) -> bool:
        """
        Decrease the priority value (increase actual priority) of an existing task.
        
        Algorithm:
        1. Find task using index mapping O(1)
        2. Update priority value
        3. Bubble up to restore heap property (since priority value decreased)
        
        Time Complexity Analysis:
        - Finding Task: O(1) with index mapping
        - Priority Update: O(1)
        - Heap Restoration: O(log n) - bubbling up
        - Total: O(log n)
        
        Args:
            task_id: Unique identifier of task to update
            new_priority: New priority level (should be higher priority than current)
            
        Returns:
            bool: True if update successful, False if task not found
        """
        if task_id not in self.task_index_map:
            return False
        
        index = self.task_index_map[task_id]
        old_priority = self.heap[index].priority
        
        # Verify this is actually a decrease in priority value
        if new_priority.value >= old_priority.value:
            return False  # Use increase_key for this operation
        
        # Update priority
        self.heap[index].priority = new_priority
        self.heap[index].update_priority()
        
        # Restore heap property by bubbling up (priority value decreased)
        self._heapify_up(index)
        
        return True
    
    def update_task_priority(self, task_id: str, new_priority: TaskPriority) -> bool:
        """
        Update priority of an existing task (automatically determines direction).
        
        Time Complexity: O(log n) - Uses either increase_key or decrease_key
        
        Args:
            task_id: Unique identifier of task to update
            new_priority: New priority level
            
        Returns:
            bool: True if update successful, False if task not found
        """
        if task_id not in self.task_index_map:
            return False
        
        index = self.task_index_map[task_id]
        old_priority = self.heap[index].priority
        
        if new_priority.value < old_priority.value:
            return self.decrease_key(task_id, new_priority)
        elif new_priority.value > old_priority.value:
            return self.increase_key(task_id, new_priority)
        else:
            # Same priority - just update composite priority
            self.heap[index].priority = new_priority
            self.heap[index].update_priority()
            return True
    
    def build_heap(self, tasks: List[Task]) -> None:
        """Build heap from list of tasks in O(n) time"""
        self.size = min(len(tasks), self.capacity)
        
        # Copy tasks to heap array
        for i in range(self.size):
            self.heap[i] = tasks[i]
        
        # Heapify from last non-leaf node to root
        for i in range(self._parent(self.size - 1), -1, -1):
            self._heapify_down(i)
    
    def enqueue_batch(self, tasks: List[Task]) -> None:
        """Efficiently add multiple tasks"""
        # Ensure capacity
        while self.size + len(tasks) > self.capacity:
            self._resize()
        
        start_size = self.size
        
        # Add all tasks to the end
        for task in tasks:
            self.heap[self.size] = task
            self.size += 1
        
        # Heapify from the last non-leaf node
        for i in range(self._parent(self.size - 1), -1, -1):
            self._heapify_down(i)
    
    def get_all_tasks(self) -> List[Task]:
        """Return a copy of all tasks (for debugging/monitoring)"""
        return [self.heap[i] for i in range(self.size)]
    
    def print_heap(self) -> None:
        """Print heap structure for debugging"""
        print("Current heap contents:")
        for i in range(self.size):
            task = self.heap[i]
            level = int(i.bit_length()) - 1 if i > 0 else 0
            indent = "  " * level
            print(f"{indent}[{i}] {task}")


class PythonHeapqPriorityQueue:
    """
    Alternative implementation using Python's built-in heapq module.
    Demonstrates the same concepts with library support.
    """
    
    def __init__(self):
        self.heap: List[Task] = []
        self.task_count = 0
    
    def enqueue(self, task: Task) -> None:
        """Add a task to the priority queue"""
        heapq.heappush(self.heap, task)
        self.task_count += 1
    
    def dequeue(self) -> Optional[Task]:
        """Remove and return the highest priority task"""
        if self.is_empty():
            return None
        
        task = heapq.heappop(self.heap)
        self.task_count -= 1
        return task
    
    def peek(self) -> Optional[Task]:
        """Return the highest priority task without removing it"""
        return self.heap[0] if not self.is_empty() else None
    
    def is_empty(self) -> bool:
        """Check if the queue is empty"""
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Return the number of tasks in the queue"""
        return len(self.heap)
    
    def update_priorities(self) -> None:
        """Recalculate all task priorities and re-heapify"""
        for task in self.heap:
            task.update_priority()
        heapq.heapify(self.heap)


class TaskScheduler:
    """
    High-level task scheduler that demonstrates priority queue usage
    in a realistic scheduling scenario.
    """
    
    def __init__(self, use_custom_heap: bool = True):
        if use_custom_heap:
            self.priority_queue = ArrayBasedPriorityQueue()
        else:
            self.priority_queue = PythonHeapqPriorityQueue()
        
        self.completed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
    
    def add_task(self, task: Task) -> None:
        """Add a task to the scheduler"""
        if hasattr(self.priority_queue, 'insert'):
            self.priority_queue.insert(task)
        else:
            self.priority_queue.enqueue(task)
        print(f"Added task: {task}")
    
    def can_execute(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        return all(dep_id in self.completed_tasks for dep_id in task.dependencies)
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next executable task considering dependencies"""
        # For simplicity, assume no dependencies in this example
        # In a real system, you'd need to check dependencies
        if hasattr(self.priority_queue, 'extract_min'):
            return self.priority_queue.extract_min()
        else:
            return self.priority_queue.dequeue()
    
    def execute_task(self, task: Task) -> bool:
        """Simulate task execution"""
        if not self.can_execute(task):
            print(f"Cannot execute {task.task_id}: dependencies not met")
            return False
        
        print(f"Executing task: {task}")
        task.status = TaskStatus.RUNNING
        self.running_tasks.add(task.task_id)
        
        # Simulate execution (in real system, this would be actual work)
        # For demo, we'll just mark as completed
        task.status = TaskStatus.COMPLETED
        self.running_tasks.remove(task.task_id)
        self.completed_tasks.add(task.task_id)
        
        print(f"Completed task: {task.task_id}")
        return True
    
    def run_scheduling_cycle(self, max_tasks: int = 5) -> None:
        """Run a scheduling cycle, executing up to max_tasks"""
        executed = 0
        
        while executed < max_tasks and not self.priority_queue.is_empty():
            task = self.get_next_task()
            if task and self.execute_task(task):
                executed += 1
        
        print(f"Scheduling cycle completed. Executed {executed} tasks.")
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        queue_size = (self.priority_queue.get_size() 
                     if hasattr(self.priority_queue, 'get_size')
                     else self.priority_queue.size())
        
        return {
            "pending_tasks": queue_size,
            "completed_tasks": len(self.completed_tasks),
            "running_tasks": len(self.running_tasks),
            "failed_tasks": len(self.failed_tasks)
        }


def create_sample_tasks() -> List[Task]:
    """Create sample tasks for demonstration"""
    base_time = datetime.now()
    
    tasks = [
        Task("SYS001", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(minutes=1), 30,
             "Critical system backup"),
        
        Task("USER001", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(minutes=10), 120,
             "User data processing"),
        
        Task("BATCH001", TaskPriority.LOW, base_time,
             base_time + timedelta(hours=1), 300,
             "Batch report generation"),
        
        Task("URGENT001", TaskPriority.HIGH, base_time,
             base_time + timedelta(minutes=5), 60,
             "Urgent user request"),
        
        Task("BG001", TaskPriority.BACKGROUND, base_time,
             base_time + timedelta(hours=2), 600,
             "Background maintenance"),
        
        Task("DEADLINE001", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(minutes=2), 90,
             "Task with tight deadline"),
    ]
    
    return tasks


def demonstrate_priority_queue():
    """Demonstrate the priority queue implementation"""
    print("=" * 60)
    print("Priority Queue Implementation Demonstration")
    print("=" * 60)
    
    # Create scheduler with custom array-based heap
    print("\n1. Creating scheduler with custom array-based heap...")
    scheduler = TaskScheduler(use_custom_heap=True)
    
    # Create and add sample tasks
    print("\n2. Adding sample tasks...")
    tasks = create_sample_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"\n3. Scheduler status: {scheduler.get_status()}")
    
    # Show heap structure
    print("\n4. Current heap structure:")
    if hasattr(scheduler.priority_queue, 'print_heap'):
        scheduler.priority_queue.print_heap()
    
    # Execute scheduling cycles
    print("\n5. Running scheduling cycle...")
    scheduler.run_scheduling_cycle(max_tasks=3)
    
    print(f"\n6. Scheduler status after execution: {scheduler.get_status()}")
    
    # Demonstrate priority update
    print("\n7. Demonstrating priority update...")
    if hasattr(scheduler.priority_queue, 'update_task_priority'):
        updated = scheduler.priority_queue.update_task_priority("BATCH001", TaskPriority.CRITICAL)
        print(f"Updated BATCH001 priority to CRITICAL: {updated}")
    
    # Execute remaining tasks
    print("\n8. Executing remaining tasks...")
    scheduler.run_scheduling_cycle(max_tasks=10)
    
    print(f"\n9. Final scheduler status: {scheduler.get_status()}")
    
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_priority_queue()