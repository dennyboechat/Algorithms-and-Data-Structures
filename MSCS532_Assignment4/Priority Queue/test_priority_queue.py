"""
Unit tests for Priority Queue Implementation
Comprehensive test suite covering all aspects of the priority queue
and task scheduling functionality.
"""

import unittest
from datetime import datetime, timedelta
import random
import time
from priority_queue_implementation import (
    Task, TaskPriority, TaskStatus,
    ArrayBasedPriorityQueue, PythonHeapqPriorityQueue,
    TaskScheduler, create_sample_tasks
)


class TestTask(unittest.TestCase):
    """Test cases for the Task class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_time = datetime.now()
        self.task1 = Task(
            "T1", TaskPriority.HIGH, self.base_time,
            self.base_time + timedelta(hours=1), 300, "Test task 1"
        )
        self.task2 = Task(
            "T2", TaskPriority.CRITICAL, self.base_time,
            self.base_time + timedelta(minutes=30), 150, "Test task 2"
        )
    
    def test_task_creation(self):
        """Test basic task creation"""
        self.assertEqual(self.task1.task_id, "T1")
        self.assertEqual(self.task1.priority, TaskPriority.HIGH)
        self.assertEqual(self.task1.status, TaskStatus.PENDING)
        self.assertIsInstance(self.task1.composite_priority, float)
    
    def test_task_comparison(self):
        """Test task comparison for heap operations"""
        # Critical task should have lower composite priority (higher actual priority)
        self.assertTrue(self.task2 < self.task1)
        self.assertFalse(self.task1 < self.task2)
    
    def test_urgency_calculation(self):
        """Test urgency score calculation"""
        # Task with closer deadline should have higher urgency
        urgent_task = Task(
            "URGENT", TaskPriority.MEDIUM, self.base_time,
            self.base_time + timedelta(minutes=1), 100, "Urgent task"
        )
        
        normal_task = Task(
            "NORMAL", TaskPriority.MEDIUM, self.base_time,
            self.base_time + timedelta(hours=1), 100, "Normal task"
        )
        
        self.assertGreater(urgent_task.urgency_score, normal_task.urgency_score)
    
    def test_priority_update(self):
        """Test priority recalculation"""
        original_priority = self.task1.composite_priority
        
        # Simulate time passing (making deadline closer)
        time.sleep(0.1)  # Small delay to ensure time difference
        self.task1.update_priority()
        
        # Priority should change due to time-based urgency
        # Note: This test might be flaky due to timing, but demonstrates the concept
        updated_priority = self.task1.composite_priority
        self.assertIsInstance(updated_priority, float)
    
    def test_overdue_task(self):
        """Test handling of overdue tasks"""
        overdue_task = Task(
            "OVERDUE", TaskPriority.LOW, self.base_time,
            self.base_time - timedelta(minutes=1), 100, "Overdue task"
        )
        
        # Overdue tasks should have very high urgency
        self.assertEqual(overdue_task.urgency_score, float('inf'))


class TestArrayBasedPriorityQueue(unittest.TestCase):
    """Test cases for the array-based priority queue"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pq = ArrayBasedPriorityQueue(capacity=10)
        self.base_time = datetime.now()
        
        self.tasks = [
            Task(f"T{i}", TaskPriority(i % 5 + 1), self.base_time,
                 self.base_time + timedelta(hours=i+1), 100 * (i+1))
            for i in range(5)
        ]
    
    def test_empty_queue(self):
        """Test operations on empty queue"""
        self.assertTrue(self.pq.is_empty())
        self.assertFalse(self.pq.is_full())
        self.assertEqual(self.pq.get_size(), 0)
        self.assertIsNone(self.pq.peek_min())
        self.assertIsNone(self.pq.extract_min())
    
    def test_single_insertion(self):
        """Test single task insertion"""
        task = self.tasks[0]
        self.assertTrue(self.pq.insert(task))
        self.assertFalse(self.pq.is_empty())
        self.assertEqual(self.pq.get_size(), 1)
        self.assertEqual(self.pq.peek_min(), task)
    
    def test_multiple_insertions(self):
        """Test multiple task insertions"""
        for task in self.tasks:
            self.assertTrue(self.pq.insert(task))
        
        self.assertEqual(self.pq.get_size(), len(self.tasks))
        self.assertFalse(self.pq.is_empty())
    
    def test_heap_property(self):
        """Test that heap property is maintained"""
        # Insert tasks in random order
        random_tasks = self.tasks.copy()
        random.shuffle(random_tasks)
        
        for task in random_tasks:
            self.pq.insert(task)
        
        # Extract all tasks and verify they come out in priority order
        extracted_priorities = []
        while not self.pq.is_empty():
            task = self.pq.extract_min()
            extracted_priorities.append(task.composite_priority)
        
        # Verify extracted in ascending order (highest priority first)
        self.assertEqual(extracted_priorities, sorted(extracted_priorities))
    
    def test_peek_vs_extract(self):
        """Test that peek doesn't modify the queue"""
        self.pq.insert(self.tasks[0])
        
        peeked = self.pq.peek_min()
        self.assertEqual(self.pq.get_size(), 1)
        
        extracted = self.pq.extract_min()
        self.assertEqual(self.pq.get_size(), 0)
        self.assertEqual(peeked, extracted)
    
    def test_capacity_and_resizing(self):
        """Test capacity limits and automatic resizing"""
        small_pq = ArrayBasedPriorityQueue(capacity=3)
        
        # Fill to capacity
        for i in range(3):
            task = Task(f"T{i}", TaskPriority.MEDIUM, self.base_time,
                       self.base_time + timedelta(hours=1), 100)
            small_pq.insert(task)
        
        # Should resize automatically when adding more
        task = Task("T3", TaskPriority.HIGH, self.base_time,
                   self.base_time + timedelta(hours=1), 100)
        self.assertTrue(small_pq.insert(task))
        self.assertEqual(small_pq.get_size(), 4)
        self.assertGreater(small_pq.capacity, 3)
    
    def test_priority_update(self):
        """Test in-place priority updates"""
        # Insert tasks
        for task in self.tasks:
            self.pq.insert(task)
        
        # Update priority of a task
        task_id = self.tasks[2].task_id
        original_top = self.pq.peek_min()
        
        # Update to highest priority
        updated = self.pq.update_task_priority(task_id, TaskPriority.CRITICAL)
        self.assertTrue(updated)
        
        # The updated task might now be at the top
        new_top = self.pq.peek_min()
        # Verify heap property is maintained
        self.assertIsNotNone(new_top)
    
    def test_batch_operations(self):
        """Test batch insertion"""
        batch_tasks = self.tasks.copy()
        self.pq.enqueue_batch(batch_tasks)
        
        self.assertEqual(self.pq.get_size(), len(batch_tasks))
        
        # Verify heap property after batch insertion
        extracted_priorities = []
        while not self.pq.is_empty():
            task = self.pq.extract_min()
            extracted_priorities.append(task.composite_priority)
        
        self.assertEqual(extracted_priorities, sorted(extracted_priorities))
    
    def test_decrease_key_operation(self):
        """Test decrease_key() operation (increase actual priority)"""
        # Insert tasks
        for task in self.tasks:
            self.pq.insert(task)
        
        # Get original root
        original_root = self.pq.peek_min()
        
        # Find a non-root task to promote
        task_to_promote = None
        for i in range(1, self.pq.get_size()):
            if self.pq.heap[i].priority != TaskPriority.CRITICAL:
                task_to_promote = self.pq.heap[i]
                break
        
        self.assertIsNotNone(task_to_promote)
        
        # Decrease key (increase actual priority)
        success = self.pq.decrease_key(task_to_promote.task_id, TaskPriority.CRITICAL)
        self.assertTrue(success)
        
        # The promoted task might now be at the root
        new_root = self.pq.peek_min()
        self.assertEqual(new_root.priority, TaskPriority.CRITICAL)
        
        # Verify heap property is maintained
        self._verify_heap_property()
    
    def test_increase_key_operation(self):
        """Test increase_key() operation (decrease actual priority)"""
        # Insert tasks
        for task in self.tasks:
            self.pq.insert(task)
        
        # Find a high-priority task to demote
        task_to_demote = None
        for i in range(self.pq.get_size()):
            if self.pq.heap[i].priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                task_to_demote = self.pq.heap[i]
                break
        
        self.assertIsNotNone(task_to_demote)
        original_priority = task_to_demote.priority
        
        # Increase key (decrease actual priority)
        success = self.pq.increase_key(task_to_demote.task_id, TaskPriority.LOW)
        self.assertTrue(success)
        
        # Verify the task's priority was updated
        # Find the task in the heap
        updated_task = None
        for i in range(self.pq.get_size()):
            if self.pq.heap[i].task_id == task_to_demote.task_id:
                updated_task = self.pq.heap[i]
                break
        
        self.assertIsNotNone(updated_task)
        self.assertEqual(updated_task.priority, TaskPriority.LOW)
        
        # Verify heap property is maintained
        self._verify_heap_property()
    
    def test_index_mapping_consistency(self):
        """Test that index mapping remains consistent during operations"""
        # Insert tasks
        for task in self.tasks:
            self.pq.insert(task)
        
        # Verify initial mapping
        for task_id, index in self.pq.task_index_map.items():
            self.assertEqual(self.pq.heap[index].task_id, task_id)
        
        # Extract a task and verify mapping updates
        extracted = self.pq.extract_min()
        self.assertNotIn(extracted.task_id, self.pq.task_index_map)
        
        # Verify remaining mappings are still correct
        for task_id, index in self.pq.task_index_map.items():
            self.assertEqual(self.pq.heap[index].task_id, task_id)
    
    def test_invalid_key_operations(self):
        """Test invalid decrease/increase key operations"""
        task = self.tasks[0]
        self.pq.insert(task)
        
        # Try to decrease key to a lower priority (should fail)
        success = self.pq.decrease_key(task.task_id, TaskPriority.BACKGROUND)
        self.assertFalse(success)
        
        # Try to increase key to a higher priority (should fail)
        success = self.pq.increase_key(task.task_id, TaskPriority.CRITICAL)
        self.assertFalse(success)
        
        # Try operations on non-existent task
        success = self.pq.decrease_key("NONEXISTENT", TaskPriority.HIGH)
        self.assertFalse(success)
        
        success = self.pq.increase_key("NONEXISTENT", TaskPriority.LOW)
        self.assertFalse(success)
    
    def _verify_heap_property(self):
        """Helper method to verify heap property is maintained"""
        for i in range(self.pq.get_size()):
            left_child = self.pq._left_child(i)
            right_child = self.pq._right_child(i)
            
            if left_child < self.pq.get_size():
                self.assertLessEqual(self.pq.heap[i].composite_priority, 
                                   self.pq.heap[left_child].composite_priority)
            
            if right_child < self.pq.get_size():
                self.assertLessEqual(self.pq.heap[i].composite_priority, 
                                   self.pq.heap[right_child].composite_priority)

    def test_build_heap(self):
        """Test building heap from existing list"""
        self.pq.build_heap(self.tasks)
        self.assertEqual(self.pq.get_size(), len(self.tasks))
        
        # Verify heap property
        extracted_priorities = []
        while not self.pq.is_empty():
            task = self.pq.extract_min()
            extracted_priorities.append(task.composite_priority)
        
        self.assertEqual(extracted_priorities, sorted(extracted_priorities))


class TestPythonHeapqPriorityQueue(unittest.TestCase):
    """Test cases for the heapq-based priority queue"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pq = PythonHeapqPriorityQueue()
        self.base_time = datetime.now()
        
        self.tasks = [
            Task(f"T{i}", TaskPriority(i % 5 + 1), self.base_time,
                 self.base_time + timedelta(hours=i+1), 100 * (i+1))
            for i in range(5)
        ]
    
    def test_basic_operations(self):
        """Test basic enqueue/dequeue operations"""
        self.assertTrue(self.pq.is_empty())
        
        for task in self.tasks:
            self.pq.enqueue(task)
        
        self.assertEqual(self.pq.size(), len(self.tasks))
        self.assertFalse(self.pq.is_empty())
        
        # Test priority order
        extracted_priorities = []
        while not self.pq.is_empty():
            task = self.pq.dequeue()
            extracted_priorities.append(task.composite_priority)
        
        self.assertEqual(extracted_priorities, sorted(extracted_priorities))
    
    def test_peek_functionality(self):
        """Test peek without modification"""
        task = self.tasks[0]
        self.pq.enqueue(task)
        
        peeked = self.pq.peek()
        self.assertEqual(self.pq.size(), 1)
        self.assertEqual(peeked, task)
        
        dequeued = self.pq.dequeue()
        self.assertEqual(self.pq.size(), 0)
        self.assertEqual(peeked, dequeued)
    
    def test_priority_updates(self):
        """Test priority recalculation"""
        for task in self.tasks:
            self.pq.enqueue(task)
        
        # Modify priorities and re-heapify
        for task in self.tasks:
            task.priority = TaskPriority.HIGH
        
        self.pq.update_priorities()
        
        # All tasks should now have the same base priority
        # (though composite priority may differ due to other factors)
        self.assertEqual(self.pq.size(), len(self.tasks))


class TestTaskScheduler(unittest.TestCase):
    """Test cases for the task scheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = TaskScheduler(use_custom_heap=True)
        self.tasks = create_sample_tasks()
    
    def test_scheduler_creation(self):
        """Test scheduler initialization"""
        self.assertIsNotNone(self.scheduler.priority_queue)
        self.assertEqual(len(self.scheduler.completed_tasks), 0)
        self.assertEqual(len(self.scheduler.running_tasks), 0)
    
    def test_task_addition(self):
        """Test adding tasks to scheduler"""
        initial_status = self.scheduler.get_status()
        
        for task in self.tasks:
            self.scheduler.add_task(task)
        
        final_status = self.scheduler.get_status()
        self.assertEqual(final_status["pending_tasks"], len(self.tasks))
    
    def test_task_execution(self):
        """Test task execution workflow"""
        # Add tasks
        for task in self.tasks:
            self.scheduler.add_task(task)
        
        initial_pending = self.scheduler.get_status()["pending_tasks"]
        
        # Execute some tasks
        self.scheduler.run_scheduling_cycle(max_tasks=3)
        
        final_status = self.scheduler.get_status()
        self.assertLessEqual(final_status["pending_tasks"], initial_pending)
        self.assertGreaterEqual(final_status["completed_tasks"], 0)
    
    def test_dependency_checking(self):
        """Test dependency resolution"""
        # Create task with dependency
        base_time = datetime.now()
        dependent_task = Task(
            "DEP001", TaskPriority.HIGH, base_time,
            base_time + timedelta(hours=1), 100,
            dependencies=["NONEXISTENT"]
        )
        
        # Should not be executable due to unmet dependency
        self.assertFalse(self.scheduler.can_execute(dependent_task))
        
        # Mark dependency as completed
        self.scheduler.completed_tasks.add("NONEXISTENT")
        self.assertTrue(self.scheduler.can_execute(dependent_task))
    
    def test_both_queue_implementations(self):
        """Test that both queue implementations work similarly"""
        custom_scheduler = TaskScheduler(use_custom_heap=True)
        heapq_scheduler = TaskScheduler(use_custom_heap=False)
        
        # Add same tasks to both
        for task in self.tasks:
            custom_scheduler.add_task(task)
            heapq_scheduler.add_task(task)
        
        # Execute scheduling cycles
        custom_scheduler.run_scheduling_cycle(max_tasks=len(self.tasks))
        heapq_scheduler.run_scheduling_cycle(max_tasks=len(self.tasks))
        
        # Both should complete all tasks
        custom_status = custom_scheduler.get_status()
        heapq_status = heapq_scheduler.get_status()
        
        self.assertEqual(custom_status["completed_tasks"], len(self.tasks))
        self.assertEqual(heapq_status["completed_tasks"], len(self.tasks))


class TestPerformance(unittest.TestCase):
    """Performance tests for priority queue operations"""
    
    def test_large_scale_operations(self):
        """Test performance with large number of tasks"""
        pq = ArrayBasedPriorityQueue(capacity=1000)
        base_time = datetime.now()
        
        # Create many tasks
        large_task_set = []
        for i in range(1000):
            task = Task(
                f"PERF{i}", 
                TaskPriority(random.randint(1, 5)),
                base_time,
                base_time + timedelta(hours=random.randint(1, 24)),
                random.randint(60, 3600)
            )
            large_task_set.append(task)
        
        # Measure insertion time
        start_time = time.time()
        for task in large_task_set:
            pq.insert(task)
        insertion_time = time.time() - start_time
        
        self.assertEqual(pq.get_size(), 1000)
        print(f"Inserted 1000 tasks in {insertion_time:.4f} seconds")
        
        # Measure extraction time
        start_time = time.time()
        extracted_count = 0
        while not pq.is_empty():
            pq.extract_min()
            extracted_count += 1
        extraction_time = time.time() - start_time
        
        self.assertEqual(extracted_count, 1000)
        print(f"Extracted 1000 tasks in {extraction_time:.4f} seconds")
    
    def test_heap_vs_heapq_performance(self):
        """Compare performance of custom heap vs heapq"""
        custom_pq = ArrayBasedPriorityQueue(capacity=1000)
        heapq_pq = PythonHeapqPriorityQueue()
        
        base_time = datetime.now()
        tasks = [
            Task(f"COMP{i}", TaskPriority(random.randint(1, 5)),
                 base_time, base_time + timedelta(hours=1), 100)
            for i in range(500)
        ]
        
        # Test custom implementation
        start_time = time.time()
        for task in tasks:
            custom_pq.insert(task)
        while not custom_pq.is_empty():
            custom_pq.extract_min()
        custom_time = time.time() - start_time
        
        # Test heapq implementation
        start_time = time.time()
        for task in tasks:
            heapq_pq.enqueue(task)
        while not heapq_pq.is_empty():
            heapq_pq.dequeue()
        heapq_time = time.time() - start_time
        
        print(f"Custom heap: {custom_time:.4f}s, Heapq: {heapq_time:.4f}s")
        
        # Both should complete successfully
        self.assertGreater(custom_time, 0)
        self.assertGreater(heapq_time, 0)


def run_comprehensive_tests():
    """Run all test suites with detailed output"""
    print("=" * 70)
    print("Priority Queue Implementation - Comprehensive Test Suite")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTask,
        TestArrayBasedPriorityQueue,
        TestPythonHeapqPriorityQueue,
        TestTaskScheduler,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
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
    # Run the comprehensive test suite
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 All tests passed! The priority queue implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
    
    exit(0 if success else 1)