"""
Priority Queue Demo and Examples
Practical demonstrations of the priority queue implementation
for various real-world scenarios.
"""

from datetime import datetime, timedelta
from priority_queue_implementation import (
    Task, TaskPriority, TaskScheduler, ArrayBasedPriorityQueue
)


def demo_basic_usage():
    """Basic usage example of the priority queue"""
    print("=" * 50)
    print("DEMO 1: Basic Priority Queue Usage")
    print("=" * 50)
    
    # Create priority queue
    pq = ArrayBasedPriorityQueue()
    base_time = datetime.now()
    
    # Create some tasks with different priorities
    tasks = [
        Task("EMAIL", TaskPriority.LOW, base_time, 
             base_time + timedelta(hours=2), 60, "Send newsletter"),
        Task("BACKUP", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(minutes=10), 300, "Database backup"),
        Task("REPORT", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(hours=1), 120, "Generate daily report"),
        Task("URGENT_FIX", TaskPriority.HIGH, base_time,
             base_time + timedelta(minutes=30), 90, "Fix production bug"),
    ]
    
    print("Adding tasks to queue:")
    for task in tasks:
        pq.insert(task)
        print(f"  Added: {task}")
    
    print(f"\nQueue size: {pq.get_size()}")
    print("\nExtracting tasks in priority order:")
    
    while not pq.is_empty():
        task = pq.extract_min()
        print(f"  Executing: {task}")
    
    print("\nDemo 1 completed!\n")


def demo_operating_system_scheduler():
    """Simulate an operating system task scheduler"""
    print("=" * 50)
    print("DEMO 2: Operating System Task Scheduler")
    print("=" * 50)
    
    scheduler = TaskScheduler(use_custom_heap=True)
    base_time = datetime.now()
    
    # Create system tasks with different characteristics
    system_tasks = [
        # Critical system processes
        Task("KERNEL_001", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(milliseconds=100), 10, "Kernel interrupt handler"),
        
        Task("MEMORY_MGR", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(milliseconds=200), 20, "Memory management"),
        
        # High priority system services
        Task("NET_STACK", TaskPriority.HIGH, base_time,
             base_time + timedelta(seconds=1), 50, "Network stack processing"),
        
        Task("FILE_SYS", TaskPriority.HIGH, base_time,
             base_time + timedelta(seconds=2), 80, "File system operations"),
        
        # User applications
        Task("WEB_BROWSER", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(seconds=5), 200, "Web browser rendering"),
        
        Task("TEXT_EDITOR", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(seconds=10), 150, "Text editor operations"),
        
        # Background tasks
        Task("ANTIVIRUS", TaskPriority.LOW, base_time,
             base_time + timedelta(minutes=1), 500, "Antivirus scan"),
        
        Task("INDEX_SEARCH", TaskPriority.BACKGROUND, base_time,
             base_time + timedelta(minutes=5), 1000, "Search index update"),
    ]
    
    print("System startup - scheduling initial tasks:")
    for task in system_tasks:
        scheduler.add_task(task)
    
    print(f"\nSystem status: {scheduler.get_status()}")
    
    # Simulate multiple scheduling rounds
    print("\n--- CPU Scheduling Cycles ---")
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        scheduler.run_scheduling_cycle(max_tasks=3)
        print(f"Status: {scheduler.get_status()}")
    
    print("\nDemo 2 completed!\n")


def demo_web_server_request_handling():
    """Simulate a web server handling requests with different priorities"""
    print("=" * 50)
    print("DEMO 3: Web Server Request Handling")
    print("=" * 50)
    
    scheduler = TaskScheduler(use_custom_heap=True)
    base_time = datetime.now()
    
    # Simulate incoming web requests
    web_requests = [
        # Critical API endpoints
        Task("API_AUTH", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(seconds=1), 100, "User authentication API"),
        
        Task("API_PAYMENT", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(seconds=2), 200, "Payment processing API"),
        
        # High priority user-facing requests
        Task("PAGE_HOME", TaskPriority.HIGH, base_time,
             base_time + timedelta(seconds=3), 150, "Homepage request"),
        
        Task("PAGE_DASHBOARD", TaskPriority.HIGH, base_time,
             base_time + timedelta(seconds=4), 180, "User dashboard"),
        
        # Medium priority content requests
        Task("IMG_THUMB", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(seconds=10), 50, "Thumbnail generation"),
        
        Task("SEARCH_QUERY", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(seconds=15), 300, "Search request"),
        
        # Low priority background tasks
        Task("ANALYTICS", TaskPriority.LOW, base_time,
             base_time + timedelta(minutes=1), 400, "Analytics processing"),
        
        Task("CACHE_WARM", TaskPriority.BACKGROUND, base_time,
             base_time + timedelta(minutes=5), 600, "Cache warming"),
    ]
    
    print("Incoming web requests:")
    for request in web_requests:
        scheduler.add_task(request)
        print(f"  Received: {request}")
    
    print(f"\nServer status: {scheduler.get_status()}")
    
    # Process requests in priority order
    print("\n--- Request Processing ---")
    while scheduler.get_status()["pending_tasks"] > 0:
        print(f"\nProcessing batch...")
        scheduler.run_scheduling_cycle(max_tasks=2)
        status = scheduler.get_status()
        print(f"Remaining: {status['pending_tasks']}, Completed: {status['completed_tasks']}")
    
    print("\nAll requests processed!")
    print("Demo 3 completed!\n")


def demo_hospital_emergency_system():
    """Simulate a hospital emergency department task prioritization"""
    print("=" * 50)
    print("DEMO 4: Hospital Emergency Department")
    print("=" * 50)
    
    scheduler = TaskScheduler(use_custom_heap=True)
    base_time = datetime.now()
    
    # Medical emergency tasks with different urgency levels
    emergency_tasks = [
        # Life-threatening emergencies (Critical)
        Task("CARDIAC_ARREST", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(minutes=2), 1800, "Cardiac arrest patient"),
        
        Task("SEVERE_TRAUMA", TaskPriority.CRITICAL, base_time,
             base_time + timedelta(minutes=3), 2400, "Major trauma case"),
        
        # Urgent but stable (High)
        Task("CHEST_PAIN", TaskPriority.HIGH, base_time,
             base_time + timedelta(minutes=15), 900, "Chest pain evaluation"),
        
        Task("SEVERE_ASTHMA", TaskPriority.HIGH, base_time,
             base_time + timedelta(minutes=20), 600, "Severe asthma attack"),
        
        # Semi-urgent (Medium)
        Task("BROKEN_BONE", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(hours=2), 3600, "Suspected fracture"),
        
        Task("MODERATE_BURN", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(hours=1), 1800, "Second-degree burn"),
        
        # Non-urgent (Low)
        Task("MINOR_CUT", TaskPriority.LOW, base_time,
             base_time + timedelta(hours=4), 900, "Minor laceration"),
        
        Task("COLD_SYMPTOMS", TaskPriority.BACKGROUND, base_time,
             base_time + timedelta(hours=6), 600, "Cold/flu symptoms"),
    ]
    
    print("Emergency department - incoming patients:")
    for task in emergency_tasks:
        scheduler.add_task(task)
        print(f"  Triage: {task}")
    
    print(f"\nED Status: {scheduler.get_status()}")
    print("Priority triage order:")
    
    # Show the order patients will be seen
    temp_pq = ArrayBasedPriorityQueue()
    for task in emergency_tasks:
        temp_pq.insert(task)
    
    order = 1
    while not temp_pq.is_empty():
        patient = temp_pq.extract_min()
        print(f"  {order}. {patient.task_id} ({patient.priority.name}) - {patient.description}")
        order += 1
    
    print("\n--- Treatment Processing ---")
    # Process patients in priority order
    scheduler.run_scheduling_cycle(max_tasks=len(emergency_tasks))
    
    print(f"\nFinal status: {scheduler.get_status()}")
    print("Demo 4 completed!\n")


def demo_priority_updates():
    """Demonstrate dynamic priority updates"""
    print("=" * 50)
    print("DEMO 5: Dynamic Priority Updates")
    print("=" * 50)
    
    pq = ArrayBasedPriorityQueue()
    base_time = datetime.now()
    
    # Create tasks with initial priorities
    tasks = [
        Task("TASK_A", TaskPriority.LOW, base_time,
             base_time + timedelta(hours=1), 300, "Background processing"),
        Task("TASK_B", TaskPriority.MEDIUM, base_time,
             base_time + timedelta(minutes=30), 200, "Regular maintenance"),
        Task("TASK_C", TaskPriority.HIGH, base_time,
             base_time + timedelta(minutes=15), 150, "User request"),
    ]
    
    print("Initial task priorities:")
    for task in tasks:
        pq.insert(task)
        print(f"  {task}")
    
    print(f"\nCurrent top priority: {pq.peek_min()}")
    
    # Simulate a situation where a low-priority task becomes critical
    print("\n--- SITUATION CHANGE ---")
    print("TASK_A discovered to be critical for system stability!")
    
    # Update priority
    success = pq.update_task_priority("TASK_A", TaskPriority.CRITICAL)
    print(f"Priority update successful: {success}")
    
    print(f"\nNew top priority: {pq.peek_min()}")
    
    print("\nExecuting tasks in new priority order:")
    while not pq.is_empty():
        task = pq.extract_min()
        print(f"  Executing: {task}")
    
    print("Demo 5 completed!\n")


def demo_batch_operations():
    """Demonstrate batch operations for efficiency"""
    print("=" * 50)
    print("DEMO 6: Batch Operations")
    print("=" * 50)
    
    pq = ArrayBasedPriorityQueue()
    base_time = datetime.now()
    
    # Create a large batch of tasks
    batch_tasks = []
    for i in range(20):
        priority = TaskPriority(i % 5 + 1)  # Cycle through priorities
        task = Task(
            f"BATCH_{i:02d}", priority, base_time,
            base_time + timedelta(minutes=i*5), 100 + i*10,
            f"Batch task number {i}"
        )
        batch_tasks.append(task)
    
    print(f"Creating batch of {len(batch_tasks)} tasks...")
    
    # Use batch insertion for efficiency
    pq.enqueue_batch(batch_tasks)
    print(f"Batch inserted! Queue size: {pq.get_size()}")
    
    # Extract first few to show they're properly prioritized
    print("\nFirst 5 tasks in priority order:")
    for i in range(5):
        if not pq.is_empty():
            task = pq.extract_min()
            print(f"  {i+1}. {task}")
    
    print(f"\nRemaining tasks in queue: {pq.get_size()}")
    print("Demo 6 completed!\n")


def run_all_demos():
    """Run all demonstration scenarios"""
    print("🚀 Priority Queue Implementation - Complete Demo Suite")
    print("=" * 70)
    
    demos = [
        demo_basic_usage,
        demo_operating_system_scheduler,
        demo_web_server_request_handling,
        demo_hospital_emergency_system,
        demo_priority_updates,
        demo_batch_operations
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"\n📋 Running Demo {i}/{len(demos)}")
        demo()
        
        if i < len(demos):
            input("Press Enter to continue to next demo...")
    
    print("=" * 70)
    print("🎉 All demos completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. Array-based heap provides O(log n) insertion and extraction")
    print("2. Min-heap ensures highest priority tasks are processed first")
    print("3. Composite priority allows for complex scheduling decisions")
    print("4. Dynamic priority updates maintain heap property efficiently")
    print("5. Batch operations provide better performance for bulk insertions")
    print("6. The implementation scales well for real-world applications")


if __name__ == "__main__":
    # Run all demonstrations
    run_all_demos()