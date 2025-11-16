# Data Structures Implementation

This repository contains Python implementations of fundamental data structures including Arrays, Matrices, Stacks, Queues, and Linked Lists.

## Files

- `data_structures.py` - Main implementation file containing all data structure classes

## Requirements

- Python 3.6 or higher
- No external dependencies required (uses only Python standard library)

## How to Run the Code

### 1. Basic Usage

You can import and use the data structures in your Python scripts:

```python
from data_structures import Array, Matrix, Stack, Queue, SinglyLinkedList

# Example usage will be shown below
```

### 2. Running Interactive Examples

Open a Python interpreter in the project directory:

```bash
python3 -i data_structures.py
```

Or start Python and import the module:

```bash
python3
>>> from data_structures import *
```

### 3. Example Usage

#### Arrays
```python
# Create and use an array
arr = Array()
arr.insert(10)
arr.insert(20)
arr.insert(30)
print(arr.access(1))  # Output: 20
arr.delete(1)
print(arr.access(1))  # Output: 30
```

#### Matrices
```python
# Create a 3x3 matrix
matrix = Matrix(3, 3)
matrix.insert(0, 0, 5)
matrix.insert(1, 1, 10)
matrix.insert(2, 2, 15)
print(matrix.access(1, 1))  # Output: 10
```

#### Stacks
```python
# Create and use a stack
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.peek())  # Output: 3
print(stack.pop())   # Output: 3
print(stack.pop())   # Output: 2
```

#### Queues
```python
# Create and use a queue
queue = Queue()
queue.enqueue("first")
queue.enqueue("second")
queue.enqueue("third")
print(queue.front())     # Output: "first"
print(queue.dequeue())   # Output: "first"
print(queue.front())     # Output: "second"
```

#### Linked Lists
```python
# Create and use a linked list
ll = SinglyLinkedList()
ll.insert(1)
ll.insert(2)
ll.insert(3)
print(ll.traverse())     # Output: [1, 2, 3]
ll.delete(2)
print(ll.traverse())     # Output: [1, 3]
```

### 4. Testing the Implementation

You can create a simple test script to verify all operations work correctly:

```python
# test_data_structures.py
from data_structures import Array, Matrix, Stack, Queue, SinglyLinkedList

def test_all_structures():
    # Test Array
    print("Testing Array...")
    arr = Array()
    arr.insert(10)
    arr.insert(20)
    assert arr.access(0) == 10
    assert arr.delete(0) == 10
    print("Array tests passed!")
    
    # Test Matrix
    print("Testing Matrix...")
    matrix = Matrix(2, 2)
    matrix.insert(0, 0, 5)
    assert matrix.access(0, 0) == 5
    print("Matrix tests passed!")
    
    # Test Stack
    print("Testing Stack...")
    stack = Stack()
    stack.push(1)
    stack.push(2)
    assert stack.peek() == 2
    assert stack.pop() == 2
    print("Stack tests passed!")
    
    # Test Queue
    print("Testing Queue...")
    queue = Queue()
    queue.enqueue("a")
    queue.enqueue("b")
    assert queue.front() == "a"
    assert queue.dequeue() == "a"
    print("Queue tests passed!")
    
    # Test Linked List
    print("Testing Linked List...")
    ll = SinglyLinkedList()
    ll.insert(1)
    ll.insert(2)
    assert ll.traverse() == [1, 2]
    assert ll.delete(1) == True
    print("Linked List tests passed!")
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_all_structures()
```

Run the test script:

```bash
python3 test_data_structures.py
```
