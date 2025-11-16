# Data Structures Implementation in Python

# 1. Arrays and Matrices
class Array:
    def __init__(self):
        self.data = []

    def insert(self, value):
        self.data.append(value)

    def delete(self, index):
        if 0 <= index < len(self.data):
            return self.data.pop(index)
        raise IndexError('Index out of range')

    def access(self, index):
        if 0 <= index < len(self.data):
            return self.data[index]
        raise IndexError('Index out of range')

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for _ in range(cols)] for _ in range(rows)]

    def insert(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row][col] = value
        else:
            raise IndexError('Index out of range')

    def access(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.data[row][col]
        raise IndexError('Index out of range')

    def delete(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row][col] = 0
        else:
            raise IndexError('Index out of range')

# 2. Stacks and Queues using Arrays
class Stack:
    def __init__(self):
        self.data = []

    def push(self, value):
        self.data.append(value)

    def pop(self):
        if not self.is_empty():
            return self.data.pop()
        raise IndexError('Pop from empty stack')

    def peek(self):
        if not self.is_empty():
            return self.data[-1]
        raise IndexError('Peek from empty stack')

    def is_empty(self):
        return len(self.data) == 0

class Queue:
    def __init__(self):
        self.data = []

    def enqueue(self, value):
        self.data.append(value)

    def dequeue(self):
        if not self.is_empty():
            return self.data.pop(0)
        raise IndexError('Dequeue from empty queue')

    def front(self):
        if not self.is_empty():
            return self.data[0]
        raise IndexError('Front from empty queue')

    def is_empty(self):
        return len(self.data) == 0

# 3. Singly Linked List
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def insert(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def delete(self, value):
        current = self.head
        prev = None
        while current:
            if current.value == value:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return True
            prev = current
            current = current.next
        return False

    def traverse(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.value)
            current = current.next
        return elements
