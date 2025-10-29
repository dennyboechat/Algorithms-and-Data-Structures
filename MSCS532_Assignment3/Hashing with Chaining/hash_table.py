"""
Hash Table with Chaining Implementation

This module implements a hash table using chaining for collision resolution.
The implementation uses a universal hash function family to minimize collisions
and supports efficient insert, search, and delete operations.
"""

import random
from typing import Any, Optional, List, Tuple


class HashNode:
    """
    Node class for the chaining implementation.
    Each node stores a key-value pair and a reference to the next node.
    """
    
    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value
        self.next: Optional['HashNode'] = None


class UniversalHashFunction:
    """
    Universal hash function family implementation.
    Uses the formula: h(k) = ((a * k + b) mod p) mod m
    where p is a large prime, a and b are random coefficients.
    """
    
    def __init__(self, table_size: int):
        self.table_size = table_size
        # Use a large prime number greater than the universe of keys
        self.prime = 2**31 - 1  # Mersenne prime
        # Random coefficients for universal hashing
        self.a = random.randint(1, self.prime - 1)
        self.b = random.randint(0, self.prime - 1)
    
    def hash(self, key: Any) -> int:
        """
        Hash function that converts any key to an index in the hash table.
        
        Args:
            key: The key to hash (can be any hashable type)
            
        Returns:
            int: Hash value in range [0, table_size)
        """
        # Convert key to integer using Python's built-in hash function
        key_int = hash(key)
        # Ensure positive value
        if key_int < 0:
            key_int = abs(key_int)
        
        # Apply universal hash function
        return ((self.a * key_int + self.b) % self.prime) % self.table_size


class HashTableChaining:
    """
    Hash table implementation using chaining for collision resolution.
    
    Features:
    - Universal hash function family to minimize collisions
    - Dynamic resizing when load factor exceeds threshold
    - Efficient insert, search, and delete operations
    - Support for any hashable key type
    """
    
    def __init__(self, initial_size: int = 11, load_factor_threshold: float = 0.75):
        """
        Initialize the hash table.
        
        Args:
            initial_size: Initial size of the hash table (default: 11)
            load_factor_threshold: Threshold for resizing (default: 0.75)
        """
        self.size = initial_size
        self.count = 0
        self.load_factor_threshold = load_factor_threshold
        self.table: List[Optional[HashNode]] = [None] * self.size
        self.hash_function = UniversalHashFunction(self.size)
        
        # Statistics for analysis
        self.collision_count = 0
        self.resize_count = 0
    
    def _hash(self, key: Any) -> int:
        """Get hash value for a key."""
        return self.hash_function.hash(key)
    
    def _resize(self):
        """
        Resize the hash table when load factor exceeds threshold.
        Uses the next prime number approximately double the current size.
        """
        old_table = self.table
        old_size = self.size
        
        # Find next prime approximately double the size
        self.size = self._next_prime(self.size * 2)
        self.table = [None] * self.size
        self.hash_function = UniversalHashFunction(self.size)
        
        # Reset count and rehash all elements
        old_count = self.count
        self.count = 0
        self.resize_count += 1
        
        # Rehash all existing elements
        for head in old_table:
            current = head
            while current:
                self._insert_without_resize(current.key, current.value)
                current = current.next
        
        print(f"Resized hash table from {old_size} to {self.size} "
              f"(rehashed {old_count} elements)")
    
    def _next_prime(self, n: int) -> int:
        """Find the next prime number greater than or equal to n."""
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def _insert_without_resize(self, key: Any, value: Any):
        """Insert without checking for resize (used during resizing)."""
        index = self._hash(key)
        
        if self.table[index] is None:
            # No collision
            self.table[index] = HashNode(key, value)
        else:
            # Collision - check if key already exists
            current = self.table[index]
            while current:
                if current.key == key:
                    # Key exists, update value
                    current.value = value
                    return
                if current.next is None:
                    break
                current = current.next
            
            # Key doesn't exist, add new node at the end
            current.next = HashNode(key, value)
            self.collision_count += 1
        
        self.count += 1
    
    def insert(self, key: Any, value: Any):
        """
        Insert a key-value pair into the hash table.
        
        Args:
            key: The key to insert (must be hashable)
            value: The value associated with the key
            
        Time Complexity: O(1) average, O(n) worst case
        """
        # Check if resize is needed
        load_factor = (self.count + 1) / self.size
        if load_factor > self.load_factor_threshold:
            self._resize()
        
        index = self._hash(key)
        
        if self.table[index] is None:
            # No collision
            self.table[index] = HashNode(key, value)
            self.count += 1
        else:
            # Collision - traverse the chain
            current = self.table[index]
            while current:
                if current.key == key:
                    # Key already exists, update value
                    current.value = value
                    return
                if current.next is None:
                    break
                current = current.next
            
            # Key doesn't exist, add new node at the end of chain
            current.next = HashNode(key, value)
            self.count += 1
            self.collision_count += 1
    
    def search(self, key: Any) -> Optional[Any]:
        """
        Search for a value associated with the given key.
        
        Args:
            key: The key to search for
            
        Returns:
            The value associated with the key, or None if not found
            
        Time Complexity: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        current = self.table[index]
        
        while current:
            if current.key == key:
                return current.value
            current = current.next
        
        return None
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key-value pair from the hash table.
        
        Args:
            key: The key to delete
            
        Returns:
            True if the key was found and deleted, False otherwise
            
        Time Complexity: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        current = self.table[index]
        
        # If the chain is empty
        if current is None:
            return False
        
        # If the first node contains the key
        if current.key == key:
            self.table[index] = current.next
            self.count -= 1
            return True
        
        # Search in the rest of the chain
        while current.next:
            if current.next.key == key:
                current.next = current.next.next
                self.count -= 1
                return True
            current = current.next
        
        return False
    
    def contains(self, key: Any) -> bool:
        """
        Check if a key exists in the hash table.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        index = self._hash(key)
        current = self.table[index]
        
        while current:
            if current.key == key:
                return True
            current = current.next
        
        return False
    
    def get_load_factor(self) -> float:
        """Get the current load factor of the hash table."""
        return self.count / self.size if self.size > 0 else 0
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the hash table performance.
        
        Returns:
            Dictionary containing various statistics
        """
        # Calculate chain lengths
        chain_lengths = []
        max_chain_length = 0
        non_empty_slots = 0
        
        for head in self.table:
            length = 0
            current = head
            while current:
                length += 1
                current = current.next
            
            chain_lengths.append(length)
            if length > 0:
                non_empty_slots += 1
            max_chain_length = max(max_chain_length, length)
        
        avg_chain_length = sum(chain_lengths) / len(chain_lengths)
        
        return {
            'size': self.size,
            'count': self.count,
            'load_factor': self.get_load_factor(),
            'collision_count': self.collision_count,
            'resize_count': self.resize_count,
            'max_chain_length': max_chain_length,
            'avg_chain_length': avg_chain_length,
            'non_empty_slots': non_empty_slots,
            'utilization': non_empty_slots / self.size
        }
    
    def display(self):
        """Display the current state of the hash table."""
        print(f"\nHash Table (size: {self.size}, count: {self.count}, "
              f"load factor: {self.get_load_factor():.2f})")
        print("-" * 50)
        
        for i, head in enumerate(self.table):
            if head is not None:
                chain = []
                current = head
                while current:
                    chain.append(f"({current.key}: {current.value})")
                    current = current.next
                print(f"Slot {i:2d}: {' -> '.join(chain)}")
            else:
                print(f"Slot {i:2d}: Empty")
    
    def __len__(self) -> int:
        """Return the number of key-value pairs in the hash table."""
        return self.count
    
    def __contains__(self, key: Any) -> bool:
        """Support 'in' operator for checking key existence."""
        return self.contains(key)
    
    def __getitem__(self, key: Any) -> Any:
        """Support bracket notation for getting values."""
        result = self.search(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found")
        return result
    
    def __setitem__(self, key: Any, value: Any):
        """Support bracket notation for setting values."""
        self.insert(key, value)
    
    def __delitem__(self, key: Any):
        """Support del statement for removing keys."""
        if not self.delete(key):
            raise KeyError(f"Key '{key}' not found")
    
    def keys(self):
        """Return an iterator over all keys in the hash table."""
        for head in self.table:
            current = head
            while current:
                yield current.key
                current = current.next
    
    def values(self):
        """Return an iterator over all values in the hash table."""
        for head in self.table:
            current = head
            while current:
                yield current.value
                current = current.next
    
    def items(self):
        """Return an iterator over all key-value pairs in the hash table."""
        for head in self.table:
            current = head
            while current:
                yield (current.key, current.value)
                current = current.next


if __name__ == "__main__":
    # Basic demonstration
    print("Hash Table with Chaining - Basic Demo")
    print("=" * 50)
    
    # Create hash table
    ht = HashTableChaining(initial_size=7)
    
    # Insert some data
    test_data = [
        ("apple", 5),
        ("banana", 3),
        ("orange", 8),
        ("grape", 12),
        ("kiwi", 6),
        ("mango", 9),
        ("pear", 4)
    ]
    
    print("Inserting test data...")
    for key, value in test_data:
        ht.insert(key, value)
        print(f"Inserted ({key}: {value})")
    
    ht.display()
    
    # Test search operations
    print("\nTesting search operations:")
    for key, expected_value in test_data[:3]:
        found_value = ht.search(key)
        print(f"Search '{key}': {found_value} (expected: {expected_value})")
    
    # Test delete operations
    print("\nTesting delete operations:")
    keys_to_delete = ["banana", "grape"]
    for key in keys_to_delete:
        success = ht.delete(key)
        print(f"Delete '{key}': {'Success' if success else 'Failed'}")
    
    ht.display()
    
    # Show statistics
    print("\nHash Table Statistics:")
    stats = ht.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")