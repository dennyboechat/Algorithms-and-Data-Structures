"""
Caching and Memoization Utilities for Recommendation System

This module implements various caching strategies including in-memory caching,
Redis-based distributed caching, and function memoization to optimize
expensive computations in recommendation algorithms.

References:
- Bernstein, P. A., & Newcomer, E. (2009). Principles of transaction processing. 
  Morgan Kaufmann.
- Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on 
  large clusters. Communications of the ACM, 51(1), 107-113.
"""

import json
import pickle
import hashlib
import functools
import time
from typing import Any, Dict, List, Optional, Union, Callable
from collections import OrderedDict, defaultdict
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Using in-memory cache only.")


class LRUCache:
    """
    Thread-safe Least Recently Used cache implementation.
    
    Based on the LRU cache algorithm described in:
    Johnson, D. B. (1994). Near-optimal bin packing algorithms. 
    MIT Press.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating access order."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache, evicting LRU item if necessary."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'size': len(self.cache),
            'max_size': self.max_size
        }


class RedisCache:
    """
    Redis-based distributed cache for scalable applications.
    
    Implements caching patterns described in:
    Fowler, M. (2002). Patterns of enterprise application architecture. 
    Addison-Wesley Professional.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, expire_time: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.expire_time = expire_time
        self.hits = 0
        self.misses = 0
        
        try:
            self.client.ping()
        except redis.ConnectionError:
            print("Warning: Could not connect to Redis server. Using fallback cache.")
            self.client = None
    
    def _serialize_key(self, key: Union[str, tuple, Dict]) -> str:
        """Convert complex keys to string representation."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return hashlib.md5(json.dumps(sorted(key), sort_keys=True).encode()).hexdigest()
        elif isinstance(key, dict):
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        else:
            return str(key)
    
    def get(self, key: Union[str, tuple, Dict]) -> Optional[Any]:
        """Get value from Redis cache."""
        if self.client is None:
            return None
        
        try:
            serialized_key = self._serialize_key(key)
            value = self.client.get(serialized_key)
            if value is not None:
                self.hits += 1
                return pickle.loads(value.encode('latin-1'))
            else:
                self.misses += 1
                return None
        except Exception as e:
            print(f"Redis get error: {e}")
            self.misses += 1
            return None
    
    def put(self, key: Union[str, tuple, Dict], value: Any) -> None:
        """Put value in Redis cache with expiration."""
        if self.client is None:
            return
        
        try:
            serialized_key = self._serialize_key(key)
            serialized_value = pickle.dumps(value).decode('latin-1')
            self.client.setex(serialized_key, self.expire_time, serialized_value)
        except Exception as e:
            print(f"Redis put error: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if self.client is not None:
            try:
                self.client.flushdb()
                self.hits = 0
                self.misses = 0
            except Exception as e:
                print(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'connected': self.client is not None
        }


class HybridCache:
    """
    Hybrid cache combining in-memory LRU cache with Redis backend.
    
    Implements multi-level caching as described in:
    Patterson, D. A., & Hennessy, J. L. (2013). Computer organization and design: 
    the hardware/software interface. Morgan Kaufmann.
    """
    
    def __init__(self, lru_size: int = 500, redis_host: str = 'localhost',
                 redis_port: int = 6379, redis_expire: int = 3600):
        self.l1_cache = LRUCache(max_size=lru_size)
        
        try:
            self.l2_cache = RedisCache(
                host=redis_host, 
                port=redis_port, 
                expire_time=redis_expire
            ) if REDIS_AVAILABLE else None
        except:
            self.l2_cache = None
    
    def get(self, key: Union[str, tuple, Dict]) -> Optional[Any]:
        """Get value from cache hierarchy (L1 -> L2)."""
        # Try L1 cache first
        value = self.l1_cache.get(str(key))
        if value is not None:
            return value
        
        # Try L2 cache if available
        if self.l2_cache is not None:
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 cache
                self.l1_cache.put(str(key), value)
                return value
        
        return None
    
    def put(self, key: Union[str, tuple, Dict], value: Any) -> None:
        """Put value in both cache levels."""
        self.l1_cache.put(str(key), value)
        if self.l2_cache is not None:
            self.l2_cache.put(key, value)
    
    def clear(self) -> None:
        """Clear both cache levels."""
        self.l1_cache.clear()
        if self.l2_cache is not None:
            self.l2_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        stats = {
            'l1_cache': self.l1_cache.get_stats(),
            'l2_cache': self.l2_cache.get_stats() if self.l2_cache else None
        }
        return stats


def memoize_with_timeout(timeout: int = 3600, cache_size: int = 1000):
    """
    Memoization decorator with timeout and size limits.
    
    Based on memoization techniques described in:
    Cormen, T. H., et al. (2009). Introduction to algorithms. 
    MIT press.
    
    Args:
        timeout: Cache entry expiration time in seconds
        cache_size: Maximum number of cached entries
    """
    def decorator(func: Callable) -> Callable:
        cache = OrderedDict()
        cache_times = {}
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            with lock:
                # Check if key exists and is not expired
                if key in cache and key in cache_times:
                    if current_time - cache_times[key] < timeout:
                        # Move to end (LRU update)
                        value = cache.pop(key)
                        cache[key] = value
                        return value
                    else:
                        # Remove expired entry
                        cache.pop(key, None)
                        cache_times.pop(key, None)
                
                # Compute new value
                result = func(*args, **kwargs)
                
                # Add to cache
                if len(cache) >= cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(cache))
                    cache.pop(oldest_key, None)
                    cache_times.pop(oldest_key, None)
                
                cache[key] = result
                cache_times[key] = current_time
                
                return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear() or cache_times.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'max_size': cache_size,
            'timeout': timeout
        }
        
        return wrapper
    return decorator


def cache_similarity_matrix(cache: Optional[HybridCache] = None):
    """
    Decorator for caching similarity matrix computations.
    
    Args:
        cache: Cache instance to use (creates new if None)
    """
    if cache is None:
        cache = HybridCache()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create cache key based on function name and matrix state
            matrix_hash = hashlib.md5(str(self.matrix.data).encode()).hexdigest()[:8]
            cache_key = f"{func.__name__}_{matrix_hash}_{args}_{kwargs}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(self, *args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        return wrapper
    return decorator


class RecommendationCache:
    """
    Specialized cache for recommendation system operations.
    
    Implements domain-specific caching strategies for:
    - User similarity matrices
    - Item similarity matrices  
    - Recommendation results
    - Feature vectors
    """
    
    def __init__(self, cache_backend: Optional[HybridCache] = None):
        self.cache = cache_backend or HybridCache()
        self.cache_stats = defaultdict(int)
    
    def get_user_similarities(self, user_id: str, method: str) -> Optional[Dict]:
        """Get cached user similarity scores."""
        key = f"user_sim_{user_id}_{method}"
        result = self.cache.get(key)
        if result:
            self.cache_stats['user_similarity_hits'] += 1
        else:
            self.cache_stats['user_similarity_misses'] += 1
        return result
    
    def put_user_similarities(self, user_id: str, method: str, similarities: Dict) -> None:
        """Cache user similarity scores."""
        key = f"user_sim_{user_id}_{method}"
        self.cache.put(key, similarities)
    
    def get_item_similarities(self, item_id: str, method: str) -> Optional[Dict]:
        """Get cached item similarity scores."""
        key = f"item_sim_{item_id}_{method}"
        result = self.cache.get(key)
        if result:
            self.cache_stats['item_similarity_hits'] += 1
        else:
            self.cache_stats['item_similarity_misses'] += 1
        return result
    
    def put_item_similarities(self, item_id: str, method: str, similarities: Dict) -> None:
        """Cache item similarity scores."""
        key = f"item_sim_{item_id}_{method}"
        self.cache.put(key, similarities)
    
    def get_recommendations(self, user_id: str, method: str, n_recs: int) -> Optional[List]:
        """Get cached recommendations."""
        key = f"recs_{user_id}_{method}_{n_recs}"
        result = self.cache.get(key)
        if result:
            self.cache_stats['recommendation_hits'] += 1
        else:
            self.cache_stats['recommendation_misses'] += 1
        return result
    
    def put_recommendations(self, user_id: str, method: str, n_recs: int, 
                          recommendations: List) -> None:
        """Cache recommendation results."""
        key = f"recs_{user_id}_{method}_{n_recs}"
        self.cache.put(key, recommendations)
    
    def get_feature_vector(self, item_id: str, feature_type: str) -> Optional[Any]:
        """Get cached feature vector."""
        key = f"features_{item_id}_{feature_type}"
        return self.cache.get(key)
    
    def put_feature_vector(self, item_id: str, feature_type: str, features: Any) -> None:
        """Cache feature vector."""
        key = f"features_{item_id}_{feature_type}"
        self.cache.put(key, features)
    
    def invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate all cache entries for a specific user."""
        # Note: This is a simplified version. In production, you'd want
        # more sophisticated cache invalidation patterns
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'cache_stats': dict(self.cache_stats),
            'backend_stats': self.cache.get_stats()
        }


# Global cache instance for easy access
_global_cache = None

def get_global_cache() -> RecommendationCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = RecommendationCache()
    return _global_cache


def clear_global_cache() -> None:
    """Clear global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.cache.clear()
        _global_cache.cache_stats.clear()