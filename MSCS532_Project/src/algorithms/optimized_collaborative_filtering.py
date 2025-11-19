"""
Optimized Collaborative Filtering Algorithms with Caching and Memoization

This module implements enhanced user-based and item-based collaborative filtering 
algorithms with advanced caching, memoization, and optimization techniques for
improved performance and scalability.

Performance optimizations include:
- LRU caching for similarity computations
- Memoization of expensive calculations
- Sparse matrix operations
- Parallel processing support
- Memory-efficient data structures

References:
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques 
  for recommender systems. Computer, 42(8), 30-37.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based 
  collaborative filtering recommendation algorithms. WWW '01.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, linalg
import concurrent.futures
import threading
import time
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Import our caching utilities
try:
    from ..utils.caching import (
        memoize_with_timeout, 
        cache_similarity_matrix, 
        RecommendationCache,
        get_global_cache
    )
except ImportError:
    # Fallback for standalone execution
    from functools import wraps
    
    def memoize_with_timeout(timeout=3600, cache_size=1000):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def cache_similarity_matrix(cache=None):
        def decorator(func):
            return func
        return decorator
    
    class RecommendationCache:
        def __init__(self): pass
        def get_user_similarities(self, *args): return None
        def put_user_similarities(self, *args): pass
        def get_recommendations(self, *args): return None
        def put_recommendations(self, *args): pass
    
    def get_global_cache():
        return RecommendationCache()


class OptimizedUserBasedCollaborativeFiltering:
    """
    Enhanced user-based collaborative filtering with advanced optimizations.
    
    Key optimizations:
    - Cached similarity computations
    - Memoized predictions
    - Sparse matrix operations
    - Parallel neighborhood computation
    - Memory-efficient data structures
    
    Based on the algorithms described in:
    Herlocker, J. L., et al. (1999). Algorithmic framework for performing 
    collaborative filtering. SIGIR '99.
    """
    
    def __init__(self, k_neighbors: int = 20, similarity_metric: str = 'cosine',
                 min_common_items: int = 2, cache_size: int = 1000,
                 use_parallel: bool = True, n_jobs: int = -1):
        """
        Initialize optimized user-based collaborative filtering.
        
        Args:
            k_neighbors: Number of similar users to consider
            similarity_metric: 'cosine', 'pearson', or 'jaccard'
            min_common_items: Minimum number of common items for similarity
            cache_size: Size of similarity cache
            use_parallel: Enable parallel processing
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.min_common_items = min_common_items
        self.cache_size = cache_size
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs
        
        # Core data structures
        self.user_item_matrix = None
        self.sparse_matrix = None
        self.user_similarity_matrix = None
        self.user_means = None
        self.user_std = None
        self.fitted = False
        
        # Caching infrastructure
        self.cache = get_global_cache()
        self._similarity_cache = {}
        self._prediction_cache = {}
        self._lock = threading.Lock()
        
        # Performance monitoring
        self.computation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    @memoize_with_timeout(timeout=3600, cache_size=1000)
    def _compute_user_similarity(self, user1_idx: int, user2_idx: int) -> float:
        """
        Compute similarity between two users with memoization.
        
        Args:
            user1_idx: Index of first user
            user2_idx: Index of second user
            
        Returns:
            Similarity score
        """
        if user1_idx == user2_idx:
            return 1.0
        
        # Get user vectors from sparse matrix
        user1_vector = self.sparse_matrix[user1_idx].toarray().flatten()
        user2_vector = self.sparse_matrix[user2_idx].toarray().flatten()
        
        # Find common rated items
        common_mask = (user1_vector > 0) & (user2_vector > 0)
        common_items = np.sum(common_mask)
        
        if common_items < self.min_common_items:
            return 0.0
        
        user1_common = user1_vector[common_mask]
        user2_common = user2_vector[common_mask]
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(user1_common, user2_common)
            norm1 = np.linalg.norm(user1_common)
            norm2 = np.linalg.norm(user2_common)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        elif self.similarity_metric == 'pearson':
            # Pearson correlation
            if len(user1_common) < 2:
                return 0.0
            return np.corrcoef(user1_common, user2_common)[0, 1] if not np.isnan(np.corrcoef(user1_common, user2_common)[0, 1]) else 0.0
        
        elif self.similarity_metric == 'jaccard':
            # Jaccard similarity (binary)
            intersection = np.sum((user1_vector > 0) & (user2_vector > 0))
            union = np.sum((user1_vector > 0) | (user2_vector > 0))
            return intersection / union if union > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def _compute_similarities_parallel(self, user_indices: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Compute similarities in parallel for better performance.
        
        Args:
            user_indices: List of user indices to compute similarities for
            
        Returns:
            Dictionary of similarity scores
        """
        similarities = {}
        
        if not self.use_parallel:
            # Sequential computation
            for i, user1_idx in enumerate(user_indices):
                for user2_idx in user_indices[i+1:]:
                    similarity = self._compute_user_similarity(user1_idx, user2_idx)
                    similarities[(user1_idx, user2_idx)] = similarity
                    similarities[(user2_idx, user1_idx)] = similarity
            return similarities
        
        # Parallel computation
        tasks = []
        for i, user1_idx in enumerate(user_indices):
            for user2_idx in user_indices[i+1:]:
                tasks.append((user1_idx, user2_idx))
        
        max_workers = self.n_jobs if self.n_jobs > 0 else None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._compute_user_similarity, user1_idx, user2_idx): (user1_idx, user2_idx)
                for user1_idx, user2_idx in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                user1_idx, user2_idx = future_to_task[future]
                try:
                    similarity = future.result()
                    similarities[(user1_idx, user2_idx)] = similarity
                    similarities[(user2_idx, user1_idx)] = similarity
                except Exception as e:
                    print(f"Error computing similarity for users {user1_idx}, {user2_idx}: {e}")
                    similarities[(user1_idx, user2_idx)] = 0.0
                    similarities[(user2_idx, user1_idx)] = 0.0
        
        return similarities
    
    @cache_similarity_matrix()
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Fit the model with optimized matrix operations and caching.
        
        Args:
            user_item_matrix: DataFrame with users as rows and items as columns
        """
        start_time = time.time()
        
        print(f"Fitting optimized user-based CF with {len(user_item_matrix)} users and {len(user_item_matrix.columns)} items...")
        
        # Store original matrix
        self.user_item_matrix = user_item_matrix.fillna(0)
        
        # Convert to sparse matrix for memory efficiency
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Compute user statistics
        self.user_means = self.user_item_matrix.mean(axis=1)
        self.user_std = self.user_item_matrix.std(axis=1).fillna(0)
        
        # Build similarity matrix with caching
        n_users = len(self.user_item_matrix)
        user_indices = list(range(n_users))
        
        # Check if similarity matrix is cached
        cache_key = f"user_similarities_{self.similarity_metric}_{n_users}"
        cached_similarities = self.cache.get_user_similarities(cache_key, self.similarity_metric)
        
        if cached_similarities is not None:
            print("Loading similarity matrix from cache...")
            self.user_similarity_matrix = pd.DataFrame(
                cached_similarities,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
            self.cache_hits += 1
        else:
            print("Computing user similarity matrix...")
            
            # Compute similarities (potentially in parallel)
            if n_users > 1000:
                # For large datasets, compute similarities in chunks
                chunk_size = min(500, n_users // 4)
                similarity_dict = {}
                
                for i in range(0, n_users, chunk_size):
                    chunk_indices = user_indices[i:i+chunk_size]
                    chunk_similarities = self._compute_similarities_parallel(chunk_indices)
                    similarity_dict.update(chunk_similarities)
            else:
                similarity_dict = self._compute_similarities_parallel(user_indices)
            
            # Convert to matrix format
            similarity_matrix = np.zeros((n_users, n_users))
            for (i, j), sim in similarity_dict.items():
                similarity_matrix[i, j] = sim
            
            # Add diagonal (self-similarity)
            np.fill_diagonal(similarity_matrix, 1.0)
            
            self.user_similarity_matrix = pd.DataFrame(
                similarity_matrix,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
            
            # Cache the result
            self.cache.put_user_similarities(cache_key, self.similarity_metric, similarity_matrix.tolist())
            self.cache_misses += 1
        
        self.fitted = True
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        print(f"Model fitting completed in {computation_time:.2f} seconds")
    
    @memoize_with_timeout(timeout=1800, cache_size=5000)
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating with memoization and optimized computation.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle unknown users/items
        if user_id not in self.user_item_matrix.index:
            if item_id in self.user_item_matrix.columns:
                return self.user_item_matrix[item_id].mean()
            return 3.0  # Default rating
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_means[user_id]
        
        # Check if user has already rated this item
        existing_rating = self.user_item_matrix.loc[user_id, item_id]
        if existing_rating > 0:
            return existing_rating
        
        # Get similar users who have rated this item
        user_similarities = self.user_similarity_matrix[user_id]
        
        # Filter users who have rated the item
        item_ratings = self.user_item_matrix[item_id]
        rated_users_mask = item_ratings > 0
        
        if not rated_users_mask.any():
            return self.user_means[user_id]
        
        # Get similarities for users who rated the item
        similar_users = user_similarities[rated_users_mask].sort_values(ascending=False)
        
        # Take top-k similar users
        top_similar = similar_users.head(self.k_neighbors)
        
        if len(top_similar) == 0 or top_similar.sum() == 0:
            return self.user_means[user_id]
        
        # Optimized prediction computation
        if self.similarity_metric == 'pearson':
            # Mean-centered prediction for Pearson correlation
            numerator = 0
            denominator = 0
            
            for similar_user, similarity in top_similar.items():
                if similarity > 0:
                    user_rating = self.user_item_matrix.loc[similar_user, item_id]
                    centered_rating = user_rating - self.user_means[similar_user]
                    numerator += similarity * centered_rating
                    denominator += abs(similarity)
            
            if denominator == 0:
                return self.user_means[user_id]
            
            prediction = self.user_means[user_id] + (numerator / denominator)
        else:
            # Weighted average for other metrics
            weights = top_similar[top_similar > 0]
            if len(weights) == 0:
                return self.user_means[user_id]
            
            weighted_ratings = []
            total_weight = 0
            
            for similar_user, similarity in weights.items():
                rating = self.user_item_matrix.loc[similar_user, item_id]
                weighted_ratings.append(similarity * rating)
                total_weight += similarity
            
            prediction = sum(weighted_ratings) / total_weight if total_weight > 0 else self.user_means[user_id]
        
        # Clamp to valid rating range
        return max(1.0, min(5.0, prediction))
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate optimized recommendations with caching.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude seen items
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Check cache first
        cache_key = f"{user_id}_{n_recommendations}_{exclude_seen}"
        cached_recs = self.cache.get_recommendations(user_id, "user_based_cf", n_recommendations)
        if cached_recs is not None:
            self.cache_hits += 1
            return cached_recs
        
        start_time = time.time()
        
        # Get candidate items
        if exclude_seen and user_id in self.user_item_matrix.index:
            seen_items = set(self.user_item_matrix.loc[
                self.user_item_matrix.loc[user_id] > 0
            ].index)
            candidate_items = [item for item in self.user_item_matrix.columns 
                             if item not in seen_items]
        else:
            candidate_items = list(self.user_item_matrix.columns)
        
        # Batch prediction for efficiency
        predictions = []
        
        if self.use_parallel and len(candidate_items) > 100:
            # Parallel prediction for large candidate sets
            max_workers = min(4, self.n_jobs) if self.n_jobs > 0 else 4
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(self.predict_rating, user_id, item_id): item_id
                    for item_id in candidate_items
                }
                
                for future in concurrent.futures.as_completed(future_to_item):
                    item_id = future_to_item[future]
                    try:
                        predicted_rating = future.result()
                        predictions.append((item_id, predicted_rating))
                    except Exception as e:
                        print(f"Error predicting rating for {item_id}: {e}")
        else:
            # Sequential prediction
            for item_id in candidate_items:
                try:
                    predicted_rating = self.predict_rating(user_id, item_id)
                    predictions.append((item_id, predicted_rating))
                except Exception as e:
                    print(f"Error predicting rating for {item_id}: {e}")
        
        # Sort and get top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations = predictions[:n_recommendations]
        
        # Cache the results
        self.cache.put_recommendations(user_id, "user_based_cf", n_recommendations, recommendations)
        self.cache_misses += 1
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Union[int, float, List]]:
        """
        Get performance statistics and metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        cache_total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0.0,
            'total_computations': len(self.computation_times),
            'matrix_density': self.sparse_matrix.nnz / np.prod(self.sparse_matrix.shape) if self.sparse_matrix is not None else 0.0,
            'cache_stats': self.cache.get_cache_stats()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._similarity_cache.clear()
        self._prediction_cache.clear()
        self.cache.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class OptimizedItemBasedCollaborativeFiltering:
    """
    Enhanced item-based collaborative filtering with optimizations.
    
    Implements the optimized item-based algorithm described in:
    Sarwar, B., et al. (2001). Item-based collaborative filtering 
    recommendation algorithms. WWW '01.
    """
    
    def __init__(self, k_neighbors: int = 20, similarity_metric: str = 'cosine',
                 min_common_users: int = 2, cache_size: int = 1000,
                 use_parallel: bool = True):
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.min_common_users = min_common_users
        self.cache_size = cache_size
        self.use_parallel = use_parallel
        
        # Core data structures
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.item_means = None
        self.fitted = False
        
        # Caching
        self.cache = get_global_cache()
        self.computation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    @memoize_with_timeout(timeout=3600, cache_size=1000)
    def _compute_item_similarity(self, item1: str, item2: str) -> float:
        """Compute similarity between two items with memoization."""
        if item1 == item2:
            return 1.0
        
        # Get user ratings for both items
        item1_ratings = self.user_item_matrix[item1]
        item2_ratings = self.user_item_matrix[item2]
        
        # Find common users
        common_users = (item1_ratings > 0) & (item2_ratings > 0)
        
        if common_users.sum() < self.min_common_users:
            return 0.0
        
        ratings1 = item1_ratings[common_users]
        ratings2 = item2_ratings[common_users]
        
        if self.similarity_metric == 'cosine':
            dot_product = np.dot(ratings1, ratings2)
            norm1 = np.linalg.norm(ratings1)
            norm2 = np.linalg.norm(ratings2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        elif self.similarity_metric == 'pearson':
            if len(ratings1) < 2:
                return 0.0
            correlation = np.corrcoef(ratings1, ratings2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        elif self.similarity_metric == 'jaccard':
            intersection = ((item1_ratings > 0) & (item2_ratings > 0)).sum()
            union = ((item1_ratings > 0) | (item2_ratings > 0)).sum()
            return intersection / union if union > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """Fit the item-based model with optimizations."""
        start_time = time.time()
        
        print(f"Fitting optimized item-based CF with {len(user_item_matrix.columns)} items...")
        
        self.user_item_matrix = user_item_matrix.fillna(0)
        self.item_means = self.user_item_matrix.mean(axis=0)
        
        # Check for cached item similarities
        n_items = len(self.user_item_matrix.columns)
        cache_key = f"item_similarities_{self.similarity_metric}_{n_items}"
        cached_similarities = self.cache.get_item_similarities(cache_key, self.similarity_metric)
        
        if cached_similarities is not None:
            print("Loading item similarity matrix from cache...")
            self.item_similarity_matrix = pd.DataFrame(
                cached_similarities,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
            self.cache_hits += 1
        else:
            print("Computing item similarity matrix...")
            
            items = list(self.user_item_matrix.columns)
            n_items = len(items)
            
            # Initialize similarity matrix
            similarity_matrix = np.eye(n_items)
            
            if self.use_parallel and n_items > 50:
                # Parallel computation for item similarities
                tasks = [(items[i], items[j]) for i in range(n_items) 
                        for j in range(i+1, n_items)]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_pair = {
                        executor.submit(self._compute_item_similarity, item1, item2): (i, j)
                        for i, (item1, item2) in enumerate(tasks)
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_pair):
                        try:
                            similarity = future.result()
                            # Find indices in the original tasks list
                            task_idx = future_to_pair[future]
                            item_pair = tasks[task_idx[1]]  # Get the actual item pair
                            
                            # Find matrix indices
                            i = items.index(item_pair[0])
                            j = items.index(item_pair[1])
                            
                            similarity_matrix[i, j] = similarity
                            similarity_matrix[j, i] = similarity
                        except Exception as e:
                            print(f"Error in parallel similarity computation: {e}")
            else:
                # Sequential computation
                for i, item1 in enumerate(items):
                    for j, item2 in enumerate(items[i+1:], i+1):
                        similarity = self._compute_item_similarity(item1, item2)
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
            
            self.item_similarity_matrix = pd.DataFrame(
                similarity_matrix,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
            
            # Cache the result
            self.cache.put_item_similarities(cache_key, self.similarity_metric, similarity_matrix.tolist())
            self.cache_misses += 1
        
        self.fitted = True
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        print(f"Item-based model fitting completed in {computation_time:.2f} seconds")
    
    @memoize_with_timeout(timeout=1800, cache_size=5000)
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating with optimized computation."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self.item_means[item_id] if item_id in self.item_means.index else 3.0
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()
        
        # Check existing rating
        existing_rating = self.user_item_matrix.loc[user_id, item_id]
        if existing_rating > 0:
            return existing_rating
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            return self.item_means[item_id]
        
        # Get similarities with target item
        item_similarities = self.item_similarity_matrix[item_id]
        similar_items = item_similarities[rated_items.index]
        
        # Filter positive similarities
        positive_similarities = similar_items[similar_items > 0]
        
        if len(positive_similarities) == 0:
            return self.item_means[item_id]
        
        # Take top-k similar items
        top_similar = positive_similarities.nlargest(self.k_neighbors)
        
        # Weighted prediction
        numerator = sum(sim * rated_items[item] for item, sim in top_similar.items())
        denominator = sum(abs(sim) for sim in top_similar.values())
        
        if denominator == 0:
            return self.item_means[item_id]
        
        prediction = numerator / denominator
        
        # Clamp to valid range
        return max(1.0, min(5.0, prediction))
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generate optimized item-based recommendations."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Check cache
        cached_recs = self.cache.get_recommendations(user_id, "item_based_cf", n_recommendations)
        if cached_recs is not None:
            self.cache_hits += 1
            return cached_recs
        
        start_time = time.time()
        
        # Get candidate items
        if exclude_seen and user_id in self.user_item_matrix.index:
            seen_items = set(self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ])
            candidate_items = [item for item in self.user_item_matrix.columns 
                             if item not in seen_items]
        else:
            candidate_items = list(self.user_item_matrix.columns)
        
        # Predict ratings
        predictions = []
        for item_id in candidate_items:
            try:
                predicted_rating = self.predict_rating(user_id, item_id)
                predictions.append((item_id, predicted_rating))
            except Exception as e:
                print(f"Error predicting rating for {item_id}: {e}")
        
        # Sort and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations = predictions[:n_recommendations]
        
        # Cache results
        self.cache.put_recommendations(user_id, "item_based_cf", n_recommendations, recommendations)
        self.cache_misses += 1
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Union[int, float, List]]:
        """Get performance statistics."""
        cache_total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0.0,
            'total_computations': len(self.computation_times),
            'cache_stats': self.cache.get_cache_stats()
        }