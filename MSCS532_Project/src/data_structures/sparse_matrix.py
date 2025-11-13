"""
Sparse User-Item Matrix Implementation

This module implements a sparse matrix representation for user-item interactions,
which is at the core of collaborative filtering algorithms.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SparseUserItemMatrix:
    """
    A sparse matrix representation of user-item interactions optimized for memory efficiency
    and fast computation in collaborative filtering algorithms.
    """
    
    def __init__(self, interaction_df: pd.DataFrame):
        """
        Initialize sparse matrix from interaction DataFrame.
        
        Args:
            interaction_df: DataFrame with columns ['user_id', 'product_id', 'rating']
        """
        self.interaction_df = interaction_df.copy()
        self.user_to_index = {}
        self.index_to_user = {}
        self.item_to_index = {}
        self.index_to_item = {}
        self.matrix = None
        self.dense_matrix = None
        
        self._build_mappings()
        self._build_sparse_matrix()
    
    def _build_mappings(self) -> None:
        """Build user and item ID to index mappings."""
        unique_users = self.interaction_df['user_id'].unique()
        unique_items = self.interaction_df['product_id'].unique()
        
        # User mappings
        for idx, user_id in enumerate(unique_users):
            self.user_to_index[user_id] = idx
            self.index_to_user[idx] = user_id
        
        # Item mappings
        for idx, item_id in enumerate(unique_items):
            self.item_to_index[item_id] = idx
            self.index_to_item[idx] = item_id
    
    def _build_sparse_matrix(self) -> None:
        """Build the sparse CSR matrix from interaction data."""
        # Filter for rating interactions
        rating_data = self.interaction_df[
            (self.interaction_df['interaction_type'] == 'rating') &
            (self.interaction_df['rating'].notna())
        ].copy()
        
        if rating_data.empty:
            # If no rating data, create binary interaction matrix
            rating_data = self.interaction_df.copy()
            rating_data['rating'] = 1.0
        
        # Convert IDs to indices
        rating_data['user_idx'] = rating_data['user_id'].map(self.user_to_index)
        rating_data['item_idx'] = rating_data['product_id'].map(self.item_to_index)
        
        # Create coordinate format matrix
        row_indices = rating_data['user_idx'].values
        col_indices = rating_data['item_idx'].values
        data = rating_data['rating'].values
        
        # Build sparse matrix
        n_users = len(self.user_to_index)
        n_items = len(self.item_to_index)
        
        self.matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_users, n_items)
        )
        
        # Handle duplicate entries by taking the mean
        self.matrix.sum_duplicates()
    
    def get_user_vector(self, user_id: str) -> np.ndarray:
        """Get the rating vector for a specific user."""
        if user_id not in self.user_to_index:
            raise ValueError(f"User {user_id} not found in matrix")
        
        user_idx = self.user_to_index[user_id]
        return self.matrix[user_idx].toarray().flatten()
    
    def get_item_vector(self, item_id: str) -> np.ndarray:
        """Get the rating vector for a specific item."""
        if item_id not in self.item_to_index:
            raise ValueError(f"Item {item_id} not found in matrix")
        
        item_idx = self.item_to_index[item_id]
        return self.matrix[:, item_idx].toarray().flatten()
    
    def get_user_item_rating(self, user_id: str, item_id: str) -> float:
        """Get the rating for a specific user-item pair."""
        if user_id not in self.user_to_index or item_id not in self.item_to_index:
            return 0.0
        
        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]
        return self.matrix[user_idx, item_idx]
    
    def get_dense_matrix(self) -> np.ndarray:
        """Convert sparse matrix to dense format."""
        if self.dense_matrix is None:
            self.dense_matrix = self.matrix.toarray()
        return self.dense_matrix
    
    def get_nonzero_users_for_item(self, item_id: str) -> List[str]:
        """Get users who have rated/interacted with a specific item."""
        if item_id not in self.item_to_index:
            return []
        
        item_idx = self.item_to_index[item_id]
        user_indices = self.matrix[:, item_idx].nonzero()[0]
        return [self.index_to_user[idx] for idx in user_indices]
    
    def get_nonzero_items_for_user(self, user_id: str) -> List[str]:
        """Get items that a user has rated/interacted with."""
        if user_id not in self.user_to_index:
            return []
        
        user_idx = self.user_to_index[user_id]
        item_indices = self.matrix[user_idx].nonzero()[1]
        return [self.index_to_item[idx] for idx in item_indices]
    
    def compute_user_similarity(self, user1_id: str, user2_id: str, 
                               method: str = 'cosine') -> float:
        """
        Compute similarity between two users.
        
        Args:
            user1_id: First user ID
            user2_id: Second user ID
            method: 'cosine', 'pearson', or 'jaccard'
        """
        if user1_id not in self.user_to_index or user2_id not in self.user_to_index:
            return 0.0
        
        user1_vector = self.get_user_vector(user1_id)
        user2_vector = self.get_user_vector(user2_id)
        
        if method == 'cosine':
            return self._cosine_similarity(user1_vector, user2_vector)
        elif method == 'pearson':
            return self._pearson_correlation(user1_vector, user2_vector)
        elif method == 'jaccard':
            return self._jaccard_similarity(user1_vector, user2_vector)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def compute_item_similarity(self, item1_id: str, item2_id: str,
                               method: str = 'cosine') -> float:
        """
        Compute similarity between two items.
        
        Args:
            item1_id: First item ID
            item2_id: Second item ID
            method: 'cosine', 'pearson', or 'jaccard'
        """
        if item1_id not in self.item_to_index or item2_id not in self.item_to_index:
            return 0.0
        
        item1_vector = self.get_item_vector(item1_id)
        item2_vector = self.get_item_vector(item2_id)
        
        if method == 'cosine':
            return self._cosine_similarity(item1_vector, item2_vector)
        elif method == 'pearson':
            return self._pearson_correlation(item1_vector, item2_vector)
        elif method == 'jaccard':
            return self._jaccard_similarity(item1_vector, item2_vector)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Handle zero vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vector1, vector2) / (norm1 * norm2)
    
    def _pearson_correlation(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute Pearson correlation coefficient between two vectors."""
        # Only consider items rated by both users
        mask = (vector1 != 0) & (vector2 != 0)
        
        if np.sum(mask) < 2:  # Need at least 2 common items
            return 0.0
        
        v1_common = vector1[mask]
        v2_common = vector2[mask]
        
        return np.corrcoef(v1_common, v2_common)[0, 1] if len(v1_common) > 1 else 0.0
    
    def _jaccard_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute Jaccard similarity between two binary vectors."""
        # Convert to binary
        binary1 = (vector1 != 0).astype(int)
        binary2 = (vector2 != 0).astype(int)
        
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_top_similar_users(self, user_id: str, top_k: int = 10,
                             method: str = 'cosine') -> List[Tuple[str, float]]:
        """
        Get top-k most similar users to a given user.
        
        Args:
            user_id: Target user ID
            top_k: Number of similar users to return
            method: Similarity method
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_to_index:
            return []
        
        similarities = []
        for other_user_id in self.user_to_index.keys():
            if other_user_id != user_id:
                sim = self.compute_user_similarity(user_id, other_user_id, method)
                similarities.append((other_user_id, sim))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_top_similar_items(self, item_id: str, top_k: int = 10,
                             method: str = 'cosine') -> List[Tuple[str, float]]:
        """
        Get top-k most similar items to a given item.
        
        Args:
            item_id: Target item ID
            top_k: Number of similar items to return
            method: Similarity method
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_to_index:
            return []
        
        similarities = []
        for other_item_id in self.item_to_index.keys():
            if other_item_id != item_id:
                sim = self.compute_item_similarity(item_id, other_item_id, method)
                similarities.append((other_item_id, sim))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_matrix_stats(self) -> Dict:
        """Get statistics about the sparse matrix."""
        return {
            'shape': self.matrix.shape,
            'nnz': self.matrix.nnz,
            'sparsity': 1.0 - (self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1])),
            'density': self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]),
            'avg_rating': self.matrix.data.mean() if self.matrix.nnz > 0 else 0,
            'rating_std': self.matrix.data.std() if self.matrix.nnz > 0 else 0,
            'num_users': self.matrix.shape[0],
            'num_items': self.matrix.shape[1]
        }
    
    def add_interaction(self, user_id: str, item_id: str, rating: float) -> None:
        """
        Add a new user-item interaction to the matrix.
        Note: This rebuilds the matrix, so use sparingly for large datasets.
        """
        # Add new interaction to DataFrame
        new_interaction = pd.DataFrame({
            'user_id': [user_id],
            'product_id': [item_id],
            'interaction_type': ['rating'],
            'rating': [rating]
        })
        self.interaction_df = pd.concat([self.interaction_df, new_interaction], ignore_index=True)
        
        # Check if we need to add new user/item mappings
        rebuild_needed = False
        
        if user_id not in self.user_to_index:
            new_user_idx = len(self.user_to_index)
            self.user_to_index[user_id] = new_user_idx
            self.index_to_user[new_user_idx] = user_id
            rebuild_needed = True
        
        if item_id not in self.item_to_index:
            new_item_idx = len(self.item_to_index)
            self.item_to_index[item_id] = new_item_idx
            self.index_to_item[new_item_idx] = item_id
            rebuild_needed = True
        
        if rebuild_needed:
            self._build_sparse_matrix()
        else:
            # Update existing matrix
            user_idx = self.user_to_index[user_id]
            item_idx = self.item_to_index[item_id]
            self.matrix[user_idx, item_idx] = rating
        
        # Clear cached dense matrix
        self.dense_matrix = None