"""
Collaborative Filtering Algorithms

This module implements user-based and item-based collaborative filtering algorithms
using scikit-learn and custom implementations for recommendation systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class UserBasedCollaborativeFiltering:
    """
    User-based collaborative filtering using neighborhood methods.
    
    Recommends items by finding similar users and suggesting items
    they liked that the target user hasn't seen yet.
    """
    
    def __init__(self, k_neighbors: int = 20, similarity_metric: str = 'cosine',
                 min_common_items: int = 2):
        """
        Initialize user-based collaborative filtering.
        
        Args:
            k_neighbors: Number of similar users to consider
            similarity_metric: 'cosine', 'pearson', or 'jaccard'
            min_common_items: Minimum number of common items for similarity calculation
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_means = None
        self.fitted = False
    
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Fit the model with user-item interaction matrix.
        
        Args:
            user_item_matrix: DataFrame with users as rows and items as columns
        """
        self.user_item_matrix = user_item_matrix.fillna(0)
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Compute user similarity matrix
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif self.similarity_metric == 'pearson':
            # Center the data for Pearson correlation
            centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0)
            self.user_similarity_matrix = cosine_similarity(centered_matrix)
        elif self.similarity_metric == 'jaccard':
            # Binary matrix for Jaccard similarity
            binary_matrix = (self.user_item_matrix > 0).astype(int)
            self.user_similarity_matrix = 1 - pairwise_distances(
                binary_matrix, metric='jaccard'
            )
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Convert to DataFrame for easier indexing
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.fitted = True
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix[item_id].mean() if item_id in self.user_item_matrix.columns else 0.0
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_means[user_id]
        
        # Get similar users who have rated this item
        user_similarities = self.user_similarity_matrix[user_id].copy()
        
        # Filter users who have rated the item
        rated_users = self.user_item_matrix[
            self.user_item_matrix[item_id] > 0
        ][item_id].index
        
        similar_users = user_similarities[rated_users].sort_values(ascending=False)
        
        # Take top-k similar users
        top_similar = similar_users.head(self.k_neighbors)
        
        if len(top_similar) == 0:
            return self.user_means[user_id]
        
        # Weighted average prediction
        numerator = 0
        denominator = 0
        
        for similar_user, similarity in top_similar.items():
            if similarity > 0:
                if self.similarity_metric == 'pearson':
                    # Mean-centered prediction
                    user_rating = self.user_item_matrix.loc[similar_user, item_id]
                    centered_rating = user_rating - self.user_means[similar_user]
                    numerator += similarity * centered_rating
                else:
                    numerator += similarity * self.user_item_matrix.loc[similar_user, item_id]
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means[user_id]
        
        if self.similarity_metric == 'pearson':
            prediction = self.user_means[user_id] + (numerator / denominator)
        else:
            prediction = numerator / denominator
        
        # Clamp to valid rating range (assuming 1-5 scale)
        return max(1.0, min(5.0, prediction))
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude items user has already rated
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            # For new users, recommend popular items
            item_popularity = self.user_item_matrix.mean(axis=0)
            top_items = item_popularity.sort_values(ascending=False).head(n_recommendations)
            return [(item, rating) for item, rating in top_items.items()]
        
        # Get items to predict
        if exclude_seen:
            seen_items = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
            candidate_items = [item for item in self.user_item_matrix.columns 
                             if item not in seen_items]
        else:
            candidate_items = list(self.user_item_matrix.columns)
        
        # Predict ratings for all candidate items
        predictions = []
        for item in candidate_items:
            predicted_rating = self.predict_rating(user_id, item)
            predictions.append((item, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class ItemBasedCollaborativeFiltering:
    """
    Item-based collaborative filtering using item similarity.
    
    Recommends items similar to those the user has already liked.
    """
    
    def __init__(self, k_neighbors: int = 20, similarity_metric: str = 'cosine'):
        """
        Initialize item-based collaborative filtering.
        
        Args:
            k_neighbors: Number of similar items to consider
            similarity_metric: 'cosine', 'pearson', or 'jaccard'
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.item_means = None
        self.fitted = False
    
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Fit the model with user-item interaction matrix.
        
        Args:
            user_item_matrix: DataFrame with users as rows and items as columns
        """
        self.user_item_matrix = user_item_matrix.fillna(0)
        self.item_means = self.user_item_matrix.mean(axis=0)
        
        # Transpose for item-item similarity
        item_user_matrix = self.user_item_matrix.T
        
        # Compute item similarity matrix
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        elif self.similarity_metric == 'pearson':
            # Center the data for Pearson correlation
            centered_matrix = item_user_matrix.sub(self.item_means, axis=0)
            self.item_similarity_matrix = cosine_similarity(centered_matrix)
        elif self.similarity_metric == 'jaccard':
            # Binary matrix for Jaccard similarity
            binary_matrix = (item_user_matrix > 0).astype(int)
            self.item_similarity_matrix = 1 - pairwise_distances(
                binary_matrix, metric='jaccard'
            )
        
        # Convert to DataFrame for easier indexing
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )
        
        self.fitted = True
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self.item_means[item_id] if item_id in self.item_means else 0.0
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()
        
        # Get items rated by this user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            return self.item_means[item_id]
        
        # Get similarities to target item
        item_similarities = self.item_similarity_matrix[item_id][rated_items.index]
        
        # Take top-k similar items
        top_similar = item_similarities.sort_values(ascending=False).head(self.k_neighbors)
        
        if len(top_similar) == 0:
            return self.item_means[item_id]
        
        # Weighted average prediction
        numerator = 0
        denominator = 0
        
        for similar_item, similarity in top_similar.items():
            if similarity > 0:
                numerator += similarity * rated_items[similar_item]
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.item_means[item_id]
        
        prediction = numerator / denominator
        
        # Clamp to valid rating range
        return max(1.0, min(5.0, prediction))
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude items user has already rated
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            # For new users, recommend popular items
            item_popularity = self.user_item_matrix.mean(axis=0)
            top_items = item_popularity.sort_values(ascending=False).head(n_recommendations)
            return [(item, rating) for item, rating in top_items.items()]
        
        # Get items to predict
        if exclude_seen:
            seen_items = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
            candidate_items = [item for item in self.user_item_matrix.columns 
                             if item not in seen_items]
        else:
            candidate_items = list(self.user_item_matrix.columns)
        
        # Predict ratings for all candidate items
        predictions = []
        for item in candidate_items:
            predicted_rating = self.predict_rating(user_id, item)
            predictions.append((item, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class MatrixFactorizationCF:
    """
    Matrix Factorization-based Collaborative Filtering using Non-negative Matrix Factorization (NMF)
    and Singular Value Decomposition (SVD).
    """
    
    def __init__(self, n_components: int = 50, method: str = 'nmf',
                 max_iter: int = 200, random_state: int = 42):
        """
        Initialize matrix factorization model.
        
        Args:
            n_components: Number of latent factors
            method: 'nmf' or 'svd'
            max_iter: Maximum iterations for optimization
            random_state: Random seed
        """
        self.n_components = n_components
        self.method = method
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.fitted = False
    
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Fit the matrix factorization model.
        
        Args:
            user_item_matrix: DataFrame with users as rows and items as columns
        """
        self.user_item_matrix = user_item_matrix.fillna(0)
        
        if self.method == 'nmf':
            self.model = NMF(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=self.random_state,
                init='random'
            )
            self.user_factors = self.model.fit_transform(self.user_item_matrix)
            self.item_factors = self.model.components_
        
        elif self.method == 'svd':
            self.model = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state
            )
            self.user_factors = self.model.fit_transform(self.user_item_matrix)
            self.item_factors = self.model.components_
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.fitted = True
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix[item_id].mean() if item_id in self.user_item_matrix.columns else 0.0
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Reconstruct rating from factors
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[:, item_idx])
        
        # For NMF, predictions are non-negative, so we scale to rating range
        if self.method == 'nmf':
            # Scale to 1-5 range
            max_prediction = np.max(np.dot(self.user_factors, self.item_factors))
            if max_prediction > 0:
                prediction = 1 + (prediction / max_prediction) * 4
        
        # Clamp to valid rating range
        return max(1.0, min(5.0, prediction))
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude items user has already rated
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            # For new users, recommend popular items
            item_popularity = self.user_item_matrix.mean(axis=0)
            top_items = item_popularity.sort_values(ascending=False).head(n_recommendations)
            return [(item, rating) for item, rating in top_items.items()]
        
        # Get items to predict
        if exclude_seen:
            seen_items = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
            candidate_items = [item for item in self.user_item_matrix.columns 
                             if item not in seen_items]
        else:
            candidate_items = list(self.user_item_matrix.columns)
        
        # Predict ratings for all candidate items
        predictions = []
        for item in candidate_items:
            predicted_rating = self.predict_rating(user_id, item)
            predictions.append((item, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_user_factors(self, user_id: str) -> np.ndarray:
        """Get latent factors for a user."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return np.zeros(self.n_components)
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        return self.user_factors[user_idx]
    
    def get_item_factors(self, item_id: str) -> np.ndarray:
        """Get latent factors for an item."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if item_id not in self.user_item_matrix.columns:
            return np.zeros(self.n_components)
        
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        return self.item_factors[:, item_idx]


class EnsembleCollaborativeFiltering:
    """
    Ensemble of multiple collaborative filtering methods for improved performance.
    """
    
    def __init__(self, methods: List[str] = None, weights: List[float] = None):
        """
        Initialize ensemble model.
        
        Args:
            methods: List of methods to ensemble ('user_based', 'item_based', 'matrix_factorization')
            weights: Weights for combining predictions
        """
        if methods is None:
            methods = ['user_based', 'item_based', 'matrix_factorization']
        
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        
        self.methods = methods
        self.weights = weights
        self.models = {}
        self.fitted = False
    
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        """
        Fit all ensemble models.
        
        Args:
            user_item_matrix: DataFrame with users as rows and items as columns
        """
        for method in self.methods:
            if method == 'user_based':
                model = UserBasedCollaborativeFiltering()
            elif method == 'item_based':
                model = ItemBasedCollaborativeFiltering()
            elif method == 'matrix_factorization':
                model = MatrixFactorizationCF()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            model.fit(user_item_matrix)
            self.models[method] = model
        
        self.fitted = True
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating using ensemble of models.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Ensemble predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        for method, weight in zip(self.methods, self.weights):
            prediction = self.models[method].predict_rating(user_id, item_id)
            predictions.append(weight * prediction)
        
        return sum(predictions) / sum(self.weights)
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate ensemble recommendations for a user.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude items user has already rated
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get recommendations from all models
        all_recommendations = {}
        
        for method, weight in zip(self.methods, self.weights):
            model_recommendations = self.models[method].recommend(
                user_id, n_recommendations * 2, exclude_seen
            )
            
            for item_id, rating in model_recommendations:
                if item_id not in all_recommendations:
                    all_recommendations[item_id] = 0
                all_recommendations[item_id] += weight * rating
        
        # Normalize by weights
        for item_id in all_recommendations:
            all_recommendations[item_id] /= sum(self.weights)
        
        # Sort and return top N
        recommendations = [(item, rating) for item, rating in all_recommendations.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]