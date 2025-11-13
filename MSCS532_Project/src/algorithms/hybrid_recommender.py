"""
Hybrid Recommendation System

This module implements a hybrid approach that combines collaborative filtering,
content-based filtering, and graph-based methods to provide improved recommendations
with better coverage and accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules (relative imports would be used in actual deployment)
try:
    from ..algorithms.collaborative_filtering import (
        UserBasedCollaborativeFiltering, 
        ItemBasedCollaborativeFiltering,
        MatrixFactorizationCF,
        EnsembleCollaborativeFiltering
    )
    from ..algorithms.content_based import (
        ContentBasedRecommender,
        PreferenceBasedRecommender,
        AdvancedContentBasedRecommender
    )
    from ..data_structures.graph_structure import BipartiteRecommendationGraph
    from ..data_structures.sparse_matrix import SparseUserItemMatrix
except ImportError:
    # For standalone testing, we'll define placeholder classes
    class UserBasedCollaborativeFiltering:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): pass
        def recommend(self, *args, **kwargs): return []
    
    class ItemBasedCollaborativeFiltering:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): pass
        def recommend(self, *args, **kwargs): return []
    
    class MatrixFactorizationCF:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): pass
        def recommend(self, *args, **kwargs): return []
    
    class ContentBasedRecommender:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): pass
        def recommend(self, *args, **kwargs): return []
    
    class BipartiteRecommendationGraph:
        def __init__(self, *args, **kwargs): pass
        def random_walk_recommendation(self, *args, **kwargs): return {}
        def personalized_pagerank_recommendation(self, *args, **kwargs): return {}


class WeightedHybridRecommender:
    """
    Weighted hybrid recommender that combines multiple recommendation approaches
    with configurable weights.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize weighted hybrid recommender.
        
        Args:
            weights: Dictionary mapping method names to weights
                   {'collaborative': 0.5, 'content': 0.3, 'graph': 0.2}
        """
        if weights is None:
            weights = {
                'collaborative': 0.4,
                'content': 0.4,
                'graph': 0.2
            }
        
        self.weights = weights
        self.collaborative_model = None
        self.content_model = None
        self.graph_model = None
        self.fitted = False
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame,
            users_df: pd.DataFrame = None) -> None:
        """
        Fit all component models.
        
        Args:
            products_df: Product data
            interactions_df: User-item interactions
            users_df: User data (optional)
        """
        print("Fitting hybrid recommendation models...")
        
        # Prepare user-item matrix for collaborative filtering
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_id', 
            values='rating',
            fill_value=0
        )
        
        # Fit collaborative filtering model
        if 'collaborative' in self.weights and self.weights['collaborative'] > 0:
            print("Training collaborative filtering model...")
            self.collaborative_model = UserBasedCollaborativeFiltering(k_neighbors=10)
            self.collaborative_model.fit(user_item_matrix)
        
        # Fit content-based model
        if 'content' in self.weights and self.weights['content'] > 0:
            print("Training content-based model...")
            self.content_model = ContentBasedRecommender()
            
            # Define feature columns (adjust based on your data)
            text_columns = ['name'] if 'name' in products_df.columns else []
            categorical_columns = ['category', 'brand'] if all(col in products_df.columns for col in ['category', 'brand']) else []
            numerical_columns = ['price', 'average_rating'] if all(col in products_df.columns for col in ['price', 'average_rating']) else []
            
            self.content_model.fit(
                products_df, interactions_df,
                text_columns=text_columns,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns
            )
        
        # Fit graph-based model
        if 'graph' in self.weights and self.weights['graph'] > 0:
            print("Training graph-based model...")
            self.graph_model = BipartiteRecommendationGraph.from_interaction_dataframe(interactions_df)
        
        self.fitted = True
        print("Hybrid model training completed!")
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating using weighted combination of methods.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        weights_sum = 0
        
        # Collaborative filtering prediction
        if (self.collaborative_model and 'collaborative' in self.weights and 
            self.weights['collaborative'] > 0):
            try:
                cf_pred = self.collaborative_model.predict_rating(user_id, item_id)
                predictions.append(cf_pred * self.weights['collaborative'])
                weights_sum += self.weights['collaborative']
            except:
                pass  # Handle cases where prediction fails
        
        # Content-based prediction
        if (self.content_model and 'content' in self.weights and 
            self.weights['content'] > 0):
            try:
                content_pred = self.content_model.predict_rating(user_id, item_id)
                predictions.append(content_pred * self.weights['content'])
                weights_sum += self.weights['content']
            except:
                pass
        
        # Graph-based prediction (simplified - could use graph similarity)
        if (self.graph_model and 'graph' in self.weights and 
            self.weights['graph'] > 0):
            try:
                # Use average rating as graph-based prediction
                graph_pred = 3.0  # Placeholder
                predictions.append(graph_pred * self.weights['graph'])
                weights_sum += self.weights['graph']
            except:
                pass
        
        # Return weighted average
        if predictions and weights_sum > 0:
            return sum(predictions) / weights_sum
        else:
            return 3.0  # Default rating
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude seen items
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        all_recommendations = {}
        
        # Get recommendations from each model
        models_and_weights = [
            (self.collaborative_model, 'collaborative'),
            (self.content_model, 'content')
        ]
        
        for model, method_name in models_and_weights:
            if model and method_name in self.weights and self.weights[method_name] > 0:
                try:
                    method_recs = model.recommend(
                        user_id, 
                        n_recommendations * 2,  # Get more to have options
                        exclude_seen
                    )
                    
                    weight = self.weights[method_name]
                    for item_id, score in method_recs:
                        if item_id not in all_recommendations:
                            all_recommendations[item_id] = 0
                        all_recommendations[item_id] += weight * score
                except Exception as e:
                    print(f"Error getting recommendations from {method_name}: {e}")
                    continue
        
        # Get graph-based recommendations
        if (self.graph_model and 'graph' in self.weights and 
            self.weights['graph'] > 0):
            try:
                graph_recs = self.graph_model.random_walk_recommendation(
                    user_id, num_walks=50
                )
                weight = self.weights['graph']
                
                for item_id, score in graph_recs.items():
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = 0
                    all_recommendations[item_id] += weight * score
            except Exception as e:
                print(f"Error getting graph recommendations: {e}")
        
        # Normalize scores
        if all_recommendations:
            max_score = max(all_recommendations.values())
            if max_score > 0:
                for item_id in all_recommendations:
                    all_recommendations[item_id] = (all_recommendations[item_id] / max_score) * 5.0
        
        # Sort and return top recommendations
        recommendations = [(item, score) for item, score in all_recommendations.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]


class SwitchingHybridRecommender:
    """
    Switching hybrid recommender that selects the best method based on
    user/item characteristics or confidence scores.
    """
    
    def __init__(self):
        self.collaborative_model = None
        self.content_model = None
        self.graph_model = None
        self.user_item_counts = {}
        self.item_user_counts = {}
        self.fitted = False
    
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """
        Fit all models and compute switching criteria.
        
        Args:
            products_df: Product data
            interactions_df: User-item interactions
        """
        print("Fitting switching hybrid models...")
        
        # Compute interaction counts for switching logic
        self.user_item_counts = interactions_df.groupby('user_id')['product_id'].count().to_dict()
        self.item_user_counts = interactions_df.groupby('product_id')['user_id'].count().to_dict()
        
        # Prepare user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_id', 
            values='rating',
            fill_value=0
        )
        
        # Fit models
        self.collaborative_model = UserBasedCollaborativeFiltering()
        self.collaborative_model.fit(user_item_matrix)
        
        self.content_model = ContentBasedRecommender()
        text_columns = ['name'] if 'name' in products_df.columns else []
        categorical_columns = ['category', 'brand'] if all(col in products_df.columns for col in ['category', 'brand']) else []
        numerical_columns = ['price', 'average_rating'] if all(col in products_df.columns for col in ['price', 'average_rating']) else []
        
        self.content_model.fit(
            products_df, interactions_df,
            text_columns=text_columns,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns
        )
        
        self.graph_model = BipartiteRecommendationGraph.from_interaction_dataframe(interactions_df)
        
        self.fitted = True
        print("Switching hybrid model training completed!")
    
    def _select_method(self, user_id: str, item_id: str = None) -> str:
        """
        Select the best method based on user/item characteristics.
        
        Args:
            user_id: Target user
            item_id: Target item (optional)
            
        Returns:
            Selected method name
        """
        user_interactions = self.user_item_counts.get(user_id, 0)
        
        # Cold start problem: use content-based for new users
        if user_interactions < 5:
            return 'content'
        
        # For users with many interactions, use collaborative filtering
        if user_interactions >= 20:
            return 'collaborative'
        
        # For medium activity users, use graph-based
        return 'graph'
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations using method switching.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude seen items
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Select best method
        method = self._select_method(user_id)
        
        print(f"Using {method} method for user {user_id}")
        
        try:
            if method == 'collaborative':
                return self.collaborative_model.recommend(user_id, n_recommendations, exclude_seen)
            elif method == 'content':
                return self.content_model.recommend(user_id, n_recommendations, exclude_seen)
            elif method == 'graph':
                graph_recs = self.graph_model.random_walk_recommendation(user_id)
                # Convert to list format and limit
                recommendations = [(item, score) for item, score in graph_recs.items()]
                recommendations.sort(key=lambda x: x[1], reverse=True)
                return recommendations[:n_recommendations]
        except Exception as e:
            print(f"Error with {method} method, falling back to content-based: {e}")
            # Fallback to content-based
            try:
                return self.content_model.recommend(user_id, n_recommendations, exclude_seen)
            except:
                # Final fallback: return empty list
                return []
        
        return []


class MixedHybridRecommender:
    """
    Mixed hybrid recommender that presents recommendations from multiple
    methods simultaneously.
    """
    
    def __init__(self, method_ratios: Dict[str, float] = None):
        """
        Initialize mixed hybrid recommender.
        
        Args:
            method_ratios: Ratio of recommendations from each method
        """
        if method_ratios is None:
            method_ratios = {
                'collaborative': 0.5,
                'content': 0.3,
                'graph': 0.2
            }
        
        self.method_ratios = method_ratios
        self.collaborative_model = None
        self.content_model = None
        self.graph_model = None
        self.fitted = False
    
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """
        Fit all models.
        
        Args:
            products_df: Product data
            interactions_df: User-item interactions
        """
        print("Fitting mixed hybrid models...")
        
        # Prepare user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_id', 
            values='rating',
            fill_value=0
        )
        
        # Fit models
        self.collaborative_model = UserBasedCollaborativeFiltering()
        self.collaborative_model.fit(user_item_matrix)
        
        self.content_model = ContentBasedRecommender()
        text_columns = ['name'] if 'name' in products_df.columns else []
        categorical_columns = ['category', 'brand'] if all(col in products_df.columns for col in ['category', 'brand']) else []
        numerical_columns = ['price', 'average_rating'] if all(col in products_df.columns for col in ['price', 'average_rating']) else []
        
        self.content_model.fit(
            products_df, interactions_df,
            text_columns=text_columns,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns
        )
        
        self.graph_model = BipartiteRecommendationGraph.from_interaction_dataframe(interactions_df)
        
        self.fitted = True
        print("Mixed hybrid model training completed!")
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> List[Tuple[str, float, str]]:
        """
        Generate mixed recommendations with method attribution.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude seen items
            
        Returns:
            List of (item_id, score, method) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        mixed_recommendations = []
        
        # Calculate number of recommendations from each method
        for method, ratio in self.method_ratios.items():
            n_method_recs = max(1, int(n_recommendations * ratio))
            
            try:
                if method == 'collaborative' and self.collaborative_model:
                    recs = self.collaborative_model.recommend(user_id, n_method_recs, exclude_seen)
                    for item_id, score in recs:
                        mixed_recommendations.append((item_id, score, 'collaborative'))
                
                elif method == 'content' and self.content_model:
                    recs = self.content_model.recommend(user_id, n_method_recs, exclude_seen)
                    for item_id, score in recs:
                        mixed_recommendations.append((item_id, score, 'content'))
                
                elif method == 'graph' and self.graph_model:
                    graph_recs = self.graph_model.random_walk_recommendation(user_id)
                    # Convert and limit
                    recs = [(item, score) for item, score in graph_recs.items()]
                    recs.sort(key=lambda x: x[1], reverse=True)
                    for item_id, score in recs[:n_method_recs]:
                        mixed_recommendations.append((item_id, score, 'graph'))
                        
            except Exception as e:
                print(f"Error getting recommendations from {method}: {e}")
                continue
        
        # Remove duplicates, keeping the first occurrence
        seen_items = set()
        unique_recommendations = []
        
        for item_id, score, method in mixed_recommendations:
            if item_id not in seen_items:
                seen_items.add(item_id)
                unique_recommendations.append((item_id, score, method))
        
        # Sort by score and return top N
        unique_recommendations.sort(key=lambda x: x[1], reverse=True)
        return unique_recommendations[:n_recommendations]


class AdaptiveHybridRecommender:
    """
    Adaptive hybrid recommender that learns optimal combination weights
    based on user feedback and performance.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize adaptive hybrid recommender.
        
        Args:
            learning_rate: Learning rate for weight adaptation
        """
        self.learning_rate = learning_rate
        self.weights = {
            'collaborative': 0.33,
            'content': 0.33,
            'graph': 0.34
        }
        self.performance_history = {method: [] for method in self.weights}
        self.collaborative_model = None
        self.content_model = None
        self.graph_model = None
        self.fitted = False
    
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """
        Fit all models.
        
        Args:
            products_df: Product data
            interactions_df: User-item interactions
        """
        # Implementation similar to other hybrid models
        # ... (fit logic would be here)
        self.fitted = True
    
    def update_weights(self, user_id: str, recommended_items: List[str],
                      actual_interactions: List[str]) -> None:
        """
        Update method weights based on user feedback.
        
        Args:
            user_id: User who received recommendations
            recommended_items: Items that were recommended
            actual_interactions: Items user actually interacted with
        """
        # Calculate hit rate for each method (simplified)
        # In practice, you'd track which method contributed each recommendation
        
        hit_rate = len(set(recommended_items) & set(actual_interactions)) / len(recommended_items)
        
        # Update weights based on performance (simplified adaptation logic)
        for method in self.weights:
            self.performance_history[method].append(hit_rate)
            
            # Keep only recent performance history
            if len(self.performance_history[method]) > 100:
                self.performance_history[method] = self.performance_history[method][-100:]
            
            # Adapt weights based on recent performance
            recent_performance = np.mean(self.performance_history[method][-10:])
            self.weights[method] += self.learning_rate * (recent_performance - 0.5)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: max(0.1, v/total_weight) for k, v in self.weights.items()}
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Generate adaptive recommendations.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        # Use current weights for recommendation (similar to weighted hybrid)
        # Implementation would combine methods using current adaptive weights
        return []