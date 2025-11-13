"""
Content-Based Filtering Algorithms

This module implements content-based recommendation algorithms that suggest items
based on product features and user preferences, utilizing machine learning models
from scikit-learn for feature analysis and similarity computation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extract and process features from product data for content-based filtering.
    """
    
    def __init__(self):
        self.tfidf_vectorizers = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def extract_features(self, products_df: pd.DataFrame, 
                        text_columns: List[str] = None,
                        categorical_columns: List[str] = None,
                        numerical_columns: List[str] = None) -> np.ndarray:
        """
        Extract and combine features from product DataFrame.
        
        Args:
            products_df: Product data
            text_columns: Columns containing text data
            categorical_columns: Columns with categorical data
            numerical_columns: Columns with numerical data
            
        Returns:
            Feature matrix
        """
        if text_columns is None:
            text_columns = []
        if categorical_columns is None:
            categorical_columns = []
        if numerical_columns is None:
            numerical_columns = []
        
        features_list = []
        self.feature_names = []
        
        # Process text features
        for col in text_columns:
            if col in products_df.columns:
                # Clean text data
                text_data = products_df[col].fillna('').astype(str)
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2)
                )
                
                tfidf_features = vectorizer.fit_transform(text_data).toarray()
                features_list.append(tfidf_features)
                
                # Store vectorizer for future use
                self.tfidf_vectorizers[col] = vectorizer
                
                # Add feature names
                feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])]
                self.feature_names.extend(feature_names)
        
        # Process categorical features
        for col in categorical_columns:
            if col in products_df.columns:
                # Handle missing values
                cat_data = products_df[col].fillna('Unknown').astype(str)
                
                # Label encoding
                encoder = LabelEncoder()
                encoded_features = encoder.fit_transform(cat_data).reshape(-1, 1)
                features_list.append(encoded_features)
                
                # Store encoder
                self.label_encoders[col] = encoder
                self.feature_names.append(f"{col}_encoded")
        
        # Process numerical features
        for col in numerical_columns:
            if col in products_df.columns:
                # Handle missing values
                num_data = products_df[col].fillna(products_df[col].median())
                
                # Standardize numerical features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(num_data.values.reshape(-1, 1))
                features_list.append(scaled_features)
                
                # Store scaler
                self.scalers[col] = scaler
                self.feature_names.append(f"{col}_scaled")
        
        # Combine all features
        if features_list:
            combined_features = np.hstack(features_list)
        else:
            # If no features specified, create dummy features
            combined_features = np.ones((len(products_df), 1))
            self.feature_names = ['dummy_feature']
        
        return combined_features
    
    def transform_new_product(self, product_data: Dict) -> np.ndarray:
        """
        Transform a new product's features using fitted transformers.
        
        Args:
            product_data: Dictionary containing product features
            
        Returns:
            Transformed feature vector
        """
        features_list = []
        
        # Transform text features
        for col, vectorizer in self.tfidf_vectorizers.items():
            text = product_data.get(col, '')
            if isinstance(text, str):
                tfidf_features = vectorizer.transform([text]).toarray()
                features_list.append(tfidf_features[0])
        
        # Transform categorical features
        for col, encoder in self.label_encoders.items():
            cat_value = product_data.get(col, 'Unknown')
            try:
                encoded_value = encoder.transform([str(cat_value)])[0]
            except ValueError:
                # Handle unseen categories
                encoded_value = 0
            features_list.append([encoded_value])
        
        # Transform numerical features
        for col, scaler in self.scalers.items():
            num_value = product_data.get(col, 0.0)
            scaled_value = scaler.transform([[float(num_value)]])[0][0]
            features_list.append([scaled_value])
        
        if features_list:
            return np.concatenate(features_list)
        else:
            return np.array([1.0])  # dummy feature


class ContentBasedRecommender:
    """
    Content-based recommendation system using product features and user preferences.
    """
    
    def __init__(self, similarity_metric: str = 'cosine'):
        """
        Initialize content-based recommender.
        
        Args:
            similarity_metric: 'cosine', 'euclidean', or 'manhattan'
        """
        self.similarity_metric = similarity_metric
        self.feature_extractor = FeatureExtractor()
        self.product_features = None
        self.product_ids = None
        self.item_similarity_matrix = None
        self.user_profiles = {}
        self.fitted = False
    
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame,
            text_columns: List[str] = None, categorical_columns: List[str] = None,
            numerical_columns: List[str] = None) -> None:
        """
        Fit the content-based model.
        
        Args:
            products_df: Product features DataFrame
            interactions_df: User-item interactions DataFrame
            text_columns: Text feature columns
            categorical_columns: Categorical feature columns
            numerical_columns: Numerical feature columns
        """
        # Extract product features
        self.product_features = self.feature_extractor.extract_features(
            products_df, text_columns, categorical_columns, numerical_columns
        )
        self.product_ids = products_df['product_id'].tolist()
        
        # Compute item-item similarity matrix
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(self.product_features)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(self.product_features)
            # Convert distances to similarities
            max_distance = np.max(distances)
            self.item_similarity_matrix = 1 - (distances / max_distance)
        else:
            # Default to cosine similarity
            self.item_similarity_matrix = cosine_similarity(self.product_features)
        
        # Build user profiles based on interactions
        self._build_user_profiles(interactions_df)
        
        self.fitted = True
    
    def _build_user_profiles(self, interactions_df: pd.DataFrame) -> None:
        """
        Build user profiles based on interaction history.
        
        Args:
            interactions_df: User-item interactions
        """
        # Group interactions by user
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            # Get products the user has interacted with
            user_products = user_interactions['product_id'].tolist()
            user_ratings = user_interactions.get('rating', [1.0] * len(user_products))
            
            # Create weighted average of product features as user profile
            user_feature_vector = np.zeros(self.product_features.shape[1])
            total_weight = 0
            
            for product_id, rating in zip(user_products, user_ratings):
                if product_id in self.product_ids:
                    product_idx = self.product_ids.index(product_id)
                    weight = rating if pd.notna(rating) else 1.0
                    user_feature_vector += weight * self.product_features[product_idx]
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                user_feature_vector /= total_weight
            
            self.user_profiles[user_id] = user_feature_vector
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair based on content similarity.
        
        Args:
            user_id: Target user
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_profiles:
            # For new users, return average rating
            return 3.0
        
        if item_id not in self.product_ids:
            # For new items, return average rating
            return 3.0
        
        # Get user profile and item features
        user_profile = self.user_profiles[user_id]
        item_idx = self.product_ids.index(item_id)
        item_features = self.product_features[item_idx]
        
        # Compute similarity between user profile and item
        similarity = cosine_similarity([user_profile], [item_features])[0][0]
        
        # Convert similarity to rating (1-5 scale)
        # Similarity ranges from -1 to 1, we map to 1-5
        rating = 1 + (similarity + 1) / 2 * 4
        
        return max(1.0, min(5.0, rating))
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True, 
                 seen_items: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations for a user.
        
        Args:
            user_id: Target user
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude items user has seen
            seen_items: List of items user has already seen
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_profiles:
            # For new users, recommend popular items (placeholder)
            recommendations = [(item_id, 3.0) for item_id in self.product_ids[:n_recommendations]]
            return recommendations
        
        # Determine items to exclude
        if exclude_seen and seen_items:
            exclude_set = set(seen_items)
        else:
            exclude_set = set()
        
        # Get candidate items
        candidate_items = [item_id for item_id in self.product_ids 
                         if item_id not in exclude_set]
        
        # Predict ratings for all candidates
        predictions = []
        for item_id in candidate_items:
            predicted_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def find_similar_items(self, item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Find items similar to a given item based on content features.
        
        Args:
            item_id: Target item
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if item_id not in self.product_ids:
            return []
        
        item_idx = self.product_ids.index(item_id)
        similarities = self.item_similarity_matrix[item_idx]
        
        # Get indices of most similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        similar_items = []
        for idx in similar_indices:
            similar_item_id = self.product_ids[idx]
            similarity_score = similarities[idx]
            similar_items.append((similar_item_id, similarity_score))
        
        return similar_items
    
    def update_user_profile(self, user_id: str, item_id: str, rating: float) -> None:
        """
        Update user profile based on new interaction.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            rating: User rating for the item
        """
        if item_id not in self.product_ids:
            return
        
        item_idx = self.product_ids.index(item_id)
        item_features = self.product_features[item_idx]
        
        if user_id in self.user_profiles:
            # Update existing profile with weighted average
            current_profile = self.user_profiles[user_id]
            # Simple update: mix 90% old profile with 10% new item
            alpha = 0.1
            new_profile = (1 - alpha) * current_profile + alpha * rating * item_features
            self.user_profiles[user_id] = new_profile
        else:
            # Create new profile
            self.user_profiles[user_id] = rating * item_features


class PreferenceBasedRecommender:
    """
    Recommendation system based on explicit user preferences and product categories.
    """
    
    def __init__(self):
        self.user_preferences = {}
        self.category_features = {}
        self.preference_model = None
        self.fitted = False
    
    def fit(self, users_df: pd.DataFrame, products_df: pd.DataFrame,
            interactions_df: pd.DataFrame) -> None:
        """
        Fit preference-based model.
        
        Args:
            users_df: User data with preferences
            products_df: Product data
            interactions_df: User-item interactions
        """
        # Extract user preferences
        for _, user in users_df.iterrows():
            user_id = user['user_id']
            preferred_categories = user.get('preferred_categories', [])
            if isinstance(preferred_categories, str):
                # Handle case where preferences are stored as comma-separated string
                preferred_categories = [cat.strip() for cat in preferred_categories.split(',')]
            self.user_preferences[user_id] = preferred_categories
        
        # Build category-based features
        categories = products_df['category'].unique()
        for category in categories:
            category_products = products_df[products_df['category'] == category]
            
            # Aggregate features for category
            numerical_cols = ['price', 'average_rating']
            category_features = {}
            
            for col in numerical_cols:
                if col in category_products.columns:
                    category_features[col] = {
                        'mean': category_products[col].mean(),
                        'std': category_products[col].std(),
                        'min': category_products[col].min(),
                        'max': category_products[col].max()
                    }
            
            self.category_features[category] = category_features
        
        self.fitted = True
    
    def recommend(self, user_id: str, products_df: pd.DataFrame,
                 n_recommendations: int = 10, exclude_seen: bool = True,
                 seen_items: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate preference-based recommendations.
        
        Args:
            user_id: Target user
            products_df: Product data
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Items user has already seen
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        user_prefs = self.user_preferences.get(user_id, [])
        if not user_prefs:
            # Return popular items for users without preferences
            popular_items = products_df.nlargest(n_recommendations, 'average_rating')
            return [(item_id, rating) for item_id, rating in 
                   zip(popular_items['product_id'], popular_items['average_rating'])]
        
        # Score items based on user preferences
        exclude_set = set(seen_items) if seen_items else set()
        scored_items = []
        
        for _, product in products_df.iterrows():
            item_id = product['product_id']
            
            if exclude_seen and item_id in exclude_set:
                continue
            
            # Calculate preference score
            category = product.get('category', 'Unknown')
            base_score = product.get('average_rating', 3.0)
            
            # Boost score if category is in user preferences
            if category in user_prefs:
                preference_boost = 1.5
            else:
                preference_boost = 1.0
            
            final_score = base_score * preference_boost
            scored_items.append((item_id, final_score))
        
        # Sort and return top recommendations
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:n_recommendations]


class AdvancedContentBasedRecommender:
    """
    Advanced content-based recommender using machine learning models
    to predict user preferences.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize advanced content-based recommender.
        
        Args:
            model_type: 'linear', 'ridge', 'random_forest'
        """
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        self.preference_model = None
        self.product_features = None
        self.product_ids = None
        self.fitted = False
    
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame,
            text_columns: List[str] = None, categorical_columns: List[str] = None,
            numerical_columns: List[str] = None) -> None:
        """
        Fit advanced content-based model using ML.
        
        Args:
            products_df: Product features
            interactions_df: User-item interactions with ratings
            text_columns: Text feature columns
            categorical_columns: Categorical feature columns
            numerical_columns: Numerical feature columns
        """
        # Extract product features
        self.product_features = self.feature_extractor.extract_features(
            products_df, text_columns, categorical_columns, numerical_columns
        )
        self.product_ids = products_df['product_id'].tolist()
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for _, interaction in interactions_df.iterrows():
            if pd.notna(interaction.get('rating')):
                product_id = interaction['product_id']
                if product_id in self.product_ids:
                    product_idx = self.product_ids.index(product_id)
                    features = self.product_features[product_idx]
                    rating = interaction['rating']
                    
                    X_train.append(features)
                    y_train.append(rating)
        
        if not X_train:
            raise ValueError("No valid training data found")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train preference model
        if self.model_type == 'linear':
            self.preference_model = LinearRegression()
        elif self.model_type == 'ridge':
            self.preference_model = Ridge(alpha=1.0)
        elif self.model_type == 'random_forest':
            self.preference_model = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.preference_model.fit(X_train, y_train)
        self.fitted = True
    
    def predict_rating(self, item_id: str) -> float:
        """
        Predict rating for an item using the trained model.
        
        Args:
            item_id: Target item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if item_id not in self.product_ids:
            return 3.0  # Default rating
        
        item_idx = self.product_ids.index(item_id)
        item_features = self.product_features[item_idx].reshape(1, -1)
        
        predicted_rating = self.preference_model.predict(item_features)[0]
        return max(1.0, min(5.0, predicted_rating))
    
    def recommend(self, n_recommendations: int = 10, 
                 exclude_seen: bool = True,
                 seen_items: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate ML-based content recommendations.
        
        Args:
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Items to exclude
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        exclude_set = set(seen_items) if seen_items else set()
        
        # Predict ratings for all items
        predictions = []
        for item_id in self.product_ids:
            if exclude_seen and item_id in exclude_set:
                continue
            
            predicted_rating = self.predict_rating(item_id)
            predictions.append((item_id, predicted_rating))
        
        # Sort and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]