"""
Data Management Module for E-commerce Recommendation System

This module provides classes for managing user data, product data, and interaction data
using pandas DataFrames as the base storage structures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime


class UserDataManager:
    """
    Manages user data including profiles and preferences using pandas DataFrames.
    """
    
    def __init__(self):
        self.users_df = pd.DataFrame(columns=[
            'user_id', 'age', 'gender', 'location', 'registration_date',
            'preferred_categories', 'spending_level'
        ])
        
    def add_user(self, user_id: str, age: int, gender: str, location: str,
                 preferred_categories: List[str], spending_level: str) -> None:
        """Add a new user to the dataset."""
        new_user = pd.DataFrame({
            'user_id': [user_id],
            'age': [age],
            'gender': [gender],
            'location': [location],
            'registration_date': [datetime.now()],
            'preferred_categories': [preferred_categories],
            'spending_level': [spending_level]
        })
        self.users_df = pd.concat([self.users_df, new_user], ignore_index=True)
    
    def get_user(self, user_id: str) -> Optional[pd.Series]:
        """Retrieve a specific user's data."""
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        return user_data.iloc[0] if not user_data.empty else None
    
    def get_users_by_category(self, category: str) -> pd.DataFrame:
        """Get all users who prefer a specific category."""
        return self.users_df[
            self.users_df['preferred_categories'].apply(
                lambda x: category in x if isinstance(x, list) else False
            )
        ]
    
    def get_user_demographics(self, user_id: str) -> Dict:
        """Get demographic information for a user."""
        user = self.get_user(user_id)
        if user is not None:
            return {
                'age': user['age'],
                'gender': user['gender'],
                'location': user['location'],
                'spending_level': user['spending_level']
            }
        return {}


class ProductDataManager:
    """
    Manages product catalog and features using pandas DataFrames.
    """
    
    def __init__(self):
        self.products_df = pd.DataFrame(columns=[
            'product_id', 'name', 'category', 'subcategory', 'price',
            'brand', 'features', 'average_rating', 'num_reviews'
        ])
    
    def add_product(self, product_id: str, name: str, category: str,
                   subcategory: str, price: float, brand: str,
                   features: Dict, average_rating: float = 0.0,
                   num_reviews: int = 0) -> None:
        """Add a new product to the catalog."""
        new_product = pd.DataFrame({
            'product_id': [product_id],
            'name': [name],
            'category': [category],
            'subcategory': [subcategory],
            'price': [price],
            'brand': [brand],
            'features': [features],
            'average_rating': [average_rating],
            'num_reviews': [num_reviews]
        })
        self.products_df = pd.concat([self.products_df, new_product], ignore_index=True)
    
    def get_product(self, product_id: str) -> Optional[pd.Series]:
        """Retrieve a specific product's data."""
        product_data = self.products_df[self.products_df['product_id'] == product_id]
        return product_data.iloc[0] if not product_data.empty else None
    
    def get_products_by_category(self, category: str) -> pd.DataFrame:
        """Get all products in a specific category."""
        return self.products_df[self.products_df['category'] == category]
    
    def get_products_by_price_range(self, min_price: float, max_price: float) -> pd.DataFrame:
        """Get products within a price range."""
        return self.products_df[
            (self.products_df['price'] >= min_price) &
            (self.products_df['price'] <= max_price)
        ]
    
    def get_product_features(self, product_id: str) -> Dict:
        """Get feature vector for a product."""
        product = self.get_product(product_id)
        if product is not None:
            features = product['features'].copy() if isinstance(product['features'], dict) else {}
            features.update({
                'price': product['price'],
                'category': product['category'],
                'brand': product['brand'],
                'average_rating': product['average_rating']
            })
            return features
        return {}


class InteractionDataManager:
    """
    Manages user-item interactions using pandas DataFrames.
    """
    
    def __init__(self):
        self.interactions_df = pd.DataFrame(columns=[
            'user_id', 'product_id', 'interaction_type', 'rating',
            'timestamp', 'session_id'
        ])
    
    def add_interaction(self, user_id: str, product_id: str, 
                       interaction_type: str, rating: Optional[float] = None,
                       session_id: Optional[str] = None) -> None:
        """Add a new user-product interaction."""
        new_interaction = pd.DataFrame({
            'user_id': [user_id],
            'product_id': [product_id],
            'interaction_type': [interaction_type],  # 'view', 'purchase', 'rating', 'cart'
            'rating': [rating],
            'timestamp': [datetime.now()],
            'session_id': [session_id]
        })
        self.interactions_df = pd.concat([self.interactions_df, new_interaction], ignore_index=True)
    
    def get_user_interactions(self, user_id: str) -> pd.DataFrame:
        """Get all interactions for a specific user."""
        return self.interactions_df[self.interactions_df['user_id'] == user_id]
    
    def get_product_interactions(self, product_id: str) -> pd.DataFrame:
        """Get all interactions for a specific product."""
        return self.interactions_df[self.interactions_df['product_id'] == product_id]
    
    def get_user_purchases(self, user_id: str) -> pd.DataFrame:
        """Get purchase history for a user."""
        return self.interactions_df[
            (self.interactions_df['user_id'] == user_id) &
            (self.interactions_df['interaction_type'] == 'purchase')
        ]
    
    def get_user_ratings(self, user_id: str) -> pd.DataFrame:
        """Get rating history for a user."""
        return self.interactions_df[
            (self.interactions_df['user_id'] == user_id) &
            (self.interactions_df['interaction_type'] == 'rating') &
            (self.interactions_df['rating'].notna())
        ]
    
    def get_interaction_matrix(self, interaction_type: str = 'rating') -> pd.DataFrame:
        """Create a user-item matrix for specified interaction type."""
        filtered_data = self.interactions_df[
            self.interactions_df['interaction_type'] == interaction_type
        ]
        
        if interaction_type == 'rating':
            matrix = filtered_data.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                fill_value=0
            )
        else:
            # For other interaction types, use binary values (1 if interaction exists)
            filtered_data = filtered_data.copy()
            filtered_data['interaction_value'] = 1
            matrix = filtered_data.pivot_table(
                index='user_id',
                columns='product_id',
                values='interaction_value',
                aggfunc='max',
                fill_value=0
            )
        
        return matrix
    
    def get_user_item_similarity_data(self) -> pd.DataFrame:
        """Get data formatted for similarity calculations."""
        return self.interactions_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )


class EcommerceDataManager:
    """
    Main data manager that coordinates all data sources.
    """
    
    def __init__(self):
        self.users = UserDataManager()
        self.products = ProductDataManager()
        self.interactions = InteractionDataManager()
    
    def load_sample_data(self) -> None:
        """Load sample e-commerce data for testing."""
        # Sample users
        users_data = [
            ('user_001', 25, 'Female', 'New York', ['Electronics', 'Books'], 'High'),
            ('user_002', 35, 'Male', 'California', ['Sports', 'Electronics'], 'Medium'),
            ('user_003', 28, 'Female', 'Texas', ['Fashion', 'Beauty'], 'High'),
            ('user_004', 42, 'Male', 'Florida', ['Home', 'Garden'], 'Low'),
            ('user_005', 31, 'Female', 'Illinois', ['Books', 'Electronics'], 'Medium'),
        ]
        
        for user_data in users_data:
            self.users.add_user(*user_data)
        
        # Sample products
        products_data = [
            ('prod_001', 'Smartphone X', 'Electronics', 'Mobile', 699.99, 'TechCorp', 
             {'screen_size': 6.1, 'storage': 128, 'camera_mp': 12}, 4.5, 1250),
            ('prod_002', 'Python Programming Book', 'Books', 'Technology', 49.99, 'TechBooks',
             {'pages': 450, 'language': 'English', 'difficulty': 'Intermediate'}, 4.7, 892),
            ('prod_003', 'Running Shoes', 'Sports', 'Footwear', 129.99, 'SportsBrand',
             {'size_range': '6-12', 'material': 'Mesh', 'type': 'Running'}, 4.3, 567),
            ('prod_004', 'Wireless Headphones', 'Electronics', 'Audio', 199.99, 'AudioCorp',
             {'battery_life': 30, 'noise_canceling': True, 'wireless': True}, 4.6, 2341),
            ('prod_005', 'Laptop Pro', 'Electronics', 'Computer', 1299.99, 'TechCorp',
             {'ram': 16, 'storage': 512, 'processor': 'Intel i7'}, 4.4, 678),
        ]
        
        for product_data in products_data:
            self.products.add_product(*product_data)
        
        # Sample interactions
        interactions_data = [
            ('user_001', 'prod_001', 'purchase', 5.0),
            ('user_001', 'prod_002', 'view', None),
            ('user_001', 'prod_004', 'rating', 4.0),
            ('user_002', 'prod_003', 'purchase', 4.0),
            ('user_002', 'prod_005', 'view', None),
            ('user_003', 'prod_001', 'view', None),
            ('user_003', 'prod_004', 'purchase', 5.0),
            ('user_004', 'prod_002', 'rating', 3.0),
            ('user_005', 'prod_002', 'purchase', 5.0),
            ('user_005', 'prod_005', 'rating', 4.0),
        ]
        
        for interaction_data in interactions_data:
            self.interactions.add_interaction(*interaction_data)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the dataset."""
        return {
            'num_users': len(self.users.users_df),
            'num_products': len(self.products.products_df),
            'num_interactions': len(self.interactions.interactions_df),
            'avg_rating': self.interactions.interactions_df[
                self.interactions.interactions_df['rating'].notna()
            ]['rating'].mean(),
            'categories': self.products.products_df['category'].unique().tolist()
        }