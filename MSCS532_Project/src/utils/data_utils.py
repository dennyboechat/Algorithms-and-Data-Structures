"""
Utility Functions for the Recommendation System

This module provides helper functions for data processing, feature engineering,
and common operations across the recommendation system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load datasets from files.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Dictionary containing loaded DataFrames
    """
    import os
    
    data_files = {
        'users': 'sample_users.csv',
        'products': 'sample_products.csv',
        'interactions': 'sample_interactions.csv'
    }
    
    datasets = {}
    
    for name, filename in data_files.items():
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            try:
                if name == 'users':
                    df = pd.read_csv(filepath)
                    # Parse preferred_categories if it's a string
                    if 'preferred_categories' in df.columns:
                        df['preferred_categories'] = df['preferred_categories'].apply(
                            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [x] if pd.notna(x) else []
                        )
                elif name == 'products':
                    df = pd.read_csv(filepath)
                    # Parse features if it's a string
                    if 'features' in df.columns:
                        df['features'] = df['features'].apply(
                            lambda x: eval(x) if isinstance(x, str) and x.startswith('{') else {} if pd.notna(x) else {}
                        )
                else:
                    df = pd.read_csv(filepath)
                    # Parse timestamps
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                datasets[name] = df
                print(f"✅ Loaded {name}: {len(df)} records")
                
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        else:
            print(f"⚠️  File not found: {filepath}")
    
    return datasets


def preprocess_data(interactions_df: pd.DataFrame, 
                   min_user_interactions: int = 5,
                   min_item_interactions: int = 3) -> pd.DataFrame:
    """
    Preprocess interaction data by filtering out sparse users and items.
    
    Args:
        interactions_df: Raw interactions DataFrame
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        
    Returns:
        Filtered interactions DataFrame
    """
    print("Preprocessing interaction data...")
    
    original_size = len(interactions_df)
    
    # Filter users with sufficient interactions
    user_counts = interactions_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    filtered_data = interactions_df[interactions_df['user_id'].isin(valid_users)]
    
    # Filter items with sufficient interactions
    item_counts = filtered_data['product_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    filtered_data = filtered_data[filtered_data['product_id'].isin(valid_items)]
    
    print(f"Filtered from {original_size} to {len(filtered_data)} interactions")
    print(f"Users: {interactions_df['user_id'].nunique()} → {filtered_data['user_id'].nunique()}")
    print(f"Items: {interactions_df['product_id'].nunique()} → {filtered_data['product_id'].nunique()}")
    
    return filtered_data


def create_user_item_matrix(interactions_df: pd.DataFrame, 
                           value_column: str = 'rating') -> pd.DataFrame:
    """
    Create user-item matrix from interactions.
    
    Args:
        interactions_df: Interactions DataFrame
        value_column: Column to use for matrix values
        
    Returns:
        User-item matrix DataFrame
    """
    # Filter for interactions with values
    if value_column in interactions_df.columns:
        data_with_values = interactions_df[interactions_df[value_column].notna()]
    else:
        # Create binary interaction matrix
        data_with_values = interactions_df.copy()
        data_with_values[value_column] = 1.0
    
    # Create pivot table
    user_item_matrix = data_with_values.pivot_table(
        index='user_id',
        columns='product_id',
        values=value_column,
        aggfunc='mean',  # Average if multiple ratings
        fill_value=0
    )
    
    return user_item_matrix


def calculate_sparsity(matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate matrix sparsity statistics.
    
    Args:
        matrix: User-item matrix
        
    Returns:
        Dictionary with sparsity metrics
    """
    total_elements = matrix.size
    non_zero_elements = (matrix != 0).sum().sum()
    
    sparsity = 1 - (non_zero_elements / total_elements)
    density = non_zero_elements / total_elements
    
    return {
        'total_elements': total_elements,
        'non_zero_elements': non_zero_elements,
        'sparsity': sparsity,
        'density': density,
        'sparsity_percentage': sparsity * 100
    }


def get_popular_items(interactions_df: pd.DataFrame, 
                     top_k: int = 20) -> pd.DataFrame:
    """
    Get most popular items based on interaction frequency.
    
    Args:
        interactions_df: Interactions DataFrame
        top_k: Number of popular items to return
        
    Returns:
        DataFrame with popular items
    """
    item_popularity = interactions_df.groupby('product_id').agg({
        'user_id': 'nunique',  # Number of unique users
        'rating': ['count', 'mean']  # Interaction count and avg rating
    }).round(2)
    
    # Flatten column names
    item_popularity.columns = ['unique_users', 'total_interactions', 'avg_rating']
    item_popularity = item_popularity.reset_index()
    
    # Sort by number of interactions
    popular_items = item_popularity.nlargest(top_k, 'total_interactions')
    
    return popular_items


def analyze_user_behavior(interactions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze user behavior patterns.
    
    Args:
        interactions_df: Interactions DataFrame
        
    Returns:
        Dictionary with behavior analysis
    """
    analysis = {}
    
    # Interaction type distribution
    if 'interaction_type' in interactions_df.columns:
        interaction_types = interactions_df['interaction_type'].value_counts()
        analysis['interaction_types'] = interaction_types.to_dict()
    
    # User activity distribution
    user_activity = interactions_df.groupby('user_id').size()
    analysis['user_activity'] = {
        'mean': user_activity.mean(),
        'median': user_activity.median(),
        'std': user_activity.std(),
        'min': user_activity.min(),
        'max': user_activity.max()
    }
    
    # Rating distribution
    if 'rating' in interactions_df.columns:
        ratings = interactions_df['rating'].dropna()
        if len(ratings) > 0:
            analysis['rating_distribution'] = {
                'mean': ratings.mean(),
                'median': ratings.median(),
                'std': ratings.std(),
                'min': ratings.min(),
                'max': ratings.max(),
                'count': len(ratings)
            }
    
    # Temporal patterns
    if 'timestamp' in interactions_df.columns:
        interactions_df['hour'] = pd.to_datetime(interactions_df['timestamp']).dt.hour
        interactions_df['day_of_week'] = pd.to_datetime(interactions_df['timestamp']).dt.dayofweek
        
        hourly_activity = interactions_df['hour'].value_counts().sort_index()
        daily_activity = interactions_df['day_of_week'].value_counts().sort_index()
        
        analysis['temporal_patterns'] = {
            'peak_hour': hourly_activity.idxmax(),
            'peak_day': daily_activity.idxmax(),
            'hourly_distribution': hourly_activity.to_dict(),
            'daily_distribution': daily_activity.to_dict()
        }
    
    return analysis


def create_feature_matrix(products_df: pd.DataFrame, 
                         categorical_columns: List[str] = None,
                         numerical_columns: List[str] = None,
                         text_columns: List[str] = None) -> np.ndarray:
    """
    Create feature matrix from product data.
    
    Args:
        products_df: Products DataFrame
        categorical_columns: List of categorical columns
        numerical_columns: List of numerical columns  
        text_columns: List of text columns
        
    Returns:
        Feature matrix
    """
    features_list = []
    
    if categorical_columns:
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_columns:
            if col in products_df.columns:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(products_df[col].fillna('Unknown'))
                features_list.append(encoded.reshape(-1, 1))
    
    if numerical_columns:
        from sklearn.preprocessing import StandardScaler
        for col in numerical_columns:
            if col in products_df.columns:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(products_df[col].fillna(0).values.reshape(-1, 1))
                features_list.append(scaled)
    
    if text_columns:
        from sklearn.feature_extraction.text import TfidfVectorizer
        for col in text_columns:
            if col in products_df.columns:
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                text_features = vectorizer.fit_transform(products_df[col].fillna('')).toarray()
                features_list.append(text_features)
    
    if features_list:
        return np.hstack(features_list)
    else:
        return np.ones((len(products_df), 1))  # Dummy feature


def split_temporal_data(interactions_df: pd.DataFrame, 
                       test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on temporal order (chronological split).
    
    Args:
        interactions_df: Interactions DataFrame with timestamp
        test_ratio: Ratio of data for testing
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if 'timestamp' not in interactions_df.columns:
        raise ValueError("Timestamp column required for temporal split")
    
    # Sort by timestamp
    sorted_data = interactions_df.sort_values('timestamp')
    
    # Calculate split point
    split_point = int(len(sorted_data) * (1 - test_ratio))
    
    train_data = sorted_data.iloc[:split_point]
    test_data = sorted_data.iloc[split_point:]
    
    print(f"Temporal split: {len(train_data)} train, {len(test_data)} test")
    print(f"Train period: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
    print(f"Test period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
    
    return train_data, test_data


def evaluate_cold_start(interactions_df: pd.DataFrame, 
                       min_interactions: int = 5) -> Dict[str, Any]:
    """
    Analyze cold start problems in the dataset.
    
    Args:
        interactions_df: Interactions DataFrame
        min_interactions: Threshold for cold start
        
    Returns:
        Dictionary with cold start analysis
    """
    user_counts = interactions_df['user_id'].value_counts()
    item_counts = interactions_df['product_id'].value_counts()
    
    cold_start_users = (user_counts < min_interactions).sum()
    cold_start_items = (item_counts < min_interactions).sum()
    
    total_users = len(user_counts)
    total_items = len(item_counts)
    
    analysis = {
        'cold_start_users': {
            'count': cold_start_users,
            'percentage': (cold_start_users / total_users) * 100
        },
        'cold_start_items': {
            'count': cold_start_items,
            'percentage': (cold_start_items / total_items) * 100
        },
        'total_users': total_users,
        'total_items': total_items,
        'interaction_threshold': min_interactions
    }
    
    return analysis


def save_recommendations(recommendations: Dict[str, List[Tuple[str, float]]], 
                        output_file: str) -> None:
    """
    Save recommendations to file.
    
    Args:
        recommendations: Dictionary mapping user_id to list of (item_id, score)
        output_file: Output file path
    """
    # Convert to serializable format
    serializable_recs = {}
    for user_id, recs in recommendations.items():
        serializable_recs[user_id] = [
            {'item_id': item_id, 'score': float(score)} 
            for item_id, score in recs
        ]
    
    with open(output_file, 'w') as f:
        json.dump(serializable_recs, f, indent=2)
    
    print(f"Recommendations saved to {output_file}")


def print_data_summary(users_df: pd.DataFrame, 
                      products_df: pd.DataFrame, 
                      interactions_df: pd.DataFrame) -> None:
    """
    Print comprehensive data summary.
    
    Args:
        users_df: Users DataFrame
        products_df: Products DataFrame
        interactions_df: Interactions DataFrame
    """
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"Users: {len(users_df):,}")
    print(f"Products: {len(products_df):,}")
    print(f"Interactions: {len(interactions_df):,}")
    
    # User demographics
    print(f"\n👥 USER DEMOGRAPHICS:")
    if 'age' in users_df.columns:
        print(f"Age range: {users_df['age'].min()}-{users_df['age'].max()} (avg: {users_df['age'].mean():.1f})")
    if 'gender' in users_df.columns:
        gender_dist = users_df['gender'].value_counts()
        print(f"Gender distribution: {dict(gender_dist)}")
    if 'spending_level' in users_df.columns:
        spending_dist = users_df['spending_level'].value_counts()
        print(f"Spending levels: {dict(spending_dist)}")
    
    # Product categories
    print(f"\n🛍️ PRODUCT CATALOG:")
    if 'category' in products_df.columns:
        category_dist = products_df['category'].value_counts()
        print(f"Categories: {dict(category_dist)}")
    if 'price' in products_df.columns:
        print(f"Price range: ${products_df['price'].min():.2f}-${products_df['price'].max():.2f} (avg: ${products_df['price'].mean():.2f})")
    
    # Interaction patterns
    print(f"\n🔄 INTERACTION PATTERNS:")
    if 'interaction_type' in interactions_df.columns:
        interaction_dist = interactions_df['interaction_type'].value_counts()
        print(f"Interaction types: {dict(interaction_dist)}")
    
    # Sparsity analysis
    user_item_matrix = create_user_item_matrix(interactions_df)
    sparsity_stats = calculate_sparsity(user_item_matrix)
    print(f"\n📈 SPARSITY ANALYSIS:")
    print(f"Matrix size: {user_item_matrix.shape}")
    print(f"Sparsity: {sparsity_stats['sparsity_percentage']:.1f}%")
    print(f"Density: {sparsity_stats['density']:.4f}")
    
    print("="*50)