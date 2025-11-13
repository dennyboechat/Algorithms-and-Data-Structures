"""
Evaluation Metrics for Recommendation Systems

This module implements comprehensive evaluation metrics including RMSE, MAE,
precision, recall, F1-score, and other metrics commonly used to assess
recommendation system performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.
    """
    
    def __init__(self):
        self.test_data = None
        self.train_data = None
        self.ground_truth = None
        
    def prepare_evaluation_data(self, interactions_df: pd.DataFrame,
                              test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split data into train and test sets for evaluation.
        
        Args:
            interactions_df: Full interaction dataset
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        # Split data maintaining user presence in both sets
        users = interactions_df['user_id'].unique()
        
        test_interactions = []
        train_interactions = []
        
        for user in users:
            user_data = interactions_df[interactions_df['user_id'] == user]
            
            if len(user_data) >= 2:
                # Split user's interactions
                user_train, user_test = train_test_split(
                    user_data, test_size=test_size, random_state=random_state
                )
                train_interactions.append(user_train)
                test_interactions.append(user_test)
            else:
                # Put single interaction in training
                train_interactions.append(user_data)
        
        self.train_data = pd.concat(train_interactions, ignore_index=True)
        self.test_data = pd.concat(test_interactions, ignore_index=True) if test_interactions else pd.DataFrame()
        
        # Create ground truth dictionary for easy lookup
        self.ground_truth = {}
        for _, row in self.test_data.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            rating = row.get('rating', 1.0)
            
            if user_id not in self.ground_truth:
                self.ground_truth[user_id] = {}
            self.ground_truth[user_id][item_id] = rating
    
    def rating_prediction_metrics(self, model, users: List[str] = None) -> Dict[str, float]:
        """
        Evaluate rating prediction accuracy.
        
        Args:
            model: Trained recommendation model with predict_rating method
            users: List of users to evaluate (if None, use all test users)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.test_data is None or self.test_data.empty:
            raise ValueError("No test data available. Call prepare_evaluation_data first.")
        
        if users is None:
            users = list(self.ground_truth.keys())
        
        predicted_ratings = []
        actual_ratings = []
        
        for user_id in users:
            if user_id in self.ground_truth:
                for item_id, actual_rating in self.ground_truth[user_id].items():
                    try:
                        predicted_rating = model.predict_rating(user_id, item_id)
                        predicted_ratings.append(predicted_rating)
                        actual_ratings.append(actual_rating)
                    except:
                        # Skip if prediction fails
                        continue
        
        if not predicted_ratings:
            return {'RMSE': float('inf'), 'MAE': float('inf'), 'R2': 0.0}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((np.array(actual_ratings) - np.array(predicted_ratings)) ** 2)
        ss_tot = np.sum((np.array(actual_ratings) - np.mean(actual_ratings)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'num_predictions': len(predicted_ratings)
        }
    
    def ranking_metrics(self, model, k: int = 10, users: List[str] = None,
                       threshold: float = 3.0) -> Dict[str, float]:
        """
        Evaluate ranking quality with precision, recall, and F1-score.
        
        Args:
            model: Trained recommendation model with recommend method
            k: Number of recommendations to evaluate
            users: List of users to evaluate
            threshold: Rating threshold for considering items as relevant
            
        Returns:
            Dictionary of ranking metrics
        """
        if self.test_data is None or self.test_data.empty:
            raise ValueError("No test data available. Call prepare_evaluation_data first.")
        
        if users is None:
            users = list(self.ground_truth.keys())
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for user_id in users:
            if user_id not in self.ground_truth:
                continue
            
            # Get relevant items (highly rated in test set)
            relevant_items = set([
                item_id for item_id, rating in self.ground_truth[user_id].items()
                if rating >= threshold
            ])
            
            if not relevant_items:
                continue  # Skip users with no relevant items
            
            try:
                # Get recommendations
                recommendations = model.recommend(user_id, k, exclude_seen=True)
                recommended_items = set([item_id for item_id, _ in recommendations])
                
                # Calculate metrics
                tp = len(recommended_items & relevant_items)  # True positives
                fp = len(recommended_items - relevant_items)  # False positives
                fn = len(relevant_items - recommended_items)  # False negatives
                
                precision = tp / len(recommended_items) if recommended_items else 0
                recall = tp / len(relevant_items) if relevant_items else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
            except Exception as e:
                # Skip if recommendation fails
                print(f"Error getting recommendations for user {user_id}: {e}")
                continue
        
        return {
            'Precision@K': np.mean(precisions) if precisions else 0,
            'Recall@K': np.mean(recalls) if recalls else 0,
            'F1@K': np.mean(f1_scores) if f1_scores else 0,
            'num_users_evaluated': len(precisions)
        }
    
    def diversity_metrics(self, model, users: List[str] = None, k: int = 10) -> Dict[str, float]:
        """
        Evaluate recommendation diversity and coverage.
        
        Args:
            model: Trained recommendation model
            users: List of users to evaluate
            k: Number of recommendations per user
            
        Returns:
            Dictionary of diversity metrics
        """
        if users is None:
            users = list(self.ground_truth.keys())
        
        all_recommended_items = set()
        user_item_sets = []
        
        for user_id in users:
            try:
                recommendations = model.recommend(user_id, k)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                all_recommended_items.update(recommended_items)
                user_item_sets.append(set(recommended_items))
                
            except:
                continue
        
        # Calculate diversity metrics
        total_items = len(self.train_data['product_id'].unique()) if hasattr(self, 'train_data') else 1
        
        # Coverage: fraction of items that appear in recommendations
        coverage = len(all_recommended_items) / total_items if total_items > 0 else 0
        
        # Intra-list diversity: average pairwise distance within recommendation lists
        intra_list_diversities = []
        for item_set in user_item_sets:
            if len(item_set) > 1:
                # Simplified diversity calculation (Jaccard distance)
                diversity = 1.0  # Placeholder - in practice, compute based on item features
                intra_list_diversities.append(diversity)
        
        avg_intra_list_diversity = np.mean(intra_list_diversities) if intra_list_diversities else 0
        
        # Personalization: how different recommendations are across users
        personalization_scores = []
        for i, set1 in enumerate(user_item_sets):
            for j, set2 in enumerate(user_item_sets[i+1:], i+1):
                if set1 and set2:
                    # Jaccard distance between user recommendation sets
                    jaccard_sim = len(set1 & set2) / len(set1 | set2)
                    personalization_scores.append(1 - jaccard_sim)
        
        personalization = np.mean(personalization_scores) if personalization_scores else 0
        
        return {
            'Coverage': coverage,
            'Avg_Intra_List_Diversity': avg_intra_list_diversity,
            'Personalization': personalization,
            'Total_Unique_Recommendations': len(all_recommended_items)
        }
    
    def novelty_metrics(self, model, users: List[str] = None, k: int = 10) -> Dict[str, float]:
        """
        Evaluate recommendation novelty.
        
        Args:
            model: Trained recommendation model
            users: List of users to evaluate
            k: Number of recommendations per user
            
        Returns:
            Dictionary of novelty metrics
        """
        if users is None:
            users = list(self.ground_truth.keys()) if self.ground_truth else []
        
        # Calculate item popularity from training data
        if hasattr(self, 'train_data') and self.train_data is not None:
            item_popularity = self.train_data['product_id'].value_counts().to_dict()
            total_interactions = len(self.train_data)
        else:
            return {'Avg_Novelty': 0, 'Avg_Self_Information': 0}
        
        novelties = []
        self_informations = []
        
        for user_id in users:
            try:
                recommendations = model.recommend(user_id, k)
                
                user_novelties = []
                user_self_info = []
                
                for item_id, _ in recommendations:
                    # Novelty: inverse of item popularity
                    popularity = item_popularity.get(item_id, 0)
                    novelty = 1 / (popularity + 1)  # +1 to avoid division by zero
                    user_novelties.append(novelty)
                    
                    # Self-information: -log2(popularity)
                    prob = popularity / total_interactions if total_interactions > 0 else 0.001
                    self_info = -np.log2(max(prob, 1e-10))  # Avoid log(0)
                    user_self_info.append(self_info)
                
                if user_novelties:
                    novelties.extend(user_novelties)
                    self_informations.extend(user_self_info)
                    
            except:
                continue
        
        return {
            'Avg_Novelty': np.mean(novelties) if novelties else 0,
            'Avg_Self_Information': np.mean(self_informations) if self_informations else 0
        }
    
    def cold_start_evaluation(self, model, min_interactions: int = 5) -> Dict[str, float]:
        """
        Evaluate performance on cold start users/items.
        
        Args:
            model: Trained recommendation model
            min_interactions: Threshold for considering users/items as cold start
            
        Returns:
            Dictionary of cold start metrics
        """
        if not hasattr(self, 'train_data') or self.train_data is None:
            return {}
        
        # Identify cold start users
        user_interaction_counts = self.train_data['user_id'].value_counts()
        cold_start_users = user_interaction_counts[user_interaction_counts < min_interactions].index.tolist()
        warm_start_users = user_interaction_counts[user_interaction_counts >= min_interactions].index.tolist()
        
        # Identify cold start items
        item_interaction_counts = self.train_data['product_id'].value_counts()
        cold_start_items = item_interaction_counts[item_interaction_counts < min_interactions].index.tolist()
        
        results = {}
        
        # Evaluate on cold start users
        if cold_start_users:
            cold_users_in_test = [u for u in cold_start_users if u in self.ground_truth]
            if cold_users_in_test:
                cold_user_metrics = self.ranking_metrics(model, users=cold_users_in_test)
                results['Cold_Start_Users_Precision'] = cold_user_metrics['Precision@K']
                results['Cold_Start_Users_Recall'] = cold_user_metrics['Recall@K']
        
        # Evaluate on warm start users for comparison
        if warm_start_users:
            warm_users_in_test = [u for u in warm_start_users if u in self.ground_truth]
            if warm_users_in_test:
                warm_user_metrics = self.ranking_metrics(model, users=warm_users_in_test)
                results['Warm_Start_Users_Precision'] = warm_user_metrics['Precision@K']
                results['Warm_Start_Users_Recall'] = warm_user_metrics['Recall@K']
        
        # Cold start item coverage
        total_cold_items = len(cold_start_items)
        results['Cold_Start_Item_Ratio'] = total_cold_items / len(self.train_data['product_id'].unique())
        
        return results
    
    def comprehensive_evaluation(self, model, k: int = 10, 
                               rating_threshold: float = 3.0) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            model: Trained recommendation model
            k: Number of recommendations for ranking metrics
            rating_threshold: Threshold for relevance in ranking metrics
            
        Returns:
            Dictionary organized by metric category
        """
        print("Running comprehensive evaluation...")
        
        results = {}
        
        # Rating prediction metrics
        try:
            results['Rating_Prediction'] = self.rating_prediction_metrics(model)
            print("✓ Rating prediction metrics completed")
        except Exception as e:
            print(f"✗ Rating prediction metrics failed: {e}")
            results['Rating_Prediction'] = {}
        
        # Ranking metrics
        try:
            results['Ranking'] = self.ranking_metrics(model, k=k, threshold=rating_threshold)
            print("✓ Ranking metrics completed")
        except Exception as e:
            print(f"✗ Ranking metrics failed: {e}")
            results['Ranking'] = {}
        
        # Diversity metrics
        try:
            results['Diversity'] = self.diversity_metrics(model, k=k)
            print("✓ Diversity metrics completed")
        except Exception as e:
            print(f"✗ Diversity metrics failed: {e}")
            results['Diversity'] = {}
        
        # Novelty metrics
        try:
            results['Novelty'] = self.novelty_metrics(model, k=k)
            print("✓ Novelty metrics completed")
        except Exception as e:
            print(f"✗ Novelty metrics failed: {e}")
            results['Novelty'] = {}
        
        # Cold start evaluation
        try:
            results['Cold_Start'] = self.cold_start_evaluation(model)
            print("✓ Cold start evaluation completed")
        except Exception as e:
            print(f"✗ Cold start evaluation failed: {e}")
            results['Cold_Start'] = {}
        
        print("Comprehensive evaluation completed!")
        return results
    
    def print_evaluation_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Results from comprehensive_evaluation
        """
        print("\n" + "="*60)
        print("RECOMMENDATION SYSTEM EVALUATION RESULTS")
        print("="*60)
        
        for category, metrics in results.items():
            print(f"\n{category.upper()} METRICS:")
            print("-" * 40)
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    if metric_name in ['RMSE', 'MAE']:
                        print(f"{metric_name:<25}: {value:.4f}")
                    else:
                        print(f"{metric_name:<25}: {value:.4f}")
                else:
                    print(f"{metric_name:<25}: {value}")
        
        print("\n" + "="*60)


def cross_validate_model(model_class, data_manager, k_folds: int = 5,
                        model_kwargs: Dict = None) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation on a recommendation model.
    
    Args:
        model_class: Class of the recommendation model
        data_manager: Data manager with loaded data
        k_folds: Number of cross-validation folds
        model_kwargs: Keyword arguments for model initialization
        
    Returns:
        Dictionary with lists of metrics for each fold
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Get all interactions
    interactions_df = data_manager.interactions.interactions_df
    
    # Split data into k folds
    users = interactions_df['user_id'].unique()
    np.random.shuffle(users)
    
    fold_size = len(users) // k_folds
    folds_metrics = {
        'RMSE': [], 'MAE': [], 'Precision@10': [], 'Recall@10': [], 'F1@10': []
    }
    
    for fold in range(k_folds):
        print(f"Running fold {fold + 1}/{k_folds}...")
        
        # Split users
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k_folds - 1 else len(users)
        test_users = users[start_idx:end_idx]
        train_users = users[np.isin(users, test_users, invert=True)]
        
        # Split data
        train_data = interactions_df[interactions_df['user_id'].isin(train_users)]
        test_data = interactions_df[interactions_df['user_id'].isin(test_users)]
        
        # Train model
        try:
            model = model_class(**model_kwargs)
            
            if hasattr(model, 'fit'):
                user_item_matrix = train_data.pivot_table(
                    index='user_id',
                    columns='product_id',
                    values='rating',
                    fill_value=0
                )
                model.fit(user_item_matrix)
            
            # Evaluate
            evaluator = RecommendationEvaluator()
            evaluator.train_data = train_data
            evaluator.test_data = test_data
            evaluator.ground_truth = {}
            
            for _, row in test_data.iterrows():
                user_id = row['user_id']
                item_id = row['product_id']
                rating = row.get('rating', 1.0)
                
                if user_id not in evaluator.ground_truth:
                    evaluator.ground_truth[user_id] = {}
                evaluator.ground_truth[user_id][item_id] = rating
            
            # Get metrics
            rating_metrics = evaluator.rating_prediction_metrics(model)
            ranking_metrics = evaluator.ranking_metrics(model, k=10)
            
            folds_metrics['RMSE'].append(rating_metrics.get('RMSE', float('inf')))
            folds_metrics['MAE'].append(rating_metrics.get('MAE', float('inf')))
            folds_metrics['Precision@10'].append(ranking_metrics.get('Precision@K', 0))
            folds_metrics['Recall@10'].append(ranking_metrics.get('Recall@K', 0))
            folds_metrics['F1@10'].append(ranking_metrics.get('F1@K', 0))
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            # Add default values for failed fold
            for metric in folds_metrics:
                folds_metrics[metric].append(0 if metric != 'RMSE' and metric != 'MAE' else float('inf'))
    
    return folds_metrics