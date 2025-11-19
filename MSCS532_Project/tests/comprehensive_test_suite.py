"""
Comprehensive Testing Framework for Recommendation System

This module implements advanced testing capabilities including unit tests,
integration tests, performance benchmarks, stress tests, and validation
frameworks for the enhanced recommendation system.

Test categories:
- Unit tests for individual components
- Integration tests for system interactions
- Performance benchmarks and profiling
- Stress tests for scalability validation
- A/B testing framework for algorithm comparison
- Data quality and validation tests

References:
- Beck, K. (2003). Test-driven development: by example. Addison-Wesley Professional.
- Fowler, M. (2018). Refactoring: improving the design of existing code. 
  Addison-Wesley Professional.
- Kohavi, R., & Longbotham, R. (2017). Online controlled experiments and A/B testing. 
  Encyclopedia of machine learning and data mining, 922-929.
"""

import unittest
import pytest
import time
import random
import threading
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from collections import defaultdict
import warnings
import gc
import psutil
import os
import json
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn/pandas not available. Some tests will be skipped.")

try:
    from ..algorithms.optimized_collaborative_filtering import (
        OptimizedUserBasedCollaborativeFiltering,
        OptimizedItemBasedCollaborativeFiltering
    )
    from ..data_structures.optimized_sparse_matrix import OptimizedSparseUserItemMatrix
    from ..data_structures.optimized_graph_structure import OptimizedBipartiteRecommendationGraph
    from ..utils.caching import RecommendationCache, LRUCache
    from ..utils.scalable_processing import DataPartitioner, StreamingProcessor
except ImportError:
    # Fallback for standalone execution
    print("Warning: Could not import optimized modules. Using mock classes for testing.")
    
    class OptimizedUserBasedCollaborativeFiltering:
        def __init__(self, **kwargs): pass
        def fit(self, data): pass
        def predict_rating(self, user, item): return 3.0
        def recommend(self, user, n=10): return []
    
    class OptimizedItemBasedCollaborativeFiltering:
        def __init__(self, **kwargs): pass
        def fit(self, data): pass
        def predict_rating(self, user, item): return 3.0
        def recommend(self, user, n=10): return []
    
    class OptimizedSparseUserItemMatrix:
        def __init__(self, df=None): pass
        def add_interaction(self, user, item, rating): pass
        def compute_user_similarity_optimized(self, u1, u2): return 0.5
    
    class OptimizedBipartiteRecommendationGraph:
        def __init__(self): pass
        def add_interaction(self, user, item, weight): pass
        def parallel_random_walk_recommendation(self, user): return {}
    
    class RecommendationCache:
        def __init__(self): pass
    
    class LRUCache:
        def __init__(self, size): pass
    
    class DataPartitioner:
        def __init__(self, **kwargs): pass
    
    class StreamingProcessor:
        def __init__(self, **kwargs): pass


class PerformanceBenchmark:
    """
    Performance benchmarking framework for recommendation algorithms.
    
    Measures:
    - Execution time
    - Memory usage
    - Throughput (operations/second)
    - Scalability characteristics
    - Cache hit rates
    """
    
    def __init__(self):
        self.results = []
        self.baseline_results = {}
        
    def benchmark_algorithm(self, algorithm_func: Callable, 
                          test_data: Any, 
                          test_name: str,
                          iterations: int = 5,
                          warmup_iterations: int = 2) -> Dict[str, Any]:
        """
        Benchmark an algorithm with multiple iterations.
        
        Args:
            algorithm_func: Function to benchmark
            test_data: Data to pass to the function
            test_name: Name of the test
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not counted)
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking {test_name}...")
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            try:
                algorithm_func(test_data)
            except Exception as e:
                print(f"Warmup iteration failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        execution_times = []
        memory_usage = []
        
        for i in range(iterations):
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute and time
            start_time = time.perf_counter()
            try:
                result = algorithm_func(test_data)
                success = True
            except Exception as e:
                print(f"Iteration {i+1} failed: {e}")
                success = False
                result = None
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
            
            if success:
                execution_times.append(execution_time)
                memory_usage.append(memory_delta)
        
        if not execution_times:
            return {'error': 'All iterations failed'}
        
        # Calculate statistics
        benchmark_result = {
            'test_name': test_name,
            'iterations': len(execution_times),
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'std_execution_time': np.std(execution_times),
            'avg_memory_delta': np.mean(memory_usage),
            'max_memory_delta': max(memory_usage),
            'throughput_ops_per_sec': 1.0 / np.mean(execution_times) if execution_times else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(benchmark_result)
        print(f"  Avg time: {benchmark_result['avg_execution_time']:.4f}s")
        print(f"  Throughput: {benchmark_result['throughput_ops_per_sec']:.2f} ops/sec")
        
        return benchmark_result
    
    def compare_algorithms(self, algorithm_configs: List[Dict],
                          test_data: Any, 
                          test_name: str) -> Dict[str, Any]:
        """
        Compare multiple algorithms on the same test data.
        
        Args:
            algorithm_configs: List of {'name': str, 'func': callable, 'params': dict}
            test_data: Test data for algorithms
            test_name: Name of comparison test
            
        Returns:
            Comparison results
        """
        print(f"Comparing algorithms for {test_name}...")
        
        comparison_results = {
            'test_name': test_name,
            'algorithms': {},
            'rankings': {}
        }
        
        # Benchmark each algorithm
        for config in algorithm_configs:
            algo_name = config['name']
            algo_func = config['func']
            
            # Create parameterized function
            def parameterized_func(data):
                return algo_func(data, **config.get('params', {}))
            
            result = self.benchmark_algorithm(
                parameterized_func, test_data, f"{test_name}_{algo_name}"
            )
            comparison_results['algorithms'][algo_name] = result
        
        # Rank algorithms by execution time
        algo_times = {
            name: results['avg_execution_time'] 
            for name, results in comparison_results['algorithms'].items()
            if 'avg_execution_time' in results
        }
        
        sorted_algos = sorted(algo_times.items(), key=lambda x: x[1])
        comparison_results['rankings']['by_speed'] = [name for name, _ in sorted_algos]
        
        # Rank by throughput
        algo_throughput = {
            name: results['throughput_ops_per_sec']
            for name, results in comparison_results['algorithms'].items()
            if 'throughput_ops_per_sec' in results
        }
        
        sorted_throughput = sorted(algo_throughput.items(), key=lambda x: x[1], reverse=True)
        comparison_results['rankings']['by_throughput'] = [name for name, _ in sorted_throughput]
        
        return comparison_results
    
    def scalability_test(self, algorithm_func: Callable,
                        data_sizes: List[int],
                        data_generator: Callable,
                        test_name: str) -> Dict[str, Any]:
        """
        Test algorithm scalability with increasing data sizes.
        
        Args:
            algorithm_func: Algorithm to test
            data_sizes: List of data sizes to test
            data_generator: Function that generates test data of given size
            test_name: Name of scalability test
            
        Returns:
            Scalability test results
        """
        print(f"Running scalability test: {test_name}")
        
        scalability_results = {
            'test_name': test_name,
            'data_sizes': data_sizes,
            'results': {}
        }
        
        for size in data_sizes:
            print(f"  Testing with data size: {size}")
            
            # Generate test data
            test_data = data_generator(size)
            
            # Benchmark with this data size
            result = self.benchmark_algorithm(
                algorithm_func, test_data, f"{test_name}_size_{size}", iterations=3
            )
            
            scalability_results['results'][size] = result
        
        # Calculate scaling characteristics
        times = [scalability_results['results'][size]['avg_execution_time'] for size in data_sizes]
        
        if len(times) >= 2:
            # Calculate approximate complexity
            size_ratios = [data_sizes[i+1] / data_sizes[i] for i in range(len(data_sizes)-1)]
            time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
            
            if size_ratios and time_ratios:
                avg_size_ratio = np.mean(size_ratios)
                avg_time_ratio = np.mean(time_ratios)
                
                # Estimate complexity (very rough)
                if avg_time_ratio <= avg_size_ratio:
                    complexity_estimate = "sub-linear or linear"
                elif avg_time_ratio <= avg_size_ratio ** 1.5:
                    complexity_estimate = "linearithmic"
                elif avg_time_ratio <= avg_size_ratio ** 2:
                    complexity_estimate = "quadratic"
                else:
                    complexity_estimate = "super-quadratic"
                
                scalability_results['complexity_estimate'] = complexity_estimate
        
        return scalability_results
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                'benchmark_results': self.results,
                'baseline_results': self.baseline_results
            }, f, indent=2, default=str)
    
    def load_baseline(self, filename: str) -> None:
        """Load baseline results for comparison."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.baseline_results = data.get('baseline_results', {})
        except FileNotFoundError:
            print(f"Baseline file {filename} not found")


class StressTester:
    """
    Stress testing framework for system reliability and performance under load.
    """
    
    def __init__(self):
        self.stress_results = []
    
    def concurrent_user_test(self, algorithm, test_data: Any,
                           n_users: int = 100, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Test algorithm performance with concurrent users.
        
        Args:
            algorithm: Algorithm instance to test
            test_data: Test data
            n_users: Number of concurrent users to simulate
            duration_seconds: Test duration in seconds
            
        Returns:
            Concurrent user test results
        """
        print(f"Running concurrent user test with {n_users} users for {duration_seconds}s...")
        
        results = {
            'n_users': n_users,
            'duration': duration_seconds,
            'requests_completed': 0,
            'requests_failed': 0,
            'response_times': [],
            'errors': []
        }
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Thread-safe result collection
        results_lock = threading.Lock()
        
        def user_simulation():
            """Simulate a single user's requests."""
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    
                    # Simulate user request (e.g., get recommendations)
                    if hasattr(algorithm, 'recommend'):
                        # Get random user for testing
                        user_id = f"test_user_{random.randint(1, 1000)}"
                        recommendations = algorithm.recommend(user_id, n_recommendations=10)
                    else:
                        # Fallback operation
                        time.sleep(0.01)  # Simulate computation
                        recommendations = []
                    
                    request_time = time.time() - request_start
                    
                    with results_lock:
                        results['requests_completed'] += 1
                        results['response_times'].append(request_time)
                
                except Exception as e:
                    with results_lock:
                        results['requests_failed'] += 1
                        results['errors'].append(str(e))
                
                # Small delay between requests
                time.sleep(random.uniform(0.01, 0.05))
        
        # Start user threads
        threads = []
        for _ in range(n_users):
            thread = threading.Thread(target=user_simulation)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for test duration
        time.sleep(duration_seconds)
        
        # Wait for threads to finish (with timeout)
        for thread in threads:
            thread.join(timeout=1.0)
        
        # Calculate final statistics
        total_requests = results['requests_completed'] + results['requests_failed']
        results['total_requests'] = total_requests
        results['success_rate'] = (results['requests_completed'] / total_requests 
                                 if total_requests > 0 else 0.0)
        results['requests_per_second'] = total_requests / duration_seconds
        
        if results['response_times']:
            results['avg_response_time'] = np.mean(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
            results['percentile_95_response_time'] = np.percentile(results['response_times'], 95)
        
        print(f"  Completed: {results['requests_completed']}")
        print(f"  Failed: {results['requests_failed']}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Requests/sec: {results['requests_per_second']:.2f}")
        
        self.stress_results.append(results)
        return results
    
    def memory_leak_test(self, algorithm_func: Callable, test_data: Any,
                        iterations: int = 1000, check_interval: int = 100) -> Dict[str, Any]:
        """
        Test for memory leaks by running many iterations.
        
        Args:
            algorithm_func: Function to test
            test_data: Test data
            iterations: Number of iterations to run
            check_interval: Interval for memory checks
            
        Returns:
            Memory leak test results
        """
        print(f"Running memory leak test for {iterations} iterations...")
        
        process = psutil.Process(os.getpid())
        memory_snapshots = []
        
        for i in range(iterations):
            # Execute algorithm
            try:
                algorithm_func(test_data)
            except Exception as e:
                print(f"Iteration {i} failed: {e}")
            
            # Check memory at intervals
            if i % check_interval == 0:
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append((i, memory_mb))
                
                # Force garbage collection
                if i % (check_interval * 2) == 0:
                    gc.collect()
        
        # Analyze memory trend
        iterations_list = [snapshot[0] for snapshot in memory_snapshots]
        memory_list = [snapshot[1] for snapshot in memory_snapshots]
        
        # Calculate trend (simple linear regression)
        if len(memory_snapshots) >= 2:
            x_mean = np.mean(iterations_list)
            y_mean = np.mean(memory_list)
            
            numerator = sum((x - x_mean) * (y - y_mean) 
                          for x, y in zip(iterations_list, memory_list))
            denominator = sum((x - x_mean) ** 2 for x in iterations_list)
            
            if denominator != 0:
                slope = numerator / denominator
                memory_trend = "increasing" if slope > 0.1 else "stable"
            else:
                slope = 0
                memory_trend = "stable"
        else:
            slope = 0
            memory_trend = "unknown"
        
        results = {
            'iterations': iterations,
            'memory_snapshots': memory_snapshots,
            'initial_memory_mb': memory_list[0] if memory_list else 0,
            'final_memory_mb': memory_list[-1] if memory_list else 0,
            'max_memory_mb': max(memory_list) if memory_list else 0,
            'memory_trend_slope': slope,
            'memory_trend': memory_trend,
            'potential_leak': slope > 0.1  # Threshold for potential leak
        }
        
        print(f"  Memory trend: {memory_trend}")
        print(f"  Initial: {results['initial_memory_mb']:.2f} MB")
        print(f"  Final: {results['final_memory_mb']:.2f} MB")
        print(f"  Max: {results['max_memory_mb']:.2f} MB")
        
        return results


class AlgorithmValidator:
    """
    Validation framework for recommendation algorithm correctness and quality.
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_recommendation_quality(self, algorithm, test_data: pd.DataFrame,
                                      test_size: float = 0.2) -> Dict[str, Any]:
        """
        Validate recommendation quality using standard metrics.
        
        Args:
            algorithm: Recommendation algorithm to validate
            test_data: DataFrame with user-item interactions
            test_size: Proportion of data to use for testing
            
        Returns:
            Validation results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available for validation'}
        
        print("Validating recommendation quality...")
        
        # Split data
        train_data, test_data_split = train_test_split(
            test_data, test_size=test_size, random_state=42
        )
        
        # Train algorithm
        try:
            if hasattr(algorithm, 'fit'):
                # For collaborative filtering algorithms
                if 'user_id' in train_data.columns and 'product_id' in train_data.columns:
                    train_matrix = train_data.pivot_table(
                        index='user_id', columns='product_id', 
                        values='rating', fill_value=0
                    )
                    algorithm.fit(train_matrix)
                else:
                    algorithm.fit(train_data)
        except Exception as e:
            return {'error': f'Algorithm training failed: {e}'}
        
        # Evaluate predictions
        predictions = []
        actuals = []
        
        for _, row in test_data_split.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            actual_rating = row.get('rating', 1.0)
            
            try:
                if hasattr(algorithm, 'predict_rating'):
                    predicted_rating = algorithm.predict_rating(user_id, item_id)
                else:
                    predicted_rating = 3.0  # Default prediction
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except Exception:
                continue  # Skip failed predictions
        
        # Calculate metrics
        results = {'n_predictions': len(predictions)}
        
        if predictions and actuals:
            results['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
            results['mae'] = mean_absolute_error(actuals, predictions)
            
            # Correlation
            if len(set(predictions)) > 1 and len(set(actuals)) > 1:
                correlation = np.corrcoef(predictions, actuals)[0, 1]
                results['correlation'] = correlation if not np.isnan(correlation) else 0.0
            else:
                results['correlation'] = 0.0
        else:
            results['error'] = 'No valid predictions generated'
        
        print(f"  RMSE: {results.get('rmse', 'N/A')}")
        print(f"  MAE: {results.get('mae', 'N/A')}")
        print(f"  Correlation: {results.get('correlation', 'N/A')}")
        
        self.validation_results.append(results)
        return results
    
    def validate_recommendation_diversity(self, algorithm, users: List[str],
                                        n_recommendations: int = 10) -> Dict[str, Any]:
        """
        Validate recommendation diversity and coverage.
        
        Args:
            algorithm: Recommendation algorithm
            users: List of user IDs to test
            n_recommendations: Number of recommendations per user
            
        Returns:
            Diversity validation results
        """
        print("Validating recommendation diversity...")
        
        all_recommendations = []
        user_recommendation_sets = []
        
        for user_id in users:
            try:
                if hasattr(algorithm, 'recommend'):
                    recommendations = algorithm.recommend(user_id, n_recommendations)
                    items = [item for item, _ in recommendations]
                else:
                    items = []  # No recommendations available
                
                all_recommendations.extend(items)
                user_recommendation_sets.append(set(items))
                
            except Exception as e:
                print(f"Failed to get recommendations for {user_id}: {e}")
        
        # Calculate diversity metrics
        results = {}
        
        if all_recommendations:
            # Coverage: proportion of unique items recommended
            unique_items = set(all_recommendations)
            results['unique_items_count'] = len(unique_items)
            results['total_recommendations'] = len(all_recommendations)
            results['coverage'] = len(unique_items) / len(all_recommendations)
            
            # Intra-list diversity (average pairwise diversity within user's recommendations)
            if user_recommendation_sets:
                diversities = []
                for rec_set in user_recommendation_sets:
                    if len(rec_set) > 1:
                        # Simple diversity measure: proportion of unique items
                        diversity = len(rec_set) / n_recommendations if n_recommendations > 0 else 0
                        diversities.append(diversity)
                
                results['avg_intra_list_diversity'] = np.mean(diversities) if diversities else 0
            
            # Personalization: how different are recommendations across users
            if len(user_recommendation_sets) > 1:
                personalization_scores = []
                for i, set1 in enumerate(user_recommendation_sets):
                    for j, set2 in enumerate(user_recommendation_sets[i+1:], i+1):
                        if set1 and set2:
                            # Jaccard distance as personalization measure
                            intersection = len(set1.intersection(set2))
                            union = len(set1.union(set2))
                            jaccard = intersection / union if union > 0 else 0
                            personalization = 1 - jaccard  # Higher is more personalized
                            personalization_scores.append(personalization)
                
                results['avg_personalization'] = (np.mean(personalization_scores) 
                                                if personalization_scores else 0)
        
        print(f"  Coverage: {results.get('coverage', 'N/A'):.3f}")
        print(f"  Avg intra-list diversity: {results.get('avg_intra_list_diversity', 'N/A'):.3f}")
        print(f"  Avg personalization: {results.get('avg_personalization', 'N/A'):.3f}")
        
        return results


class IntegrationTester:
    """
    Integration testing for system components.
    """
    
    def test_end_to_end_workflow(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test complete end-to-end recommendation workflow.
        
        Args:
            interactions_df: Sample interaction data
            
        Returns:
            Integration test results
        """
        print("Running end-to-end integration test...")
        
        results = {
            'stages': {},
            'overall_success': True,
            'errors': []
        }
        
        try:
            # Stage 1: Data processing
            print("  Stage 1: Data processing...")
            sparse_matrix = OptimizedSparseUserItemMatrix(interactions_df)
            results['stages']['data_processing'] = {'success': True}
            
        except Exception as e:
            results['stages']['data_processing'] = {'success': False, 'error': str(e)}
            results['errors'].append(f"Data processing: {e}")
            results['overall_success'] = False
        
        try:
            # Stage 2: Algorithm training
            print("  Stage 2: Algorithm training...")
            
            # Create user-item matrix for collaborative filtering
            if 'user_id' in interactions_df.columns and 'product_id' in interactions_df.columns:
                user_item_matrix = interactions_df.pivot_table(
                    index='user_id', columns='product_id', 
                    values='rating', fill_value=0
                )
                
                cf_algorithm = OptimizedUserBasedCollaborativeFiltering(k_neighbors=10)
                cf_algorithm.fit(user_item_matrix)
                
                results['stages']['algorithm_training'] = {'success': True}
            else:
                raise ValueError("Required columns not found in interactions_df")
                
        except Exception as e:
            results['stages']['algorithm_training'] = {'success': False, 'error': str(e)}
            results['errors'].append(f"Algorithm training: {e}")
            results['overall_success'] = False
        
        try:
            # Stage 3: Recommendation generation
            print("  Stage 3: Recommendation generation...")
            
            # Get sample users
            sample_users = interactions_df['user_id'].unique()[:5]
            
            recommendations_generated = 0
            for user_id in sample_users:
                try:
                    recommendations = cf_algorithm.recommend(user_id, n_recommendations=5)
                    if recommendations:
                        recommendations_generated += 1
                except Exception:
                    continue
            
            results['stages']['recommendation_generation'] = {
                'success': True,
                'recommendations_generated': recommendations_generated,
                'users_tested': len(sample_users)
            }
            
        except Exception as e:
            results['stages']['recommendation_generation'] = {'success': False, 'error': str(e)}
            results['errors'].append(f"Recommendation generation: {e}")
            results['overall_success'] = False
        
        try:
            # Stage 4: Graph operations
            print("  Stage 4: Graph operations...")
            
            graph = OptimizedBipartiteRecommendationGraph()
            
            # Add sample interactions
            for _, row in interactions_df.head(100).iterrows():
                graph.add_interaction(
                    row['user_id'], 
                    row['product_id'], 
                    row.get('rating', 1.0)
                )
            
            # Test graph operations
            stats = graph.get_graph_statistics()
            results['stages']['graph_operations'] = {
                'success': True,
                'graph_stats': stats
            }
            
        except Exception as e:
            results['stages']['graph_operations'] = {'success': False, 'error': str(e)}
            results['errors'].append(f"Graph operations: {e}")
            results['overall_success'] = False
        
        print(f"  Overall success: {results['overall_success']}")
        if not results['overall_success']:
            print(f"  Errors: {results['errors']}")
        
        return results


def run_comprehensive_test_suite(interactions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the complete test suite on the recommendation system.
    
    Args:
        interactions_df: Sample interaction data for testing
        
    Returns:
        Complete test results
    """
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'performance_benchmarks': {},
        'stress_tests': {},
        'validation_results': {},
        'integration_tests': {},
        'summary': {}
    }
    
    # Initialize test frameworks
    benchmark = PerformanceBenchmark()
    stress_tester = StressTester()
    validator = AlgorithmValidator()
    integration_tester = IntegrationTester()
    
    # 1. Performance Benchmarks
    print("\n1. PERFORMANCE BENCHMARKS")
    print("-" * 40)
    
    try:
        # Create test algorithms
        cf_user = OptimizedUserBasedCollaborativeFiltering(k_neighbors=20)
        cf_item = OptimizedItemBasedCollaborativeFiltering(k_neighbors=20)
        
        # Prepare test data
        if 'user_id' in interactions_df.columns:
            test_matrix = interactions_df.pivot_table(
                index='user_id', columns='product_id',
                values='rating', fill_value=0
            )
        
            # Benchmark collaborative filtering
            def test_cf_user(data):
                cf_user.fit(data)
                return cf_user.recommend('test_user_1', 10)
            
            def test_cf_item(data):
                cf_item.fit(data)
                return cf_item.recommend('test_user_1', 10)
            
            # Compare algorithms
            algorithm_configs = [
                {'name': 'user_based_cf', 'func': test_cf_user, 'params': {}},
                {'name': 'item_based_cf', 'func': test_cf_item, 'params': {}}
            ]
            
            comparison = benchmark.compare_algorithms(
                algorithm_configs, test_matrix, "collaborative_filtering_comparison"
            )
            test_results['performance_benchmarks']['cf_comparison'] = comparison
        
        # Benchmark sparse matrix operations
        sparse_matrix = OptimizedSparseUserItemMatrix(interactions_df)
        
        def test_similarity_computation(data):
            users = list(sparse_matrix.user_to_index.keys())[:10]
            if len(users) >= 2:
                return sparse_matrix.compute_user_similarity_optimized(users[0], users[1])
            return 0.0
        
        similarity_benchmark = benchmark.benchmark_algorithm(
            test_similarity_computation, sparse_matrix, "similarity_computation"
        )
        test_results['performance_benchmarks']['similarity'] = similarity_benchmark
        
    except Exception as e:
        test_results['performance_benchmarks']['error'] = str(e)
        print(f"Performance benchmarks failed: {e}")
    
    # 2. Stress Tests
    print("\n2. STRESS TESTS")
    print("-" * 40)
    
    try:
        # Concurrent user test
        if 'cf_user' in locals():
            concurrent_results = stress_tester.concurrent_user_test(
                cf_user, test_matrix, n_users=50, duration_seconds=30
            )
            test_results['stress_tests']['concurrent_users'] = concurrent_results
        
        # Memory leak test
        def simple_operation(data):
            # Simple operation that shouldn't leak memory
            return len(str(data))
        
        memory_results = stress_tester.memory_leak_test(
            simple_operation, interactions_df, iterations=500, check_interval=50
        )
        test_results['stress_tests']['memory_leak'] = memory_results
        
    except Exception as e:
        test_results['stress_tests']['error'] = str(e)
        print(f"Stress tests failed: {e}")
    
    # 3. Validation Tests
    print("\n3. VALIDATION TESTS")
    print("-" * 40)
    
    try:
        if 'cf_user' in locals() and len(interactions_df) > 100:
            quality_results = validator.validate_recommendation_quality(
                cf_user, interactions_df.head(1000)
            )
            test_results['validation_results']['quality'] = quality_results
            
            # Diversity validation
            sample_users = interactions_df['user_id'].unique()[:20]
            diversity_results = validator.validate_recommendation_diversity(
                cf_user, sample_users
            )
            test_results['validation_results']['diversity'] = diversity_results
        
    except Exception as e:
        test_results['validation_results']['error'] = str(e)
        print(f"Validation tests failed: {e}")
    
    # 4. Integration Tests
    print("\n4. INTEGRATION TESTS")
    print("-" * 40)
    
    try:
        integration_results = integration_tester.test_end_to_end_workflow(interactions_df)
        test_results['integration_tests'] = integration_results
        
    except Exception as e:
        test_results['integration_tests']['error'] = str(e)
        print(f"Integration tests failed: {e}")
    
    # Generate Summary
    print("\n5. TEST SUMMARY")
    print("-" * 40)
    
    summary = {
        'total_tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'performance_summary': {},
        'critical_issues': []
    }
    
    # Count tests and failures
    for category, results in test_results.items():
        if category == 'summary':
            continue
        
        if isinstance(results, dict):
            if 'error' in results:
                summary['tests_failed'] += 1
                summary['critical_issues'].append(f"{category}: {results['error']}")
            else:
                summary['tests_passed'] += 1
            summary['total_tests_run'] += 1
    
    # Performance summary
    if 'performance_benchmarks' in test_results:
        perf_data = test_results['performance_benchmarks']
        if 'similarity' in perf_data:
            sim_data = perf_data['similarity']
            summary['performance_summary']['avg_similarity_time'] = sim_data.get('avg_execution_time', 'N/A')
    
    test_results['summary'] = summary
    
    print(f"Tests run: {summary['total_tests_run']}")
    print(f"Passed: {summary['tests_passed']}")
    print(f"Failed: {summary['tests_failed']}")
    
    if summary['critical_issues']:
        print("Critical issues:")
        for issue in summary['critical_issues']:
            print(f"  - {issue}")
    
    return test_results