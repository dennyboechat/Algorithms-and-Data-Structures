"""
Scalable Data Processing for Large-Scale Recommendation Systems

This module implements advanced data partitioning, streaming processing, and
distributed computing techniques for handling massive datasets that don't fit
in memory or require real-time processing capabilities.

Key features:
- Chunked data processing for memory efficiency
- Streaming data ingestion and processing
- Distributed computing with MapReduce patterns
- Incremental model updates
- Load balancing and fault tolerance

References:
- Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on 
  large clusters. Communications of the ACM, 51(1), 107-113.
- Zaharia, M., et al. (2010). Spark: Cluster computing with working sets. 
  HotCloud '10.
- Chen, C., et al. (2012). Large-scale behavioral targeting with a social twist. 
  CIKM '12.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Iterator, Callable, Any
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import gc
import pickle
import os
import hashlib
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

try:
    from ..utils.caching import get_global_cache
    from ..data_structures.optimized_sparse_matrix import OptimizedSparseUserItemMatrix
except ImportError:
    def get_global_cache():
        class DummyCache:
            def get(self, key): return None
            def put(self, key, value): pass
        return DummyCache()
    
    class OptimizedSparseUserItemMatrix:
        def __init__(self, *args, **kwargs): pass


class DataPartitioner:
    """
    Intelligent data partitioning for distributed processing.
    
    Implements partitioning strategies based on:
    - User-based partitioning (horizontal)
    - Item-based partitioning (vertical)  
    - Hash-based distribution
    - Range-based partitioning
    - Hybrid approaches for load balancing
    
    Based on techniques described in:
    Karau, H., et al. (2015). Learning Spark: Lightning-fast big data analysis. 
    O'Reilly Media.
    """
    
    def __init__(self, partitioning_strategy: str = 'hash', 
                 n_partitions: int = None, load_balance: bool = True):
        """
        Initialize data partitioner.
        
        Args:
            partitioning_strategy: 'hash', 'range', 'user_based', 'item_based', 'hybrid'
            n_partitions: Number of partitions (auto-detect if None)
            load_balance: Enable load balancing across partitions
        """
        self.strategy = partitioning_strategy
        self.n_partitions = n_partitions or mp.cpu_count()
        self.load_balance = load_balance
        
        # Partition metadata
        self.partition_info = {}
        self.partition_sizes = defaultdict(int)
        self.partition_stats = defaultdict(dict)
        
        # Load balancing
        self.partition_weights = defaultdict(float)
        self.rebalance_threshold = 0.3  # 30% imbalance triggers rebalancing
        
    def partition_interactions(self, interactions_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Partition user-item interactions based on strategy.
        
        Args:
            interactions_df: DataFrame with user-item interactions
            
        Returns:
            Dictionary mapping partition_id -> DataFrame
        """
        print(f"Partitioning {len(interactions_df)} interactions using {self.strategy} strategy...")
        
        partitions = {}
        
        if self.strategy == 'hash':
            partitions = self._hash_partition(interactions_df, 'user_id')
        elif self.strategy == 'user_based':
            partitions = self._user_based_partition(interactions_df)
        elif self.strategy == 'item_based':
            partitions = self._item_based_partition(interactions_df)
        elif self.strategy == 'range':
            partitions = self._range_partition(interactions_df, 'user_id')
        elif self.strategy == 'hybrid':
            partitions = self._hybrid_partition(interactions_df)
        else:
            raise ValueError(f"Unknown partitioning strategy: {self.strategy}")
        
        # Collect partition statistics
        self._update_partition_stats(partitions)
        
        # Apply load balancing if enabled
        if self.load_balance:
            partitions = self._rebalance_partitions(partitions)
        
        print(f"Created {len(partitions)} partitions with sizes: {dict(self.partition_sizes)}")
        return partitions
    
    def _hash_partition(self, df: pd.DataFrame, column: str) -> Dict[int, pd.DataFrame]:
        """Hash-based partitioning for even distribution."""
        partitions = defaultdict(list)
        
        for _, row in df.iterrows():
            # Hash the column value to determine partition
            hash_value = hashlib.md5(str(row[column]).encode()).hexdigest()
            partition_id = int(hash_value, 16) % self.n_partitions
            partitions[partition_id].append(row)
        
        # Convert to DataFrames
        return {pid: pd.DataFrame(rows) for pid, rows in partitions.items() if rows}
    
    def _user_based_partition(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """User-based partitioning keeping user data together."""
        user_partitions = defaultdict(int)
        partitions = defaultdict(list)
        
        # Assign users to partitions
        unique_users = df['user_id'].unique()
        for i, user_id in enumerate(unique_users):
            partition_id = i % self.n_partitions
            user_partitions[user_id] = partition_id
        
        # Assign interactions to partitions based on user
        for _, row in df.iterrows():
            partition_id = user_partitions[row['user_id']]
            partitions[partition_id].append(row)
        
        return {pid: pd.DataFrame(rows) for pid, rows in partitions.items() if rows}
    
    def _item_based_partition(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Item-based partitioning keeping item data together."""
        item_partitions = defaultdict(int)
        partitions = defaultdict(list)
        
        # Assign items to partitions
        unique_items = df['product_id'].unique()
        for i, item_id in enumerate(unique_items):
            partition_id = i % self.n_partitions
            item_partitions[item_id] = partition_id
        
        # Assign interactions to partitions based on item
        for _, row in df.iterrows():
            partition_id = item_partitions[row['product_id']]
            partitions[partition_id].append(row)
        
        return {pid: pd.DataFrame(rows) for pid, rows in partitions.items() if rows}
    
    def _range_partition(self, df: pd.DataFrame, column: str) -> Dict[int, pd.DataFrame]:
        """Range-based partitioning for ordered data."""
        # Sort by column and split into equal ranges
        sorted_df = df.sort_values(column)
        chunk_size = len(sorted_df) // self.n_partitions
        
        partitions = {}
        for i in range(self.n_partitions):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.n_partitions - 1 else len(sorted_df)
            partitions[i] = sorted_df.iloc[start_idx:end_idx].copy()
        
        return partitions
    
    def _hybrid_partition(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Hybrid partitioning combining multiple strategies.
        
        Uses user-based partitioning with hash-based load balancing.
        """
        # Start with user-based partitioning
        user_partitions = self._user_based_partition(df)
        
        # Calculate load imbalance
        sizes = [len(partition) for partition in user_partitions.values()]
        max_size = max(sizes)
        min_size = min(sizes)
        imbalance = (max_size - min_size) / max_size if max_size > 0 else 0
        
        # If imbalanced, apply hash-based redistribution
        if imbalance > self.rebalance_threshold:
            print(f"Load imbalance detected ({imbalance:.2%}), applying hash redistribution...")
            return self._hash_partition(df, 'user_id')
        
        return user_partitions
    
    def _update_partition_stats(self, partitions: Dict[int, pd.DataFrame]) -> None:
        """Update partition statistics and metadata."""
        for partition_id, partition_df in partitions.items():
            self.partition_sizes[partition_id] = len(partition_df)
            
            # Collect detailed statistics
            self.partition_stats[partition_id] = {
                'n_interactions': len(partition_df),
                'n_users': partition_df['user_id'].nunique(),
                'n_items': partition_df['product_id'].nunique(),
                'density': self._calculate_partition_density(partition_df),
                'avg_rating': partition_df.get('rating', pd.Series([0])).mean()
            }
    
    def _calculate_partition_density(self, partition_df: pd.DataFrame) -> float:
        """Calculate interaction density for a partition."""
        n_users = partition_df['user_id'].nunique()
        n_items = partition_df['product_id'].nunique()
        n_interactions = len(partition_df)
        
        if n_users == 0 or n_items == 0:
            return 0.0
        
        return n_interactions / (n_users * n_items)
    
    def _rebalance_partitions(self, partitions: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Rebalance partitions to reduce load skew.
        
        Implements the load balancing algorithm described in:
        Gufler, B., et al. (2012). Load balancing in MapReduce based on scalable 
        cardinality estimates. ICDE '12.
        """
        # Calculate current load distribution
        sizes = [len(partition) for partition in partitions.values()]
        if not sizes:
            return partitions
        
        avg_size = sum(sizes) / len(sizes)
        
        # Find overloaded and underloaded partitions
        overloaded = []
        underloaded = []
        
        for partition_id, size in enumerate(sizes):
            if size > avg_size * (1 + self.rebalance_threshold):
                overloaded.append((partition_id, size))
            elif size < avg_size * (1 - self.rebalance_threshold):
                underloaded.append((partition_id, size))
        
        # Rebalance if needed
        if overloaded and underloaded:
            print(f"Rebalancing {len(overloaded)} overloaded partitions...")
            partitions = self._redistribute_data(partitions, overloaded, underloaded)
        
        return partitions
    
    def _redistribute_data(self, partitions: Dict[int, pd.DataFrame],
                          overloaded: List[Tuple[int, int]], 
                          underloaded: List[Tuple[int, int]]) -> Dict[int, pd.DataFrame]:
        """Redistribute data between partitions for load balancing."""
        # Simple redistribution: move excess data from overloaded to underloaded
        for overloaded_id, overloaded_size in overloaded:
            if not underloaded:
                break
            
            # Calculate excess data to move
            avg_size = sum(len(p) for p in partitions.values()) / len(partitions)
            excess = overloaded_size - int(avg_size)
            
            if excess <= 0:
                continue
            
            # Find best underloaded partition
            underloaded_id, underloaded_size = underloaded.pop(0)
            
            # Move data
            overloaded_df = partitions[overloaded_id]
            data_to_move = overloaded_df.tail(excess).copy()
            
            partitions[overloaded_id] = overloaded_df.head(len(overloaded_df) - excess)
            partitions[underloaded_id] = pd.concat([partitions[underloaded_id], data_to_move], 
                                                  ignore_index=True)
        
        return partitions
    
    def get_partition_info(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed information about all partitions."""
        return dict(self.partition_stats)


class StreamingProcessor:
    """
    Real-time streaming processor for continuous data ingestion.
    
    Handles:
    - Continuous data streams
    - Incremental model updates
    - Real-time recommendations
    - Stream windowing and aggregation
    - Fault tolerance and recovery
    
    Based on streaming processing concepts from:
    Akidau, T., et al. (2015). The dataflow model: A practical approach to 
    balancing correctness, latency, and cost in massive-scale, unbounded, 
    out-of-order data processing. VLDB '15.
    """
    
    def __init__(self, window_size: int = 1000, buffer_size: int = 10000,
                 update_interval: float = 60.0, max_workers: int = 4):
        """
        Initialize streaming processor.
        
        Args:
            window_size: Size of processing window
            buffer_size: Maximum buffer size before processing
            update_interval: Model update interval (seconds)
            max_workers: Number of worker threads
        """
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.max_workers = max_workers
        
        # Streaming infrastructure
        self.data_buffer = deque(maxlen=buffer_size)
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Threading and synchronization
        self.is_running = False
        self.workers = []
        self.lock = threading.Lock()
        
        # Model and state
        self.current_model = None
        self.window_data = deque(maxlen=window_size)
        self.last_update = time.time()
        
        # Statistics
        self.processed_items = 0
        self.processing_times = []
        self.throughput_history = []
        
        # Cache for real-time access
        self.cache = get_global_cache()
    
    def start_processing(self) -> None:
        """Start the streaming processor."""
        if self.is_running:
            return
        
        print(f"Starting streaming processor with {self.max_workers} workers...")
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        # Start main processing loop
        main_thread = threading.Thread(target=self._main_loop)
        main_thread.daemon = True
        main_thread.start()
        self.workers.append(main_thread)
    
    def stop_processing(self) -> None:
        """Stop the streaming processor."""
        print("Stopping streaming processor...")
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
        
        self.workers.clear()
    
    def add_interaction(self, user_id: str, item_id: str, rating: float = None,
                       timestamp: float = None) -> None:
        """
        Add new interaction to the stream.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            rating: Interaction rating (optional)
            timestamp: Interaction timestamp (optional, uses current time)
        """
        interaction = {
            'user_id': user_id,
            'product_id': item_id,
            'rating': rating or 1.0,
            'timestamp': timestamp or time.time(),
            'interaction_type': 'rating' if rating else 'view'
        }
        
        with self.lock:
            self.data_buffer.append(interaction)
            
        # Trigger processing if buffer is full
        if len(self.data_buffer) >= self.buffer_size:
            self._trigger_processing()
    
    def _trigger_processing(self) -> None:
        """Trigger batch processing of buffered data."""
        with self.lock:
            if self.data_buffer:
                # Move buffer data to processing queue
                batch_data = list(self.data_buffer)
                self.data_buffer.clear()
                self.processing_queue.put(batch_data)
    
    def _main_loop(self) -> None:
        """Main processing loop for streaming data."""
        while self.is_running:
            try:
                # Check for periodic model updates
                current_time = time.time()
                if current_time - self.last_update >= self.update_interval:
                    self._update_model()
                    self.last_update = current_time
                
                # Trigger processing for partial buffers
                if len(self.data_buffer) > 0:
                    time.sleep(1.0)  # Wait for more data
                    if len(self.data_buffer) >= self.window_size // 2:
                        self._trigger_processing()
                
                time.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                print(f"Error in main processing loop: {e}")
                time.sleep(1.0)
    
    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread loop for processing batches."""
        while self.is_running:
            try:
                # Get batch from queue (with timeout)
                batch_data = self.processing_queue.get(timeout=1.0)
                
                # Process batch
                start_time = time.time()
                self._process_batch(batch_data, worker_id)
                processing_time = time.time() - start_time
                
                # Update statistics
                with self.lock:
                    self.processed_items += len(batch_data)
                    self.processing_times.append(processing_time)
                    
                    # Calculate throughput
                    throughput = len(batch_data) / processing_time if processing_time > 0 else 0
                    self.throughput_history.append(throughput)
                    
                    # Keep only recent history
                    if len(self.throughput_history) > 100:
                        self.throughput_history = self.throughput_history[-50:]
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
                time.sleep(1.0)
    
    def _process_batch(self, batch_data: List[Dict], worker_id: int) -> None:
        """
        Process a batch of interactions.
        
        Args:
            batch_data: List of interaction dictionaries
            worker_id: ID of processing worker
        """
        if not batch_data:
            return
        
        # Convert to DataFrame for processing
        batch_df = pd.DataFrame(batch_data)
        
        # Add to sliding window
        with self.lock:
            self.window_data.extend(batch_data)
        
        # Update model incrementally if possible
        if self.current_model and hasattr(self.current_model, 'add_interactions'):
            try:
                for interaction in batch_data:
                    self.current_model.add_interaction(
                        interaction['user_id'],
                        interaction['product_id'], 
                        interaction['rating']
                    )
            except Exception as e:
                print(f"Error in incremental model update: {e}")
        
        # Cache recent interactions for real-time access
        self._cache_recent_interactions(batch_data)
    
    def _cache_recent_interactions(self, batch_data: List[Dict]) -> None:
        """Cache recent interactions for fast access."""
        for interaction in batch_data:
            user_id = interaction['user_id']
            item_id = interaction['product_id']
            
            # Cache user's recent items
            recent_items_key = f"recent_items_{user_id}"
            recent_items = self.cache.get_user_similarities(recent_items_key, 'recent') or []
            
            # Add new item and keep last N items
            recent_items.append(item_id)
            recent_items = recent_items[-50:]  # Keep last 50 items
            
            self.cache.put_user_similarities(recent_items_key, 'recent', recent_items)
    
    def _update_model(self) -> None:
        """Periodic model update with accumulated window data."""
        if not self.window_data:
            return
        
        print(f"Updating model with {len(self.window_data)} interactions...")
        
        try:
            # Convert window data to DataFrame
            window_df = pd.DataFrame(list(self.window_data))
            
            # Create or update sparse matrix
            if self.current_model is None:
                self.current_model = OptimizedSparseUserItemMatrix(window_df)
            else:
                # Incremental update (simplified)
                for interaction in self.window_data:
                    self.current_model.add_interaction(
                        interaction['user_id'],
                        interaction['product_id'],
                        interaction['rating']
                    )
            
            print(f"Model updated successfully")
            
        except Exception as e:
            print(f"Error updating model: {e}")
    
    def get_real_time_recommendations(self, user_id: str, n_recs: int = 10) -> List[Tuple[str, float]]:
        """
        Get real-time recommendations for a user.
        
        Uses cached data and current model for fast response.
        """
        if self.current_model is None:
            return []
        
        try:
            # Check for cached recommendations
            cached_recs = self.cache.get_recommendations(user_id, "streaming", n_recs)
            if cached_recs is not None:
                return cached_recs
            
            # Get user's recent interactions
            recent_items_key = f"recent_items_{user_id}"
            recent_items = self.cache.get_user_similarities(recent_items_key, 'recent') or []
            
            # Simple recommendation based on recent activity and model
            # This is a simplified version - in practice you'd use more sophisticated methods
            recommendations = []
            
            if hasattr(self.current_model, 'get_user_vector'):
                # Use existing model for recommendations
                try:
                    user_vector = self.current_model.get_user_vector(user_id)
                    if len(user_vector) > 0:
                        # Find similar items to user's recent interactions
                        for item_id in recent_items[-5:]:  # Last 5 items
                            similar_items = self.current_model.compute_top_k_similar_items(
                                item_id, k=3, method='cosine'
                            )
                            for sim_item, score in similar_items:
                                recommendations.append((sim_item, score))
                except Exception as e:
                    print(f"Error getting recommendations from model: {e}")
            
            # Remove duplicates and sort
            seen_items = set(recent_items)
            unique_recs = []
            for item, score in recommendations:
                if item not in seen_items:
                    unique_recs.append((item, score))
                    seen_items.add(item)
            
            unique_recs.sort(key=lambda x: x[1], reverse=True)
            final_recs = unique_recs[:n_recs]
            
            # Cache results
            self.cache.put_recommendations(user_id, "streaming", n_recs, final_recs)
            
            return final_recs
            
        except Exception as e:
            print(f"Error getting real-time recommendations: {e}")
            return []
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming processor statistics."""
        current_time = time.time()
        uptime = current_time - self.last_update if hasattr(self, 'start_time') else 0
        
        avg_throughput = (np.mean(self.throughput_history) 
                         if self.throughput_history else 0.0)
        
        return {
            'is_running': self.is_running,
            'processed_items': self.processed_items,
            'buffer_size': len(self.data_buffer),
            'window_size': len(self.window_data),
            'avg_processing_time': (np.mean(self.processing_times) 
                                  if self.processing_times else 0.0),
            'avg_throughput': avg_throughput,
            'uptime_seconds': uptime,
            'active_workers': len([w for w in self.workers if w.is_alive()])
        }


class DistributedProcessor:
    """
    Distributed processing for large-scale recommendation systems.
    
    Implements MapReduce patterns and distributed algorithms for:
    - Parallel similarity computations
    - Distributed matrix factorization
    - Fault-tolerant processing
    - Load balancing across workers
    
    Based on distributed computing patterns from:
    White, T. (2012). Hadoop: The definitive guide. O'Reilly Media.
    """
    
    def __init__(self, n_workers: int = None, use_processes: bool = True):
        """
        Initialize distributed processor.
        
        Args:
            n_workers: Number of worker processes/threads
            use_processes: Use processes instead of threads for CPU-bound tasks
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.use_processes = use_processes
        
        # Job management
        self.active_jobs = {}
        self.job_results = {}
        self.job_counter = 0
        
        # Performance tracking
        self.job_times = []
        self.worker_stats = defaultdict(list)
        
    def parallel_similarity_computation(self, matrix: OptimizedSparseUserItemMatrix,
                                      entity_pairs: List[Tuple[str, str]],
                                      similarity_method: str = 'cosine') -> Dict[Tuple[str, str], float]:
        """
        Compute similarities for entity pairs in parallel.
        
        Args:
            matrix: Sparse matrix instance
            entity_pairs: List of (entity1, entity2) pairs
            similarity_method: Similarity computation method
            
        Returns:
            Dictionary mapping pairs to similarity scores
        """
        if not entity_pairs:
            return {}
        
        print(f"Computing {len(entity_pairs)} similarities using {self.n_workers} workers...")
        
        # Split pairs into chunks for workers
        chunk_size = max(1, len(entity_pairs) // self.n_workers)
        chunks = [entity_pairs[i:i + chunk_size] 
                 for i in range(0, len(entity_pairs), chunk_size)]
        
        # Choose executor type
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = {}
        start_time = time.time()
        
        with executor_class(max_workers=self.n_workers) as executor:
            # Submit jobs
            future_to_chunk = {
                executor.submit(
                    self._compute_similarities_chunk,
                    matrix, chunk, similarity_method
                ): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.update(chunk_results)
                except Exception as e:
                    print(f"Error in chunk {chunk_idx}: {e}")
        
        computation_time = time.time() - start_time
        self.job_times.append(computation_time)
        
        print(f"Parallel similarity computation completed in {computation_time:.2f}s")
        return results
    
    @staticmethod
    def _compute_similarities_chunk(matrix: OptimizedSparseUserItemMatrix,
                                   entity_pairs: List[Tuple[str, str]],
                                   similarity_method: str) -> Dict[Tuple[str, str], float]:
        """
        Compute similarities for a chunk of entity pairs.
        
        This is a static method to support multiprocessing.
        """
        results = {}
        
        for entity1, entity2 in entity_pairs:
            try:
                # Determine if entities are users or items based on matrix
                if (entity1 in matrix.user_to_index and 
                    entity2 in matrix.user_to_index):
                    similarity = matrix.compute_user_similarity_optimized(
                        entity1, entity2, similarity_method
                    )
                elif (entity1 in matrix.item_to_index and 
                      entity2 in matrix.item_to_index):
                    similarity = matrix.compute_item_similarity_optimized(
                        entity1, entity2, similarity_method  
                    )
                else:
                    similarity = 0.0
                
                results[(entity1, entity2)] = similarity
                
            except Exception as e:
                print(f"Error computing similarity for {entity1}, {entity2}: {e}")
                results[(entity1, entity2)] = 0.0
        
        return results
    
    def distributed_matrix_factorization(self, matrices: Dict[int, OptimizedSparseUserItemMatrix],
                                       n_factors: int = 50) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform matrix factorization across distributed partitions.
        
        Args:
            matrices: Dictionary mapping partition_id -> matrix
            n_factors: Number of latent factors
            
        Returns:
            Dictionary mapping partition_id -> (U, V) factor matrices
        """
        print(f"Performing distributed matrix factorization with {n_factors} factors...")
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = {}
        start_time = time.time()
        
        with executor_class(max_workers=self.n_workers) as executor:
            # Submit factorization jobs
            future_to_partition = {
                executor.submit(
                    self._factorize_matrix_partition,
                    partition_id, matrix, n_factors
                ): partition_id
                for partition_id, matrix in matrices.items()
            }
            
            # Collect results
            for future in as_completed(future_to_partition):
                partition_id = future_to_partition[future]
                try:
                    factors = future.result()
                    results[partition_id] = factors
                except Exception as e:
                    print(f"Error in matrix factorization for partition {partition_id}: {e}")
        
        computation_time = time.time() - start_time
        print(f"Distributed matrix factorization completed in {computation_time:.2f}s")
        
        return results
    
    @staticmethod
    def _factorize_matrix_partition(partition_id: int, 
                                   matrix: OptimizedSparseUserItemMatrix,
                                   n_factors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform matrix factorization on a single partition.
        
        Static method to support multiprocessing.
        """
        try:
            U, V = matrix.matrix_factorization(n_factors)
            return U, V
        except Exception as e:
            print(f"Matrix factorization failed for partition {partition_id}: {e}")
            # Return zero matrices as fallback
            return (np.zeros((matrix.n_users, n_factors)), 
                   np.zeros((n_factors, matrix.n_items)))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get distributed processing performance statistics."""
        return {
            'n_workers': self.n_workers,
            'use_processes': self.use_processes,
            'completed_jobs': len(self.job_times),
            'avg_job_time': np.mean(self.job_times) if self.job_times else 0.0,
            'total_processing_time': sum(self.job_times),
            'active_jobs': len(self.active_jobs)
        }