"""
Enhanced Graph-Based Algorithms for Large-Scale Recommendation Systems

This module implements optimized graph algorithms with parallel processing,
approximate methods, and advanced graph mining techniques for scalable
recommendation systems handling millions of users and items.

Key optimizations:
- Parallel graph traversal algorithms
- Approximate graph algorithms for large-scale processing
- Advanced centrality measures
- Community detection algorithms
- Incremental graph updates
- Memory-efficient graph representations

References:
- Page, L., et al. (1999). The PageRank citation ranking: Bringing order to the web. 
  Stanford InfoLab Technical Report.
- Fortunato, S. (2010). Community detection in graphs. Physics Reports, 486(3-5), 75-174.
- Leskovec, J., et al. (2014). Mining of massive datasets. Cambridge University Press.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator
from collections import defaultdict, deque
import heapq
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import networkx as nx
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    pd = None
    nx = None
    print("Warning: scipy/networkx not available. Using basic implementations.")

try:
    from ..utils.caching import memoize_with_timeout, get_global_cache
except ImportError:
    def memoize_with_timeout(timeout=3600, cache_size=1000):
        def decorator(func):
            return func
        return decorator
    
    def get_global_cache():
        class DummyCache:
            def get(self, key): return None
            def put(self, key, value): pass
        return DummyCache()


class OptimizedBipartiteRecommendationGraph:
    """
    Highly optimized bipartite graph for recommendation systems.
    
    Features:
    - Memory-efficient adjacency list representation
    - Parallel graph algorithms
    - Approximate methods for large-scale graphs
    - Incremental updates
    - Advanced centrality measures
    - Community detection
    
    Based on algorithms described in:
    Latapy, M., et al. (2008). Basic notions for the analysis of large two-mode 
    networks. Social Networks, 30(1), 31-48.
    """
    
    def __init__(self, use_sparse_matrix: bool = True, enable_caching: bool = True,
                 max_cache_size: int = 10000):
        """
        Initialize optimized bipartite graph.
        
        Args:
            use_sparse_matrix: Use sparse matrix for large graphs
            enable_caching: Enable caching for expensive computations
            max_cache_size: Maximum cache size
        """
        # Graph structure
        self.adjacency_list = defaultdict(dict)  # node -> {neighbor: weight}
        self.user_nodes = set()
        self.item_nodes = set()
        self.edge_weights = defaultdict(float)
        self.node_attributes = defaultdict(dict)
        
        # Sparse matrix representation for large graphs
        self.use_sparse_matrix = use_sparse_matrix
        self.sparse_matrix = None
        self.node_to_index = {}
        self.index_to_node = {}
        
        # Caching and optimization
        self.enable_caching = enable_caching
        self.cache = get_global_cache() if enable_caching else None
        self.max_cache_size = max_cache_size
        self._computation_cache = {}
        
        # Performance tracking
        self.algorithm_times = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Graph statistics (cached)
        self._stats_cache = {}
        self._stats_dirty = True
    
    def add_user(self, user_id: str, attributes: Optional[Dict] = None) -> None:
        """Add user node with thread safety."""
        with self._lock:
            self.user_nodes.add(user_id)
            if attributes:
                self.node_attributes[user_id].update(attributes)
            self._stats_dirty = True
    
    def add_item(self, item_id: str, attributes: Optional[Dict] = None) -> None:
        """Add item node with thread safety."""
        with self._lock:
            self.item_nodes.add(item_id)
            if attributes:
                self.node_attributes[item_id].update(attributes)
            self._stats_dirty = True
    
    def add_interaction(self, user_id: str, item_id: str, weight: float = 1.0,
                       interaction_type: str = 'rating') -> None:
        """
        Add interaction with optimized updates.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            weight: Interaction weight
            interaction_type: Type of interaction
        """
        with self._lock:
            # Add nodes if they don't exist
            self.user_nodes.add(user_id)
            self.item_nodes.add(item_id)
            
            # Add bidirectional edges
            self.adjacency_list[user_id][item_id] = weight
            self.adjacency_list[item_id][user_id] = weight
            
            # Store edge metadata
            edge_key = (user_id, item_id)
            self.edge_weights[edge_key] = weight
            
            # Mark statistics as dirty
            self._stats_dirty = True
            
            # Invalidate relevant caches
            if self.enable_caching:
                self._invalidate_node_cache(user_id)
                self._invalidate_node_cache(item_id)
    
    def build_sparse_representation(self) -> None:
        """
        Build sparse matrix representation for large graphs.
        
        This enables efficient linear algebra operations on the graph.
        """
        if not self.use_sparse_matrix:
            return
        
        print("Building sparse matrix representation...")
        start_time = time.time()
        
        all_nodes = list(self.user_nodes.union(self.item_nodes))
        n_nodes = len(all_nodes)
        
        # Create node mappings
        self.node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
        self.index_to_node = {idx: node for idx, node in enumerate(all_nodes)}
        
        # Build sparse matrix
        if SCIPY_AVAILABLE:
            row_indices = []
            col_indices = []
            data = []
            
            for node, neighbors in self.adjacency_list.items():
                if node in self.node_to_index:
                    node_idx = self.node_to_index[node]
                    for neighbor, weight in neighbors.items():
                        if neighbor in self.node_to_index:
                            neighbor_idx = self.node_to_index[neighbor]
                            row_indices.append(node_idx)
                            col_indices.append(neighbor_idx)
                            data.append(weight)
            
            self.sparse_matrix = csr_matrix(
                (data, (row_indices, col_indices)), 
                shape=(n_nodes, n_nodes)
            )
            
            build_time = time.time() - start_time
            self.algorithm_times['matrix_build'].append(build_time)
            print(f"Sparse matrix built in {build_time:.2f}s")
        else:
            print("Scipy not available, skipping sparse matrix build")
    
    @memoize_with_timeout(timeout=3600, cache_size=1000)
    def compute_node_centrality(self, node_id: str, centrality_type: str = 'degree') -> float:
        """
        Compute node centrality with caching.
        
        Args:
            node_id: Target node
            centrality_type: 'degree', 'betweenness', 'closeness', 'pagerank'
            
        Returns:
            Centrality score
        """
        start_time = time.time()
        
        if centrality_type == 'degree':
            centrality = self._compute_degree_centrality(node_id)
        elif centrality_type == 'betweenness':
            centrality = self._compute_betweenness_centrality(node_id)
        elif centrality_type == 'closeness':
            centrality = self._compute_closeness_centrality(node_id)
        elif centrality_type == 'pagerank':
            centrality = self._compute_pagerank_centrality(node_id)
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
        
        computation_time = time.time() - start_time
        self.algorithm_times[f'{centrality_type}_centrality'].append(computation_time)
        
        return centrality
    
    def _compute_degree_centrality(self, node_id: str) -> float:
        """Compute normalized degree centrality."""
        if node_id not in self.adjacency_list:
            return 0.0
        
        degree = len(self.adjacency_list[node_id])
        
        # Normalize by maximum possible degree
        if node_id in self.user_nodes:
            max_degree = len(self.item_nodes)
        elif node_id in self.item_nodes:
            max_degree = len(self.user_nodes)
        else:
            max_degree = 1
        
        return degree / max_degree if max_degree > 0 else 0.0
    
    def _compute_betweenness_centrality(self, node_id: str) -> float:
        """
        Compute betweenness centrality using approximate algorithm.
        
        Uses sampling for large graphs to maintain performance.
        """
        if node_id not in self.adjacency_list:
            return 0.0
        
        # For large graphs, use sampling
        all_nodes = list(self.user_nodes.union(self.item_nodes))
        n_nodes = len(all_nodes)
        
        if n_nodes > 1000:
            # Sample nodes for approximate betweenness
            sample_size = min(100, n_nodes // 10)
            sampled_nodes = random.sample(all_nodes, sample_size)
        else:
            sampled_nodes = all_nodes
        
        betweenness = 0.0
        total_paths = 0
        
        # Compute shortest paths through sampled nodes
        for source in sampled_nodes:
            if source == node_id:
                continue
            
            for target in sampled_nodes:
                if source == target or target == node_id:
                    continue
                
                # Find shortest paths from source to target
                paths = self._find_shortest_paths(source, target)
                paths_through_node = [p for p in paths if node_id in p[1:-1]]
                
                if paths:
                    total_paths += 1
                    betweenness += len(paths_through_node) / len(paths)
        
        # Normalize
        n_pairs = len(sampled_nodes) * (len(sampled_nodes) - 1)
        return betweenness / n_pairs if n_pairs > 0 else 0.0
    
    def _compute_closeness_centrality(self, node_id: str) -> float:
        """Compute closeness centrality."""
        if node_id not in self.adjacency_list:
            return 0.0
        
        # Compute distances to all other nodes
        distances = self._compute_shortest_distances(node_id)
        
        if not distances:
            return 0.0
        
        # Sum of distances
        total_distance = sum(distances.values())
        n_reachable = len(distances)
        
        if total_distance == 0:
            return 0.0
        
        # Normalize by number of reachable nodes
        return (n_reachable - 1) / total_distance
    
    def _compute_pagerank_centrality(self, node_id: str, damping: float = 0.85,
                                   max_iter: int = 100, tolerance: float = 1e-6) -> float:
        """
        Compute PageRank centrality with power iteration.
        
        Implements the algorithm described in:
        Page, L., et al. (1999). The PageRank citation ranking.
        """
        if self.sparse_matrix is not None:
            return self._pagerank_sparse(node_id, damping, max_iter, tolerance)
        else:
            return self._pagerank_iterative(node_id, damping, max_iter, tolerance)
    
    def _pagerank_sparse(self, node_id: str, damping: float, 
                        max_iter: int, tolerance: float) -> float:
        """PageRank using sparse matrix operations."""
        if not SCIPY_AVAILABLE or node_id not in self.node_to_index:
            return 0.0
        
        n_nodes = self.sparse_matrix.shape[0]
        
        # Initialize PageRank vector
        pagerank = np.ones(n_nodes) / n_nodes
        
        # Normalize adjacency matrix (column-stochastic)
        # Convert to column-stochastic matrix
        col_sums = np.array(self.sparse_matrix.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        
        # Create diagonal matrix for normalization
        D_inv = csr_matrix((1.0 / col_sums, (range(n_nodes), range(n_nodes))))
        M = self.sparse_matrix @ D_inv
        
        # Power iteration
        for _ in range(max_iter):
            prev_pagerank = pagerank.copy()
            pagerank = damping * M @ pagerank + (1 - damping) / n_nodes
            
            # Check convergence
            if np.linalg.norm(pagerank - prev_pagerank) < tolerance:
                break
        
        # Return PageRank for target node
        node_idx = self.node_to_index[node_id]
        return float(pagerank[node_idx])
    
    def _pagerank_iterative(self, node_id: str, damping: float,
                           max_iter: int, tolerance: float) -> float:
        """PageRank using iterative method for small graphs."""
        all_nodes = list(self.user_nodes.union(self.item_nodes))
        n_nodes = len(all_nodes)
        
        if n_nodes == 0 or node_id not in all_nodes:
            return 0.0
        
        # Initialize PageRank scores
        pagerank = {node: 1.0 / n_nodes for node in all_nodes}
        
        # Power iteration
        for _ in range(max_iter):
            prev_pagerank = pagerank.copy()
            
            for node in all_nodes:
                rank = (1 - damping) / n_nodes
                
                # Sum contributions from neighbors
                for neighbor in self.adjacency_list.get(node, {}):
                    neighbor_degree = len(self.adjacency_list.get(neighbor, {}))
                    if neighbor_degree > 0:
                        rank += damping * prev_pagerank[neighbor] / neighbor_degree
                
                pagerank[node] = rank
            
            # Check convergence
            diff = sum(abs(pagerank[node] - prev_pagerank[node]) for node in all_nodes)
            if diff < tolerance:
                break
        
        return pagerank.get(node_id, 0.0)
    
    def parallel_random_walk_recommendation(self, user_id: str, n_walks: int = 1000,
                                          walk_length: int = 10, 
                                          n_workers: int = 4) -> Dict[str, float]:
        """
        Parallel random walk for recommendations.
        
        Uses multiple threads to perform random walks and aggregate results.
        
        Args:
            user_id: Starting user for walks
            n_walks: Total number of random walks
            walk_length: Maximum length of each walk
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary of item_id -> recommendation score
        """
        if user_id not in self.user_nodes:
            return {}
        
        start_time = time.time()
        
        # Distribute walks among workers
        walks_per_worker = n_walks // n_workers
        remaining_walks = n_walks % n_workers
        
        # Thread-safe result collection
        result_lock = threading.Lock()
        combined_results = defaultdict(float)
        
        def worker_random_walks(n_worker_walks: int) -> None:
            """Worker function for parallel random walks."""
            local_results = defaultdict(int)
            
            for _ in range(n_worker_walks):
                # Perform single random walk
                visited_items = self._single_random_walk(user_id, walk_length)
                
                # Count visited items
                for item_id in visited_items:
                    if item_id in self.item_nodes:
                        local_results[item_id] += 1
            
            # Merge results thread-safely
            with result_lock:
                for item_id, count in local_results.items():
                    combined_results[item_id] += count
        
        # Start worker threads
        threads = []
        for i in range(n_workers):
            worker_walks = walks_per_worker + (1 if i < remaining_walks else 0)
            thread = threading.Thread(target=worker_random_walks, args=(worker_walks,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Normalize scores
        total_visits = sum(combined_results.values())
        if total_visits > 0:
            normalized_results = {
                item_id: count / total_visits 
                for item_id, count in combined_results.items()
            }
        else:
            normalized_results = {}
        
        computation_time = time.time() - start_time
        self.algorithm_times['parallel_random_walk'].append(computation_time)
        
        return dict(normalized_results)
    
    def _single_random_walk(self, start_node: str, max_length: int) -> List[str]:
        """Perform a single random walk starting from a node."""
        if start_node not in self.adjacency_list:
            return []
        
        walk = [start_node]
        current_node = start_node
        visited_items = []
        
        for _ in range(max_length):
            neighbors = list(self.adjacency_list.get(current_node, {}).keys())
            
            if not neighbors:
                break
            
            # Weighted random selection
            weights = [self.adjacency_list[current_node][neighbor] 
                      for neighbor in neighbors]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                next_node = np.random.choice(neighbors, p=probabilities)
            else:
                next_node = random.choice(neighbors)
            
            walk.append(next_node)
            current_node = next_node
            
            # Track items visited
            if current_node in self.item_nodes:
                visited_items.append(current_node)
        
        return visited_items
    
    def detect_communities(self, method: str = 'louvain', 
                          resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in the bipartite graph.
        
        Args:
            method: Community detection method ('louvain', 'label_propagation')
            resolution: Resolution parameter for community detection
            
        Returns:
            Dictionary mapping node_id -> community_id
        """
        start_time = time.time()
        
        if method == 'louvain':
            communities = self._louvain_communities(resolution)
        elif method == 'label_propagation':
            communities = self._label_propagation_communities()
        else:
            raise ValueError(f"Unknown community detection method: {method}")
        
        computation_time = time.time() - start_time
        self.algorithm_times[f'{method}_communities'].append(computation_time)
        
        return communities
    
    def _louvain_communities(self, resolution: float) -> Dict[str, int]:
        """
        Simplified Louvain community detection.
        
        Implements a basic version of the algorithm described in:
        Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks.
        """
        # Initialize each node in its own community
        all_nodes = list(self.user_nodes.union(self.item_nodes))
        communities = {node: i for i, node in enumerate(all_nodes)}
        
        improved = True
        iteration = 0
        max_iterations = 100
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Shuffle nodes for random order processing
            shuffled_nodes = all_nodes.copy()
            random.shuffle(shuffled_nodes)
            
            for node in shuffled_nodes:
                current_community = communities[node]
                best_community = current_community
                best_gain = 0.0
                
                # Check neighboring communities
                neighboring_communities = set()
                for neighbor in self.adjacency_list.get(node, {}):
                    neighboring_communities.add(communities[neighbor])
                
                for community in neighboring_communities:
                    if community == current_community:
                        continue
                    
                    # Calculate modularity gain
                    gain = self._calculate_modularity_gain(
                        node, current_community, community, communities, resolution
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
                
                # Move node if beneficial
                if best_community != current_community:
                    communities[node] = best_community
                    improved = True
        
        # Relabel communities to be consecutive integers
        unique_communities = list(set(communities.values()))
        community_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_communities)}
        
        return {node: community_mapping[comm_id] for node, comm_id in communities.items()}
    
    def _calculate_modularity_gain(self, node: str, current_comm: int, 
                                  target_comm: int, communities: Dict[str, int],
                                  resolution: float) -> float:
        """Calculate modularity gain for moving a node between communities."""
        # Simplified modularity calculation
        # In practice, you'd want a more sophisticated implementation
        
        node_degree = len(self.adjacency_list.get(node, {}))
        
        # Count edges to current and target communities
        edges_to_current = 0
        edges_to_target = 0
        
        for neighbor, weight in self.adjacency_list.get(node, {}).items():
            if communities[neighbor] == current_comm:
                edges_to_current += weight
            elif communities[neighbor] == target_comm:
                edges_to_target += weight
        
        # Simple gain calculation (actual modularity is more complex)
        gain = (edges_to_target - edges_to_current) * resolution
        
        return gain
    
    def _label_propagation_communities(self, max_iterations: int = 100) -> Dict[str, int]:
        """
        Label propagation community detection.
        
        Each node adopts the most common label among its neighbors.
        """
        all_nodes = list(self.user_nodes.union(self.item_nodes))
        
        # Initialize with unique labels
        labels = {node: i for i, node in enumerate(all_nodes)}
        
        for iteration in range(max_iterations):
            new_labels = {}
            changed = False
            
            # Shuffle nodes for random order
            shuffled_nodes = all_nodes.copy()
            random.shuffle(shuffled_nodes)
            
            for node in shuffled_nodes:
                # Count neighbor labels
                neighbor_labels = defaultdict(float)
                
                for neighbor, weight in self.adjacency_list.get(node, {}).items():
                    neighbor_labels[labels[neighbor]] += weight
                
                if neighbor_labels:
                    # Choose most common label (weighted by edge weights)
                    best_label = max(neighbor_labels.items(), key=lambda x: x[1])[0]
                    new_labels[node] = best_label
                    
                    if best_label != labels[node]:
                        changed = True
                else:
                    new_labels[node] = labels[node]
            
            labels = new_labels
            
            # Stop if no changes
            if not changed:
                break
        
        # Relabel to consecutive integers
        unique_labels = list(set(labels.values()))
        label_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_labels)}
        
        return {node: label_mapping[label] for node, label in labels.items()}
    
    def _find_shortest_paths(self, source: str, target: str, 
                           max_paths: int = 10) -> List[List[str]]:
        """Find shortest paths between two nodes using BFS."""
        if source == target:
            return [[source]]
        
        if source not in self.adjacency_list or target not in self.adjacency_list:
            return []
        
        # BFS to find shortest path length
        queue = deque([(source, [source])])
        visited = {source: 0}
        paths = []
        shortest_length = None
        
        while queue and len(paths) < max_paths:
            current_node, path = queue.popleft()
            
            # If we found a longer path, stop
            if shortest_length is not None and len(path) > shortest_length:
                break
            
            for neighbor in self.adjacency_list.get(current_node, {}):
                new_path = path + [neighbor]
                
                if neighbor == target:
                    paths.append(new_path)
                    if shortest_length is None:
                        shortest_length = len(new_path)
                elif neighbor not in visited or visited[neighbor] >= len(new_path):
                    visited[neighbor] = len(new_path)
                    queue.append((neighbor, new_path))
        
        return paths
    
    def _compute_shortest_distances(self, source: str, 
                                   max_distance: int = 6) -> Dict[str, int]:
        """Compute shortest distances from source to all reachable nodes."""
        if source not in self.adjacency_list:
            return {}
        
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            current_node = queue.popleft()
            current_distance = distances[current_node]
            
            if current_distance >= max_distance:
                continue
            
            for neighbor in self.adjacency_list.get(current_node, {}):
                if neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        # Remove source from distances
        distances.pop(source, None)
        return distances
    
    def _invalidate_node_cache(self, node_id: str) -> None:
        """Invalidate cache entries related to a specific node."""
        if not self.enable_caching:
            return
        
        keys_to_remove = [k for k in self._computation_cache.keys() if node_id in str(k)]
        for key in keys_to_remove:
            del self._computation_cache[key]
    
    def get_graph_statistics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.
        
        Args:
            force_refresh: Force recomputation of statistics
            
        Returns:
            Dictionary of graph statistics
        """
        if not force_refresh and not self._stats_dirty and self._stats_cache:
            return self._stats_cache
        
        start_time = time.time()
        
        n_users = len(self.user_nodes)
        n_items = len(self.item_nodes)
        n_edges = len(self.edge_weights)
        
        # Calculate density
        max_edges = n_users * n_items if n_users > 0 and n_items > 0 else 1
        density = n_edges / max_edges
        
        # Degree statistics
        user_degrees = [len(self.adjacency_list.get(user, {})) for user in self.user_nodes]
        item_degrees = [len(self.adjacency_list.get(item, {})) for item in self.item_nodes]
        
        all_degrees = user_degrees + item_degrees
        
        stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_edges': n_edges,
            'density': density,
            'avg_user_degree': np.mean(user_degrees) if user_degrees else 0.0,
            'avg_item_degree': np.mean(item_degrees) if item_degrees else 0.0,
            'avg_degree': np.mean(all_degrees) if all_degrees else 0.0,
            'max_degree': max(all_degrees) if all_degrees else 0,
            'min_degree': min(all_degrees) if all_degrees else 0,
            'degree_std': np.std(all_degrees) if all_degrees else 0.0
        }
        
        # Cache statistics
        self._stats_cache = stats
        self._stats_dirty = False
        
        computation_time = time.time() - start_time
        self.algorithm_times['statistics'].append(computation_time)
        
        return stats
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for all algorithms."""
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) 
                             if (self.cache_hits + self.cache_misses) > 0 else 0.0),
            'algorithm_times': {}
        }
        
        # Calculate average times for each algorithm
        for algorithm, times in self.algorithm_times.items():
            if times:
                stats['algorithm_times'][algorithm] = {
                    'count': len(times),
                    'avg_time': np.mean(times),
                    'total_time': sum(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return stats
    
    @classmethod
    def from_interaction_dataframe(cls, interactions_df, **kwargs) -> 'OptimizedBipartiteRecommendationGraph':
        """
        Create graph from interaction DataFrame.
        
        Args:
            interactions_df: DataFrame with user_id, product_id, rating columns
            **kwargs: Additional arguments for graph initialization
            
        Returns:
            OptimizedBipartiteRecommendationGraph instance
        """
        if pd is None:
            raise ImportError("pandas is required for DataFrame operations")
        
        graph = cls(**kwargs)
        
        print(f"Creating graph from {len(interactions_df)} interactions...")
        
        # Add all interactions
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            weight = row.get('rating', 1.0)
            
            graph.add_interaction(user_id, item_id, weight)
        
        # Build sparse representation if enabled
        if graph.use_sparse_matrix:
            graph.build_sparse_representation()
        
        print(f"Graph created with {len(graph.user_nodes)} users and {len(graph.item_nodes)} items")
        
        return graph