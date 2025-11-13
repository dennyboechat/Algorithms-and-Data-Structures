"""
Tree-Based Structures for Efficient Nearest-Neighbor Searches

This module implements KD-trees and Ball-trees for efficient nearest-neighbor
searches in recommendation systems, particularly useful for content-based filtering
and high-dimensional feature spaces.
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import heapq
from abc import ABC, abstractmethod


@dataclass
class DataPoint:
    """Represents a data point with features and metadata."""
    features: np.ndarray
    item_id: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NearestNeighborTree(ABC):
    """Abstract base class for nearest neighbor tree structures."""
    
    @abstractmethod
    def build(self, data_points: List[DataPoint]) -> None:
        """Build the tree from data points."""
        pass
    
    @abstractmethod
    def query(self, query_point: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to query point."""
        pass


class KDTreeNode:
    """Node in a KD-tree structure."""
    
    def __init__(self, point: DataPoint = None, split_dim: int = 0,
                 split_value: float = 0.0, left=None, right=None):
        self.point = point
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.is_leaf = point is not None


class KDTree(NearestNeighborTree):
    """
    K-Dimensional Tree for efficient nearest neighbor searches.
    
    Best for low to medium dimensional data (typically d < 20).
    Time complexity: O(log n) average, O(n) worst case for queries.
    """
    
    def __init__(self, max_depth: int = 50):
        self.root = None
        self.max_depth = max_depth
        self.dimension = None
    
    def build(self, data_points: List[DataPoint]) -> None:
        """Build KD-tree from data points."""
        if not data_points:
            self.root = None
            return
        
        self.dimension = len(data_points[0].features)
        self.root = self._build_recursive(data_points, depth=0)
    
    def _build_recursive(self, points: List[DataPoint], depth: int) -> Optional[KDTreeNode]:
        """Recursively build the KD-tree."""
        if not points or depth > self.max_depth:
            return None
        
        if len(points) == 1:
            return KDTreeNode(point=points[0])
        
        # Choose splitting dimension (cycle through dimensions)
        split_dim = depth % self.dimension
        
        # Sort points by the splitting dimension
        points.sort(key=lambda p: p.features[split_dim])
        
        # Find median
        median_idx = len(points) // 2
        median_point = points[median_idx]
        split_value = median_point.features[split_dim]
        
        # Split points
        left_points = points[:median_idx]
        right_points = points[median_idx + 1:]
        
        # Create node
        node = KDTreeNode(
            point=median_point,
            split_dim=split_dim,
            split_value=split_value
        )
        
        # Recursively build subtrees
        node.left = self._build_recursive(left_points, depth + 1)
        node.right = self._build_recursive(right_points, depth + 1)
        
        return node
    
    def query(self, query_point: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest neighbors using KD-tree search."""
        if self.root is None:
            return []
        
        # Use a max-heap to keep track of k nearest neighbors
        best_neighbors = []  # List of (-distance, item_id) tuples
        
        def search_recursive(node: KDTreeNode, query: np.ndarray) -> None:
            if node is None:
                return
            
            # Calculate distance to current node's point
            if node.point is not None:
                distance = np.linalg.norm(query - node.point.features)
                
                if len(best_neighbors) < k:
                    heapq.heappush(best_neighbors, (-distance, node.point.item_id))
                elif distance < -best_neighbors[0][0]:
                    heapq.heapreplace(best_neighbors, (-distance, node.point.item_id))
            
            # Determine which subtree to search first
            if query[node.split_dim] <= node.split_value:
                near_subtree, far_subtree = node.left, node.right
            else:
                near_subtree, far_subtree = node.right, node.left
            
            # Search the near subtree
            search_recursive(near_subtree, query)
            
            # Check if we need to search the far subtree
            if (len(best_neighbors) < k or 
                abs(query[node.split_dim] - node.split_value) < -best_neighbors[0][0]):
                search_recursive(far_subtree, query)
        
        search_recursive(self.root, query_point)
        
        # Convert back to (item_id, distance) format and sort
        result = [(item_id, -distance) for distance, item_id in best_neighbors]
        result.sort(key=lambda x: x[1])
        
        return result


class BallTreeNode:
    """Node in a Ball-tree structure."""
    
    def __init__(self, points: List[DataPoint] = None, center: np.ndarray = None,
                 radius: float = 0.0, left=None, right=None):
        self.points = points or []
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right
        self.is_leaf = len(self.points) <= 1


class BallTree(NearestNeighborTree):
    """
    Ball Tree for efficient nearest neighbor searches in high-dimensional spaces.
    
    Better than KD-tree for high-dimensional data (d > 20).
    Time complexity: O(log n) for queries.
    """
    
    def __init__(self, leaf_size: int = 30, max_depth: int = 50):
        self.root = None
        self.leaf_size = leaf_size
        self.max_depth = max_depth
    
    def build(self, data_points: List[DataPoint]) -> None:
        """Build Ball-tree from data points."""
        if not data_points:
            self.root = None
            return
        
        self.root = self._build_recursive(data_points, depth=0)
    
    def _build_recursive(self, points: List[DataPoint], depth: int) -> Optional[BallTreeNode]:
        """Recursively build the Ball-tree."""
        if not points or depth > self.max_depth:
            return None
        
        # Calculate center and radius
        features_matrix = np.array([p.features for p in points])
        center = np.mean(features_matrix, axis=0)
        
        distances = [np.linalg.norm(p.features - center) for p in points]
        radius = max(distances) if distances else 0.0
        
        # Create leaf node if small enough
        if len(points) <= self.leaf_size:
            return BallTreeNode(points=points, center=center, radius=radius)
        
        # Find the dimension with maximum spread
        spreads = np.var(features_matrix, axis=0)
        split_dim = np.argmax(spreads)
        
        # Sort points by the splitting dimension
        points.sort(key=lambda p: p.features[split_dim])
        
        # Split points at median
        median_idx = len(points) // 2
        left_points = points[:median_idx]
        right_points = points[median_idx:]
        
        # Create internal node
        node = BallTreeNode(center=center, radius=radius)
        
        # Recursively build subtrees
        node.left = self._build_recursive(left_points, depth + 1)
        node.right = self._build_recursive(right_points, depth + 1)
        
        return node
    
    def query(self, query_point: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest neighbors using Ball-tree search."""
        if self.root is None:
            return []
        
        # Use a max-heap to keep track of k nearest neighbors
        best_neighbors = []  # List of (-distance, item_id) tuples
        
        def search_recursive(node: BallTreeNode, query: np.ndarray) -> None:
            if node is None:
                return
            
            # Calculate distance to ball center
            center_distance = np.linalg.norm(query - node.center)
            
            # Pruning: if we have k neighbors and the ball is too far, skip
            if (len(best_neighbors) == k and 
                center_distance - node.radius >= -best_neighbors[0][0]):
                return
            
            # If leaf node, check all points
            if node.is_leaf:
                for point in node.points:
                    distance = np.linalg.norm(query - point.features)
                    
                    if len(best_neighbors) < k:
                        heapq.heappush(best_neighbors, (-distance, point.item_id))
                    elif distance < -best_neighbors[0][0]:
                        heapq.heapreplace(best_neighbors, (-distance, point.item_id))
            else:
                # Recursively search children
                # Search closer child first
                left_distance = np.linalg.norm(query - node.left.center) if node.left else float('inf')
                right_distance = np.linalg.norm(query - node.right.center) if node.right else float('inf')
                
                if left_distance <= right_distance:
                    search_recursive(node.left, query)
                    search_recursive(node.right, query)
                else:
                    search_recursive(node.right, query)
                    search_recursive(node.left, query)
        
        search_recursive(self.root, query_point)
        
        # Convert back to (item_id, distance) format and sort
        result = [(item_id, -distance) for distance, item_id in best_neighbors]
        result.sort(key=lambda x: x[1])
        
        return result


class FeatureBasedNearestNeighbor:
    """
    High-level interface for feature-based nearest neighbor search
    supporting both KD-tree and Ball-tree implementations.
    """
    
    def __init__(self, algorithm: str = 'auto', leaf_size: int = 30, max_depth: int = 50):
        """
        Initialize the nearest neighbor searcher.
        
        Args:
            algorithm: 'kdtree', 'balltree', or 'auto'
            leaf_size: Leaf size for tree construction
            max_depth: Maximum depth for trees
        """
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.tree = None
        self.data_points = []
        self.feature_dimension = None
    
    def fit(self, features: np.ndarray, item_ids: List[str], metadata: List[dict] = None) -> None:
        """
        Fit the nearest neighbor model.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            item_ids: List of item identifiers
            metadata: Optional metadata for each item
        """
        if metadata is None:
            metadata = [{}] * len(item_ids)
        
        # Create data points
        self.data_points = [
            DataPoint(features[i], item_ids[i], metadata[i])
            for i in range(len(item_ids))
        ]
        
        self.feature_dimension = features.shape[1]
        
        # Choose algorithm automatically if needed
        if self.algorithm == 'auto':
            # Use KD-tree for low-dimensional data, Ball-tree for high-dimensional
            chosen_algorithm = 'kdtree' if self.feature_dimension < 20 else 'balltree'
        else:
            chosen_algorithm = self.algorithm
        
        # Build appropriate tree
        if chosen_algorithm == 'kdtree':
            self.tree = KDTree(max_depth=self.max_depth)
        elif chosen_algorithm == 'balltree':
            self.tree = BallTree(leaf_size=self.leaf_size, max_depth=self.max_depth)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.tree.build(self.data_points)
    
    def query(self, query_features: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for query features.
        
        Args:
            query_features: Query feature vector
            k: Number of neighbors to return
            
        Returns:
            List of (item_id, distance) tuples
        """
        if self.tree is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.tree.query(query_features, k)
    
    def query_radius(self, query_features: np.ndarray, radius: float) -> List[Tuple[str, float]]:
        """
        Find all neighbors within a given radius.
        
        Args:
            query_features: Query feature vector
            radius: Search radius
            
        Returns:
            List of (item_id, distance) tuples within radius
        """
        # Get a large number of neighbors and filter by radius
        candidates = self.query(query_features, k=min(1000, len(self.data_points)))
        return [(item_id, distance) for item_id, distance in candidates if distance <= radius]
    
    def get_feature_importance(self, query_features: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Estimate feature importance based on nearest neighbors.
        
        Args:
            query_features: Query feature vector
            k: Number of neighbors to consider
            
        Returns:
            Feature importance scores
        """
        neighbors = self.query(query_features, k)
        
        if not neighbors:
            return np.zeros(self.feature_dimension)
        
        # Get neighbor features
        neighbor_features = []
        for item_id, _ in neighbors:
            for point in self.data_points:
                if point.item_id == item_id:
                    neighbor_features.append(point.features)
                    break
        
        if not neighbor_features:
            return np.zeros(self.feature_dimension)
        
        neighbor_matrix = np.array(neighbor_features)
        
        # Calculate feature variance among neighbors
        feature_variance = np.var(neighbor_matrix, axis=0)
        
        # Higher variance means less importance for similarity
        # So we return inverse variance (with smoothing)
        importance = 1.0 / (feature_variance + 1e-8)
        
        # Normalize to [0, 1]
        importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + 1e-8)
        
        return importance


class HierarchicalClustering:
    """
    Hierarchical clustering for organizing items into a tree structure
    that can be used for efficient recommendation browsing.
    """
    
    def __init__(self, linkage: str = 'ward', distance_metric: str = 'euclidean'):
        """
        Initialize hierarchical clustering.
        
        Args:
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            distance_metric: Distance metric to use
        """
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.cluster_tree = None
        self.item_ids = None
    
    def fit(self, features: np.ndarray, item_ids: List[str]) -> None:
        """
        Build hierarchical clustering tree.
        
        Args:
            features: Feature matrix
            item_ids: Item identifiers
        """
        self.item_ids = item_ids
        
        # Simple agglomerative clustering implementation
        n_samples = features.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [{i} for i in range(n_samples)]
        
        # Build dendrogram (simplified version)
        merge_history = []
        
        while len(clusters) > 1:
            # Find closest pair of clusters
            min_distance = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._cluster_distance(clusters[i], clusters[j], features)
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Merge clusters
            new_cluster = clusters[merge_i] | clusters[merge_j]
            merge_history.append((merge_i, merge_j, min_distance))
            
            # Remove old clusters and add new one
            clusters = [cluster for k, cluster in enumerate(clusters) if k not in {merge_i, merge_j}]
            clusters.append(new_cluster)
        
        self.cluster_tree = merge_history
    
    def _cluster_distance(self, cluster1: set, cluster2: set, features: np.ndarray) -> float:
        """Calculate distance between two clusters."""
        if self.linkage == 'single':
            # Minimum distance
            min_dist = float('inf')
            for i in cluster1:
                for j in cluster2:
                    dist = np.linalg.norm(features[i] - features[j])
                    min_dist = min(min_dist, dist)
            return min_dist
        
        elif self.linkage == 'complete':
            # Maximum distance
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    dist = np.linalg.norm(features[i] - features[j])
                    max_dist = max(max_dist, dist)
            return max_dist
        
        elif self.linkage == 'average':
            # Average distance
            total_dist = 0
            count = 0
            for i in cluster1:
                for j in cluster2:
                    dist = np.linalg.norm(features[i] - features[j])
                    total_dist += dist
                    count += 1
            return total_dist / count if count > 0 else 0
        
        else:
            # Default to average
            return self._cluster_distance(cluster1, cluster2, features)
    
    def get_clusters(self, n_clusters: int) -> List[List[str]]:
        """
        Get clusters at a specific level.
        
        Args:
            n_clusters: Number of clusters to return
            
        Returns:
            List of clusters, each containing item IDs
        """
        # This is a simplified implementation
        # In practice, you'd traverse the dendrogram to get the right number of clusters
        return []  # Placeholder implementation