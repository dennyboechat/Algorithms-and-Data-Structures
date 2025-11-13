"""
Graph-Based Structure for User-Item Relationships

This module implements a graph representation where users and items are nodes,
and their interactions are edges. This structure supports various graph algorithms
for recommendation systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import heapq


class BipartiteRecommendationGraph:
    """
    A bipartite graph representation of user-item interactions where:
    - Users and items are different types of nodes
    - Edges represent interactions with weights (ratings, frequencies, etc.)
    """
    
    def __init__(self):
        self.graph = defaultdict(dict)  # Adjacency list with weights
        self.user_nodes = set()
        self.item_nodes = set()
        self.edge_weights = defaultdict(float)
        self.node_attributes = defaultdict(dict)
    
    def add_user(self, user_id: str, attributes: Optional[Dict] = None) -> None:
        """Add a user node to the graph."""
        self.user_nodes.add(user_id)
        if attributes:
            self.node_attributes[user_id] = attributes
    
    def add_item(self, item_id: str, attributes: Optional[Dict] = None) -> None:
        """Add an item node to the graph."""
        self.item_nodes.add(item_id)
        if attributes:
            self.node_attributes[item_id] = attributes
    
    def add_interaction(self, user_id: str, item_id: str, weight: float = 1.0,
                       interaction_type: str = 'rating') -> None:
        """
        Add an interaction edge between user and item.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            weight: Edge weight (rating, frequency, etc.)
            interaction_type: Type of interaction
        """
        self.user_nodes.add(user_id)
        self.item_nodes.add(item_id)
        
        # Add bidirectional edge
        self.graph[user_id][item_id] = weight
        self.graph[item_id][user_id] = weight
        
        # Store edge metadata
        edge_key = (user_id, item_id)
        self.edge_weights[edge_key] = weight
        
        # Store interaction type as edge attribute
        if 'edge_types' not in self.node_attributes:
            self.node_attributes['edge_types'] = {}
        self.node_attributes['edge_types'][edge_key] = interaction_type
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node."""
        return list(self.graph[node_id].keys())
    
    def get_user_items(self, user_id: str) -> List[str]:
        """Get all items interacted with by a user."""
        if user_id not in self.user_nodes:
            return []
        return [item for item in self.graph[user_id].keys() if item in self.item_nodes]
    
    def get_item_users(self, item_id: str) -> List[str]:
        """Get all users who interacted with an item."""
        if item_id not in self.item_nodes:
            return []
        return [user for user in self.graph[item_id].keys() if user in self.user_nodes]
    
    def get_edge_weight(self, node1: str, node2: str) -> float:
        """Get the weight of edge between two nodes."""
        return self.graph[node1].get(node2, 0.0)
    
    def compute_user_similarity_graph_based(self, user1: str, user2: str) -> float:
        """
        Compute similarity between two users based on common neighbors (items).
        Uses Jaccard similarity on the neighborhood.
        """
        if user1 not in self.user_nodes or user2 not in self.user_nodes:
            return 0.0
        
        items1 = set(self.get_user_items(user1))
        items2 = set(self.get_user_items(user2))
        
        if not items1 or not items2:
            return 0.0
        
        intersection = len(items1 & items2)
        union = len(items1 | items2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_item_similarity_graph_based(self, item1: str, item2: str) -> float:
        """
        Compute similarity between two items based on common neighbors (users).
        Uses Jaccard similarity on the neighborhood.
        """
        if item1 not in self.item_nodes or item2 not in self.item_nodes:
            return 0.0
        
        users1 = set(self.get_item_users(item1))
        users2 = set(self.get_item_users(item2))
        
        if not users1 or not users2:
            return 0.0
        
        intersection = len(users1 & users2)
        union = len(users1 | users2)
        
        return intersection / union if union > 0 else 0.0
    
    def random_walk_recommendation(self, user_id: str, walk_length: int = 10,
                                  num_walks: int = 100) -> Dict[str, float]:
        """
        Perform random walks starting from a user to discover item recommendations.
        
        Args:
            user_id: Starting user
            walk_length: Length of each random walk
            num_walks: Number of walks to perform
            
        Returns:
            Dictionary mapping item IDs to recommendation scores
        """
        if user_id not in self.user_nodes:
            return {}
        
        item_visit_counts = defaultdict(int)
        
        for _ in range(num_walks):
            current_node = user_id
            
            for step in range(walk_length):
                neighbors = self.get_neighbors(current_node)
                if not neighbors:
                    break
                
                # Choose next node based on edge weights
                weights = [self.get_edge_weight(current_node, neighbor) for neighbor in neighbors]
                total_weight = sum(weights)
                
                if total_weight == 0:
                    break
                
                # Weighted random selection
                probabilities = [w / total_weight for w in weights]
                next_node = np.random.choice(neighbors, p=probabilities)
                
                # If we landed on an item (and it's not the starting user), count it
                if next_node in self.item_nodes and step > 0:
                    item_visit_counts[next_node] += 1
                
                current_node = next_node
        
        # Convert counts to scores (normalize by number of walks)
        recommendations = {item: count / num_walks for item, count in item_visit_counts.items()}
        
        # Remove items the user has already interacted with
        user_items = set(self.get_user_items(user_id))
        recommendations = {item: score for item, score in recommendations.items()
                         if item not in user_items}
        
        return recommendations
    
    def shortest_path_recommendation(self, user_id: str, max_path_length: int = 3) -> Dict[str, float]:
        """
        Find item recommendations based on shortest paths in the graph.
        Items with shorter paths from the user get higher scores.
        
        Args:
            user_id: Target user
            max_path_length: Maximum path length to consider
            
        Returns:
            Dictionary mapping item IDs to recommendation scores
        """
        if user_id not in self.user_nodes:
            return {}
        
        # BFS to find shortest paths
        queue = deque([(user_id, 0)])  # (node, distance)
        visited = {user_id}
        item_distances = {}
        
        while queue:
            current_node, distance = queue.popleft()
            
            if distance >= max_path_length:
                continue
            
            for neighbor in self.get_neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_distance = distance + 1
                    
                    # If it's an item and we haven't seen it before
                    if neighbor in self.item_nodes and neighbor not in item_distances:
                        item_distances[neighbor] = new_distance
                    
                    queue.append((neighbor, new_distance))
        
        # Convert distances to scores (closer items get higher scores)
        user_items = set(self.get_user_items(user_id))
        recommendations = {}
        
        for item, distance in item_distances.items():
            if item not in user_items:  # Exclude items already interacted with
                # Higher score for shorter distance
                score = 1.0 / (distance + 1)
                recommendations[item] = score
        
        return recommendations
    
    def personalized_pagerank_recommendation(self, user_id: str, damping_factor: float = 0.85,
                                           max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Perform Personalized PageRank starting from a user to get item recommendations.
        
        Args:
            user_id: Starting user for personalization
            damping_factor: PageRank damping factor
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary mapping item IDs to PageRank scores
        """
        if user_id not in self.user_nodes:
            return {}
        
        # Get all nodes
        all_nodes = list(self.user_nodes | self.item_nodes)
        node_to_index = {node: i for i, node in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        
        # Initialize PageRank scores
        scores = np.ones(n_nodes) / n_nodes
        personalization = np.zeros(n_nodes)
        personalization[node_to_index[user_id]] = 1.0
        
        # Build transition matrix
        transition_matrix = np.zeros((n_nodes, n_nodes))
        
        for i, node in enumerate(all_nodes):
            neighbors = self.get_neighbors(node)
            if neighbors:
                total_weight = sum(self.get_edge_weight(node, neighbor) for neighbor in neighbors)
                for neighbor in neighbors:
                    j = node_to_index[neighbor]
                    weight = self.get_edge_weight(node, neighbor)
                    transition_matrix[j, i] = weight / total_weight
        
        # PageRank iterations
        for iteration in range(max_iterations):
            new_scores = (1 - damping_factor) * personalization + \
                        damping_factor * transition_matrix.dot(scores)
            
            # Check convergence
            if np.linalg.norm(new_scores - scores, 1) < tolerance:
                break
            
            scores = new_scores
        
        # Extract item recommendations
        user_items = set(self.get_user_items(user_id))
        recommendations = {}
        
        for node, score in zip(all_nodes, scores):
            if node in self.item_nodes and node not in user_items:
                recommendations[node] = score
        
        return recommendations
    
    def find_similar_users_graph(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar users using graph-based similarity."""
        if user_id not in self.user_nodes:
            return []
        
        similarities = []
        for other_user in self.user_nodes:
            if other_user != user_id:
                sim = self.compute_user_similarity_graph_based(user_id, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_items_graph(self, item_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar items using graph-based similarity."""
        if item_id not in self.item_nodes:
            return []
        
        similarities = []
        for other_item in self.item_nodes:
            if other_item != item_id:
                sim = self.compute_item_similarity_graph_based(item_id, other_item)
                if sim > 0:
                    similarities.append((other_item, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the graph structure."""
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        
        user_degrees = [len(self.get_user_items(user)) for user in self.user_nodes]
        item_degrees = [len(self.get_item_users(item)) for item in self.item_nodes]
        
        return {
            'num_users': len(self.user_nodes),
            'num_items': len(self.item_nodes),
            'num_edges': total_edges,
            'avg_user_degree': np.mean(user_degrees) if user_degrees else 0,
            'avg_item_degree': np.mean(item_degrees) if item_degrees else 0,
            'max_user_degree': max(user_degrees) if user_degrees else 0,
            'max_item_degree': max(item_degrees) if item_degrees else 0,
            'graph_density': total_edges / (len(self.user_nodes) * len(self.item_nodes)) if self.user_nodes and self.item_nodes else 0
        }
    
    @classmethod
    def from_interaction_dataframe(cls, interactions_df: pd.DataFrame) -> 'BipartiteRecommendationGraph':
        """
        Create a graph from an interaction DataFrame.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'product_id', 'rating', 'interaction_type']
        """
        graph = cls()
        
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            rating = row.get('rating', 1.0)
            interaction_type = row.get('interaction_type', 'interaction')
            
            # Use rating if available, otherwise use 1.0
            weight = rating if pd.notna(rating) else 1.0
            
            graph.add_interaction(user_id, item_id, weight, interaction_type)
        
        return graph