"""
Data Loader Module for MuseRecSys

This module provides a DataLoader class that generates Mock data based on KuaiRec dataset structure.
It includes both basic features and LLM semantic features for recommendation system training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import random


@dataclass
class UserFeatures:
    """Data class for user features."""
    user_id: int
    age: int
    gender: int  # 0 for male, 1 for female
    user_active_degree: float  # Value between 0 and 1
    feature_vector: np.ndarray  # Additional feature vector for model input


@dataclass
class ItemFeatures:
    """Data class for item features."""
    item_id: int
    category_id: int
    tags: List[str]
    video_duration: int  # in seconds
    feature_vector: np.ndarray  # Additional feature vector for model input


@dataclass
class UserHistory:
    """Data class for user history behavior sequence."""
    user_id: int
    history_item_ids: List[int]
    history_timestamps: Optional[List[int]] = None


class DataLoader:
    """
    Data Loader class for MuseRecSys recommendation system.

    Provides Mock data and feature interfaces based on KuaiRec dataset structure.
    Includes basic features (user/item attributes) and LLM semantic features
    (user state embeddings and item semantic embeddings).

    Attributes:
        num_users (int): Number of users in the dataset.
        num_items (int): Number of items in the dataset.
        emb_dim (int): Dimension of semantic embeddings (default: 2560).
        user_features (Dict[int, UserFeatures]): Dictionary mapping user_id to features.
        item_features (Dict[int, ItemFeatures]): Dictionary mapping item_id to features.
        user_history (Dict[int, UserHistory]): Dictionary mapping user_id to history.
        user_state_embs (np.ndarray): User state embeddings [N_users, 5, 2560].
        item_semantic_embs (np.ndarray): Item semantic embeddings [N_items, 2560].
    """

    # User state embedding dimension names (5 semantic vectors)
    USER_STATE_DIMENSIONS = [
        "long_term_intent",      # User's long-term interest intention
        "life_stage",            # User's life stage characteristics
        "psychological_demand",  # User's psychological needs
        "retrieval_suggestions", # Suggestions for retrieval
        "interest_growth_points" # User's potential interest growth areas
    ]

    # Mock item categories (based on typical video content)
    ITEM_CATEGORIES = [
        "Entertainment", "Education", "Gaming", "Sports",
        "Music", "News", "Technology", "Lifestyle",
        "Comedy", "Science"
    ]

    # Mock tags for items
    ITEM_TAGS = [
        "viral", "trending", "popular", "new", "classic",
        "tutorial", "review", "vlog", "challenge", "live",
        "short", "long-form", "series", "standalone", "collaboration"
    ]

    def __init__(self, num_users: int = 1000, num_items: int = 5000, emb_dim: int = 2560, seed: int = 42):
        """
        Initialize the DataLoader with Mock data.

        Args:
            num_users (int): Number of users to generate.
            num_items (int): Number of items to generate.
            emb_dim (int): Dimension of semantic embeddings.
            seed (int): Random seed for reproducibility.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Initialize data structures
        self._user_features: Dict[int, UserFeatures] = {}
        self._item_features: Dict[int, ItemFeatures] = {}
        self._user_history: Dict[int, UserHistory] = {}
        self._user_state_embs: Optional[np.ndarray] = None
        self._item_semantic_embs: Optional[np.ndarray] = None

        # Generate Mock data
        self._generate_mock_data()

    def _generate_mock_data(self) -> None:
        """Generate all Mock data for users and items."""
        self._generate_user_features()
        self._generate_item_features()
        self._generate_user_history()
        self._generate_user_state_embs()
        self._generate_item_semantic_embs()

    def _generate_user_features(self) -> None:
        """Generate Mock user features."""
        for user_id in range(self.num_users):
            # Age: distribution skewed towards younger users (18-45)
            age = int(np.random.normal(loc=28, scale=8))
            age = np.clip(age, 18, 65)

            # Gender: balanced distribution
            gender = np.random.randint(0, 2)

            # Active degree: beta distribution for more realistic spread
            active_degree = float(np.random.beta(a=2, b=5))

            # Feature vector for model input (32-dim)
            feature_vector = np.random.randn(32).astype(np.float32)

            self._user_features[user_id] = UserFeatures(
                user_id=user_id,
                age=age,
                gender=gender,
                user_active_degree=active_degree,
                feature_vector=feature_vector
            )

    def _generate_item_features(self) -> None:
        """Generate Mock item features."""
        for item_id in range(self.num_items):
            # Category: random selection from predefined categories
            category_id = np.random.randint(0, len(self.ITEM_CATEGORIES))

            # Tags: 2-5 random tags per item
            num_tags = np.random.randint(2, 6)
            tags = random.sample(self.ITEM_TAGS, num_tags)

            # Video duration: log-normal distribution (most videos short, some long)
            duration = int(np.random.lognormal(mean=4, sigma=1))
            duration = np.clip(duration, 30, 7200)  # 30 seconds to 2 hours

            # Feature vector for model input (32-dim)
            feature_vector = np.random.randn(32).astype(np.float32)

            self._item_features[item_id] = ItemFeatures(
                item_id=item_id,
                category_id=category_id,
                tags=tags,
                video_duration=duration,
                feature_vector=feature_vector
            )

    def _generate_user_history(self) -> None:
        """Generate Mock user behavior history."""
        for user_id in range(self.num_users):
            # Each user has 5-50 historical interactions
            num_interactions = np.random.randint(5, 51)

            # Sample random items (with replacement for some items)
            history_items = [
                np.random.randint(0, self.num_items)
                for _ in range(num_interactions)
            ]

            # Generate timestamps (more recent activity is more frequent)
            timestamps = sorted(
                np.random.randint(0, 1000000, size=num_interactions).tolist()
            )

            self._user_history[user_id] = UserHistory(
                user_id=user_id,
                history_item_ids=history_items,
                history_timestamps=timestamps
            )

    def _generate_user_state_embs(self) -> None:
        """
        Generate Mock LLM user state embeddings.

        Shape: [N_users, 5, 2560]
        The 5 dimensions represent different semantic aspects of user state.
        """
        # Generate normalized embeddings
        embeddings = np.random.randn(self.num_users, 5, self.emb_dim).astype(np.float32)

        # L2 normalize each embedding vector
        norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
        self._user_state_embs = embeddings / (norms + 1e-8)

    def _generate_item_semantic_embs(self) -> None:
        """
        Generate Mock LLM item semantic embeddings.

        Shape: [N_items, 2560]
        These represent semantic vectors from item title/description text.
        """
        # Generate normalized embeddings
        embeddings = np.random.randn(self.num_items, self.emb_dim).astype(np.float32)

        # L2 normalize each embedding vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._item_semantic_embs = embeddings / (norms + 1e-8)

    def get_user_features(self, user_id: int) -> UserFeatures:
        """
        Get features for a specific user.

        Args:
            user_id (int): The user identifier.

        Returns:
            UserFeatures: User feature object.

        Raises:
            IndexError: If user_id is out of range.
        """
        if user_id not in self._user_features:
            raise IndexError(f"User ID {user_id} not found. Valid range: 0-{self.num_users-1}")
        return self._user_features[user_id]

    def get_item_features(self, item_id: int) -> ItemFeatures:
        """
        Get features for a specific item.

        Args:
            item_id (int): The item identifier.

        Returns:
            ItemFeatures: Item feature object.

        Raises:
            IndexError: If item_id is out of range.
        """
        if item_id not in self._item_features:
            raise IndexError(f"Item ID {item_id} not found. Valid range: 0-{self.num_items-1}")
        return self._item_features[item_id]

    def get_user_history(self, user_id: int, max_len: int = 20) -> List[int]:
        """
        Get user behavior history sequence.

        Args:
            user_id (int): The user identifier.
            max_len (int): Maximum length of history to return.

        Returns:
            List[int]: List of item IDs representing user's history.

        Raises:
            IndexError: If user_id is out of range.
        """
        if user_id not in self._user_history:
            raise IndexError(f"User ID {user_id} not found. Valid range: 0-{self.num_users-1}")

        history = self._user_history[user_id].history_item_ids

        # Return the most recent max_len items
        return history[-max_len:] if len(history) > max_len else history

    def get_user_state_embs(self, user_id: int) -> np.ndarray:
        """
        Get LLM semantic embeddings for a specific user.

        Args:
            user_id (int): The user identifier.

        Returns:
            np.ndarray: User state embeddings with shape [5, 2560].

        Raises:
            IndexError: If user_id is out of range.
        """
        if user_id >= self.num_users or user_id < 0:
            raise IndexError(f"User ID {user_id} not found. Valid range: 0-{self.num_users-1}")

        return self._user_state_embs[user_id]

    def get_item_semantic_embs(self, item_id: int) -> np.ndarray:
        """
        Get LLM semantic embeddings for a specific item.

        Args:
            item_id (int): The item identifier.

        Returns:
            np.ndarray: Item semantic embedding with shape [2560].

        Raises:
            IndexError: If item_id is out of range.
        """
        if item_id >= self.num_items or item_id < 0:
            raise IndexError(f"Item ID {item_id} not found. Valid range: 0-{self.num_items-1}")

        return self._item_semantic_embs[item_id]

    def get_all_item_semantic_embs(self) -> np.ndarray:
        """
        Get all item semantic embeddings.

        This is typically used for building Faiss index.

        Returns:
            np.ndarray: All item semantic embeddings with shape [N_items, 2560].
        """
        return self._item_semantic_embs

    def get_batch_user_features(self, user_ids: List[int]) -> List[UserFeatures]:
        """
        Get features for multiple users at once.

        Args:
            user_ids (List[int]): List of user identifiers.

        Returns:
            List[UserFeatures]: List of user feature objects.
        """
        return [self.get_user_features(uid) for uid in user_ids]

    def get_batch_item_features(self, item_ids: List[int]) -> List[ItemFeatures]:
        """
        Get features for multiple items at once.

        Args:
            item_ids (List[int]): List of item identifiers.

        Returns:
            List[ItemFeatures]: List of item feature objects.
        """
        return [self.get_item_features(iid) for iid in item_ids]

    def get_batch_user_state_embs(self, user_ids: List[int]) -> np.ndarray:
        """
        Get LLM semantic embeddings for multiple users.

        Args:
            user_ids (List[int]): List of user identifiers.

        Returns:
            np.ndarray: User state embeddings with shape [len(user_ids), 5, 2560].
        """
        return self._user_state_embs[user_ids]

    def get_batch_item_semantic_embs(self, item_ids: List[int]) -> np.ndarray:
        """
        Get LLM semantic embeddings for multiple items.

        Args:
            item_ids (List[int]): List of item identifiers.

        Returns:
            np.ndarray: Item semantic embeddings with shape [len(item_ids), 2560].
        """
        return self._item_semantic_embs[item_ids]

    # ========================================================================
    # Methods for compatibility with Recall Layer
    # ========================================================================

    def load_item_semantic_embs(self) -> np.ndarray:
        """
        Load all item semantic embeddings (for Faiss indexing).

        Returns:
            np.ndarray: All item semantic embeddings with shape [N_items, 2560].
        """
        return self._item_semantic_embs

    def load_item_co_occurrence(self) -> Dict[Tuple[int, int], int]:
        """
        Compute and load item co-occurrence statistics for Item2Item recall.

        Returns:
            Dict[Tuple[int, int], int]: Mapping (item_i, item_j) -> co-occurrence count.
        """
        co_occurrence = {}
        for user_id in range(self.num_users):
            history = self._user_history[user_id].history_item_ids
            for i, item_i in enumerate(history):
                for item_j in history[i+1:]:
                    key = (min(item_i, item_j), max(item_i, item_j))
                    co_occurrence[key] = co_occurrence.get(key, 0) + 1
        return co_occurrence

    def load_hot_items(self) -> List[int]:
        """
        Generate hot/trending items list based on popularity.

        Returns:
            List[int]: Item IDs sorted by popularity (descending).
        """
        # Score items by a simple popularity metric
        item_scores = []
        for item_id in range(self.num_items):
            # Mock score: combination of random factors
            score = np.random.random() * 100
            item_scores.append((item_id, score))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in item_scores]

    def load_user_states(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load user state vectors for LLM semantic recall.

        Returns:
            Dict[int, Dict[str, np.ndarray]]: Mapping user_id to dict of 5 state vectors.
        """
        user_states = {}
        state_names = self.USER_STATE_DIMENSIONS  # ["long_term_intent", "life_stage", ...]

        for user_id in range(self.num_users):
            user_states[user_id] = {
                state_names[0]: self._user_state_embs[user_id][0],  # long_term_intent
                state_names[1]: self._user_state_embs[user_id][1],  # life_stage
                state_names[2]: self._user_state_embs[user_id][2],  # psychological_demand
                state_names[3]: self._user_state_embs[user_id][3],  # retrieval_suggestions
                state_names[4]: self._user_state_embs[user_id][4],  # interest_growth_points
            }
        return user_states

    def sample_positive_item(self, user_id: int) -> int:
        """
        Sample a positive item (item from user's history).

        Args:
            user_id (int): The user identifier.

        Returns:
            int: An item ID from user's history.
        """
        history = self._user_history[user_id].history_item_ids
        return random.choice(history)

    def sample_negative_item(self, user_id: int, num_samples: int = 1) -> Union[int, List[int]]:
        """
        Sample negative item(s) (items not in user's history).

        Args:
            user_id (int): The user identifier.
            num_samples (int): Number of negative samples to return.

        Returns:
            Union[int, List[int]]: One or more item IDs not in user's history.
        """
        history_set = set(self._user_history[user_id].history_item_ids)

        # All possible items
        all_items = set(range(self.num_items))
        negative_pool = list(all_items - history_set)

        if num_samples == 1:
            return random.choice(negative_pool)
        return random.sample(negative_pool, min(num_samples, len(negative_pool)))

    def get_statistics(self) -> Dict[str, any]:
        """
        Get dataset statistics.

        Returns:
            Dict[str, any]: Dictionary containing dataset statistics.
        """
        history_lengths = [len(h.history_item_ids) for h in self._user_history.values()]

        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dimension": self.emb_dim,
            "avg_history_length": np.mean(history_lengths),
            "min_history_length": np.min(history_lengths),
            "max_history_length": np.max(history_lengths),
            "user_state_embedding_shape": self._user_state_embs.shape,
            "item_semantic_embedding_shape": self._item_semantic_embs.shape,
        }

    def __repr__(self) -> str:
        """String representation of the DataLoader."""
        stats = self.get_statistics()
        return (
            f"DataLoader(num_users={stats['num_users']}, "
            f"num_items={stats['num_items']}, "
            f"emb_dim={stats['embedding_dimension']}, "
            f"avg_history_length={stats['avg_history_length']:.2f})"
        )


def create_dataloader(config: Optional[Dict] = None) -> DataLoader:
    """
    Factory function to create a DataLoader with optional configuration.

    Args:
        config (Dict, optional): Configuration dictionary. Supports keys:
            - num_users (int): Number of users (default: 1000)
            - num_items (int): Number of items (default: 5000)
            - emb_dim (int): Embedding dimension (default: 2560)
            - seed (int): Random seed (default: 42)

    Returns:
        DataLoader: Initialized DataLoader instance.

    Example:
        >>> loader = create_dataloader({"num_users": 500, "num_items": 2000})
        >>> print(loader.get_statistics())
    """
    if config is None:
        config = {}

    return DataLoader(
        num_users=config.get("num_users", 1000),
        num_items=config.get("num_items", 5000),
        emb_dim=config.get("emb_dim", 2560),
        seed=config.get("seed", 42)
    )


if __name__ == "__main__":
    # Demo and testing
    print("=" * 60)
    print("MuseRecSys DataLoader - Demo")
    print("=" * 60)

    # Create DataLoader with small sample size for demo
    loader = DataLoader(num_users=10, num_items=50, emb_dim=2560)

    print(f"\n{loader}")

    print("\n--- Dataset Statistics ---")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n--- User Features Example ---")
    user_feat = loader.get_user_features(0)
    print(f"User 0: {user_feat}")

    print("\n--- Item Features Example ---")
    item_feat = loader.get_item_features(0)
    print(f"Item 0: {item_feat}")

    print("\n--- User History Example ---")
    history = loader.get_user_history(0, max_len=10)
    print(f"User 0 history (last 10 items): {history}")

    print("\n--- User State Embeddings Example ---")
    user_state = loader.get_user_state_embs(0)
    print(f"User 0 state embeddings shape: {user_state.shape}")
    print(f"User 0 state embeddings sample (first dim, first 5 values): {user_state[0][:5]}")

    print("\n--- Item Semantic Embeddings Example ---")
    item_sem = loader.get_item_semantic_embs(0)
    print(f"Item 0 semantic embedding shape: {item_sem.shape}")
    print(f"Item 0 semantic embedding sample (first 5 values): {item_sem[:5]}")

    print("\n--- All Item Semantic Embeddings ---")
    all_item_sem = loader.get_all_item_semantic_embs()
    print(f"All items semantic embeddings shape: {all_item_sem.shape}")

    print("\n--- Sampling Example ---")
    positive = loader.sample_positive_item(0)
    negative = loader.sample_negative_item(0)
    print(f"Positive sample for User 0: {positive}")
    print(f"Negative sample for User 0: {negative}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
