## buffer.py
import torch
import numpy as np
from typing import List, Tuple, Optional
import random

class ReplayBuffer:
    """
    Buffer management for off-policy exploration and low-energy sample storage.
    Implements FIFO eviction policy with optional prioritization based on sample energies.
    """

    def __init__(self, capacity: int, prioritize: bool = True, priority_k: float = 0.01):
        """
        Initializes the replay buffer.
        Args:
            capacity (int): Maximum number of samples to store in the buffer.
            prioritize (bool): Whether to use priority-based sampling.
            priority_k (float): Hyperparameter for rank-based priority (k=0.01 default).
        """
        self.capacity: int = capacity
        self.prioritize: bool = prioritize
        self.priority_k: float = priority_k

        # Data storage
        self.samples: torch.Tensor = torch.empty((capacity, 0))  # Will be initialized on first add
        self.energies: torch.Tensor = torch.empty((capacity,), dtype=torch.float32)
        self.conditions: Optional[torch.Tensor] = None  # For conditional models; optional

        # Queue tracking for FIFO eviction
        self.next_idx: int = 0
        self.size: int = 0

        # Auxiliary data for priority ranking
        self._sorted_indices: torch.Tensor = torch.tensor([], dtype=torch.long)
        self._sorted_energies: torch.Tensor = torch.tensor([], dtype=torch.float32)

    def initialize(self, input_dim: int, condition_dim: Optional[int] = None):
        """
        Initialize tensors after knowing input dimensions.
        """
        self.samples = torch.empty((self.capacity, input_dim))
        self.energies = torch.full((self.capacity,), float('inf'))  # initialize with high energies
        if condition_dim is not None and condition_dim > 0:
            self.conditions = torch.empty((self.capacity, condition_dim))
        else:
            self.conditions = None

    def add(self, sample: torch.Tensor, energy: float, condition: Optional[torch.Tensor] = None):
        """
        Add a new sample with its energy (and optional condition) to the buffer.
        Evicts oldest sample if buffer is full.
        """
        # If buffer uninitialized, allocate memory
        if self.samples.shape[1] == 0:
            input_dim = sample.shape[1]
            self.initialize(input_dim, condition_dim=condition.shape[1] if condition is not None else None)

        idx = self.next_idx
        # Store sample
        self.samples[idx] = sample.detach().cpu()
        self.energies[idx] = energy
        if self.conditions is not None and condition is not None:
            self.conditions[idx] = condition.detach().cpu()

        # Update sorted energies and indices for priority sampling
        self._update_priorities()

        # Update FIFO pointer
        self.next_idx = (self.next_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def _update_priorities(self):
        """
        Update a sorted view of energies for priority sampling (rank-based).
        """
        if self.size == 0:
            self._sorted_indices = torch.tensor([], dtype=torch.long)
            self._sorted_energies = torch.tensor([], dtype=torch.float32)
            return
        # Get valid energies and indices
        valid_energies = self.energies[:self.size]
        # Argsort for ascending order (lower energy = higher priority)
        self._sorted_energies, self._sorted_indices = torch.sort(valid_energies)
        # Store indices relative to buffer
        self._sorted_indices = self._sorted_indices

    def sample(self, batch_size: int, prioritized: bool = True, sample_condition: bool = False) -> List[Tuple[torch.Tensor, float, Optional[torch.Tensor]]]:
        """
        Sample a batch of samples from the buffer.
        Args:
            batch_size (int): Number of samples to retrieve.
            prioritized (bool): Whether to sample according to priority.
            sample_condition (bool): Whether to return conditions.
        Returns:
            List of tuples: (sample tensor, energy, condition or None)
        """
        if self.size == 0:
            raise ValueError("Buffer is empty. Add samples before sampling.")

        if prioritized:
            # Compute probability distribution based on rank
            # Priority p(x) ∝ (k * |D| + rank(x))^-1
            # Rank is inverse of order in sorted energies; highest priority for lowest energy
            energies = self.energies[:self.size]
            # Generate ranks: 1 (best) to size (worst)
            ranks = torch.argsort(energies).argsort() + 1  # ranks start at 1
            weights = (self.priority_k * self.size + ranks.float()).pow(-1)
            probs = weights / torch.sum(weights)
            # Sample indices according to probs
            sampled_indices = np.random.choice(self.size, size=batch_size, replace=True, p=probs.cpu().numpy())
        else:
            # Uniform sampling
            sampled_indices = np.random.choice(self.size, size=batch_size, replace=True)

        batch_samples = []
        for idx in sampled_indices:
            sample = self.samples[idx]
            energy = float(self.energies[idx])
            condition = None
            if self.conditions is not None:
                condition = self.conditions[idx]
            batch_samples.append((sample.clone(), energy, condition))
        return batch_samples

    def maintain(self):
        """
        Placeholder for maintenance routines.
        """
        # For FIFO, eviction is handled during add itself.
        # Optionally, implement re-prioritization, pruning, or recalculate priorities.
        pass

    def get_all_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return all stored samples and energies as tensors.
        """
        return self.samples[:self.size], self.energies[:self.size]

    def update_priorities(self):
        """
        Recompute the sorted energies and indices.
        Can be called externally if energies are updated post hoc.
        """
        self._update_priorities()
