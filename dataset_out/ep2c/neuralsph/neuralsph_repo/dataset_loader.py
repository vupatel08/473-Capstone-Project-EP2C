## dataset_loader.py
import os
import numpy as np
from typing import Optional, Tuple, List, Dict


class Dataset:
    """
    Simple dataset object to store sequences of positions, velocities,
    particle types, and optional external forces.
    """
    def __init__(self,
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 types: np.ndarray,
                 external_forces: Optional[np.ndarray] = None):
        self.positions = positions  # shape: (N_seq, seq_len, N_particles, dim)
        self.velocities = velocities  # shape: (N_seq, seq_len, N_particles, dim)
        self.types = types  # shape: (N_seq, N_particles)
        self.external_forces = external_forces  # shape: same as positions


class DatasetLoader:
    def __init__(self, dataset_path: str, config: Dict):
        """
        Loads dataset from the specified path, processes it, and stores in memory.

        Args:
            dataset_path (str): Path to dataset directory or files.
            config (Dict): Configuration dictionary with keys:
                - sequence_length (int): length of sequences for evaluation.
                - training_subsequence_interval (int): interval for training sampling.
        """
        self.dataset_path = dataset_path
        self.sequence_length = config.get('dataset', {}).get('sequence_length', 400)
        self.subsample_interval = config.get('dataset', {}).get('training_subsequence_interval', 100)

        # Internal storage
        self.positions_all = None  # shape: (total_samples, max_seq_len, N_particles, dim)
        self.velocities_all = None
        self.types_all = None
        self.forces_all = None

        # Load data
        self._load_data()

        # After loading, determine number of sequences
        self.total_sequences = self.positions_all.shape[0]

    def _load_data(self):
        """
        Loads dataset files from the path, supports npz or npy, or custom format.
        Assumes data stored with keys: 'positions', optionally 'forces', 'types'.
        """
        # Find data files in directory
        files = []
        if os.path.isdir(self.dataset_path):
            for fname in os.listdir(self.dataset_path):
                if fname.endswith('.npz') or fname.endswith('.npy'):
                    files.append(os.path.join(self.dataset_path, fname))
        elif os.path.isfile(self.dataset_path):
            files = [self.dataset_path]
        else:
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found.")

        # For simplicity, if multiple files, load and concatenate
        pos_list = []
        force_list = []
        type_list = []

        for fpath in files:
            if fpath.endswith('.npz'):
                data = np.load(fpath)
                positions = data['positions']  # shape: (N_seq, seq_len, N_particles, dim)
                if 'forces' in data:
                    forces = data['forces']
                else:
                    forces = None
                if 'types' in data:
                    types = data['types']
                else:
                    types = None
            elif fpath.endswith('.npy'):
                # Assume npy stores sequence of positions
                positions = np.load(fpath)  # shape: (N_seq, seq_len, N_particles, dim)
                forces = None
                types = None
            else:
                continue  # unsupported format

            pos_list.append(positions)
            if forces is not None:
                force_list.append(forces)
            if types is not None:
                type_list.append(types)

        if len(pos_list) == 0:
            raise ValueError("No valid dataset files found.")

        # Concatenate data from all files
        self.positions_all = np.concatenate(pos_list, axis=0)
        if force_list:
            self.forces_all = np.concatenate(force_list, axis=0)
        else:
            self.forces_all = None
        if type_list:
            self.types_all = np.concatenate(type_list, axis=0)
        else:
            # If no types provided, assign default type 0
            num_particles = self.positions_all.shape[2]
            self.types_all = np.zeros((self.positions_all.shape[0], num_particles), dtype=np.int32)

        # Validate data shapes
        N_seq, seq_len, N_particles, dim = self.positions_all.shape
        assert self.types_all.shape[0] == N_seq
        assert self.types_all.shape[1] == N_particles
        if self.forces_all is not None:
            assert self.forces_all.shape == (N_seq, seq_len, N_particles, dim)
        # Velocities can be derived or stored; here, we'll compute during getitem

    def get_sequence(self, index: int) -> Dict:
        """
        Returns a full sequence sample at index, including positions, velocities,
        types, and external forces if available.

        Args:
            index (int): index of the sequence.
        Returns:
            dict with keys:
                - 'positions': (seq_len, N_particles, dim)
                - 'velocities': (seq_len, N_particles, dim)
                - 'types': (N_particles,)
                - 'external_forces': (seq_len, N_particles, dim) or None
        """
        if index < 0 or index >= self.total_sequences:
            raise IndexError("Sequence index out of bounds.")

        pos_seq = self.positions_all[index]  # shape: (seq_len, N_particles, dim)
        # Derive velocities via finite differences, shape: (seq_len, N_particles, dim)
        velocities = np.zeros_like(pos_seq)
        velocities[1:] = pos_seq[1:] - pos_seq[:-1]
        velocities[0] = velocities[1]  # assign first timestep same as second for consistency

        types_seq = self.types_all[index]  # shape: (N_particles,)

        if self.forces_all is not None:
            forces_seq = self.forces_all[index]
        else:
            forces_seq = None

        sample = {
            'positions': pos_seq.astype(np.float32),
            'velocities': velocities.astype(np.float32),
            'types': types_seq.astype(np.int32),
            'external_forces': forces_seq.astype(np.float32) if forces_seq is not None else None
        }
        return sample

    def get_subsequence(self, index: int, start_time: int) -> Dict:
        """
        Get a subsequence of length self.sequence_length starting from start_time.

        Args:
            index (int): sequence index.
            start_time (int): starting timestep.

        Returns:
            dict with same keys as get_sequence, but truncated to subsequence length.
        """
        seq = self.get_sequence(index)
        end_time = start_time + self.sequence_length
        # Clip if necessary
        if end_time > seq['positions'].shape[0]:
            raise ValueError("Subsequence end exceeds sequence length.")
        subseq = {
            'positions': seq['positions'][start_time:end_time],
            'velocities': seq['velocities'][start_time:end_time],
            'types': seq['types'],
            'external_forces': None
        }
        if seq['external_forces'] is not None:
            subseq['external_forces'] = seq['external_forces'][start_time:end_time]
        return subseq

    def get_random_batch(self, batch_size: int) -> List[Dict]:
        """
        Randomly sample a list of sequences for batch training.

        Args:
            batch_size (int): number of sequences to sample.

        Returns:
            list of sample dictionaries.
        """
        indices = np.random.choice(self.total_sequences, size=batch_size, replace=False)
        return [self.get_sequence(i) for i in indices]
