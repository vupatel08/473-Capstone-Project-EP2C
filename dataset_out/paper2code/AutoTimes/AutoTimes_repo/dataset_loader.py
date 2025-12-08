## dataset_loader.py
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class TimeSeriesSample:
    series: np.ndarray             # Shape: (T, C)
    timestamps: np.ndarray         # Shape: (T,), dtype: object or datetime
    series_name: str
    series_metadata: Dict = field(default_factory=dict)

class DataLoader:
    def __init__(self, dataset_paths: Dict[str, str], segment_size: int = 96):
        """
        Initialize DataLoader with dataset paths and configuration.
        Args:
            dataset_paths (dict): Keys are dataset identifiers, values are CSV file paths.
            segment_size (int): Length of segments for tokenization.
        """
        self.dataset_paths = dataset_paths
        self.segment_size = segment_size
        # Containers for loaded datasets
        self.datasets: Dict[str, List[TimeSeriesSample]] = {}
        # Store normalization parameters per dataset
        self.norm_params: Dict[str, Dict] = {}

    def load_all(self):
        """
        Load all datasets specified in dataset_paths.
        Returns:
            dict: Keys are dataset identifiers, values are list of TimeSeriesSample objects.
        """
        for name, path in self.dataset_paths.items():
            self.datasets[name] = self._load_dataset(path, name)
        return self.datasets

    def _load_dataset(self, path: str, dataset_name: str) -> List[TimeSeriesSample]:
        """
        Load a single dataset CSV file.
        Args:
            path (str): Path to CSV file.
            dataset_name (str): Name identifier for dataset.
        Returns:
            List[TimeSeriesSample]: List of time series samples.
        """
        data_samples: List[TimeSeriesSample] = []

        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Load the dataset CSV
        df = pd.read_csv(path)

        # Assume the dataset includes a 'timestamp' column and variate columns
        # If columns differ, user should modify accordingly.
        if 'timestamp' not in df.columns:
            raise ValueError(f"'timestamp' column not found in {path}")

        time_col = 'timestamp'
        var_cols = [col for col in df.columns if col != time_col]
        # Convert timestamp column to datetime
        df[time_col] = pd.to_datetime(df[time_col])

        # Extract series as numpy array
        series_data = df[var_cols].values.astype(np.float32)  # shape (T, C)
        timestamps = df[time_col].values  # ndarray of pandas Timestamps

        # Normalize series: fit on entire series initially, or defer normalization to train later
        # Here, for simplicity, normalize using min-max (per series)
        # For realistic training, normalization is fit on training set only
        # Extension: normalization on training dataset only is recommended when used in custome train scripts
        series_min = series_data.min(axis=0)
        series_max = series_data.max(axis=0)
        # Avoid division by zero
        denom = series_max - series_min
        denom[denom == 0] = 1.0
        normalized_series = (series_data - series_min) / denom

        # Store normalization params for potential future use
        self.norm_params[dataset_name] = {
            'min': series_min,
            'max': series_max
        }

        # Segment the series into non-overlapping segments
        T = normalized_series.shape[0]
        total_segments = T // self.segment_size
        segments = []
        segment_timestamps = []

        for i in range(total_segments):
            start_idx = i * self.segment_size
            end_idx = start_idx + self.segment_size
            segment = normalized_series[start_idx:end_idx, :]  # shape (S, C)
            segments.append(segment)

            # For the segment's timestamp, use the starting timestamp
            segment_start_time = timestamps[start_idx]
            segment_timestamps.append(segment_start_time)

        # Create TimeSeriesSample
        series_full = normalized_series  # shape (T, C)
        # Pack all info
        sample = TimeSeriesSample(
            series=series_full,
            timestamps=np.array(timestamps),  # store original timestamps
            series_name=dataset_name,
            series_metadata={'segment_timestamps': np.array(segment_timestamps)}
        )

        return [sample]

    def get_normalization_params(self, dataset_name: str):
        """
        Retrieve normalization parameters for a dataset.
        Args:
            dataset_name (str): Dataset identifier.
        Returns:
            dict: containing 'min' and 'max' arrays.
        """
        return self.norm_params.get(dataset_name, None)
