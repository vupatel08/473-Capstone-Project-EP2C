## dataset_loader.py
import os
import glob
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import torch

# Import our custom tokenization module - assuming it's implemented as per design
from tokenization import Tokenizer

class DatasetLoader:
    """
    DatasetLoader handles loading, preprocessing, and partitioning datasets into training,
    validation, and test sets. It supports multiple datasets, heterogeneity, and hierarchical
    dataset organization for large-scale pretraining.
    """
    def __init__(self, dataset_paths: List[str], config: dict):
        """
        Args:
            dataset_paths (List[str]): List of dataset directories or file paths.
            config (dict): Configuration dictionary with preprocessing parameters.
        """
        self.dataset_paths = dataset_paths
        self.config = config
        self.train_ratio = self.config.get('dataset', {}).get('dataset_split_ratio', 0.8)
        # Initiate internal storage
        self.raw_data = []  # List of dataset dicts
        self.train_series = []  # List of normalized training series
        self.val_series = []    # List of normalized validation series
        self.test_series = []   # List of normalized test series
        self.series_metadata = []  # To store metadata like length, mean, std, frequency, source info

        # Load datasets
        self.load_data()

    def load_data(self):
        """
        Load raw datasets from given paths.
        Supports CSV, Parquet, or other readable formats.
        """
        # For each dataset path, load data
        for path in self.dataset_paths:
            dataset_dict = {
                'name': os.path.basename(path),
                'series_list': [],   # List of raw series (np.ndarray)
                'metadata': {}       # Dict for metadata info
            }
            # Determine file type and read accordingly
            files = []
            if os.path.isdir(path):
                # Load all csv/parquet files
                files = glob.glob(os.path.join(path, '*.csv')) + glob.glob(os.path.join(path, '*.parquet'))
            elif os.path.isfile(path):
                files = [path]
            else:
                continue  # Path does not exist

            for f in files:
                series_data, timestamp_array = self.read_series_file(f)
                # Handle missing values: linear interpolation
                if isinstance(series_data, pd.DataFrame):
                    series_array = self.preprocess_series_dataframe(series_data)
                elif isinstance(series_data, np.ndarray):
                    series_array = self.preprocess_series_array(series_data)
                else:
                    continue
                # Store the raw series and metadata
                dataset_dict['series_list'].append({
                    'series': series_array,
                    'timestamp': timestamp_array,
                    'source': dataset_dict['name']
                })
            # Save dataset info
            self.raw_data.append(dataset_dict)
        # After loading all datasets, process normalization and splitting
        self._normalize_and_split()

    def read_series_file(self, filename: str) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Read a time series file into pandas DataFrame.
        Supports CSV and Parquet with timestamp and variate columns.
        """
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filename)
        else:
            return None, None
        # Expect timestamp column if exists
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else None
        timestamp_array = None
        if timestamp_col:
            timestamp_array = pd.to_datetime(df[timestamp_col]).astype(np.float64).values
            df = df.drop(columns=[timestamp_col])
        return df, timestamp_array

    def preprocess_series_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Handle missing values via linear interpolation, convert to numpy array.
        """
        df_interpolated = df.interpolate(method='linear', limit_direction='both')
        # Convert to numpy array of shape (T, variates)
        series_array = df_interpolated.values
        return series_array

    def preprocess_series_array(self, array: np.ndarray) -> np.ndarray:
        """
        Handle missing data if necessary (assumed clean here).
        """
        # If array has NaNs, interpolate
        if np.isnan(array).any():
            for col in range(array.shape[1]):
                col_data = pd.Series(array[:, col])
                array[:, col] = col_data.interpolate(method='linear', limit_direction='both').values
        return array

    def _normalize_and_split(self):
        """
        Normalize all series in training set, then split into train/val/test.
        """
        # Gather all training series
        train_series_all = []
        for dataset in self.raw_data:
            for item in dataset['series_list']:
                train_series_all.append(item['series'])
        # Compute mean/std on training data
        all_train_data = np.concatenate(train_series_all, axis=0)
        # Compute mean and std for each feature
        feature_mean = np.mean(all_train_data, axis=0)
        feature_std = np.std(all_train_data, axis=0) + 1e-8  # Avoid division by zero

        # Normalize and assign back
        for dataset in self.raw_data:
            for item in dataset['series_list']:
                series = item['series']
                normalized_series = (series - feature_mean) / feature_std
                item['series'] = normalized_series
                # Save metadata
                self.series_metadata.append({
                    'length': normalized_series.shape[0],
                    'mean': feature_mean,
                    'std': feature_std,
                    'source': item['source']
                })

        # Split each series into train/val/test based on ratio, keep chronological
        for dataset in self.raw_data:
            for item in dataset['series_list']:
                length = item['series'].shape[0]
                split_point = int(length * self.train_ratio)
                item['train'] = item['series'][:split_point]
                item['val'] = item['series'][split_point:]
                item['test'] = item['series'][split_point:]
        # Store separate lists for train/val/test series
        self.train_series = [item['train'] for d in self.raw_data for item in d['series_list']]
        self.val_series = [item['val'] for d in self.raw_data for item in d['series_list']]
        self.test_series = [item['test'] for d in self.raw_data for item in d['series_list']]

    def get_series_split(self, split: str) -> List[np.ndarray]:
        """
        Return list of series for 'train', 'val', or 'test'
        """
        if split == 'train':
            return self.train_series
        elif split == 'val':
            return self.val_series
        elif split == 'test':
            return self.test_series
        else:
            raise ValueError(f"Unknown split: {split}")

    def tokenize_series(self, series: np.ndarray, token_length: int) -> List[List[float]]:
        """
        Convert a single series into a list of tokens (sub-sequences) of token_length.
        Handles possible irregular sampling by directly segmenting.
        """
        tokens = []
        length = series.shape[0]
        # Use non-overlapping segmentation
        for start in range(0, length - token_length + 1, token_length):
            token = series[start:start + token_length]
            tokens.append(token)
        return tokens

    def prepare_tokenized_dataset(self, series_list: List[np.ndarray], token_length: int) -> List[np.ndarray]:
        """
        Tokenize all series in a list, convert to tensors.
        """
        tokenized = []
        for series in series_list:
            tokens = self.tokenize_series(series, token_length)
            for token in tokens:
                tokenized.append(torch.tensor(token, dtype=torch.float32))
        return tokenized

    def convert_to_unified_sequence(self, series_list: List[np.ndarray]) -> torch.Tensor:
        """
        Concatenate tokenized series into one long sequence of tokens.
        """
        sequence = []
        for series in series_list:
            tokens = self.tokenize_series(series, self.config.get('dataset', {}).get('segment_length', 96))
            sequence.extend(tokens)
        # Stack tokens into tensor of shape (N, S), where N is total tokens, S is token length (or feature dims)
        tensor_seq = torch.stack(sequence, dim=0)
        return tensor_seq

    def get_dataset_for_task(self, split: str, token_length: int = 96) -> List[torch.Tensor]:
        """
        Retrieve tokenized dataset for downstream task.
        Args:
            split (str): 'train', 'val', 'test'
            token_length (int): length of each token segment
        """
        series_data = self.get_series_split(split)
        return self.prepare_tokenized_dataset(series_data, token_length)

