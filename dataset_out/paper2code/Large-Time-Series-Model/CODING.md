# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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

```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.functional import mse_loss
from typing import Dict, List, Optional
import os

# Import the model class (assuming it is in model.py)
from model import TimerTransformer

class Evaluation:
    """
    Evaluation class for assessing trained Timer models on multiple downstream tasks:
    forecasting, imputation, anomaly detection, and zero-shot evaluation.
    """
    def __init__(self, 
                 model: TimerTransformer,
                 dataset_loader,
                 task: str,
                 task_params: dict,
                 config: dict):
        """
        Initialize the Evaluation instance.
        
        Args:
            model (TimerTransformer): The pre-trained or fine-tuned model to evaluate.
            dataset_loader: An object providing datasets and data iterators (e.g., DatasetLoader).
            task (str): The task type: 'forecasting', 'imputation', 'anomaly_detection', 'zero_shot'.
            task_params (dict): Task-specific parameters, e.g., dataset info, masking ratios, forecast lengths.
            config (dict): Evaluation configuration, including metrics, thresholds, logging.
        """
        self.model = model
        self.model.eval()
        self.dataset_loader = dataset_loader
        self.task = task
        self.task_params = task_params
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Load metrics configuration
        self.metrics_config = self.config.get('evaluation', {})
        self.forecast_metrics = self.metrics_config.get('forecast_metrics', ['MSE', 'MAE'])
        self.imputation_metrics = self.metrics_config.get('imputation_metrics', ['MSE'])
        self.detection_metrics = self.metrics_config.get('anomaly_detection_metrics', ['precision', 'recall', 'F1'])

        # Setup logging directory
        self.output_dir = self.config.get('logging', {}).get('save_dir', 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)

        # Store results
        self.results = {}

    def evaluate(self):
        """
        Main method to run evaluation based on task type.
        """
        if self.task == 'forecasting':
            self._evaluate_forecasting()
        elif self.task == 'imputation':
            self._evaluate_imputation()
        elif self.task == 'anomaly_detection':
            self._evaluate_anomaly_detection()
        elif self.task == 'zero_shot':
            self._evaluate_zero_shot()
        else:
            raise ValueError(f"Unknown task type: {self.task}")
        # Save results
        self._save_results()

    def _evaluate_forecasting(self):
        """
        Evaluate forecasting task using autoregressive generation.
        """
        forecast_length = self.task_params.get('forecast_length', 96)
        lookback_length = self.task_params.get('lookback_length', 672)
        dataset = self.dataset_loader

        all_mse = {metric: [] for metric in self.forecast_metrics}
        all_mae = {metric: [] for metric in self.forecast_metrics}

        for dataset_name, dataset_obj in dataset.raw_data.items():
            for series_item in dataset_obj['series_list']:
                series = series_item['series']
                start_idx = 0
                T = series.shape[0]
                while start_idx + lookback_length + forecast_length <= T:
                    context_seq = series[start_idx:start_idx + lookback_length]
                    true_future = series[start_idx + lookback_length:start_idx + lookback_length + forecast_length]
                    
                    # Prepare input tensor
                    context_tensor, timestamp_list = self._prepare_series_for_model(context_seq, series_item.get('timestamp', None))
                    context_tensor = context_tensor.unsqueeze(0).to(self.device)  # batch size 1

                    # Autoregressive generation
                    generated_sequence = self._autoregressive_generate(context_tensor, forecast_length)

                    # Extract predicted tokens (assumed last forecast_length tokens)
                    pred_future = generated_sequence.squeeze(0)[-forecast_length:].cpu().numpy()

                    # Compute metrics
                    mse_value = np.mean((pred_future - true_future) ** 2)
                    mae_value = np.mean(np.abs(pred_future - true_future))
                    
                    if 'MSE' in self.forecast_metrics:
                        all_mse['MSE'].append(mse_value)
                    if 'MAE' in self.forecast_metrics:
                        all_mae['MAE'].append(mae_value)
                    
                    start_idx += 1

        # Aggregate metrics
        self.results['forecasting'] = {
            'MSE': np.mean(all_mse.get('MSE', [np.nan])),
            'MAE': np.mean(all_mae.get('MAE', [np.nan]))
        }

    def _autoregressive_generate(self, input_tensor: torch.Tensor, forecast_length: int):
        """
        Generate forecast sequence autoregressively.
        """
        generated_seq = input_tensor
        for _ in range(forecast_length):
            preds = self.model(generated_seq)
            # preds shape: (batch_size, seq_len, output_dim)
            next_token = preds[:, -1, :]  # last token prediction
            # Append
            generated_seq = torch.cat([generated_seq, next_token.unsqueeze(1)], dim=1)
        return generated_seq

    def _prepare_series_for_model(self, series: np.ndarray, timestamp: Optional[float]):
        """
        Convert series to input tensor suitable for model.
        """
        # Convert series to sequence of token IDs or embeddings as needed
        # For inference, we assume series are already tokenized or continuous vectors
        # Here, placeholder: for forecasting, series is just a tensor
        tensor_series = torch.tensor(series, dtype=torch.float32)
        # If necessary, expand dims (batch)
        return tensor_series, [timestamp]

    def _evaluate_imputation(self):
        """
        Evaluate imputation task with masked segments.
        """
        # Retrieve datasets
        dataset = self.dataset_loader
        mask_ratio = self.task_params.get('mask_ratio', 0.25)
        total_mse = []

        for dataset_name, dataset_obj in dataset.raw_data.items():
            for series_item in dataset_obj['series_list']:
                series = series_item['series']
                # Mask segments randomly
                masked_series, masks = self._apply_mask(series, mask_ratio)
                # Prepare input
                input_tensor, timestamp_list = self._prepare_series_for_model(masked_series, series_item.get('timestamp', None))
                input_tensor = input_tensor.unsqueeze(0).to(self.device)  # batch size 1
                
                # Generate missing parts
                output = self.model(input_tensor)
                pred_series = output.squeeze(0).cpu().detach().numpy()

                # Compute MSE on masked regions
                mse_value = np.mean((pred_series[masks] - series[masks]) ** 2)
                total_mse.append(mse_value)

        avg_mse = np.mean(total_mse)
        self.results['imputation'] = {
            'MSE': avg_mse,
            'Mask Ratio': mask_ratio
        }

    def _apply_mask(self, series: np.ndarray, mask_ratio: float):
        """
        Mask a segment of the series with zeros based on mask_ratio.
        Returns masked series and boolean mask array.
        """
        length = series.shape[0]
        mask_length = int(length * mask_ratio)
        start_idx = np.random.randint(0, length - mask_length + 1)
        mask_array = np.zeros_like(series, dtype=bool)
        mask_array[start_idx:start_idx + mask_length] = True
        masked_series = np.copy(series)
        masked_series[start_idx:start_idx + mask_length] = 0  # Set masked region to zero
        return masked_series, mask_array

    def _evaluate_anomaly_detection(self):
        """
        Evaluate anomaly detection via predictive errors or likelihood.
        """
        dataset = self.dataset_loader
        all_errors = []
        all_labels = []

        # For simplicity, assume we have test series with ground truth labels indicating anomalies
        # That could be in task_params or dataset object
        for dataset_name, dataset_obj in dataset.raw_data.items():
            for series_item in dataset_obj['series_list']:
                series = series_item['series']
                # For each series, predict next tokens and compute errors
                input_series = series[:-self.task_params.get('forecast_length', 96)]
                true_series = series[-self.task_params.get('forecast_length', 96):]

                input_tensor, timestamp_list = self._prepare_series_for_model(input_series, series_item.get('timestamp', None))
                input_tensor = input_tensor.unsqueeze(0).to(self.device)

                preds = self.model(input_tensor)
                pred_series = preds.squeeze(0).cpu().detach().numpy()

                # Evaluate per segment or per step
                errors = np.mean((pred_series - true_series) ** 2, axis=1)
                # Determine error threshold at specified quantile
                quantile = self.task_params.get('quantile', 0.95)
                threshold = np.quantile(errors, quantile)
                # Predict anomalies where error > threshold
                pred_labels = (errors > threshold).astype(int)

                # Ground truth labels for anomalies could be provided in dataset; here, we simulate or load
                # For demonstration, assume no ground truth labels, so just compute the distribution
                # (In real scenario, replace with actual labels)
                ground_truth = np.zeros_like(errors)

                # Collect metrics
                all_errors.extend(errors)
                all_labels.extend(ground_truth)

        # For detection metrics, binarize errors based on threshold
        pred_anomalies = (np.array(all_errors) > np.quantile(all_errors, self.task_params.get('quantile', 0.95))).astype(int)
        true_labels = np.array(all_labels)

        if len(np.unique(true_labels)) > 1:
            precision = precision_score(true_labels, pred_anomalies)
            recall = recall_score(true_labels, pred_anomalies)
            f1 = f1_score(true_labels, pred_anomalies)
        else:
            precision = recall = f1 = np.nan  # Undefined if no positive labels

        self.results['anomaly_detection'] = {
            'precision': precision,
            'recall': recall,
            'F1': f1
        }

    def _evaluate_zero_shot(self):
        """
        Evaluate zero-shot: run model directly on new datasets without fine-tuning.
        """
        # Load datasets not used in pretraining; assume they are available via dataset_loader
        # For each dataset, perform sequence generation and compute errors
        zero_shot_errors = []
        dataset = self.dataset_loader

        for dataset_name, dataset_obj in dataset.raw_data.items():
            for series_item in dataset_obj['series_list']:
                series = series_item['series']
                input_seq = series[:self.task_params.get('context_length', 672)]
                true_future = series[self.task_params.get('context_length', 672):]

                input_tensor, timestamp_list = self._prepare_series_for_model(input_seq, series_item.get('timestamp', None))
                input_tensor = input_tensor.unsqueeze(0).to(self.device)

                # Generate predictions autoregressively
                output_seq = self.model.generate(input_tensor, max_new_tokens=len(true_future))
                pred_series = output_seq.squeeze(0).cpu().detach().numpy()

                # Compute error
                error = np.mean((pred_series - true_future) ** 2)
                zero_shot_errors.append(error)

        # Compute average error
        avg_error = np.mean(zero_shot_errors)
        self.results['zero_shot'] = {
            'average_MSE': avg_error
        }

    def _save_results(self):
        """
        Save the evaluation metrics to file.
        """
        import json
        save_path = os.path.join(self.output_dir, f"{self.task}_evaluation_results.json")
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)
```

## main.py

```python
## main.py
import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Import modules following the defined interfaces
from dataset_loader import DatasetLoader
from tokenization import Tokenizer
from model import TimerTransformer
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Parse configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Set Random seeds for reproducibility
    seed = cfg.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 3. Prepare logging/checkpoint directories
    save_dir = cfg.get('logging', {}).get('save_dir', 'checkpoints/')
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. Instantiate DatasetLoader
    dataset_paths = []
    for ds_cfg in cfg['dataset']['pretraining_datasets']:
        dataset_paths.append(ds_cfg['path'])
    dataset_loader = DatasetLoader(dataset_paths, cfg)
    
    # 5. Instantiate Tokenizer for pretraining
    hierarchy_levels = cfg['dataset'].get('dataset_species_levels', [])
    # Extract segment lengths for hierarchy levels
    segment_lengths = []
    for level in hierarchy_levels:
        level_name = level.get('name', '')
        # Default pattern; adjust if necessary
        if '96' in level_name:
            segment_lengths.append(96)
        elif '672' in level_name:
            segment_lengths.append(672)
        elif '1440' in level_name:
            segment_lengths.append(1440)
        else:
            segment_lengths.append(96)  # fallback

    tokenizer = Tokenizer(
        hierarchy_levels=['small','medium','large'],
        segment_lengths=segment_lengths,
        use_timestamps=True,
        max_sequence_length=cfg['training'].get('max_sequence_length', 1440),
        embedding_dim=cfg['model'].get('hidden_size', 512)
    )
    
    # 6. Convert datasets to tokenized sequences for pretraining
    # Gather all training series (from all datasets and split)
    train_series_list = dataset_loader.get_series_split('train')
    # Convert to tokenized sequences (list of tensors)
    tokenized_train_sequences = []
    for series in train_series_list:
        seq_obj = tokenizer.convert_series_to_sequence(series)
        seq_ids, _ = seq_obj.to_id_tensor()
        tokenized_train_sequences.append(seq_ids)
    
    # 7. Build model
    model_size = cfg['model'].get('size_m', 50)  # in millions
    size_multiplier = {
        1: 1, 50: 1, 91: 2, 311: 3, 385: 4
    }.get(model_size, 50)
    model_params = {
        'size': cfg['model'].get('hidden_size', 512),
        'num_layers': cfg['model'].get('num_layers', 6),
        'num_heads': cfg['model'].get('num_heads', 8),
        'max_position_embeddings': cfg['model'].get('max_position_embeddings', 1024),
        'dropout': cfg['model'].get('dropout_rate', 0.1),
        'input_token_length': cfg['model'].get('input_token_length', 96),
        'use_positional_embedding': True,
        'use_timestamp_embedding': True
    }
    model = TimerTransformer(**model_params).to(device)
    
    # 8. Load pretrained checkpoint if enabled (e.g., for fine-tuning)
    pretrained_path = None
    # Decide based on typical configs, or add config option
    if cfg.get('training', {}).get('pretrained_checkpoint_path'):
        pretrained_path = cfg['training']['pretrained_checkpoint_path']
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Pretrained checkpoint not found at {pretrained_path}")
    
    # 9. Initialize Trainer with dataset, model, hyperparameters
    trainer_cfg = cfg['training']
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_sequences,
        val_dataset=None,  # Could set validation sequences similarly if validation data is available
        config=trainer_cfg
    )
    
    # 10. Pretraining phase
    pretrain_flag = cfg.get('training', {}).get('do_pretrain', True)
    if pretrain_flag:
        print("Starting pretraining...")
        trainer.train()
        # Save final pretrained checkpoint
        final_ckpt_path = os.path.join(save_dir, 'pretrain_final.pt')
        trainer.save_checkpoint(final_ckpt_path)
        print(f"Pretraining completed. Checkpoint saved at {final_ckpt_path}")
        # Load the final checkpoint into model
        ckpt = torch.load(final_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Skipping pretraining as per configuration.")
    
    # 11. Fine-tuning / downstream tasks
    downstream_task = cfg.get('task', 'forecasting')  # default to forecasting
    task_params = cfg.get('task_params', {})
    # Load or prepare datasets for downstream task, similar to training but task-specific
    # For simplicity, assume datasets for downstream are prepared similarly
    # For real usage, datasets are loaded and tokenized per task
    # For this code, as per instruction, invoke evaluation only
    
    # Prepare dataset for downstream task
    # For example, for forecasting:
    downstream_dataset_sequences = []
    # Assume task datasets are stored or loaded accordingly
    # Here, we just assume test series (mocked)
    test_series_list = dataset_loader.get_series_split('test')
    # Convert each to token sequences with task-specific parameters
    for series in test_series_list:
        seq_obj = tokenizer.convert_series_to_sequence(series)
        seq_ids, _ = seq_obj.to_id_tensor()
        downstream_dataset_sequences.append(seq_ids)
    
    # Initialize Evaluation object
    eval_obj = Evaluation(
        model=model,
        dataset_loader=dataset_loader,
        task=downstream_task,
        task_params=task_params,
        config=cfg
    )
    # Run evaluation (can be forecast, impute, detect)
    eval_obj.evaluate()
    print(f"Evaluation for task '{downstream_task}' completed. Results:")
    print(eval_obj.results)
    
    # 12. Save evaluation results and optionally generate figures
    # Save results json
    results_path = os.path.join(cfg.get('logging', {}).get('save_dir', 'checkpoints/'), 'evaluation_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_obj.results, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """Single decoder (self-attention + feedforward) layer with causal masking."""
    def __init__(self, size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(size, size * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(size * 4, size)
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + ff_output)
        return x

class TimerTransformer(nn.Module):
    """
    GPT-style decoder-only transformer for large time series modeling.
    
    Supports:
    - Hierarchical scaling via model size and layers
    - Autoregressive next token prediction
    - Optional position and timestamp embeddings
    """
    def __init__(self, 
                 size: int = 512,                     # Hidden dimension D
                 num_layers: int = 6,                 # Number of decoder layers L
                 num_heads: int = 8,                  # Attention heads
                 ff_dim_multiplier: int = 2,          # FFN dimension multiplier
                 max_position_embeddings: int = 1024, # Max sequence length
                 dropout: float = 0.1,                # Dropout probability
                 input_token_length: int = 96,        # Token length S
                 use_positional_embedding: bool = True,
                 use_timestamp_embedding: bool = True):
        super().__init__()
        self.size = size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_position_embeddings
        self.input_token_length = input_token_length
        self.use_pos_emb = use_positional_embedding
        self.use_time_emb = use_timestamp_embedding
        self.ff_dim = size * ff_dim_multiplier
        self.dropout = dropout

        # Token embedding layer (mapping token IDs or continuous vectors)
        # Here, for implementation, we assume input tokens are continuous vectors,
        # so embedding layer is not required. For discrete tokens, implement nn.Embedding.
        # Let's assume we process embeddings outside; so in forward, input is already embedding.
        # For simplicity, we'll provide linear projection later if needed.

        # Positional Embedding
        if self.use_pos_emb:
            self.pos_embedding = nn.Embedding(self.max_seq_len, size)
        else:
            self.pos_embedding = None

        # Timestamp embedding (optional)
        if self.use_time_emb:
            self.timing_embedding = nn.Embedding(self.max_seq_len, size)
        else:
            self.timing_embedding = None

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(size, num_heads, dropout) for _ in range(num_layers)
        ])

        # Final linear decoder: project hidden states to token vectors (regression)
        # Since tokens are continuous values, output is of dimension equal to token dimension S
        # e.g., predicting S-dimensional token vectors directly
        self.output_dim = self.input_token_length
        self.decoder_head = nn.Linear(size, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for positional embeddings and decoder head."""
        if self.pos_embedding is not None:
            nn.init.uniform_(self.pos_embedding.weight, -0.02, 0.02)
        if self.timing_embedding is not None:
            nn.init.uniform_(self.timing_embedding.weight, -0.02, 0.02)
        nn.init.xavier_uniform_(self.decoder_head.weight)
        if self.decoder_head.bias is not None:
            nn.init.zeros_(self.decoder_head.bias)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for self-attention (size: seq_len x seq_len)."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, positional_ids: Optional[torch.Tensor] = None, timestamp_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token representations, shape (batch_size, seq_len, embed_dim)
            positional_ids (torch.Tensor): Optional positional indices, shape (batch_size, seq_len)
            timestamp_ids (torch.Tensor): Optional timestamp indices, shape (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Predicted token vectors, shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Add positional embeddings
        if self.use_pos_emb and self.pos_embedding is not None:
            if positional_ids is None:
                positional_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positional_ids)  # (batch_size, seq_len, size)
        else:
            pos_emb = 0

        # Add timestamp embeddings
        if self.use_time_emb and self.timing_embedding is not None:
            if timestamp_ids is None:
                timestamp_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            time_emb = self.timing_embedding(timestamp_ids)
        else:
            time_emb = 0

        # Add embeddings
        x = x + pos_emb + time_emb
        # Initial input: assume x is already embedded if continuous, else embed here
        # You may add an input embedding layer if input tokens are discrete indices.

        # Create causal mask for attention
        attn_mask = self._generate_causal_mask(seq_len)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Project final hidden states to produce output token vectors
        output = self.decoder_head(x)  # (batch_size, seq_len, output_dim)
        return output

    def generate(self, 
                 past_tokens: torch.Tensor, 
                 max_new_tokens: int) -> torch.Tensor:
        """
        Generate sequence autoregressively.
        Args:
            past_tokens (torch.Tensor): Initial token embedding sequence, shape (batch_size, seq_len, embed_dim)
            max_new_tokens (int): Max tokens (timesteps) to generate.
        Returns:
            generated (torch.Tensor): Generated token vectors, shape (batch_size, seq_len + max_new_tokens, output_dim)
        """
        batch_size, seq_len, _ = past_tokens.shape
        device = past_tokens.device
        current_seq = past_tokens

        for _ in range(max_new_tokens):
            # Build positional and timestamp IDs for current sequence: assume incremental
            if self.use_pos_emb:
                positional_ids = torch.arange(current_seq.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                positional_ids = None
            if self.use_time_emb:
                # For generation, timestamps increase sequentially; placeholder set to zeros or use actual timestamps
                timestamp_ids = torch.zeros_like(positional_ids, dtype=torch.long)
            else:
                timestamp_ids = None

            # Forward pass: get predictions
            pred = self.forward(current_seq, positional_ids, timestamp_ids)
            # Take last token prediction
            next_token_vec = pred[:, -1, :]  # (batch_size, output_dim)
            # Append predicted token vector as embedding for next step
            # Since we're predicting continuous vectors, no embedding lookup is needed.
            # Expand to shape (batch_size, 1, output_dim)
            next_token = next_token_vec.unsqueeze(1)
            # For consistency with input shape, create an embedding-like tensor
            # Here, we just treat the output vector as the embedding for next token
            current_seq = torch.cat([current_seq, next_token], dim=1)

        return current_seq

```

## requirements.txt

# requirements.txt

# Core Deep Learning Framework
torch==1.9.0        # For model development, training, and inference
numpy==1.21.0        # Numerical operations and array manipulations
pandas==1.3.0       # Dataset handling and DataFrame manipulation
scipy==1.7.0        # Statistical functions, metrics, and auxiliary utilities

# Transformer and NLP Tools
transformers==4.12.0   # For implementing GPT-style decoder-only Transformer architecture

# Utility & Monitoring
tqdm==4.62.0            # Progress bars during training, validation, and testing

# Configuration Parsing
pyyaml==6.0             # For handling YAML configuration files

# (Optional, if datasets are large and require streaming or memory mapping)
# - No other dependencies explicitly needed, but you can add datasets or dask if necessary

# Notes:
# - These versions are chosen for compatibility and stability aligned with the approach.
# - Use of CUDA or GPU support is assumed to be handled outside the requirements.txt file (via PyTorch's CUDA support).
# - Remember to set CUDA environment variables and install compatible CUDA toolkit for GPU acceleration if needed.

# End of requirements.txt

## tokenization.py

# tokenization.py
"""
The Tokenizer class is responsible for transforming raw multivariate or univariate time series data 
into a sequence of tokens suitable for large-scale autoregressive training of the Timer model. 
It supports hierarchical segmentation levels, continuous token embedding facilitation, optional timestamp 
embedding incorporation, and handles heterogeneity across datasets and variables.
"""

import numpy as np
from typing import List, Optional, Tuple
import math

class Token:
    """
    Represents a single token extracted from a time series segment,
    optionally with timestamp metadata.
    """
    def __init__(self, values: np.ndarray, token_id: int, timestamp: Optional[float] = None):
        """
        Args:
            values (np.ndarray): The feature vector of the token (shape: (S, V) or (S,))
            token_id (int): The unique ID of this token.
            timestamp (Optional[float]): Starting timestamp associated with this segment.
        """
        self.values = values
        self.token_id = token_id
        self.timestamp = timestamp

class Sequence:
    """
    Represents a sequence of tokens ready for embedding and model input.
    """
    def __init__(self, tokens: List[Token]):
        """
        Args:
            tokens (List[Token]): Ordered list of Token objects.
        """
        self.tokens = tokens

    def to_id_tensor(self) -> Tuple[torch.Tensor, List[Optional[float]]]:
        """
        Converts tokens to a tensor of token IDs for model input,
        and returns associated timestamps.
        Returns:
            ids (torch.Tensor): Shape (N,), sequence of token IDs.
            timestamps (List[Optional[float]]): Corresponding list of timestamps.
        """
        import torch
        ids = torch.tensor([t.token_id for t in self.tokens], dtype=torch.long)
        timestamps = [t.timestamp for t in self.tokens]
        return ids, timestamps

class Tokenizer:
    """
    Handles hierarchical segmentation, encoding, and embedding of time series data.
    """
    def __init__(self, 
                 hierarchy_levels: List[str] = ['small', 'medium', 'large'],
                 segment_lengths: List[int] = [96, 672, 1440],
                 use_timestamps: bool = True,
                 max_sequence_length: int = 1440,
                 embedding_dim: int = 512):
        """
        Args:
            hierarchy_levels (List[str]): Names or identifiers for levels of hierarchy.
            segment_lengths (List[int]): Corresponding segment lengths for each hierarchy level.
            use_timestamps (bool): Whether to include timestamp info as embeddings.
            max_sequence_length (int): Max tokens per sequence to handle large datasets.
            embedding_dim (int): Embedding size consistent with model input.
        """
        self.hierarchy_levels = hierarchy_levels
        self.segment_lengths = segment_lengths
        self.use_timestamps = use_timestamps
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        # Initialize learned embedding weights for tokens
        # For continuous tokens, we'll have a learnable embedding layer
        # Here, for simplicity, we implement as a dictionary or via numpy (in practice, use nn.Embedding)
        self.token_embedding_table = None  # Placeholder for embedding lookup

    def _initialize_embeddings(self, vocab_size: int):
        """
        Initialize embedding weights for tokens. For continuous tokens, this could be a learned lookup.
        """
        import torch
        self.token_embedding_table = torch.nn.Embedding(vocab_size, self.embedding_dim)

    def _segment_series(self, series: np.ndarray, length: int, step: int = None) -> List[np.ndarray]:
        """
        Segment a series into non-overlapping or overlapping windows.
        Args:
            series (np.ndarray): Series data, shape (T, V) or (T,).
            length (int): Segment length.
            step (int): Step size between segments; default is length for non-overlapping.
        Returns:
            List of segmented array slices.
        """
        T = series.shape[0]
        if step is None:
            step = length
        segments = []
        for start_idx in range(0, T - length + 1, step):
            segments.append(series[start_idx:start_idx + length])
        return segments

    def _normalize_series(self, series: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Apply normalization to a series.
        """
        return (series - mean) / (std + 1e-8)

    def _interpolate_missing(self, series: np.ndarray, timestamps: Optional[np.ndarray]) -> np.ndarray:
        """
        Fill missing values in series via linear interpolation.
        """
        # If series has NaNs, interpolate
        if np.isnan(series).any():
            if timestamps is None:
                # Regular series: interpolate directly
                for col in range(series.shape[1]) if series.ndim > 1 else range(series.shape[0]):
                    arr = series[:, col] if series.ndim > 1 else series
                    indices = np.arange(series.shape[0])
                    valid_mask = ~np.isnan(arr)
                    if valid_mask.sum() > 1:
                        arr[np.isnan(arr)] = np.interp(indices[np.isnan(arr)], indices[valid_mask], arr[valid_mask])
                    if series.ndim > 1:
                        series[:, col] = arr
                    else:
                        series = arr
            else:
                # Irregular sampling: interpolate along timestamps
                for col in range(series.shape[1]) if series.ndim > 1 else range(series.shape[0]):
                    arr = series[:, col] if series.ndim > 1 else series
                    valid_mask = ~np.isnan(arr)
                    if valid_mask.sum() > 1:
                        arr[np.isnan(arr)] = np.interp(timestamps[np.isnan(arr)], timestamps[valid_mask], arr[valid_mask])
                    if series.ndim > 1:
                        series[:, col] = arr
                    else:
                        series = arr
        return series

    def tokenize_series(self, 
                        series: np.ndarray, 
                        timestamp: Optional[float] = None,
                        hierarchy_level: str = 'small') -> List[Token]:
        """
        Segment a series into tokens, with optional timestamp embedding.
        Args:
            series (np.ndarray): Series data shape (T, V) or (T,)
            timestamp (Optional[float]): Starting timestamp for series, if present.
            hierarchy_level (str): The hierarchy level (small, medium, large).
        Returns:
            List[Token]: Sequence of tokens from the series.
        """
        # Select segmentation length based on hierarchy
        try:
            level_idx = self.hierarchy_levels.index(hierarchy_level)
        except ValueError:
            level_idx = 0  # default to first if unknown
        seg_length = self.segment_lengths[min(level_idx, len(self.segment_lengths)-1)]

        # Segment the series
        tokens = []
        series_T = series.shape[0]
        start_idx = 0
        while start_idx + seg_length <= series_T:
            segment = series[start_idx:start_idx + seg_length]
            # For univariate: shape (S,)
            # for multivariate: shape (S, V)
            # For token_id: treat as continuous vector (append as float array)
            token_id = self._compute_token_id(segment)
            token_timestamp = timestamp + start_idx if timestamp is not None else None
            tokens.append(Token(values=segment, token_id=token_id, timestamp=token_timestamp))
            start_idx += seg_length
        return tokens

    def _compute_token_id(self, segment: np.ndarray) -> int:
        """
        Map a segment to a token ID.
        For simplicity, we can hash the segment's bytes or compute a hash of the flattened array.
        Alternatively, maintain a learned embedding layer for continuous tokens.
        """
        # Hash the flattened segment bytes
        array_bytes = segment.tobytes()
        # Use Python's hash function or a deterministic hash
        token_hash = hash(array_bytes)
        return token_hash & 0xffffffff  # Convert to 32-bit int

    def convert_series_to_sequence(self, 
                                   series: np.ndarray, 
                                   timestamp: Optional[float] = None,
                                   hierarchy_level: str = 'small') -> Sequence:
        """
        Convert a full series into a sequence of tokens (a 'sentence').
        Args:
            series (np.ndarray): Series data, shape (T, V)
            timestamp (Optional[float]): Starting timestamp for the series
            hierarchy_level (str): The segmentation level
        Returns:
            Sequence: The sequence object containing tokens
        """
        tokens = self.tokenize_series(series=series, timestamp=timestamp, hierarchy_level=hierarchy_level)
        return Sequence(tokens)

    def build_embedding_table(self, vocab_size: int):
        """
        Initialize embedding table for token IDs, if using learned embedding
        """
        import torch
        self._initialize_embeddings(vocab_size)

    def get_token_id(self, token: np.ndarray) -> int:
        """
        Generate or lookup token ID for a segment
        """
        return self._compute_token_id(token)

    def embed_token(self, token: np.ndarray) -> torch.Tensor:
        """
        Convert a token into an embedding vector, using learned embedding.
        """
        import torch
        if self.token_embedding_table is None:
            raise RuntimeError("Embedding table not initialized. Call build_embedding_table().")
        token_id = self._compute_token_id(token)
        token_id_mod = token_id % self.token_embedding_table.num_embeddings
        return self.token_embedding_table.weight[token_id_mod]

    def process_series_for_model(self, 
                                 series: np.ndarray, 
                                 timestamp: Optional[float] = None, 
                                 hierarchy_level: str = 'small') -> Tuple[torch.Tensor, List[Optional[float]]]:
        """
        Convert a raw series into sequence IDs and timestamp list for model input
        """
        sequence = self.convert_series_to_sequence(series, timestamp, hierarchy_level)
        ids, timestamps = sequence.to_id_tensor()
        return ids, timestamps

# Note:
# - The actual implementation for embedding management (e.g., torch.nn.Embedding) should be part of
#   the larger model code, but here we provide a placeholder interface.
# - For continuous tokens, in practice, integrate with a learned embedding layer or vector quantization.
# - Handling of irregular timestamps and variable sampling rates may require resampling beforehand.
# - This class provides core logic; integration with the overall pipeline (training, inference) will use
#   process_series_for_model() and sequence construction methods.
```

## trainer.py

```python
# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import math
from typing import Optional, List, Dict

# Import the Model class
from model import TimerTransformer

class Trainer:
    """
    Trainer manages the training loop, learning rate scheduling, checkpointing,
    and evaluation for the Timer large time series model.
    """
    def __init__(self, 
                 model: TimerTransformer,
                 train_dataset: List[torch.Tensor],
                 val_dataset: Optional[List[torch.Tensor]],
                 config: Dict):
        """
        Args:
            model (TimerTransformer): The pre-initialized or randomly initialized model.
            train_dataset (List[torch.Tensor]): List of tokenized training sequences.
            val_dataset (Optional[List[torch.Tensor]]): List of tokenized validation sequences.
            config (dict): Configuration from YAML containing hyperparameters.
        """
        # Store parameters from config with defaults
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Hyperparameters
        self.batch_size: int = self.config.get('training', {}).get('batch_size', 2048)
        self.epochs: int = self.config.get('training', {}).get('epochs', 10)
        self.learning_rate: float = self.config.get('training', {}).get('learning_rate', 3e-5)
        self.warmup_steps: int = self.config.get('training', {}).get('warmup_steps', 1000)
        self.decay_strategy: str = self.config.get('training', {}).get('decay_strategy', 'exponential')
        self.decay_rate: float = self.config.get('training', {}).get('decay_rate', 0.5)
        self.save_dir: str = self.config.get('logging', {}).get('save_dir', 'checkpoints/')
        self.log_interval: int = self.config.get('logging', {}).get('log_interval', 100)
        self.save_interval: int = self.config.get('logging', {}).get('save_interval', 1)

        os.makedirs(self.save_dir, exist_ok=True)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # Learning rate scheduler with warmup + decay
        self.global_step = 0
        self._init_scheduler()

        # Prepare DataLoader
        self.train_loader = self._create_dataloader(self.train_dataset, self.batch_size)
        if self.val_dataset is not None:
            self.val_loader = self._create_dataloader(self.val_dataset, self.batch_size, shuffle=False)
        else:
            self.val_loader = None

        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        self.logger = logging.getLogger('Trainer')

        # Tracking best validation performance (e.g., lowest val loss)
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None

    def _create_dataloader(self, dataset: List[torch.Tensor], batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Create DataLoader for dataset with collate fn to handle batching.
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: List[torch.Tensor]) -> Dict:
        """
        Collate function to batch sequences, convert to tensor, create causal mask.
        """
        # Stack sequences: batch_size x seq_len
        input_ids = torch.stack(batch, dim=0)  # (B, L)
        input_ids = input_ids.to(self.device)

        # Create causal attention mask (lower triangular)
        seq_len = input_ids.shape[1]
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()

        return {
            'input_ids': input_ids,
            'attn_mask': attn_mask
        }

    def _init_scheduler(self):
        """
        Initialize learning rate scheduler as per decay strategy.
        """
        total_steps = len(self.train_loader) * self.epochs
        # Using a custom scheduler: exponential decay post warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(self.warmup_steps)
            else:
                # Steps after warmup
                decay_steps = total_steps - self.warmup_steps
                progress = float(current_step - self.warmup_steps) / decay_steps
                return self.decay_rate ** progress
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train(self):
        """
        Run the full training loop over epochs.
        """
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Starting epoch {epoch}")
            epoch_loss = 0.0
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                self.global_step += 1
                self.optimizer.zero_grad()

                input_ids = batch['input_ids']  # (B, L)
                attn_mask = batch['attn_mask']   # (L, L)

                # For autoregressive generation, model input and labels: input[:-1], target[:-1]
                # Direct prediction of next tokens: input sequence shifted
                # But in GPT, typically input is sequence, label is next token shifted by one
                # For simplicity, using full sequence; loss computed on each token shift
                logits = self.model(input_ids)

                # Shift inputs and targets for causal prediction
                target = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1, :]  # (B, L-1, vocab_size or output_dim)

                # Loss: CrossEntropy over token IDs
                loss_fn = nn.CrossEntropyLoss()
                # Reshape logits and targets
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # optional

                self.optimizer.step()
                # Step scheduler
                if self.decay_strategy == 'exponential':
                    self.lr_scheduler.step()

                epoch_loss += loss.item()

                if step % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Epoch {epoch} Step {step}/{len(self.train_loader)} "
                                     f"Loss: {loss.item():.4f} LR: {current_lr:.6f} Total Loss: {epoch_loss/step:.4f}")

            # End of epoch: evaluate & save checkpoint if needed
            avg_loss = epoch_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch} finished with average loss: {avg_loss:.4f}")

            # Optional: validation
            if self.val_loader is not None:
                val_loss = self.evaluate()
                self.logger.info(f"Validation Loss after epoch {epoch}: {val_loss:.4f}")
                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.save_dir, f"best_epoch_{epoch}.pt")
                    self.save_checkpoint(checkpoint_path)
                    self.best_checkpoint_path = checkpoint_path
            # Save checkpoint periodically
            if epoch % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f"epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path)

    def evaluate(self):
        """
        Run evaluation on validation dataset, compute average loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids']
                attn_mask = batch['attn_mask']
                logits = self.model(input_ids)
                target = input_ids[:, 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, path: str):
        """
        Save model and optimizer states.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': getattr(self, 'current_epoch', 0),
            'global_step': self.global_step
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load saved checkpoint for resuming training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\Large-Time-Series-Model\Large-Time-Series-Model_repo`
