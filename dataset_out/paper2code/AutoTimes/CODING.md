# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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
```

## evaluation.py

```python
# evaluation.py

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import random
import pandas as pd

from dataset_loader import TimeSeriesSample  # Assumed to be available
from model import Model                       # Assumed to be available
from prompt_builder import PromptBuilder      # Assumed to be available
from utils import compute_metrics               # Assumed utility functions for SMAPE, MAE, MSE

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Utility functions for handling timestamps
def sequence_to_numpy_timestamps(timestamps_np: np.ndarray) -> List[str]:
    """
    Convert sequence of timestamps (np.datetime64, pd.Timestamp, etc.) to list of formatted strings.
    """
    result = []
    for ts in timestamps_np:
        if isinstance(ts, np.datetime64):
            dt = pd.to_datetime(str(ts))
        elif isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        elif isinstance(ts, datetime.datetime):
            dt = ts
        elif isinstance(ts, float) or isinstance(ts, int):
            dt = datetime.datetime.fromtimestamp(ts)
        else:
            dt = pd.to_datetime(str(ts))
        result.append(dt.strftime("%Y/%m/%d %H:%M:%S"))
    return result

# Main evaluation class
class Evaluation:
    def __init__(self,
                 dataset_samples: List[TimeSeriesSample],
                 model_path: str,
                 dataset_name: str,
                 forecast_horizon: int = 96,
                 metric_list: List[str] = None,
                 prompt_strategy: str = 'firstF',
                 prompt_length: int = 48,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.dataset_samples = dataset_samples
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.forecast_horizon = forecast_horizon
        self.prompt_strategy = prompt_strategy
        self.prompt_length = prompt_length
        self.device = device

        # Metrics
        if metric_list is None:
            self.metrics_list = ['SMAPE', 'MAE', 'MSE']
        else:
            self.metrics_list = metric_list

        # Load model and tokenizer
        self.model_obj = None
        self.tokenizer = None
        self._load_model()

        # Initialize prompt builder
        self.prompt_builder = None
        self._init_prompt_builder()

    def _load_model(self):
        # Load the trained checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        pretrained_model_name = CONFIG['model']['pretrained_model_name']
        # Load LM model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model_obj = Model(pretrained_model_path=pretrained_model_name,
                               embedding_dim=CONFIG['model']['embedding_dim'],
                               segment_size=CONFIG['hyperparameters']['segment_size'])
        # Load checkpoint state dict
        self.model_obj.lm_model.load_state_dict(checkpoint['model_state_dict'])
        self.model_obj.eval()
        # Freeze backbone
        for param in self.model_obj.lm_model.parameters():
            param.requires_grad = False
        self.model_obj.lm_model.to(self.device)

    def _init_prompt_builder(self):
        self.prompt_builder = PromptBuilder(
            tokenizer=self.tokenizer,
            segment_size=CONFIG['hyperparameters']['segment_size'],
            prompt_strategy=self.prompt_strategy,
            prompt_length=self.prompt_length,
            timestamp_format="%Y/%m/%d %H:%M:%S"
        )

    def evaluate(self):
        """
        Perform inference over all samples, compute metrics, and optionally plot.
        Returns:
            dict: aggregated metrics (mean and std) over samples
        """
        all_metrics = {k: [] for k in self.metrics_list}
        preds_per_sample = []
        gt_series_list = []

        print(f"Starting evaluation on dataset: {self.dataset_name}")

        for sample in tqdm(self.dataset_samples):
            # For each sample, generate prediction
            pred_series = self._predict_series_for_sample(sample)
            # Get ground truth for the forecast horizon
            # Assumption: sample.series includes full series; extract last F steps for ground truth
            total_timesteps = sample.series.shape[0]
            series_dim = sample.series.shape[1] if len(sample.series.shape) > 1 else 1
            # Determine start index of prediction: last F steps
            start_idx = total_timesteps - self.forecast_horizon
            x_true = sample.series[start_idx:, ...]  # shape (F, C)
            # pred_series: shape (F, C)
            # Clip or reshape as needed
            # Convert to numpy
            x_true_np = x_true
            preds_np = pred_series

            preds_per_sample.append(preds_np)
            gt_series_list.append(x_true_np)

            # Compute metrics per sample
            for metric_name in self.metrics_list:
                if metric_name == 'SMAPE':
                    val = compute_metrics.smape(preds_np, x_true_np)
                elif metric_name == 'MAE':
                    val = compute_metrics.mae(preds_np, x_true_np)
                elif metric_name == 'MSE':
                    val = compute_metrics.mse(preds_np, x_true_np)
                else:
                    val = None
                if val is not None:
                    all_metrics[metric_name].append(val)

        # Aggregate metrics
        metrics_summary = {}
        for k in self.metrics_list:
            values = np.array(all_metrics[k])
            metrics_summary[k] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Optional: Plot a few sample predictions vs ground truth
        self._plot_sample_predictions(preds_per_sample, gt_series_list)

        # Return metrics
        return metrics_summary

    def _predict_series_for_sample(self, sample: TimeSeriesSample):
        """
        For a single sample: build prompt, perform autoregressive inference, decode.
        """
        # Extract series and timestamps
        series_np = sample.series  # shape (T, C)
        timestamps_np = sample.timestamps

        # Build prompt texts (series + timestamps)
        prompt_texts = self.prompt_builder.build_prompt(
            series=series_np,
            timestamps=timestamps_np,
            strategy=self.prompt_strategy,
            prompt_length=self.prompt_length
        )

        # Embed textual timestamps
        timestamp_embeds = self.model_obj.embed_timestamps(prompt_texts)  # (1, D)
        # Extract last lookback window
        L = CONFIG['hyperparameters']['lookback_length']
        F = self.forecast_horizon
        S = CONFIG['hyperparameters']['segment_size']
        C = series_np.shape[1] if len(series_np.shape)>1 else 1

        T_total = series_np.shape[0]
        if T_total < L:
            # Series too short, consider padding or skip
            print(f"Series shorter than lookback length ({T_total} < {L}), skipping sample.")
            return np.zeros((F, C))
        else:
            # Get last L timesteps
            lookback_series = series_np[-L:, :]  # shape (L, C)

        # For simplicity, process the series into the last segment
        last_segment = lookback_series[-S:, :]  # (S, C)
        # Use the model's segmentation embedding
        # For each variate, process separately if needed
        # For simplicity, assume multivariate: process all at once
        # Prepare input tokens
        # Here, we process batch size 1
        series_input = torch.tensor(last_segment.T, dtype=torch.float32).to(self.device).T  # shape (S, C)
        pred_series = torch.zeros((F, C))
        # Initialize current series segment for autoregression
        current_segment = series_input  # shape (S, C)
        for step in range(F):
            # For each variate independently (Channel-wise)
            # Prepare prompts per variate if needed (here, for simplification, process all together)
            # Embed the current segment for forecasting next
            # Here, pick representative segment per variate
            # For simplicity, process all variates simultaneously
            # embed current segment: shape (S, C) => shape (1, S) per variate
            # For simplicity, treat as a batch
            # Compose input embeddings
            input_embeds = self.model_obj.embed_segments(current_segment.T).to(self.device)  # shape (C, D)
            # Add timestamp embedding for each variate
            input_embeds = input_embeds + timestamp_embeds.repeat(C,1)  # shape (C, D)
            # Obtain next token embeddings
            next_token_embeds = self.model_obj.predict_next_embeddings(input_embeds, max_new_tokens=1)  # (1, D)
            # Map back to series space
            next_series_seg = self.model_obj.decode_tokens(next_token_embeds)  # shape (1, S)
            # For simplicity, average over variates
            predicted_seg = next_series_seg.squeeze(0).mean(dim=0).cpu().detach().numpy()  # shape (S,)
            # Save predicted segment
            pred_series[step, :] = predicted_seg[:C]
            # Update current_series for next iteration: shift window
            current_segment = np.vstack((current_segment[1:, :], predicted_seg[:C].reshape(1, C)))

        return pred_series

    def _plot_sample_predictions(self, preds_per_sample, gt_series_list):
        """
        Plot sample predictions vs ground truth for visualization.
        """
        num_samples = min(3, len(preds_per_sample))
        plt.figure(figsize=(12, 4 * num_samples))
        for i in range(num_samples):
            plt.subplot(num_samples,1,i+1)
            plt.plot(gt_series_list[i], label='Ground Truth')
            plt.plot(np.arange(len(gt_series_list[i])-self.forecast_horizon, len(gt_series_list[i])), preds_per_sample[i], label='Predicted')
            plt.legend()
            plt.title(f"Sample {i+1} Prediction vs Ground Truth")
        plt.tight_layout()
        plt.show()
```


## main.py

```python
# main.py
import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

# Import the modules as per design
import dataset_loader
import prompt_builder
import model
import trainer
import evaluation

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 1. Parse command-line args for optional config path
    parser = argparse.ArgumentParser(description='AutoTimes Reproduction Main')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    # 2. Load config.yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Set seed for reproducibility
    seed = 42
    set_seed(seed)

    # 4. Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 5. Load datasets using dataset_loader
    dataset_paths = config['dataset']
    segment_size = config['hyperparameters'].get('segment_size', 96)
    
    dl_module = dataset_loader.DataLoader(dataset_paths, segment_size=segment_size)
    all_datasets = dl_module.load_all()

    # For demonstration:
    # For training, pick one dataset or multiple datasets for training
    # and prepare splits with respect to realistic scenario:
    # Here, we pick one dataset e.g., "etth1" (adjust as needed)
    train_dataset_name = 'etth1'   # in practice, can be based on training setup
    val_dataset_name = 'etth1'     # For simplicity, use same dataset for val
    test_dataset_name = 'etth1'    # Similarly for test

    train_samples = all_datasets[train_dataset_name]
    val_samples = all_datasets[val_dataset_name]
    test_samples = all_datasets[test_dataset_name]

    # 6. Initialize PromptBuilder with tokenizer
    pretrained_model_name = config['model']['pretrained_model_name']
    # Using transformers tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    prompt_builder_obj = prompt_builder.PromptBuilder(
        tokenizer=tokenizer,
        segment_size=config['hyperparameters'].get('segment_size', 96),
        prompt_strategy=config['hyperparameters'].get('prompt_strategy', 'firstF'),
        prompt_length=config['hyperparameters'].get('text_prompt_length', 48)
    )

    # 7. Initialize Model (load pretrained LM, freeze backbone)
    model_obj = model.Model(
        pretrained_model_path=pretrained_model_name,
        embedding_dim=config['model'].get('embedding_dim', 768),
        segment_size=config['hyperparameters'].get('segment_size', 96)
    )
    model_obj.freeze_backbone()

    # 8. Initialize Trainer
    hyperparams = trainer.Hyperparameters(config)
    trainer_obj = trainer.Trainer(
        model=model_obj,
        dataset_samples=train_samples,
        prompt_builder=prompt_builder_obj,
        hyperparams=hyperparams,
        device=device
    )

    # 9. Setup DataLoader for training
    # Note: Implement batching with collate_fn based on your dataset loader
    # For simplicity, assuming dataset_samples are already in appropriate form
    # Here, just replicate DataLoader with batch_size from config
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_samples, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=trainer_obj._collate_fn)
    val_loader = DataLoader(val_samples, batch_size=hyperparams.batch_size, shuffle=False, collate_fn=trainer_obj._collate_fn)
    # Save the loaders in trainer
    trainer_obj.set_dataloaders(train_samples, val_samples, test_samples)
        
    # 10. Train the model
    trainer_obj.train()

    # 11. Load the best checkpoint
    checkpoint_path = os.path.join(trainer_obj.save_dir, 'best_model.pt')
    # Assuming trainer has method to load
    model_obj.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model_obj.eval()

    # 12. Inference / Evaluation on test dataset
    eval_obj = evaluation.Evaluation(
        dataset_samples=test_samples,
        model_path=checkpoint_path,
        dataset_name='Test Dataset',
        forecast_horizon=hyperparams.forecast_horizon,
        prompt_strategy=hyperparams.prompt_strategy,
        prompt_length=hyperparams.text_prompt_length,
        device=device
    )

    metrics_results = eval_obj.evaluate()
    print("Test Metrics:", metrics_results)

    # 13. Save predictions, possibly as plots or files
    # For demonstration, evaluation class already produces plots/logs

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class Model:
    def __init__(self, pretrained_model_path: str, embedding_dim: int = 768, segment_size: int = 96):
        """
        Initialize the AutoTimes model with a pre-trained decoder-only language model.
        Args:
            pretrained_model_path (str): Name or path of the pretrained LM (e.g., 'LLaMA-7B', 'gpt2', 'facebook/opt-1.3b').
            embedding_dim (int): Dimension of the LM token embeddings (D).
            segment_size (int): Number of data points per segment token (S).
        """
        # Load the pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
        self.lm_model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lm_model.to(self.device)

        # Freeze all LM parameters
        for param in self.lm_model.parameters():
            param.requires_grad = False

        # Store configuration
        self.D = embedding_dim
        self.S = segment_size

        # Initialize segmentation embedding (MLP): input -> D
        self.segment_embed = nn.Sequential(
            nn.Linear(self.S, 512),
            nn.ReLU(),
            nn.Linear(512, self.D)
        )
        # Initialize projection head for decoding token embeddings back to series segments
        self.segment_projection = nn.Sequential(
            nn.Linear(self.D, 512),
            nn.ReLU(),
            nn.Linear(512, self.S)
        )

        # Set trainable parameters: only segment_embed and segment_projection
        for param in self.segment_embed.parameters():
            param.requires_grad = True
        for param in self.segment_projection.parameters():
            param.requires_grad = True

        # Use the lm's embedding layer for timestamp text
        self.text_token_embedding = self.lm_model.transformer.wte

    def freeze_backbone(self):
        """
        Freeze parameters of the LM backbone.
        """
        for param in self.lm_model.parameters():
            param.requires_grad = False
        # Unfreeze trainable layers if needed (by default, only embed/ proj are trainable)

    def embed_segments(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Embed raw series segments into the model's latent space.
        Args:
            segments (torch.Tensor): shape (batch_size, S), raw series data.
        Returns:
            torch.Tensor: shape (batch_size, D), embedded segment vectors.
        """
        # segments shape: (B, S)
        embedded = self.segment_embed(segments)
        return embedded

    def embed_timestamps(self, timestamps: List[str]) -> torch.Tensor:
        """
        Convert timestamp strings into timestamp embeddings.
        Args:
            timestamps (List[str]): list of textual timestamps.
        Returns:
            torch.Tensor: shape (len(timestamps), D), timestamp embeddings.
        """
        # Tokenize text timestamps
        tokens = self.tokenizer(timestamps, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(self.device)  # (L, T)
        # Obtain token embeddings
        with torch.no_grad():
            token_embeds = self.text_token_embedding(input_ids)  # (L, T, D)
        # Extract the embedding for <EOS> token, assuming tokenizer.eos_token_id exists
        eos_token_id = self.tokenizer.eos_token_id
        # Find index of eos_token in input_ids to get <EOS> embedding
        eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)
        # Gather embeddings at eos positions
        eos_embeds = token_embeds[eos_indices[0], eos_indices[1], :]  # shape (L, D)
        # If multiple timestamps, return tensor
        return eos_embeds

    def embed_input(self, series_segments: torch.Tensor, timestamp_embeds: torch.Tensor) -> torch.Tensor:
        """
        Combine series segment embeddings and timestamp embeddings to form input token embeddings.
        Args:
            series_segments (torch.Tensor): shape (B, S), raw data.
            timestamp_embeds (torch.Tensor): shape (B, D)
        Returns:
            torch.Tensor: shape (B, D), combined embeddings.
        """
        segment_embeds = self.embed_segments(series_segments)  # (B, D)
        # Add timestamp embedding (broadcast over batch)
        input_embeddings = segment_embeds + timestamp_embeds
        return input_embeddings

    def predict_next_embeddings(self, input_embeddings: torch.Tensor, max_new_tokens: int = 1) -> torch.Tensor:
        """
        Given input sequence embeddings, predict next token embeddings autoregressively.
        Args:
            input_embeddings (torch.Tensor): shape (seq_len, D)
            max_new_tokens (int): how many tokens to generate
        Returns:
            torch.Tensor: predicted token embeddings for next tokens, shape (max_new_tokens, D)
        """
        seq_embeddings = input_embeddings.unsqueeze(0)  # (1, seq_len, D)
        generated_embeddings: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.lm_model(inputs_embeds=seq_embeddings)
                logits = outputs.logits  # shape (1, seq_len, vocab_size)
            # Use last token's logits
            last_logits = logits[:, -1, :]  # (1, vocab_size)
            # For simplicity, pick argmax token
            next_token_id = torch.argmax(last_logits, dim=-1)  # (1,)
            # Convert token id to embedding
            next_token_embed = self.text_token_embedding(next_token_id)  # (1, D)
            generated_embeddings.append(next_token_embed.squeeze(0))
            # Append next token embed to sequence for next iteration
            seq_embeddings = torch.cat([seq_embeddings, next_token_embed.unsqueeze(0)], dim=1)  # (1, seq+1, D)

        return torch.stack(generated_embeddings, dim=0)  # (max_new_tokens, D)

    def decode_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted token embeddings back into series segments.
        Args:
            token_embeddings (torch.Tensor): shape (T, D)
        Returns:
            torch.Tensor: shape (T, S), series segments.
        """
        # Map embeddings to series space
        series_segments = self.segment_projection(token_embeddings)  # (T, S)
        return series_segments

    def forward(self, input_series: torch.Tensor, timestamps: List[str], target_horizon: int = 96) -> torch.Tensor:
        """
        Forward pass for training: given input series and timestamps, predict the next tokens.
        Args:
            input_series (torch.Tensor): shape (B, S), current series segments.
            timestamps (List[str]): list of textual timestamp strings for the batch.
            target_horizon (int): number of segments to predict.
        Returns:
            torch.Tensor: decoded series segments, shape (B, target_horizon, S).
        """
        # Embed input series
        timestamp_embeds = self.embed_timestamps(timestamps)  # (B, D)
        input_embeds = self.embed_input(input_series, timestamp_embeds)  # (B, D)
        input_embeds = input_embeds  # (B, D)

        # Initialize sequence
        seq_embeds = input_embeds.unsqueeze(0)  # (1, B, D)
        outputs: List[torch.Tensor] = []

        # Generate target_horizon segments autoregressively
        for _ in range(target_horizon):
            with torch.no_grad():
                outputs_logits = self.lm_model(inputs_embeds=seq_embeds)  # (1, seq, vocab_size)
                last_logits = outputs_logits.logits[:, -1, :]  # (1, vocab_size)
            # Select argmax token
            next_token_id = torch.argmax(last_logits, dim=-1)  # (1,)
            # Map to embedding
            next_embed = self.text_token_embedding(next_token_id)  # (1, D)
            # Append to sequence
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)  # (1, seq+1, D)
            outputs.append(next_embed.squeeze(0))

        # Stack predicted token embeddings
        predicted_token_embeds = torch.stack(outputs, dim=0)  # (target_horizon, D)
        # Decode into series segments
        predicted_series_segments = self.decode_tokens(predicted_token_embeds)  # (T, S)
        # Reshape to (target_horizon, S)
        predicted_series_segments = predicted_series_segments.reshape(-1, self.S)
        return predicted_series_segments

    def inference(self, last_series_segment: torch.Tensor, timestamp: str, predict_steps: int) -> torch.Tensor:
        """
        Autoregressive inference: generate multiple future segments.
        Args:
            last_series_segment (torch.Tensor): last known series segment, shape (S,)
            timestamp (str): textual timestamp for the next step
            predict_steps (int): number of segments to generate
        Returns:
            torch.Tensor: predicted series segments, shape (predict_steps, S)
        """
        generated_segments: List[torch.Tensor] = []
        current_segment = last_series_segment.unsqueeze(0)  # shape (1, S)
        timestamp_emb = self.embed_timestamps([timestamp])  # shape (1, D)

        # Initialize embedded sequence
        input_embeds = self.embed_input(current_segment, timestamp_emb)  # (1, D)

        seq_embeds = input_embeds.unsqueeze(0)  # (1, 1, D)

        for _ in range(predict_steps):
            with torch.no_grad():
                outputs_logits = self.lm_model(inputs_embeds=seq_embeds)  # (1, seq, vocab_size)
                last_logits = outputs_logits.logits[:, -1, :]  # (1, vocab_size)
            next_token_id = torch.argmax(last_logits, dim=-1)  # (1,)
            next_embed = self.text_token_embedding(next_token_id)  # (1, D)
            seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)  # append new token
            # Decode token embedding into series segment
            series_token = self.decode_tokens(next_embed)  # (1, S)
            generated_segments.append(series_token.squeeze(0))
            # Update last_series_segment for next prediction if needed
            current_segment = series_token.squeeze(0)

        return torch.stack(generated_segments, dim=0)  # (predict_steps, S)
```

## prompt_builder.py

```python
## prompt_builder.py
import datetime
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer
import numpy as np
import math
import random

class PromptBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        segment_size: int = 96,
        prompt_strategy: str = "firstF",
        prompt_length: int = 48,
        timestamp_format: str = "%Y/%m/%d %H:%M:%S"
    ):
        """
        Initialize the PromptBuilder with tokenizer and hyperparameters.
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer aligned with the pre-trained LLM.
            segment_size (int): Number of data points per segment token.
            prompt_strategy (str): Strategy to select prompt segments ('firstF', 'lastF', 'recentSeries', 'random', 'out_series').
            prompt_length (int): Length of the prompt in tokens (or segments), depending on strategy.
            timestamp_format (str): Format string for textual timestamp conversion.
        """
        self.tokenizer = tokenizer
        self.segment_size = segment_size
        self.prompt_strategy = prompt_strategy
        self.prompt_length = prompt_length
        self.timestamp_format = timestamp_format

    def build_prompt(
        self,
        series: np.ndarray,
        timestamps: np.ndarray,
        strategy: Optional[str] = None,
        prompt_length: Optional[int] = None
    ) -> List[int]:
        """
        Build tokenized prompt sequence based on the strategy.
        Args:
            series (np.ndarray): Series data, shape (T, C).
            timestamps (np.ndarray): Timestamps corresponding to series, shape (T,).
            strategy (str): Optional, override class default.
            prompt_length (int): Optional, override class default.
        Returns:
            List[int]: Token IDs representing the full prompt.
        """
        strat = strategy or self.prompt_strategy
        plength = prompt_length or self.prompt_length

        # Select prompt segments and timestamps based on strategy
        prompt_info = self._select_prompt_segments(series, timestamps, strat, plength)

        # Convert series segments to text prompts
        series_prompts = [
            self._series_segment_to_text(seg) for seg in prompt_info['series_segments']
        ]

        # Convert timestamps to text prompts
        timestamp_prompts = [
            self._timestamp_to_text(ts) for ts in prompt_info['timestamps']
        ]

        # Assemble full prompt string
        full_prompt_text = self._assemble_prompt(series_prompts, timestamp_prompts)

        # Tokenize
        token_ids = self.tokenizer.encode(full_prompt_text, add_special_tokens=True)
        return token_ids

    def _select_prompt_segments(
        self,
        series: np.ndarray,
        timestamps: np.ndarray,
        strategy: str,
        prompt_length_tokens: int
    ) -> Dict[str, List]:
        """
        Select series segments and timestamps according to strategy.
        For simplicity, assume prompt_length_tokens is in number of prompt segments.
        More sophisticated strategies can be implemented.
        """
        T = series.shape[0]
        seg_size = self.segment_size

        if strategy == "firstF":
            # Take the first prompt_length segments
            num_segments = prompt_length_tokens
            start_idx = 0
            end_idx = min(num_segments * seg_size, T)
            indices = np.arange(start_idx, end_idx)
        elif strategy == "lastF":
            # Take the last prompt_length segments
            num_segments = prompt_length_tokens
            end_idx = T
            start_idx = max(end_idx - num_segments * seg_size, 0)
            indices = np.arange(start_idx, end_idx)
        elif strategy == "recentSeries":
            # Take the most recent series of length prompt_length * segment_size
            end_idx = T
            start_idx = max(end_idx - prompt_length_tokens * seg_size, 0)
            indices = np.arange(start_idx, end_idx)
        elif strategy == "random":
            # Random segments from series
            max_start = max(T - prompt_length_tokens * seg_size, 0)
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + prompt_length_tokens * seg_size
            indices = np.arange(start_idx, end_idx)
        elif strategy == "out_series":
            # For out-of-series prompt, placeholder: select random unrelated series
            # Since no external data is provided, fallback same as random for demonstration
            max_start = max(T - prompt_length_tokens * seg_size, 0)
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + prompt_length_tokens * seg_size
            indices = np.arange(start_idx, end_idx)
        else:
            # Default fallback: firstF
            num_segments = prompt_length_tokens
            start_idx = 0
            end_idx = min(num_segments * seg_size, T)
            indices = np.arange(start_idx, end_idx)

        # Extract series segments
        series_segments = []
        timestamps_for_prompt = []

        for i in range(0, len(indices), seg_size):
            seg_indices = indices[i:i + seg_size]
            if len(seg_indices) == 0:
                continue
            # Handle case if last segment is shorter than seg_size
            seg_indices = np.array(seg_indices)
            # For actual series, select the data points
            # For simplicity, select the data points or mean
            start_data_idx = seg_indices[0]
            end_data_idx = seg_indices[-1] + 1
            # Ensure indices are within bounds
            if start_data_idx < 0:
                start_data_idx = 0
            if end_data_idx > T:
                end_data_idx = T
            segment = series[start_data_idx:end_data_idx, :]
            series_segments.append(segment)

            # For timestamp, pick the start timestamp of this segment
            ts_idx = start_data_idx
            if ts_idx >= len(timestamps):
                ts_idx = len(timestamps) - 1
            timestamps_for_prompt.append(timestamps[ts_idx])

        return {
            'series_segments': series_segments,
            'timestamps': timestamps_for_prompt
        }

    def _series_segment_to_text(self, segment: np.ndarray) -> str:
        """
        Convert a numerical series segment into a string prompt.
        Example: "0.123 0.456 0.789 ..."
        """
        flat_series = segment.flatten()
        return " ".join([f"{val:.3f}" for val in flat_series])

    def _timestamp_to_text(self, timestamp) -> str:
        """
        Convert a timestamp to textual format.
        Accepts pandas Timestamp, datetime, or float (epoch).
        """
        if isinstance(timestamp, np.datetime64):
            ts = pd.to_datetime(str(timestamp))
        elif isinstance(timestamp, datetime.datetime):
            ts = timestamp
        elif isinstance(timestamp, float) or isinstance(timestamp, int):
            ts = datetime.datetime.fromtimestamp(timestamp)
        else:
            ts = pd.to_datetime(str(timestamp))
        return ts.strftime(self.timestamp_format)

    def _assemble_prompt(
        self,
        series_prompts: List[str],
        timestamp_prompts: List[str]
    ) -> str:
        """
        Concatenate series and timestamp prompts into a single prompt string.
        Format example:
        "Series: {series_prompt} Timestamp: {timestamp}\n"
        """
        prompt_lines = []
        for s_text, t_text in zip(series_prompts, timestamp_prompts):
            line = f"Series: {s_text} Timestamp: {t_text}"
            prompt_lines.append(line)
        return "\n".join(prompt_lines)
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from tqdm import tqdm
from typing import List, Dict, Any
from dataset_loader import TimeSeriesSample
from model import Model
from utils import compute_metrics  # Assume a utility function for metrics
from evaluation import Evaluation
from torch.cuda.amp import GradScaler, autocast

class Hyperparameters:
    def __init__(self, config: Dict[str, Any]):
        self.lookback_length: int = config.get('hyperparameters', {}).get('lookback_length', 672)
        self.forecast_horizon: int = config.get('hyperparameters', {}).get('forecast_horizon', 96)
        self.segment_size: int = config.get('hyperparameters', {}).get('segment_size', 96)
        self.prompt_strategy: str = config.get('hyperparameters', {}).get('prompt_strategy', 'firstF')
        self.training_epochs: int = config.get('hyperparameters', {}).get('training_epochs', 50)
        self.batch_size: int = config.get('hyperparameters', {}).get('batch_size', 224)
        self.learning_rate: float = config.get('hyperparameters', {}).get('learning_rate', 5e-5)
        self.text_prompt_length: int = config.get('hyperparameters', {}).get('text_prompt_length', 48)

class Trainer:
    def __init__(
        self,
        model: Model,
        dataset_samples: List[TimeSeriesSample],
        prompt_builder,  # Instance of PromptBuilder
        hyperparams: Hyperparameters,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir: str = './checkpoint',
        metrics_list: List[str] = ['SMAPE', 'MAE', 'MSE']
    ):
        self.model = model
        self.dataset_samples = dataset_samples
        self.prompt_builder = prompt_builder
        self.hyperparams = hyperparams
        self.device = device
        self.save_dir = save_dir
        self.metrics_list = metrics_list

        # Create directory if not exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize optimizer: only train embedding and projection head
        self._setup_optimizer()

        # Prepare train/val data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # For validation
        self.best_val_metric = float('inf')
        self.scaler = GradScaler()  # For mixed precision

    def _setup_optimizer(self):
        # Only train embedded layers: segment_embed and segment_projection
        params = list(self.model.segment_embed.parameters()) + list(self.model.segment_projection.parameters())
        self.optimizer = optim.AdamW(params, lr=self.hyperparams.learning_rate, weight_decay=0.01)

    def set_dataloaders(self, train_samples, val_samples, test_samples):
        # Wrap samples into DataLoader with collate_fn for batching
        self.train_loader = DataLoader(train_samples, batch_size=self.hyperparams.batch_size,
                                       shuffle=True, collate_fn=self._collate_fn)
        self.val_loader = DataLoader(val_samples, batch_size=self.hyperparams.eval_batch_size,
                                     shuffle=False, collate_fn=self._collate_fn)
        self.test_loader = DataLoader(test_samples, batch_size=self.hyperparams.eval_batch_size,
                                       shuffle=False, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: List[TimeSeriesSample]):
        # Batch as list of samples; process to tensors
        series_list = []
        timestamp_list = []
        series_name_list = []

        for sample in batch:
            series_list.append(sample.series)
            timestamp_list.append(sample.timestamps)
            series_name_list.append(sample.series_name)

        series_arr = np.stack(series_list)  # shape (B, T, C)
        T = series_arr.shape[1]
        batch_size = len(batch)

        # For simplicity, assume same length sequences with trimming or padding handled
        # Normally, you'd pad sequences to max in batch
        max_T = T
        series_tensor = torch.tensor(series_arr, dtype=torch.float32, device=self.device)  # (B, T, C)

        # For each sample, get the corresponding timestamps
        # For simplified implementation, use the first timestamps as prompt; in practice, handle each separately
        timestamps_batch = [ts for ts in timestamp_list]

        return {
            'series': series_tensor,                          # (B, T, C)
            'timestamps': timestamps_batch,                   # list of np.ndarray
            'series_name': series_name_list                 # list of str
        }

    def train(self):
        for epoch in range(self.hyperparams.training_epochs):
            print(f"Epoch {epoch+1}/{self.hyperparams.training_epochs}")
            train_loss = 0.0
            pbar = tqdm(self.train_loader)
            self.model.train()
            for batch in pbar:
                self.optimizer.zero_grad()

                series_batch = batch['series'].to(self.device)  # (B, T, C)
                timestamps_batch = batch['timestamps']           # list
                batch_size = series_batch.size(0)
                T_total = series_batch.shape[1]
                C_dim = series_batch.shape[2]

                # For each sample: extract lookback window
                input_series = series_batch[:, :self.hyperparams.lookback_length, :]  # (B, L, C)
                target_series = series_batch[:, self.hyperparams.lookback_length:self.hyperparams.lookback_length + int(self.hyperparams.forecast_horizon / self.model.S), :]  # shape (B, horizon/segment_size, C)

                # Prepare prompts
                # For each sample, build prompt string
                prompt_texts = []
                for i in range(batch_size):
                    # Using prompt_builder, build prompt string for sample i
                    # First, serialize series to segments as needed
                    # For simplicity, use the entire lookback series for prompt
                    # Get series data as (T, C), textual prompts via prompt_builder
                    # The prompt_builder handles segmentation and timestamp conversion
                    prompt_str = self.prompt_builder.build_prompt(
                        series=series_batch[i].cpu().numpy(),
                        timestamps=sequence_to_numpy_timestamps(batch['timestamps'][i]),
                        strategy=self.hyperparams.prompt_strategy,
                        prompt_length=self.hyperparams.text_prompt_length
                    )
                    prompt_texts.append(prompt_str)

                # Convert textual prompts to embeddings
                timestamp_embeds = self.model.embed_timestamps(prompt_texts)  # (B, D)

                # Prepare series segments: flatten input_series to (B, S) and embed
                # We need to generate the input embeddings for the prompt + series
                # For training, use input_series as ground truth (supervised)
                # Embed series segments
                # For simplicity, flatten series: shape (B, L*C), then split into segments
                # Alternatively, pass one segment at a time; here, for training, we process entire batch
                # For code simplicity, assume embedded per segment
                # We'll implement the process step-by-step below:

                # Generate input embeddings
                # For training, encode the entire lookback series for each sample
                # Here, a simplified approach: process entire batch
                # (Alternatively, process each sample separately and batch)
                # For brevity, process batch all at once:
                series_segments = input_series.reshape(batch_size, -1, C_dim)  # (B, L, C), then segment
                # For the purposes of training, sample a representative segment:
                # For example, take the last segment in the lookback series
                last_segments = series_segments[:, -self.model.S:self.model.S, :]  # (B, S, C)
                last_segments = last_segments.squeeze(1)  # (B, S)

                # Embed segments
                segment_embeds = self.model.embed_segments(last_segments.to(self.device))  # (B, D)
                # Add timestamp embeddings
                input_embeds = segment_embeds + timestamp_embeds  # (B, D)

                # For training, predict next tokens of size forecast_horizon / segment_size
                # We can run training step directly
                with autocast():
                    # Generate predicted next tokens
                    output_embeddings = self.model.predict_next_embeddings(input_embeds, max_new_tokens=int(self.hyperparams.forecast_horizon / self.model.S))
                    # Decode tokens back to series segments
                    pred_series_segments = self.model.decode_tokens(output_embeddings)  # shape (T_pred, S)
                    # For supervision, extract GT segments
                    # Target shape: (B, horizon/segment_size, C)
                    # For simplicity, process batch-wise
                    gt_segments = target_series.reshape(batch_size, -1, C_dim)  # (B, horizon/S, C)
                    # Map gt segments to embedding space comparable with predictions
                    # For supervised learning, compare predicted series segments with gt
                    # Since predicted are in series space (via decode), directly compute MSE
                    # For simplicity, flatten predictions and GT for each batch item
                    loss = nn.MSELoss()(pred_series_segments, gt_segments.reshape(-1, C_dim))
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.segment_embed.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.segment_projection.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                pbar.set_description(f"Loss: {loss.item():.4f}")

            avg_loss = train_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

            # Validation step
            val_metrics = self._validate()
            # Save best model based on validation metric (e.g., SMAPE)
            val_primary = val_metrics.get('SMAPE', 0)
            if val_primary < self.best_val_metric:
                self.best_val_metric = val_primary
                self._save_checkpoint(os.path.join(self.save_dir, 'best_model.pt'))
                print("Saved new best model at epoch", epoch+1)

    def _validate(self):
        # Run evaluation on validation set, no grads
        self.model.eval()
        all_metrics = {k: [] for k in self.metrics_list}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                series_batch = batch['series'].to(self.device)
                timestamps_batch = batch['timestamps']
                batch_size = series_batch.size(0)
                T_total = series_batch.shape[1]
                # For each sample: pick last lookback window
                input_series = series_batch[:, :self.hyperparams.lookback_length, :]
                # Build prompt strings
                prompt_texts = []
                for i in range(batch_size):
                    prompt_str = self.prompt_builder.build_prompt(
                        series=series_batch[i].cpu().numpy(),
                        timestamps=sequence_to_numpy_timestamps(timestamps_batch[i]),
                        strategy=self.hyperparams.prompt_strategy,
                        prompt_length=self.hyperparams.text_prompt_length
                    )
                    prompt_texts.append(prompt_str)
                timestamp_embeds = self.model.embed_timestamps(prompt_texts)
                last_segments = input_series[:, -self.model.S:, :].squeeze(1)
                input_embeds = self.model.embed_segments(last_segments) + timestamp_embeds

                # Generate predictions autoregressively
                pred_series_segments = self._autoregressive_predict(input_embeds, prompts_texts=prompt_texts)
                # Compare with ground truth
                gt_segments = batch['series'][:, self.hyperparams.lookback_length:self.hyperparams.lookback_length + int(self.hyperparams.forecast_horizon / self.model.S), :].reshape(batch_size, -1, C_dim)
                for metric_name in self.metrics_list:
                    metric_value = compute_metrics(
                        pred_series_segments, gt_segments, metric_name
                    )
                    all_metrics[metric_name].append(metric_value)

        # Aggregate metrics
        averaged_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        print("Validation Metrics:", averaged_metrics)
        return averaged_metrics

    def _autoregressive_predict(self, init_embeds: torch.Tensor, prompts_texts: List[str]) -> torch.Tensor:
        """
        Generate forecasts autoregressively for horizon.
        Args:
            init_embeds (torch.Tensor): shape (B, D)
            prompts_texts (list): batch prompt strings for timestamp embedding
        Returns:
            torch.Tensor: predicted series segments, shape (horizon/segment_size, S)
        """
        batch_size = init_embeds.shape[0]
        horizon_segments = int(self.hyperparams.forecast_horizon / self.model.S)
        generated_embeds = []
        current_embeds = init_embeds  # (B, D)

        for _ in range(horizon_segments):
            # Expand to batch
            output_embeddings = self.model.predict_next_embeddings(current_embeds, max_new_tokens=1)  # (1, D)
            # Decode token embeddings to series segment
            series_seg = self.model.decode_tokens(output_embeddings)  # (1, S)
            # For simplicity, take mean of batch or process per sample
            # Here, process per sample: reuse current_embeds
            # Update current_embeds for next step
            current_embeds = output_embeddings.squeeze(0)
            generated_embeds.append(series_seg.squeeze(0))  # (S,)

        # Stack final generated series segments
        forecast_segments = torch.stack(generated_embeds, dim=0)  # (horizon/segment_size, S)
        return forecast_segments

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))

# Utility function outside class (could be in utils.py)
def sequence_to_numpy_timestamps(timestamps_np: np.ndarray):
    """Convert sequence of timestamps (np.datetime64 or datetime) to list of strs."""
    # Add necessary conversion if needed
    transformed = []
    for ts in timestamps_np:
        if isinstance(ts, np.datetime64):
            dt = pd.to_datetime(str(ts))
        elif isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        elif isinstance(ts, datetime.datetime):
            dt = ts
        else:
            dt = pd.to_datetime(str(ts))
        transformed.append(dt.strftime("%Y/%m/%d %H:%M:%S"))
    return transformed
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\AutoTimes\AutoTimes_repo`
