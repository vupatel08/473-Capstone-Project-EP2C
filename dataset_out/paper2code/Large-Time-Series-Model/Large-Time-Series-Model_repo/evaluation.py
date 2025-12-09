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
