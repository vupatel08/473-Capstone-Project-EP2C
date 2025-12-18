## evaluation.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional
import os

class Evaluation:
    """
    Loads a trained model and dataset, runs inference, and computes specified evaluation metrics.
    """

    def __init__(self, config: Dict):
        """
        Initializes evaluation with configuration, loads model and dataset.
        Args:
            config (Dict): configuration loaded from 'config.yaml'
        """
        # Extract model config
        model_type: str = config['model'].get('type', 'bert-base-uncased')
        pretrained: bool = config['model'].get('pretrained', True)
        max_seq_length: int = config['model'].get('max_seq_length', 128)
        # Dataset config
        dataset_name: str = config['dataset'].get('name', 'glue')
        task_name: str = config['dataset'].get('task', 'sst2')
        split: str = 'validation' if 'validation' in config['dataset'].get('split', '') else 'test'
        tokenizer_name: str = config['dataset'].get('tokenizer_name', 'bert-base-uncased')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load dataset
        self.dataset = self._load_dataset(dataset_name, task_name, split)

        # DataLoader
        batch_size = 32  # set small batch for evaluation
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_type, pretrained)
        self.model.to(self.device)
        self.model.eval()

        # Collect metrics to compute
        self.metrics = config['evaluation'].get('metrics', ['accuracy', 'loss'])

        # Loss criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def _load_dataset(self, dataset_name: str, task_name: str, split: str):
        """
        Loads the dataset from datasets library
        """
        if dataset_name.lower() == 'glue':
            raw_ds = load_dataset('glue', task_name)
            # Choose split
            if split not in raw_ds:
                raise ValueError(f"Split '{split}' not found in dataset.")
            dataset_split = raw_ds[split]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return dataset_split

    def _load_model(self, model_type: str, pretrained: bool):
        """
        Loads the model architecture and weights from checkpoint if available.
        """
        # Load model: assume model weights are saved under a standardized path
        # For this code, replace 'checkpoint.pth' with actual path if needed
        model_path = 'checkpoint.pth'  # adjust as needed
        model = AutoModelForSequenceClassification.from_pretrained(model_type)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            # If no checkpoint, load default pretrained
            if pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(model_type)
        return model

    def evaluate(self) -> Dict[str, float]:
        """
        Runs inference over the dataset, computes metrics, returns dict.
        """
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dataloader:
                inputs = self._prepare_batch(batch).to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits

                # Compute loss if needed
                if 'loss' in self.metrics:
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item() * labels.size(0)
                else:
                    loss = None

                preds = torch.argmax(logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_samples += labels.size(0)

        # Compute metrics
        results = {}
        if 'accuracy' in self.metrics:
            acc = accuracy_score(all_labels, all_predictions)
            results['accuracy'] = acc
        if 'loss' in self.metrics:
            results['loss'] = total_loss / total_samples
        # Add other metrics if needed
        if 'f1' in self.metrics:
            f1 = f1_score(all_labels, all_predictions, average='macro')
            results['f1'] = f1

        return results

    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        Tokenizes data batch, returns model inputs dict.
        """
        # For datasets like SST-2, batch contains text data. For others, adjust accordingly.
        # Assuming batch contains 'sentence' or similar text key.
        if 'sentence' in self.dataset.features:
            texts = batch['sentence']
            encoding = self.tokenizer(
                texts,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        elif 'sentence1' in self.dataset.features or 'sentence2' in self.dataset.features:
            # e.g., QNLI, MRPC
            texts = list(zip(batch['sentence1'], batch['sentence2']))
            # For simplicity, concatenate sentence pairs
            texts = ['{} [SEP] {}'.format(s1, s2) for s1, s2 in texts]
            encoding = self.tokenizer(
                texts,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            # fallback: assume input_ids present
            # Possibly for other tasks
            return {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
