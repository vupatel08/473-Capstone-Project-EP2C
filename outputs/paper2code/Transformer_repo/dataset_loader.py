"""dataset_loader.py
This module defines the DatasetLoader class responsible for loading
and preprocessing datasets for machine translation (MT) and constituency parsing.
It uses SentencePiece for tokenization (BPE/word-piece) and groups examples into dynamic batches
based on a maximum token count per batch.
"""

import os
import logging
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import sentencepiece as spm

# Set up basic logging configuration.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def pad_collate_fn(batch: List[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads a batch of tokenized examples.
    Each example is a tuple: (source_token_ids, target_token_ids).
    Returns a dictionary containing padded tensors and corresponding masks.
    """
    src_list = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    tgt_list = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    padded_src = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    padded_tgt = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)
    src_mask = (padded_src != 0)
    tgt_mask = (padded_tgt != 0)
    return {"src": padded_src, "tgt": padded_tgt, "src_mask": src_mask, "tgt_mask": tgt_mask}


class SimpleDataset(Dataset):
    """
    A simple dataset wrapper which holds a list of tokenized examples.
    Each example is a tuple: (source_token_ids, target_token_ids).
    """
    def __init__(self, examples: List[Tuple[List[int], List[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.examples[idx]


class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that groups examples into batches based on a maximum token count threshold.
    The condition is that for a given batch, both the sum of source tokens and the sum of target tokens
    do not exceed max_tokens.
    """
    def __init__(self, data: Dataset, max_tokens: int) -> None:
        self.data = data  # data items are tuples: (src, tgt)
        self.max_tokens = max_tokens
        self.indices = list(range(len(data)))
        # Sort indices by the length of the source sequence (for better bucketing)
        self.indices.sort(key=lambda i: len(data[i][0]))

    def __iter__(self):
        batch = []
        src_sum = 0
        tgt_sum = 0

        for idx in self.indices:
            src_len = len(self.data[idx][0])
            tgt_len = len(self.data[idx][1])
            # If adding the new example surpasses the max_tokens limit on either side and batch is non-empty, yield current batch.
            if batch and (src_sum + src_len > self.max_tokens or tgt_sum + tgt_len > self.max_tokens):
                yield batch
                batch = []
                src_sum = 0
                tgt_sum = 0
            batch.append(idx)
            src_sum += src_len
            tgt_sum += tgt_len

        if batch:
            yield batch

    def __len__(self) -> int:
        # Count the number of batches by iterating over the sampler.
        count = 0
        temp_batch = []
        src_sum = 0
        tgt_sum = 0
        for idx in self.indices:
            src_len = len(self.data[idx][0])
            tgt_len = len(self.data[idx][1])
            if temp_batch and (src_sum + src_len > self.max_tokens or tgt_sum + tgt_len > self.max_tokens):
                count += 1
                temp_batch = []
                src_sum = 0
                tgt_sum = 0
            temp_batch.append(idx)
            src_sum += src_len
            tgt_sum += tgt_len
        if temp_batch:
            count += 1
        return count


class DatasetLoader:
    """
    The DatasetLoader class loads and preprocesses datasets for machine translation and constituency parsing.
    It utilizes SentencePiece for tokenization and returns DataLoader objects organized by experiment type and split.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes DatasetLoader with the given configuration.
        Extracts dataset settings, initializes tokenizers (SentencePiece),
        and sets batch size limits.
        """
        self.config = config
        # Extract batch size tokens for translation; default to 25000 tokens if not provided.
        self.batch_size_tokens: int = config.get("dataset", {}).get("translation", {}).get("batch_size_tokens", 25000)

        # Dictionaries to store SentencePieceProcessor for each dataset.
        self.mt_tokenizers: Dict[str, spm.SentencePieceProcessor] = {}
        self.parsing_tokenizers: Dict[str, spm.SentencePieceProcessor] = {}

        # Initialize tokenizers for machine translation datasets.
        translation_config = config.get("dataset", {}).get("translation", {})
        for dataset_key, dataset_spec in translation_config.items():
            if dataset_key == "batch_size_tokens":
                continue
            # Set default vocabulary sizes and determine model type based on tokenization description.
            if dataset_key == "en_de":
                vocab_size = 37000
                model_type = "bpe"  # Byte-Pair Encoding
            elif dataset_key == "en_fr":
                vocab_size = 32000
                model_type = "unigram"  # Word-piece segmentation often uses unigram models.
            else:
                vocab_size = 32000
                model_type = "bpe"
            processor = self._get_or_train_tokenizer(
                dataset_type="translation",
                dataset_key=dataset_key,
                vocab_size=vocab_size,
                model_type=model_type
            )
            self.mt_tokenizers[dataset_key] = processor

        # Initialize tokenizers for parsing datasets.
        parsing_config = config.get("dataset", {}).get("parsing", {})
        for dataset_key, dataset_spec in parsing_config.items():
            vocab_size = dataset_spec.get("vocabulary_size", 16000)
            model_type = "bpe"  # Using BPE for parsing as default.
            processor = self._get_or_train_tokenizer(
                dataset_type="parsing",
                dataset_key=dataset_key,
                vocab_size=vocab_size,
                model_type=model_type
            )
            self.parsing_tokenizers[dataset_key] = processor

    def _get_or_train_tokenizer(self, dataset_type: str, dataset_key: str, vocab_size: int, model_type: str) -> spm.SentencePieceProcessor:
        """
        Loads an existing SentencePiece model if available, otherwise trains one
        using the raw training data for the specified dataset.
        """
        model_prefix = f"spm_{dataset_key}"
        model_file = model_prefix + ".model"
        sp = spm.SentencePieceProcessor()
        if os.path.exists(model_file):
            sp.Load(model_file)
            logging.info(f"Loaded existing SentencePiece model from {model_file} for {dataset_key} ({dataset_type}).")
        else:
            logging.info(f"Training SentencePiece model for {dataset_key} ({dataset_type}) with vocab size {vocab_size} and model type {model_type}.")
            # Determine the training input file(s) based on dataset type.
            if dataset_type == "translation":
                # For translation, use both source and target training files.
                if dataset_key == "en_de":
                    src_lang, tgt_lang = "en", "de"
                elif dataset_key == "en_fr":
                    src_lang, tgt_lang = "en", "fr"
                else:
                    src_lang, tgt_lang = "en", "other"

                train_src = os.path.join("data", "translation", dataset_key, f"train.{src_lang}")
                train_tgt = os.path.join("data", "translation", dataset_key, f"train.{tgt_lang}")
                if not os.path.exists(train_src) or not os.path.exists(train_tgt):
                    logging.error(f"Training files for {dataset_key} translation not found: {train_src} or {train_tgt}")
                    raise FileNotFoundError(f"Training files for {dataset_key} translation not found.")
                # Combine source and target files into one.
                combined_file = os.path.join("data", "translation", dataset_key, "train_combined.txt")
                with open(combined_file, "w", encoding="utf-8") as outfile:
                    for fname in [train_src, train_tgt]:
                        with open(fname, "r", encoding="utf-8") as infile:
                            for line in infile:
                                outfile.write(line)
            elif dataset_type == "parsing":
                # For parsing, use the training file.
                combined_file = os.path.join("data", "parsing", dataset_key, "train.txt")
                if not os.path.exists(combined_file):
                    logging.error(f"Training file for {dataset_key} parsing not found: {combined_file}")
                    raise FileNotFoundError(f"Training file for {dataset_key} parsing not found.")
            else:
                logging.error(f"Unknown dataset type: {dataset_type}")
                raise ValueError("Unknown dataset type.")

            # Train the SentencePiece model.
            spm.SentencePieceTrainer.Train(
                input=combined_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            sp.Load(model_file)
            logging.info(f"Trained and loaded SentencePiece model from {model_file} for {dataset_key} ({dataset_type}).")
        return sp

    def load_data(self) -> Dict[str, Any]:
        """
        Loads and preprocesses the datasets for machine translation and parsing experiments.
        Returns a dictionary with the following structure:
          {
            "translation": {
                "en_de": {"train": DataLoader, "val": DataLoader, "test": DataLoader},
                "en_fr": {"train": DataLoader, "val": DataLoader, "test": DataLoader}
             },
            "parsing": {
                "wsj": {"train": DataLoader, "val": DataLoader, "test": DataLoader},
                "semi_supervised": {"train": DataLoader, ...}
             }
          }
        """
        data_dict: Dict[str, Any] = {"translation": {}, "parsing": {}}

        # Process machine translation datasets.
        translation_config = self.config.get("dataset", {}).get("translation", {})
        for dataset_key, dataset_spec in translation_config.items():
            if dataset_key == "batch_size_tokens":
                continue
            splits = ["train", "val", "test"]
            lang_pair_data = {}
            for split in splits:
                examples = self._load_translation_split(dataset_key, split)
                if len(examples) == 0:
                    logging.warning(f"No examples found for translation dataset {dataset_key} split {split}.")
                dataset = SimpleDataset(examples)
                batch_sampler = DynamicBatchSampler(dataset, self.batch_size_tokens)
                dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=pad_collate_fn)
                lang_pair_data[split] = dataloader
            data_dict["translation"][dataset_key] = lang_pair_data

        # Process constituency parsing datasets.
        parsing_config = self.config.get("dataset", {}).get("parsing", {})
        for dataset_key, dataset_spec in parsing_config.items():
            splits = ["train", "val", "test"]
            parsing_data = {}
            for split in splits:
                examples = self._load_parsing_split(dataset_key, split)
                if len(examples) == 0:
                    logging.warning(f"No examples found for parsing dataset {dataset_key} split {split}.")
                dataset = SimpleDataset(examples)
                batch_sampler = DynamicBatchSampler(dataset, self.batch_size_tokens)
                dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=pad_collate_fn)
                parsing_data[split] = dataloader
            data_dict["parsing"][dataset_key] = parsing_data

        return data_dict

    def _load_translation_split(self, dataset_key: str, split: str) -> List[Tuple[List[int], List[int]]]:
        """
        Loads a translation dataset split (train/val/test) for a given language pair.
        Reads raw source and target text files, applies tokenization using the appropriate SentencePiece model,
        and returns a list of (tokenized_src, tokenized_tgt) examples.
        """
        if dataset_key == "en_de":
            src_lang, tgt_lang = "en", "de"
        elif dataset_key == "en_fr":
            src_lang, tgt_lang = "en", "fr"
        else:
            src_lang, tgt_lang = "en", "other"

        base_dir = os.path.join("data", "translation", dataset_key)
        src_file = os.path.join(base_dir, f"{split}.{src_lang}")
        tgt_file = os.path.join(base_dir, f"{split}.{tgt_lang}")

        if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
            logging.error(f"Translation files for {dataset_key} split {split} not found: {src_file}, {tgt_file}.")
            return []

        examples: List[Tuple[List[int], List[int]]] = []
        with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()
            if len(src_lines) != len(tgt_lines):
                logging.warning(f"Source and target file lengths do not match for {dataset_key} {split}.")
            for src_line, tgt_line in zip(src_lines, tgt_lines):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                sp_processor = self.mt_tokenizers.get(dataset_key)
                src_ids = sp_processor.encode(src_line, out_type=int)
                tgt_ids = sp_processor.encode(tgt_line, out_type=int)
                examples.append((src_ids, tgt_ids))
        logging.info(f"Loaded {len(examples)} examples for translation dataset {dataset_key} split {split}.")
        return examples

    def _load_parsing_split(self, dataset_key: str, split: str) -> List[Tuple[List[int], List[int]]]:
        """
        Loads a constituency parsing dataset split (train/val/test) for a given dataset.
        Assumes each line in the raw data file is formatted as 'sentence ||| parse_tree'.
        Returns a list of (tokenized_sentence, tokenized_parse_tree) examples.
        """
        base_dir = os.path.join("data", "parsing", dataset_key)
        file_path = os.path.join(base_dir, f"{split}.txt")
        if not os.path.exists(file_path):
            logging.error(f"Parsing file for {dataset_key} split {split} not found: {file_path}.")
            return []

        examples: List[Tuple[List[int], List[int]]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Expecting a delimiter "|||"
                if "|||" in line:
                    parts = line.split("|||")
                    if len(parts) < 2:
                        continue
                    src_sentence = parts[0].strip()
                    tgt_tree = parts[1].strip()
                else:
                    continue  # Skip if delimiter not found.
                sp_processor = self.parsing_tokenizers.get(dataset_key)
                src_ids = sp_processor.encode(src_sentence, out_type=int)
                tgt_ids = sp_processor.encode(tgt_tree, out_type=int)
                examples.append((src_ids, tgt_ids))
        logging.info(f"Loaded {len(examples)} examples for parsing dataset {dataset_key} split {split}.")
        return examples
