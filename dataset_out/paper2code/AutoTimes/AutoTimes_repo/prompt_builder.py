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
