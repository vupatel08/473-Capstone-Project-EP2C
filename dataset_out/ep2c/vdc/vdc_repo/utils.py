## utils.py
import os
import yaml
import json
import time
import logging
import random
import numpy as np
import torch
from torch import nn
from PIL import Image
from typing import Tuple, List, Dict, Optional

import requests
import base64
from io import BytesIO

# Optional: import for CLIP features
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration globally for reuse
_CONFIG = None

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load YAML configuration once and cache."""
    global _CONFIG
    if _CONFIG is None:
        with open(config_path, 'r') as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG

# --------------------- Image Handling & Preprocessing ---------------------
def load_image(image_path: str) -> Image.Image:
    """Load an image from disk."""
    return Image.open(image_path).convert('RGB')

def preprocess_image(image: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    """Resize and normalize an image for model input."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(image)

def image_to_tensor(image: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    return preprocess_image(image, size, mean, std)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL image."""
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    return to_pil(tensor.cpu())

# --------------------- Poisoning & Noise augmentation ---------------------
def overlay_trigger(image: Image.Image, trigger_size: int = 4, color: Tuple[int, int, int]=(255,255,255))
    """Overlay a white square patch at bottom-right corner."""
    img = image.copy()
    width, height = img.size
    for i in range(width - trigger_size, width):
        for j in range(height - trigger_size, height):
            img.putpixel((i, j), color)
    return img

def flip_labels(labels: List[int], method: str, ratio: float, num_classes: int) -> Tuple[List[int], set, List[int]]:
    """Apply noisy label flipping (symmetric/asymmetric). Returns new labels, indices, true labels."""
    corrupted_labels = labels.copy()
    total = len(labels)
    num_noisy = int(total * ratio)
    indices = list(range(total))
    random.shuffle(indices)
    noisy_idx = set(indices[:num_noisy])
    true_labels = labels.copy()

    for idx in noisy_idx:
        true_lbl = labels[idx]
        if method == 'symmetric':
            new_lbl = true_lbl
            while new_lbl == true_lbl:
                new_lbl = random.randint(0, num_classes -1)
            corrupted_labels[idx] = new_lbl
        elif method == 'asymmetric':
            new_lbl = (true_lbl + 1) % num_classes
            corrupted_labels[idx] = new_lbl
        # Extend if needed with other strategies
    return corrupted_labels, noisy_idx, true_labels

# --------------------- API Wrappers for LLM/MLLM ---------------------
def call_llm(prompt: str, api_key: str, model: str='gpt-4', temperature: float=0.2, max_tokens: int=512) -> str:
    """
    Call OpenAI API with retries.
    """
    retry = 3
    for attempt in range(retry):
        try:
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response.choices[0].message['content'].strip()
            return answer
        except Exception as e:
            logger.warning(f"API call attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    logger.error("LLM API call failed after retries.")
    return ""

def answer_question_with_ml(image: Image.Image, question: str, ml_model, ml_processor, device='cpu') -> str:
    """
    Use local or API multimodal model to answer a question about an image.
    Args:
        image: PIL Image
        question: str
        ml_model: the loaded multimodal model (e.g., BLIP2, Otter)
        ml_processor: corresponding processor
        device: computation device
    Return:
        str: answer text
    """
    # Example implementation for models like OFA, BLIP2, Otter, etc.
    # This function needs to be adapted per specific model.
    # Here, we assume the model has a method 'generate_answer'
    # For illustration, using BLIP2 pipeline if available:
    # from transformers import pipeline
    # qa_pipeline = pipeline('image-question-answering', model=ml_model, tokenizer=ml_processor)
    # answer = qa_pipeline(image, question)
    # return answer['answer']
    # Since the design supports plug-and-play, we just raise NotImplementedError
    raise NotImplementedError("Implement model inference according to specific multimodal model used.")

# Placeholder: For openai's GPT-4 with vision, actual API call is required, which isn't publicly available yet.
# Here, just as a framework placeholder:
def answer_question_with_openai(image_path: str, question: str, api_key: str, model: str='gpt-4-vision') -> str:
    """
    Actual GPT-4 with vision API call (if accessible). Placeholder.
    """
    # TODO: Implement with appropriate API call as per provider's docs
    return ""  # Return empty string or placeholder

# --------------------- Prompt Construction ---------------------
def generate_general_question_prompts() -> List[str]:
    """Generate fixed set of general questions (from Appendix E1)."""
    return [
        "Describe the image in detail.",
        "Describe the image briefly.",
        "How would you summarize the content of the image in a few words?",
        "Provide a detailed description of the given image.",
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the given image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo’s key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."
    ]

def generate_label_specific_prompt(label: str) -> str:
    """Create prompt template for label-specific questions."""
    return (
        "Generate questions to verify if the object in the image corresponds to the label '{label}'. "
        "The questions should be answerable with 'yes' or 'no'. Focus on attributes, features, "
        "or functions characteristic of the label '{label}'."
    ).format(label=label)

def generate_evaluate_prompt(response: str, label: str) -> str:
    """Create prompt for GPT to evaluate correctness of answer."""
    prompt = (
        f"Assume you are a helpful and precise assistant. "
        f"Question: 'Does the answer \"{response}\" correctly describe the label \"{label}\"?' "
        f"Respond with 'yes' or 'no' only."
    )
    return prompt

# --------------------- Response Parsing & Scoring ---------------------
def parse_and_evaluate_response(question: str, answer: str, label: str) -> Tuple[bool, float]:
    """
    Determine correctness based on answer text and generate semantic similarity.
    For yes/no, parse answer string.
    """
    q_lower = question.lower()
    a_lower = answer.lower()

    # Use heuristics: if question is likely deterministic
    if "?" in question:
        # For open-ended, use GPT prompt (call API)
        prompt = generate_evaluate_prompt(answer, label)
        # Do API call (not implemented here)
        # is_correct = call_llm(prompt, api_key, ...)
        # Placeholder:
        is_correct = a_lower.startswith("yes")
        # For similarity, could compute via CLIP
        similarity = 0.0  # Placeholder
        return (is_correct, similarity)
    else:
        # deterministic: check string presence
        if "yes" in a_lower:
            return (True, 0.0)
        elif "no" in a_lower:
            return (False, 0.0)
        else:
            return (False, 0.0)

def compute_similarity_score(image: Image.Image, label: str, clip_model, clip_processor, device='cpu') -> float:
    """
    Compute cosine similarity between image and label text using CLIP.
    """
    # Encode label text
    text_inputs = clip_processor(text=label, return_tensors="pt", padding=True).to(device)
    text_emb = clip_model.get_text_features(**text_inputs)
    text_emb = nn.functional.normalize(text_emb, p=2, dim=-1)

    # Encode image
    image_input = clip_processor(images=image, return_tensors="pt").to(device)
    image_emb = clip_model.get_image_features(**image_input)
    image_emb = nn.functional.normalize(image_emb, p=2, dim=-1)

    similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()
    return similarity

# --------------------- Scoring and Detection ---------------------
def aggregate_sample_score(
    correctness_list: List[bool],
    similarity_score: float,
    use_similarity: bool = True,
    similarity_weight: float =0.5
) -> float:
    """Calculate sample's overall score based on correctness and similarity."""
    correct_count = sum(correctness_list)
    total_q = len(correctness_list)
    correctness_ratio = correct_count / total_q if total_q > 0 else 0.0
    if use_similarity:
        combined_score = (1 - similarity_weight) * correctness_ratio + similarity_weight * similarity_score
        return combined_score
    else:
        return correctness_ratio

def classify_sample(score: float, threshold: float=0.2) -> bool:
    """Classify sample as dirty if score below threshold."""
    return score < threshold

# --------------------- Performance Metrics Computation ---------------------
def compute_metrics(
    correctness_per_sample: List[List[bool]],
    similarity_scores: List[float],
    true_dirty_flags: List[bool]
) -> Dict:
    """Compute TPR, FPR given correctness, similarity, and ground truth."""
    scores = []
    pred_dirty = []
    for correctness_list, sim_score in zip(correctness_per_sample, similarity_scores):
        score = aggregate_sample_score(correctness_list, sim_score)
        scores.append(score)
        pred_dirty.append(score < 0.2)
    # Calculate TPR and FPR
    TP = sum(p and gt for p, gt in zip(pred_dirty, true_dirty_flags))
    FP = sum(p and not gt for p, gt in zip(pred_dirty, true_dirty_flags))
    total_dirty = sum(true_dirty_flags)
    total_clean = len(true_dirty_flags) - total_dirty
    TPR = TP / total_dirty if total_dirty > 0 else 0.0
    FPR = FP / total_clean if total_clean > 0 else 0.0
    return {
        'scores': scores,
        'pred_labels': pred_dirty,
        'TPR': TPR,
        'FPR': FPR
    }
