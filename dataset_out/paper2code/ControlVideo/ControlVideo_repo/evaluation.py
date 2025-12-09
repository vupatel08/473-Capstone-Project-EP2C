## evaluation.py
import os
import json
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class EvaluationMetrics:
    """
    This class provides methods to evaluate generated videos on:
    - CLIP similarity (prompt fidelity)
    - Temporal consistency (frame stability)
    - FID (distribution similarity to real videos)
    
    It relies on pretrained CLIP and optional video feature extractors.
    """
    def __init__(self, config: dict):
        """
        Initialize models for feature extraction and set configuration.
        
        Args:
            config (dict): Configuration dictionary (parsed from 'config.yaml').
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Load CLIP model
        self.clip_model_name = self.config.get("clip_model_name", "openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Placeholder for real dataset features for FID
        # Could be loaded from files or computed on-the-fly
        self.real_video_features = None  # Optional: precompute external
        
        # For FID, collect features of real videos if available
        # Else, compute features on the fly as needed
        
    def extract_frame_features(self, frames: list) -> np.ndarray:
        """
        Extract image features for a list of frames using CLIP.
        
        Args:
            frames (list): List of np.ndarray RGB images in [H,W,3], dtype uint8.
        
        Returns:
            np.ndarray: Array of shape (N, feature_dim), normalized.
        """
        inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)  # shape: (N, D)
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    
    def embed_prompt(self, prompt: str) -> np.ndarray:
        """
        Embed the textual prompt into CLIP text features.
        
        Args:
            prompt (str): Input text prompt.
        
        Returns:
            np.ndarray: Embedding vector (1, D), normalized.
        """
        inputs = self.clip_processor(tokenizer=prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
        return text_embeddings.cpu().numpy()
    
    def compute_clip_similarity(self, prompt_embedding: np.ndarray, frame_embeddings: np.ndarray) -> float:
        """
        Compute average cosine similarity between prompt embedding and each frame.
        
        Args:
            prompt_embedding (np.ndarray): array of shape (1, D)
            frame_embeddings (np.ndarray): array of shape (N, D)
        
        Returns:
            float: Average cosine similarity score.
        """
        prompt_tensor = torch.tensor(prompt_embedding).to(self.device)
        frame_tensors = torch.tensor(frame_embeddings).to(self.device)
        similarities = cosine_similarity(prompt_tensor, frame_tensors)  # shape: (N,)
        return similarities.mean().item()
    
    def compute_temporal_consistency(self, frames: list) -> float:
        """
        Calculate temporal consistency over the video as mean cosine similarity
        between features of consecutive frames.
        
        Args:
            frames (list): List of np.ndarray RGB images.
        
        Returns:
            float: Average similarity between consecutive frames.
        """
        if len(frames) < 2:
            return 1.0  # Trivially consistent
        # Extract features for all frames
        frame_embeddings = self.extract_frame_features(frames)
        similarities = []
        for i in range(len(frame_embeddings) - 1):
            feat1 = torch.tensor(frame_embeddings[i])
            feat2 = torch.tensor(frame_embeddings[i+1])
            sim = cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
            similarities.append(sim.item())
        return np.mean(similarities)
    
    def compute_fid(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """
        Compute Fréchet Inception Distance (FID) between real and generated features.
        
        Args:
            real_features (np.ndarray): Features of real videos, shape (N_real, D)
            gen_features (np.ndarray): Features of generated videos, shape (N_gen, D)
        
        Returns:
            float: FID score.
        """
        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(gen_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(gen_features, rowvar=False)
        # Compute squared difference of means
        diff = np.linalg.norm(mu1 - mu2) ** 2
        # Compute sqrt of product of covariances
        covmean = _sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    def evaluate_video(self, video_frames: list, prompt: str, real_video_features: np.ndarray=None) -> dict:
        """
        Evaluate a generated video's quality metrics: CLIP similarity, TC, (optional) FID.
        
        Args:
            video_frames (list): List of np.ndarray RGB images.
            prompt (str): Text prompt.
            real_video_features (np.ndarray): Precomputed features of real videos for FID (optional).
        
        Returns:
            dict: Dictionary with keys 'clip_score', 'temporal_consistency', 'fid' (if real features provided)
        """
        results = {}
        # Embed prompt
        prompt_emb = self.embed_prompt(prompt)  # shape: (1, D)
        # Extract frame features
        frame_embs = self.extract_frame_features(video_frames)  # shape: (N, D)
        # CLIP similarity
        clip_score = self.compute_clip_similarity(prompt_emb, frame_embs)
        results['clip_score'] = clip_score

        # Temporal consistency
        tc_score = self.compute_temporal_consistency(video_frames)
        results['temporal_consistency'] = tc_score

        # FID (if real features provided)
        if real_video_features is not None:
            # Compute features for generated video
            gen_features = frame_embs
            # Calculate FID
            fid_score = self.compute_fid(real_video_features, gen_features)
            results['fid'] = fid_score
        else:
            results['fid'] = None  # Placeholder if no real features provided

        return results

    def evaluate_batch(self, videos: list, prompts: list, real_features_list: list=None) -> dict:
        """
        Evaluate a batch of videos, returning aggregated metrics.
        
        Args:
            videos (list): List of list of frames (np.ndarray).
            prompts (list): List of prompt strings.
            real_features_list (list): List of real video features, or None.
        
        Returns:
            dict: Averaged metrics over batch.
        """
        clip_scores = []
        tc_scores = []
        fid_scores = []

        for i in tqdm(range(len(videos)), desc="Evaluating videos"):
            video = videos[i]
            prompt = prompts[i]
            real_feats = None if (real_features_list is None) else real_features_list[i]
            metrics = self.evaluate_video(video, prompt, real_feats)
            clip_scores.append(metrics['clip_score'])
            tc_scores.append(metrics['temporal_consistency'])
            if metrics['fid'] is not None:
                fid_scores.append(metrics['fid'])
        
        avg_results = {
            'clip_score_mean': np.mean(clip_scores),
            'clip_score_std': np.std(clip_scores),
            'temporal_consistency_mean': np.mean(tc_scores),
            'temporal_consistency_std': np.std(tc_scores),
            'fid_mean': np.mean(fid_scores) if fid_scores else None,
            'fid_std': np.std(fid_scores) if fid_scores else None
        }
        return avg_results

    def save_results(self, results: dict, output_path: str):
        """
        Save evaluation metrics results into a JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)


def _sqrtm(matrix):
    """
    Compute matrix square root using scipy.linalg.sqrtm.
    """
    from scipy.linalg import sqrtm
    return sqrtm(matrix)
