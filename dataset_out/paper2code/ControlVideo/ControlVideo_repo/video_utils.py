## video_utils.py
import cv2
import numpy as np
import torch
from typing import List
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model for CLIP similarity metrics
try:
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _clip_model.to(device)
except:
    _clip_model = None
    _clip_processor = None

def save_frames_as_video(frames: List[np.ndarray], filename: str, fps: float = 25.0) -> None:
    """
    Save a sequence of RGB frames as a video file.
    
    Args:
        frames (List[np.ndarray]): List of frames in RGB format, shape (H, W, 3), dtype uint8.
        filename (str): Output filename, e.g., 'output.mp4'.
        fps (float): Frames per second.
    """
    if not frames:
        raise ValueError("Empty frame list provided.")
    height, width = frames[0].shape[:2]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG' etc.
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()

def extract_features(frames: List[np.ndarray], feature_type: str) -> np.ndarray:
    """
    Extract features from frames for evaluation metrics.
    
    Args:
        frames (List[np.ndarray]): List of RGB frames, shape (H, W, 3), dtype uint8.
        feature_type (str): One of 'FID', 'CLIP', 'Flow'.
    
    Returns:
        np.ndarray: Feature vectors (or flow fields) of shape (N, D) or (N-1, 2,H,W) for flow.
    """
    if feature_type == "FID":
        # Use ResNet50 features as placeholder, can be replaced with InceptionV3 for real FID
        model = resnet50(pretrained=True).eval()
        preprocess = Compose([Resize((299, 299)), ToTensor(), Normalize(mean=[0.5]*3, std=[0.5]*3)])
        features = []
        with torch.no_grad():
            for frame in frames:
                img_tensor = preprocess(Image.fromarray(frame)).unsqueeze(0).to(next(model.parameters()).device)
                feat = model.conv1(img_tensor)
                feat = model.avgpool(feat)
                feat = torch.flatten(feat, 1).cpu().numpy()
                features.append(feat)
        return np.array(features)
    elif feature_type == "CLIP":
        if _clip_model is None:
            raise RuntimeError("CLIP model not initialized.")
        inputs = _clip_processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            img_feats = _clip_model.get_image_features(**inputs).cpu().numpy()
        # Normalize features
        norms = np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-6
        img_feats = img_feats / norms
        return img_feats
    elif feature_type == "Flow":
        # Compute optical flow between consecutive frames
        flow_fields = []
        for i in range(len(frames)-1):
            prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            nextf = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, nextf, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow_fields.append(flow)
        # Stack flow fields
        return np.array(flow_fields)  # shape: (N-1, H, W, 2)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

def compute_warping_error(frames: List[np.ndarray]) -> float:
    """
    Compute average warping errors based on optical flow to quantify temporal consistency.
    
    Args:
        frames (List[np.ndarray]): List of frames in RGB.
    
    Returns:
        float: Average warping error.
    """
    errors = []
    for i in range(len(frames)-1):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        nextf = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nextf, None, pyr_scale=0.5,
                                            levels=3, winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2, flags=0)
        h, w = prev.shape
        flow_map = -flow  # Flow from prev to next
        flow_map = np.clip(flow_map, -w, w)
        # Warp previous frame to next frame
        h_coords, w_coords = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (w_coords + flow_map[..., 0]).astype(np.float32)
        map_y = (h_coords + flow_map[..., 1]).astype(np.float32)
        warped_prev = cv2.remap(frames[i], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Compute L2 difference
        diff = np.mean((frames[i+1].astype(np.float32) - warped_prev.astype(np.float32)) ** 2)
        errors.append(diff)
    return float(np.mean(errors)) if errors else 0.0

def save_frames_as_video(frames: List[np.ndarray], filename: str, fps: float = 25.0) -> None:
    """
    Save sequence of RGB frames as a video file.
    """
    if not frames:
        raise ValueError("Frame list is empty.")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

def extract_feature_from_video(video_path: str, feature_type: str) -> np.ndarray:
    """
    Extract features from a video file by reading frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        success, frame = cap.read()
    cap.release()
    return extract_features(frames, feature_type)
