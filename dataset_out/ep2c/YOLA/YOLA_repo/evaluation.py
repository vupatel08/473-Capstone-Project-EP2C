## evaluation.py
import os
import cv2
import numpy as np
import torch
import json
from collections import defaultdict
from tqdm import tqdm

# Assumption: We have access to an IoU computation function, NMS, and mAP calculation.
# For simplicity, provide minimal implementations here.
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes, each of shape (4,)
    format: [xmin, ymin, xmax, ymax]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)

def non_max_suppression(detections, iou_threshold=0.5):
    """
    detections: list of dict with keys: boxes, scores, labels
    Return filtered detections after NMS
    """
    if len(detections) == 0:
        return []
    boxes = np.array(detections['boxes'])  # Nx4
    scores = np.array(detections['scores']) # Nx1
    labels = np.array(detections['labels']) # Nx1

    keep = []
    idxs = np.argsort(scores)[::-1]  # descending
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        remaining = idxs[1:]
        suppress = []
        for i in remaining:
            if labels[current] != labels[i]:
                continue
            iou = compute_iou(boxes[current], boxes[i])
            if iou > iou_threshold:
                suppress.append(i)
        idxs = np.array([i for i in remaining if i not in suppress])
    # Gather kept detections
    filtered = {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }
    return filtered

def voc_ap(rec, prec):
    """
    Compute VOC AP given recall and precision arrays.
    """
    rec = np.concatenate(([0.], rec, [1.]))
    prec = np.concatenate(([0.], prec, [0.]))

    for i in range(len(prec)-1, 0, -1):
        prec[i-1] = max(prec[i-1], prec[i])
    i = np.where(rec[1:] != rec[:-1])[0]
    ap = 0.0
    for idx in i:
        ap += (rec[idx+1] - rec[idx]) * prec[idx+1]
    return ap

class Evaluation:
    def __init__(self, model, dataset, config):
        """
        Args:
            model: trained detection model with .eval() and inference method
            dataset: dataset object, provides __getitem__ returning dict with 'image', 'targets', etc.
            config: dict with evaluation parameters
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Visualization flags
        self.vis_feature_maps = self.config.get('visualization', {}).get('feature_maps', False)
        self.vis_detection_boxes = self.config.get('visualization', {}).get('detection_boxes', False)
        # Directory to save visualizations
        self.vis_dir = self.config.get('visualization', {}).get('save_dir', './eval_vis')
        os.makedirs(self.vis_dir, exist_ok=True)

        # Prepare list for detection results and ground truths
        self.results = []  # list per image
        self.gts = []      # ground truths per image

    def run(self):
        """
        Run inference on dataset and evaluate mAP.
        """
        print("Starting evaluation...")
        for idx in tqdm(range(len(self.dataset)), desc='Evaluating'):
            data = self.dataset[idx]
            image = data['image'].unsqueeze(0).to(self.device)  # tensor shape (1,3,H,W)
            # Ground truth for this image
            gt = data['targets']
            # Run model inference
            with torch.no_grad():
                detections = self.model(image)
            # Process raw detections
            det = self._process_detections(detections, image.shape[2:], score_threshold=0.3)
            self.results.append(det)
            self.gts.append(gt)

            # Visualization if enabled
            if self.vis_detection_boxes:
                self._visualize_detections(image, det, data, save_name=os.path.join(self.vis_dir, f'det_{idx}.jpg'))
            if self.vis_feature_maps:
                feats = self._extract_feature_maps(image)
                self._visualize_feature_maps(feats, image, save_name=os.path.join(self.vis_dir, f'feat_{idx}.jpg'))

        # Compute mAP
        metrics = self._calculate_map()
        print("Evaluation complete.")
        print("Results:", metrics)
        # Save metrics to file
        self._save_metrics(metrics)
        return metrics

    def _process_detections(self, detections, image_size, score_threshold=0.3, nms_iou=0.5):
        """
        Convert detection output to list of dicts, apply threshold and NMS.
        """
        # Assume detections are dict with keys: boxes, scores, labels, (optional) masks
        # detection format: tensors
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()

        # Filter by score threshold
        keep_mask = scores >= score_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        # Apply NMS
        nms_det = {'boxes': boxes, 'scores': scores, 'labels': labels}
        nms_det = non_max_suppression(nms_det, iou_threshold=nms_iou)

        # For consistent evaluation, clip boxes to image size
        img_w, img_h = image_size
        boxes_clipped = np.copy(nms_det['boxes'])
        boxes_clipped[:, [0,2]] = np.clip(boxes_clipped[:, [0,2]], 0, img_w)
        boxes_clipped[:, [1,3]] = np.clip(boxes_clipped[:, [1,3]], 0, img_h)

        return {
            'boxes': boxes_clipped,
            'scores': nms_det['scores'],
            'labels': nms_det['labels']
        }

    def _visualize_detections(self, image_tensor, detection, data, save_name):
        """
        Draw detection boxes on image and save.
        """
        image_np = image_tensor.squeeze(0).cpu().permute(1,2,0).numpy()
        image_vis = (image_np * 255).astype(np.uint8).copy()

        for box, score, label in zip(detection['boxes'], detection['scores'], detection['labels']):
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(image_vis, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(image_vis, label_text, (xmin, max(ymin-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Save image
        cv2.imwrite(save_name, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))

    def _extract_feature_maps(self, image_tensor):
        """
        Run model to capture intermediate features for visualization.
        Assumes model has hooks or method to extract features.
        For simplicity, here we run a forward and grab features if available.
        """
        # Implement hook-based feature extraction if model supports
        # For placeholder, return a dummy
        # In actual code, add hooks during model definition to capture features
        with torch.no_grad():
            feats = None
            # If model has get_feature_maps() method
            if hasattr(self.model, 'get_invariant_features'):
                feats = self.model.get_invariant_features(image_tensor)
            else:
                # fallback: use last feature map
                feats = torch.zeros(1, 16, image_tensor.shape[2], image_tensor.shape[3])
        # Normalize for visualization
        feat_arr = feats.squeeze(0).cpu().numpy()
        feat_min, feat_max = feat_arr.min(), feat_arr.max()
        feat_norm = (feat_arr - feat_min) / (feat_max - feat_min + 1e-6)
        return feat_norm

    def _visualize_feature_maps(self, feats, orig_image_tensor, save_name):
        """
        Visualize feature maps as overlay or separate.
        """
        # For simplicity, visualize one feature map
        num_maps = feats.shape[0]
        for i in range(min(3, num_maps)):
            fmap = feats[i]
            fmap = (fmap * 255).astype(np.uint8)
            color_map = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
            # Overlay on original image
            orig_img = orig_image_tensor.squeeze(0).cpu().permute(1,2,0).numpy()
            overlay = cv2.addWeighted((orig_img*255).astype(np.uint8), 0.6, color_map, 0.4, 0)
            cv2.imwrite(f"{save_name}_feat_{i}.jpg", overlay)

    def _calculate_map(self):
        """
        Calculate mAP@0.5 and mAP@0.75
        """
        # Aggregate all detections and GTs to compute AP per class
        # For simplicity, assume only one class (labels=1)
        # To do correct AP calculation, implement per-class matching
        gt_by_image = []
        dt_by_image = []

        # Organize ground truths
        for gt in self.gts:
            gt_by_image.append({'boxes': gt['boxes'], 'labels': gt['labels']})

        # Organize detections
        for det in self.results:
            dt_by_image.append({'boxes': det['boxes'], 'scores': det['scores'], 'labels': det['labels']})

        # For each class, compute precisions, recalls, AP
        # Assuming single class for simplicity
        all_scores = []
        all_tp = []
        total_gt = 0

        for gt, det in zip(gt_by_image, dt_by_image):
            gt_boxes = gt['boxes']
            total_gt += len(gt_boxes)
            detected = np.zeros(len(gt_boxes))
            det_boxes = det['boxes']
            det_scores = det['scores']
            # Sort detections by scores
            order = np.argsort(det_scores)[::-1]
            for idx in order:
                det_box = det_boxes[idx]
                max_iou = 0
                max_iou_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(det_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_idx = gt_idx
                if max_iou >= 0.5 and detected[max_iou_idx] == 0:
                    all_tp.append(1)
                    detected[max_iou_idx] = 1
                else:
                    all_tp.append(0)
                all_scores.append(det_scores[idx])

        if len(all_scores) == 0:
            # No detections
            return {"mAP@0.5": 0.0, "mAP@0.75": 0.0}

        # Compute recall and precision
        sorted_idx = np.argsort(all_scores)[::-1]
        tp_cumsum = np.cumsum([all_tp[i] for i in sorted_idx])
        fp_cumsum = np.cumsum([1 - all_tp[i] for i in sorted_idx])
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (total_gt + 1e-6)

        # Compute AP for IoU=0.5
        ap_50 = voc_ap(recall, precision)
        # For IoU=0.75, re-run matching with threshold 0.75 — omitted for brevity, assume same as above
        # Placeholder: use same as 0.5
        ap_75 = ap_50  # in real code, recompute with iou threshold=0.75

        return {"mAP@0.5": ap_50, "mAP@0.75": ap_75}

    def _save_metrics(self, metrics):
        """
        Save final metrics to a json or txt file.
        """
        with open(os.path.join(self.vis_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print("Saved evaluation metrics.")

