# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset.py

**Logic Analysis for dataset.py — Implementing the Dataset Class for Scribble-Supervised Semantic Segmentation**

---

### Overview
The dataset.py module must define a Dataset class tailored for the Pascal VOC 2012 dataset with scribble annotations, supporting data loading, parsing, augmentation, and serving data in a format compatible with PyTorch DataLoader. The class should handle:
- Loading images and annotations (including scribble masks and labels).
- Applying data augmentations consistent with training experiments.
- Returning image-tensor and label-pixel pairs for training and evaluation.
- Supporting optional inference with or without scribble labels and class guidance.

---

### 1. Data Loading and Initialization
- **Dataset Path Configuration:**
  - Use paths specified in the config.yaml: `dataset_path` and `scribble_annotations_path`.
  - Dataset folder structure typically includes:
    - Images: `.../JPEGImages/` or similar.
    - Annotations: Pascal VOC-style segmentation masks, here, scribble masks stored separately, possibly in another directory structure.
- **Indexing Files:**
  - Generate a list of image IDs (e.g., `train`, `val` splits).
  - For each ID, map:
    - Image file (e.g., JPEG or PNG).
    - Scribble annotation file.
    - Ground truth label image for evaluation (if available).
- **Data for each sample:**
  - Raw image: To be loaded as `uint8` array.
  - Scribble annotation mask: Tells which pixels are labeled (with class IDs) and which are ignored.
  - Possibly, a class-wise label indicator to facilitate partial cross entropy computation.

---

### 2. Parsing and Preprocessing
- **Image Loading:**
  - Read the image using `PIL.Image.open()` or OpenCV.
  - Convert to RGB tensor.
- **Scribble Mask Loading:**
  - Load scribble mask images (e.g., PNGs). These could encode:
    - Labeled pixels: class IDs.
    - Unlabeled pixels: special ignore index (e.g., -1 or 255).
  - Ensure correct datatype (integers), normalize labels if necessary.

- **Ground Truth for Evaluation:**
  - Load per-image semantic label maps for validation.
  - Use during evaluation, not for training.

---

### 3. Data Augmentation
Apply consistent augmentations during training:
- **Random Scaling:**
  - Use albumentations’ `RandomResizedCrop` or custom scaling within the range [0.5, 2.0].
- **Random Rotation:**
  - Rotate images randomly within [-10, 10] degrees.
- **Horizontal Flipping:**
  - Random flip.
- **Gaussian Blur:**
  - Apply with a probability; implement with albumentations.
- **Cropping:**
  - Final crop to 512×512.
- **Transform Pipeline:**
  - Encapsulate all augmentations into a single albumentations Compose object.
  - Apply identically to image and scribble label mask to maintain pixel correspondence.
- **Additional Preprocessing:**
  - Normalize images (mean, std) if model requires.
  - Convert images and masks to tensors (`torch.FloatTensor`, `torch.LongTensor`).

---

### 4. Returning Data Samples
- Each `__getitem__()` call must return:
  - **Image Tensor:** shape `(3, H, W)` as float (after augmentation), normalized.
  - **Labels Tensor:** pixel-wise class labels, shape `(H, W)`.
    - For labeled pixels: class IDs.
    - For unlabeled pixels: ignore index (-1 or 255).
- **Sample Data Structure:**
```python
{
  'image': image_tensor,
  'label': label_tensor,
  'mask': scribble_mask (optional),
  'image_path': file_path (for debugging),
  'anno_path': annotation_path (optional),
}
```

---

### 5. Supporting Different Modes
- **Training Mode:**
  - Load images with the augmentation pipeline.
  - Return both image and scribble label masks.
- **Validation Mode:**
  - Load original images without augmentation.
  - Return the full label map for mIoU calculation.
- **Inference Mode:**
  - Possibly disable augmentation.
  - Return images only, with optional support for guidance if needed (e.g., class labels).

---

### 6. Handling Scribble Annotations
- **Format Assumptions:**
  - Typically stored as PNG images with class IDs for scribbled pixels.
  - Ignore unlabeled pixels during loss computation.
- **Storage:**
  - Store as a separate directory, with filenames matching the images.
- **Processing:**
  - Read masks to identify labeled regions.
  - Convert class IDs to tensor; set unlabeled pixels to ignore index.

---

### 7. Data Transformations & Consistency
- **Synchronization:**
  - Ensure augmentation is applied identically to both image and mask (albumentations supports mask transformations).
- **Transformation Arguments:**
  - Random scale, rotation, flip, Gaussian blur, crop.
  - Keep parameters consistent with training config.
- **Post-processing:**
  - Clip or normalize images after augmentation.
  - Convert masks to `LongTensor` for loss functions.

---

### 8. Additional Considerations
- **Caching:**
  - Optional: cache images/masks to speed up loading.
- **Shuffling & Workers:**
  - Use `DataLoader` parameters appropriately.
- **Dataset splits:**
  - Implement `split` argument (`train`, `val`) to load different subsets.
- **Debugging:**
  - Implement a method to visualize sample augmentation results to verify correctness.

---

### 9. Implementation Summary
- In the `__init__()`:
  - Store file lists, initialize augmentation pipeline.
  - Load category labels if necessary.
- In the `__getitem__()`:
  - Load image, scribble mask.
  - Apply augmentations.
  - Transform to tensors.
  - Return packed sample.
- In the `__len__()`:
  - Return dataset size.

---

### Final Notes
- The dataset class must be flexible enough to support different modes (train/val/inference).
- Ensure robustness to missing or corrupted files.
- Maintain efficient data loading compatible with batch training.
- Abstract file paths and parameters from config.yaml for flexibility.

---

This detailed logic analysis provides a clear guideline to implement the Dataset class aligned with the proposed method, experimental setup, and data handling nuances described in the paper.

## evaluation.py

### Evaluation.py Logic Analysis for `evaluate()` Function

The purpose of `evaluate()` is to load a trained semantic segmentation model, perform inference on the validation dataset, compute quantitative metrics such as per-class IoU and mean IoU, and record these results for analysis and comparison. Additionally, it may include optional prototype-based guidance during inference based on configuration.

This analysis outlines the detailed steps and considerations to implement the function correctly, aligning with the paper's methodology, experimental setup, and the provided configuration.

---

### 1. **Setup and Initialization**

- **Load Configuration Parameters:**
  - Dataset paths (validation data directory, scribble annotations if applicable).
  - Whether to utilize prototypes during inference (`use_prototypes`).
  - Whether to apply class label guidance during inference (`class_guidance`), if enabled.
  - Model architecture and checkpoint path for loading trained weights.

- **Initialize Environment:**
  - Set device: CPU or CUDA GPU.
  - Ensure reproducibility if necessary (e.g., set seed).

- **Load the Trained Model:**
  - Instantiate the model architecture specified in the config (e.g., Segformer MiT-B1).
  - Load trained weights (checkpoint file path provided or pre-specified).
  - Set the model to evaluation mode (`model.eval()`).

---

### 2. **Data Loader for Validation Set**

- **Dataset Preparation:**
  - Use the same dataset class as for training (`dataset.py`) but in 'validation' split.
  - Implement a DataLoader with batch size `batch_size` (from config).
  - Apply necessary transformations:
    - Resize or crop to `image_size` (512).
    - Avoid data augmentation or use only deterministic transforms suitable for evaluation.
    - Load the original images and the ground truth labels.

- **Scribble Annotations:**
  - During evaluation, typically, scribble annotations are not used; only the original labels are used to compute metrics.
  - However, if the model relies on scribble masks for auxiliary tasks, ignore during evaluation unless specified.

---

### 3. **Inference Loop - Per Batch**

For each batch of images:

- **Move input tensors to device.**

- **Optional Prototype Guidance:**
  - If `use_prototypes` is `True`, further steps depend on whether guidance during inference is enabled:
    - If `class_guidance` is `True`, use additional predicted class labels or classification scores to retrieve relevant prototypes from the prototype memory bank.
    - Else, proceed without guidance or with a heuristic (e.g., confidence threshold filtering of prototypes) as described in the paper.

- **Model Forward Pass:**
  - Feed the images into the model:
    - During inference, typically the forward process:
      - Generate feature maps via backbone.
      - Pass through the decoder.
      - Produce predicted segmentation logits (size: batch × classes × height × width).
  - If feature augmentation with prototypes is enabled:
    - Retrieve relevant prototypes.
    - Perform prototype-based feature augmentation:
      - For local prototypes, might involve attention-weighted features.
      - For global prototypes, may involve merging global prototype sets and augmenting features.
    - Use the augmented features to generate the final logits.

- **Prediction Extraction:**
  - Convert logits to predicted label maps:
    - Use `torch.argmax` over class dimension.
  - Move predictions to CPU for metric evaluation.

- **Store predictions and ground truths for all images.**

---

### 4. **Metric Computation**

- **Per-class IoU calculation:**
  - Use `scikit-learn` functions such as `sklearn.metrics.confusion_matrix` or manual IoU calculation:
    - For each class, compute intersection and union:
      - Intersection = count of pixels where prediction == GT == class.
      - Union = count of pixels where prediction == class OR GT == class.
    - IoU for each class = intersection / union.

- **Mean IoU:**
  - Average per-class IoU across all 21 categories.

- **Per-class and mean IoU storage:**
  - Save metrics for reporting.
  - Optionally, generate visualizations such as confusion matrices.

---

### 5. **Optional Features / Variants**

- **Prototype Guidance During Evaluation:**
  - If `use_prototypes` and `class_guidance` are enabled:
    - Load or compute the prototypes associated with the models.
    - Use the full prototype-based feature augmentation process:
      - Retrieve prototypes aligned with predicted categories.
      - Re-augment features.
      - Recompute the logits or directly modify the prediction maps.
  - This step could improve accuracy, as in the training phase, but should be used only if intended by the evaluation protocol.

- **Additional Post-processing:**
  - Optionally, apply DenseCRF or other post-processing techniques for better boundary delineation (not mandatory unless specified).
  
---

### 6. **Final Metrics and Result Recording**

- **Aggregate predictions and ground truths:**
  - Accumulate across all validation images.

- **Compute overall metrics:**
  - Per-class IoU.
  - Mean IoU.

- **Record results:**
  - Save per-class IoU results (e.g., CSV, JSON).
  - Save overall score.
  - Optionally, save some qualitative visualizations of segmentation for analysis.

---

### 7. **Return and Output**

- **Return a dictionary with:**
  - `per_class_iou`: dictionary mapping class IDs to IoU scores.
  - `mean_iou`: float for overall mean IoU.
  - Additional info (e.g., confusion matrix, time taken).

- **Logging and presentation:**
  - Print detailed metrics.
  - Save metrics to a report file if required.

---

### 8. **Error Handling & Robustness**

- Check for consistency in model and dataset loading.
- Handle cases where prototypes are not available or guidance is disabled.
- Ensure predictions are properly aligned with ground truth labels, masking ignored pixels if necessary.

---

### Summary

This evaluation.py plan ensures a proper, reproducible, and accurate assessment of the proposed prototype-guided scribble-supervised semantic segmentation method. It respects the experimental design detailed in the paper, leverages the configuration settings, and accommodates optional prototype guidance during inference for improved performance analysis.

## main.py

# Main.py Logic Analysis for Scribble-Supervised Semantic Segmentation Reproduction

The purpose of main.py is to serve as the central orchestration script that initializes all major components, manages training schedule, conducts training epochs, and performs evaluation, eventually producing performance metrics and visualization outputs. Based on the provided plan, design, task list, and configuration, this analysis outlines the detailed steps, flow control, and inter-component interactions needed for correct implementation.

---

## Overall Structure and Responsibilities

1. **Configuration Loading**
   - Load hyperparameters, model settings, data paths, augmentation parameters, prototype settings, and training schedule from the `config.yaml`.
   - Validate presence and correctness of configurations, ensuring all required parameters (learning rate, batch size, epochs, etc.) are available.

2. **Dataset Initialization**
   - Instantiate the training dataset object:
     - Load Pascal VOC 2012 images and corresponding scribble annotations.
     - Apply data augmentation (random scale, rotation, flip, Gaussian blur, crop to 512×512).
     - During data loading, parse scribble masks into a format compatible with classification labels and ignore regions (e.g., using label 255 for ignore).
   - Instantiate the validation dataset object:
     - Load images and ground truth labels for evaluation.
   - Create DataLoader instances for train and validation datasets:
     - Batch size as specified in config.
     - Shuffle training data.
     - Use appropriate collate_fn if necessary.

3. **Model Initialization**
   - Instantiate the segmentation model:
     - Use the specified backbone (`mit-b1`).
     - Load ImageNet pre-trained weights for the backbone.
     - Configure the decoder to produce segmentation logits.
   - Initialize auxiliary modules:
     - Prototype memory bank for global prototypes (number per class: 5).
     - Initialize prototype counters and fill states.
   - Implement the forward pass interface:
     - Extract multi-level features.
     - Generate initial segmentation predictions.
     - Support prototype-based feature augmentation in the forward pass.

4. **Trainer Instantiation**
   - Pass the model, datasets, and configuration parameters to the trainer.
   - Trainer manages:
     - Prototype extraction (local and global).
     - Prototype bank update operations.
     - Loss scheduling based on current epoch.
     - Forward passes with and without prototypes.
     - Computation of total loss, accumulation, and backpropagation.
     - Handling warm-up, partial, and full-prototype training phases.

5. **Training Loop**
   - For each epoch in total epochs:
     - **Phase determination**:
       - If epoch < warm-up epoch (`warmup_epochs`), disable prototype augmentation.
       - After warm-up, enable local prototypes (after first few epochs).
       - After prototypes are sufficiently reliable, enable global prototypes.
       - Adjust prototype-related flags or masks accordingly.
     - **Batch Iteration**:
       - Load batch images and scribble labels.
       - Forward pass through the model:
         - During the forward, perform prototype extraction:
           - Identify high-confidence pixels based on current model predictions.
           - Compute local prototypes:
             - Select top-K confident pixels per class (using confidence scores).
             - Average feature vectors weighted by confidence.
           - If applicable, update global prototypes:
             - Use cosine similarity to select which global prototype to update.
             - Update with running average (momentum α=0.99).
         - Optionally, augment features with prototypes:
           - Local augmentation: using local prototypes.
           - Global augmentation: using global prototypes (after fill).
         - Generate model predictions with augmented features.
       - **Loss Calculation**:
         - Compute partial cross-entropy on labeled pixels.
         - Compute consistency losses (initial vs. augmented predictions).
         - Combine losses with weights (`λ_l`, `λ_g`).
     - **Backpropagation**:
       - Perform optimizer step.
       - Step learning rate scheduler if applicable at `lr_decay_epoch`.
     - **Logging**:
       - Track individual losses, total loss, and mIoU on the validation set periodically.
     - **Checkpointing**:
       - Save model state at specified intervals or when performance improves.

6. **Evaluation**
   - After training completion:
     - Load best checkpoint (based on validation mIoU).
     - Run inference on validation set:
       - Support optional prototype augmentation if instructed.
       - Use the evaluation module to compute per-class IoU, mean IoU.
       - Save predictions visualizations, e.g., overlay segmentation masks on input images.

7. **Results Saving**
   - Store evaluation metrics (full mIoU, per-class IoU) in JSON or CSV.
   - Save sample visualizations illustrating boundary refinement, boundary regions guided by prototypes.
   - Log experimental details, including seed, hyperparameters, and dataset info, for reproducibility.

---

## Component Interaction and Data Flow

- **Initialization**:
  - Load config → datasets → dataloaders.
  - Initialize model → set to train mode.
  - Initialize prototype memory bank in trainer.
- **Each epoch**:
  - Iterate over batches:
    - Load images + scribble labels.
    - Forward pass:
      - Extract features.
      - Perform prototype extraction if phase allows.
      - Update prototypes and memory bank.
      - Perform feature augmentation with prototypes if enabled.
      - Generate segmentation predictions.
      - Compute loss functions.
    - Backward and optimize.
  - Post-epoch:
    - Calculate epoch metric (mIoU on validation).
    - Save checkpoint if improved.
    - Decay learning rate if scheduled.
- **Post-training**:
  - Run evaluation:
    - Load best model weights.
    - Run inference on validation dataset.
    - Produce final metrics and visualizations.

---

## Additional Considerations

- **Seeding**:
  - Fix random seed for reproducibility.
- **Hardware**:
  - Move models and tensors to GPU.
  - Ensure mixed precision if necessary.
- **Logging & Visualization**:
  - Record loss terms per epoch.
  - Visualize prototype influence, boundary enhancements.
  - Save logs for debugging.

---

## Summary

The main.py script will:
- Load configurations for model, data, training schedule.
- Instantiate dataset loaders with augmentation and scribble parsing.
- Build the segmentation model with a specific backbone.
- Initialize prototype memory bank and related flags.
- Manage the training loop, scheduling phases based on epoch.
- Conduct prototype extraction and update operations in each batch.
- Enable feature augmentation based on prototypes.
- Compute and optimize combined loss functions.
- Save model checkpoints.
- After training, perform evaluation with or without prototype guidance.
- Save evaluation results and sample visualizations.

This comprehensive design ensures faithful reproduction of the experimental methodology, maintaining the sequence and modular interactions as specified in the paper, plan, and JSON design.

## model.py

# Logic Analysis for model.py

## Overview
The `model.py` module encapsulates the core neural network architecture for the proposed scribble-supervised semantic segmentation system. It must include:

- A **Backbone Encoder** based on Segformer (MiT-B1 or other variants), extracting multi-level feature maps.
- A **Transformer Decoder**, processing fused features and producing segmentation logits.
- Modules for **Prototype Extraction and Feature Augmentation**, capable of handling both local and global prototypes.
- A **flexible interface** to incorporate prototypes during feature augmentation.

The design must adhere strictly to the specified interface, enabling subsequent modules (trainer, inference) to interact seamlessly.

---

## 1. Backbone Encoder: Segformer (MiT-B1)
- **Input**: RGB images of size `(B, 3, H, W)`.
- **Output**: List of multi-level feature tensors, typically four, each at different resolutions:
  - `F1`: Most detailed (highest resolution).
  - `F2`, `F3`, `F4`: Lower resolutions with richer semantic info.
- **Implementation**:
  - Use an existing Segformer implementation (e.g., from `timm` or custom).
  - Initialize with ImageNet pre-trained weights.
  - Extract features from specific transformer layers, possibly with a feature extraction head.
- **Notes**:
  - The features should have consistent channel dimensions; if not, include a linear projection layer to unify dimensions.

---

## 2. Transformer Decoder
- **Input**:
  - Multi-level features from the backbone.
  - Prototype representations (local and global, depending on stage).
- **Process**:
  - Fuse multi-level features into a compact, unified feature map (`F`).
  - Use the features as input to the decoder, which progressively upsamples to final output size.
- **Output**:
  - Segmentation logits: tensor `(B, C, H, W)` for classification over C categories.

---

## 3. Prototype Extraction and Feature Augmentation Modules
### 3.1 Prototype Extraction Interface:
- **Purpose**:
  - Obtain high-confidence feature prototypes for each class (local prototypes).
  - Use these prototypes for augmentation.
- **Implementation details**:
  - Use current prediction maps and feature maps:
    - For each class `t`, select top-K confident pixel features.
    - Compute weighted average of features based on prediction confidence.
  - Store or pass these prototypes to augmentation modules.

### 3.2 Prototype Management:
- **Local prototypes**:
  - Calculated each iteration from high-confidence regions.
  - Input to augmentation modules.
- **Global prototypes**:
  - Stored in memory banks, updated via cosine similarity comparison.
  - Managed externally, but accessible for augmentation.
  - Should be passed to the augmentation modules.

### 3.3 Feature Augmentation:
- **Design**:
  - Receives the feature tensor `F` and prototypes (`local` or `global`).
  - Computes attention weights between each feature vector and prototypes (dot product => softmax).
  - Propagates prototype information into feature maps via attention-weighted convolution or concatenation.
  - Utilize residual connections:
    - `f_aug = ReLU(f + Linear(concat(attn-weighted prototype, f)))`
- **Outputs**:
  - Augmented feature tensors that retain the same spatial size.
  - Ready to be fed into the decoder for prediction.

---

## 4. Class and Interface Design
- **`SegformerEncoder`**:
  - Method: `extract_features(x)`
  - Outputs: list of tensors `[F1, F2, F3, F4]`.
- **`TransformerDecoder`**:
  - Method: `forward(features, prototypes=None)`
  - Inputs:
    - Primary features (from encoder).
    - Optional prototypes (local/global) for feature augmentation.
  - Output:
    - Segmentation logits `(B, C, H, W)`.

- **`PrototypeExtractor`**:
  - Method: `compute_prototypes(features, predictions, labels, class_ids)`
  - Inputs:
    - Feature map: shape `(B, D, H, W)`.
    - Prediction map: `(B, C, H, W)`.
    - Labels: `(B, H, W)` (sparse labels or ignore regions).
    - Class IDs: set of present classes in current batch.
  - Output:
    - Dictionary: class-wise prototypes `(C: list of feature vectors)` or tensor `[C, K, D]`.

- **`FeatureAugmenter`**:
  - Method: `augment(features, prototypes)`
  - Inputs:
    - Feature tensor `(B, D, H, W)`.
    - Prototypes `(C, K, D)` for global, or `(C, D)` for local.
  - Output:
    - Augmented feature tensor `(B, D, H, W)`.

---

## 5. Module Implementation Details
### 5.1 Backbone
- Leverage an existing Segformer implementation:
  - E.g., import from `timm`.
  - Initialize with pre-trained weights.
  - Adjust extraction points to get multi-scale features.

### 5.2 Decoder
- Use a simple multi-scale fusion decoder:
  - Fuse feature maps (e.g., via upsampling + concatenation).
  - Use transformer blocks or convolutional layers for feature refinement.
  - Final last layer projects fused features to `C` classes.

### 5.3 Prototype Extraction
- For each class:
  - Mask the feature map regions corresponding to confidence-strong predictions.
  - Select top-K highest confidence pixels.
  - Compute weighted average and normalize.
- Keep local prototypes updated each iteration.

### 5.4 Prototype Memory Bank
- For each class:
  - Maintain a fixed-size tensor of prototypes `(K, D)`.
  - Update with momentum when new prototypes are computed:
    - `f_new = alpha * f_old + (1 - alpha) * f_current`.
  - Use cosine similarity to find the most similar prototype to replace.

### 5.5 Feature Augmentation
- For each feature vector:
  - Calculate similarity with prototypes.
  - Derive attention weights via softmax.
  - Compute weighted average prototypes.
  - Concatenate or add to original features.
  - Pass through a linear layer + ReLU + residual.
- Repeat separately for local and global prototypes (if both used).

## 6. Additional Considerations
- Ensure that features passed to augmentation modules are appropriately normalized when computing cosine similarity.
- Manage the prototype memory bank as a class-wise dictionary or tensor list.
- The network should be modular to facilitate easy toggling of prototype types (local/global/both).
- Keep track of the current training stage to control whether prototypes are used or not (warm-up vs. full phase).
- Implement methods to initialize prototypes, reset memory if needed, and validate the correctness of augmentation.

---

## Summary
The `model.py` will include classes and functions for:
- Loading and executing Segformer's encoder to get multi-level features.
- A flexible transformer decoder with optional prototype-guided feature augmentation.
- Modules for extracting prototypes from high-confidence regions.
- Maintaining and updating prototype memory banks based on cosine similarity.
- Applying prototype-based augmentation to features before decoding into segmentation logits.
- Providing interfaces for forward pass, prototype extraction, and augmentation that can be conditionally invoked based on training phase.

This detailed plan ensures a structured, modular, and faithful implementation aligned with the paper’s methodology and experimental procedures.

## trainer.py

# Logic Analysis for trainer.py

The `trainer.py` module is responsible for orchestrating the training, incorporating prototype extraction, prototype memory bank updates, loss scheduling, and the integration of prototype-based feature augmentation during forward passes. It depends heavily on the model architecture defined in `model.py`, the dataset loader in `dataset.py`, and the prototype management components.

Below is an in-depth, structured analysis of the logic flow, key functions, state management, and training procedures that should be implemented in `trainer.py`.

---

## 1. Core Class Structure and Initialization

- **Class `Trainer`**:
  - Initialize with:
    - The model object (from `model.py`)
    - The dataset or dataloader objects (`DataLoader`)
    - Configuration parameters (from `config.yaml`)
    - Optional: optimizer, scheduler, checkpoint path, device (GPU/CPU)
  
- **Attributes**:
  - `model`: neural network model, with methods to extract features, predict, and perform augmentation
  - `optimizer`: AdamW with specified learning rate
  - `scheduler`: multi-step learning rate decay
  - `proto_bank`: instance of proto memory bank managing global prototypes
  - `current_epoch`: current epoch during training
  - `training_phase`: indicator for warm-up, local-only proto usage, full proto usage phases
  - `loss_weights`: for partial CE, local, and global consistency losses
  - `phase_thresholds`: epochs marking phase changes (e.g., warm-up, full use of prototypes)
- **Initialization Tasks**:
  - Load or instantiate `PrototypeBank` with specified number of prototypes per class
  - Set training phase to warm-up initially
  - Set up optimizer, learning rate scheduler
  - Set up data loaders for training, validation
  - Initialize metrics tracking (loss history, mIoU, class IoU)

---

## 2. Forward Pass and Prototype Extraction

- **Input**:
  - Batch images and associated scribble labels (from dataset)
  - Batch prediction maps generated by model
  - Features extracted from the model's encoder(s) (via `model.extract_features`)
  
- **Process**:
  - **Initial forward pass**:
    - Run images through the model to get predictions (semantic maps)
    - Calculate partial cross-entropy loss with scribble labels (only on labeled pixels)
  
  - **Prototype extraction**:
    - During the warm-up phase, do not extract prototypes or perform augmentation
    - After warm-up:
      - For each image in batch:
        - Use the predicted probability map (confidence scores)
        - Select high-confidence pixels (e.g., top percentage specified by `interaction_topk`)
        - For each class present:
          - Gather feature vectors at these high-confidence pixels
          - Compute local prototypes (weighted average, normalized)
        - Store local prototypes per class in a temporary data structure
      - **Update global prototypes**:
        - For each local prototype (per class):
          - Search for the most similar prototype in the global memory bank
          - If global memory for the class isn't full:
            - Append the local prototype directly
          - Else:
            - Compute cosine similarities between local prototype and each in global bank
            - Replace the most similar one with an interpolated prototype:
              - `f_new = α * f_old + (1 - α) * f_local`
            - Update in place
  
- **Note**:
  - Prototype extraction is driven by the current predictions' confidence and the available scribble labels.
  - These prototypes are stored in `PrototypeBank` (or similar structure) and are used for subsequent augmentation.

---

## 3. Prototype-based Feature Augmentation

- **Local prototype augmentation**:
  - When the phase permits:
    - For each feature vector in the batch:
      - Compute attention weights with the current class's local prototype
      - Use the attention to generate an augmented feature (weighted prototype + original)
      - Residual connection: `f_aug = ReLU(f + Linear(Attention-weighted features))`
    - Replace or concatenate features for next prediction
  
- **Global prototype augmentation**:
  - When conditions satisfy (full global prototype set):
    - For each class:
      - Retrieve set of prototypes
      - Merge prototypes if needed
      - Perform similar attention-weighted augmentation as above
    - Use in combination with local prototypes (possibly as an ensemble)
  
- **During inference**:
  - If enabled, augment features with prototypes similarly to training (depending on test-time strategy).
  - Possibly integrate class guidance or filtering (if class label info is available or predicted).

---

## 4. Loss Calculation and Scheduling

- **Primary Loss (partial cross entropy)**:
  - Compute between predicted logits and scribble labels
  - Apply only on labeled pixels (`Ω_L`)
  - Scale by `partial_ce_scale`

- **Prototype-consistency Loss**:
  - When using prototypes:
    - Get augmented predictions from augmented features
    - Compute mean squared error (MSE) between initial prediction probabilities and augmented predictions (`L_con_l` and `L_con_g`)
    - Scale losses with respective weights (`λ_l`, `λ_g`)
  
- **Combined Loss**:
  - **Warm-up Epochs**:
    - Loss = `L_pce` only
  - **Phase 1 (local prototypes only)**:
    - Loss = `L_pce + λ_l * L_con_l`
  - **Phase 2 (local + global prototypes)**:
    - Loss = `L_pce + λ_l * L_con_l + λ_g * L_con_g`
  - **Phase transition management**:
    - Based on `epoch` or validation performance
    - Enable prototype augmentation accordingly
  
- **Loss optimization**:
  - Backpropagate total loss
  - Step optimizer and scheduler

---

## 5. Epoch & Phase Management

- **Phase thresholds**:
  - **Warm-up**: first `warmup_epochs`
  - **Prototype utilization start**:
    - After warm-up, enable prototype extraction and augmentation
  - **Full prototype usage**:
    - When global prototypes are fully filled
- **Transition triggers**:
  - Epoch count
  - Prototypes filled (for global prototypes)
- **At each epoch**:
  - Check and update `training_phase`
  - Adjust loss weights and whether to perform prototype-based augmentation
  - Possibly evaluate on validation set and log metrics

---

## 6. Checkpoints & Logging

- Save model checkpoints at regular intervals or best mIoU
- Log:
  - Loss components (partial CE, consistency)
  - mIoU per epoch
  - Prototype bank status
  - Sample visualizations of predictions, prototypes influence

---

## 7. Additional Considerations

- **Handling ambiguous or wrong prototypes**:
  - Filter prototypes from low confidence predictions
  - Possibly use pseudo labels or external classifiers (e.g., from classification models, as described in inference section)
  
- **Date/Time Management**:
  - Save training logs with timestamps
  - Use early stopping if needed based on validation performance
  
- **Device Management**:
  - Ensure model, prototype banks, and data are on GPU/CPU as appropriate
  - Handle DataParallel/DataDistributed if required

---

## Summary = Actionable Steps

1. Initialize `Trainer` with model, dataloader, configs.
2. For each epoch:
   - Determine current training phase.
   - For each batch:
     - Forward pass through model to get predictions, features.
     - Compute partial CE loss.
     - If out of warm-up:
       - Extract high-confidence prototypes per class.
       - Update prototype memory banks.
       - If conditions allow, perform feature augmentation:
         - Local prototypes augmentation.
         - Global prototypes augmentation (if set).
       - Generate augmented predictions.
       - Compute consistency losses between initial and augmented predictions.
     - Combine losses as per current phase.
     - Backpropagate and step optimizer.
   - Schedule learning rate decay at epoch `lr_decay_epoch`.
   - Log metrics and save checkpoints.
3. After training:
   - Run evaluation on validation set.
   - Visualize results.

These steps articulate the detailed logic flow needed for `trainer.py` to effectively manage training with dynamic prototype extraction, update, augmentation, and loss scheduling as outlined in the paper.

