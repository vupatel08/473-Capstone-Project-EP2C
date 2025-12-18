# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

{
  "dataset_loader.py": [
    {
      "component": "Class: ProteinTextDataset",
      "description": "Defines a unified dataset class to handle loading, preprocessing, and serving data samples for different phases (pretraining, finetuning, testing).",
      "key_functions": [
        " __init__(self, data_path: str, mode: str, config: dict)": Initialize dataset with file path, mode (pretraining/finetuning/test), and configuration.",
        " __len__(self)": Return dataset size.",
        " __getitem__(self, idx: int)": Return a preprocessed sample, including protein sequence, textual annotations, mutation info, and optional labels.",
        " _load_data(self)": Internal method to parse raw data files, converting raw entries into structured samples.",
        " _preprocess_sample(self, raw_sample: dict)": Convert raw data into model-ready format (tokenized sequences, texts, labels).",
        " _enrich_text(self, text: str, literature_refs: list)": Use literature references and prompts to expand low-quality annotations, via data enrichment routines."
      ],
      "inputs": [
        "Dataset files in JSONL/CSV format containing fields: 'protein_sequence', 'mutation_info', 'mutational_effect_text', 'literature_refs', 'species', 'sequence_length', etc.",
        "Optional: enriched textual data stored separately."
      ],
      "outputs": [
        "PyTorch Dataset objects yielding tokenized protein sequences, textual annotations, and mutation labels.",
        "Processed and tokenized fields: sequence_ids, attention_mask, text_ids, text_attention_mask, mutation_labels (positions, amino acids)."
      ],
      "logic": [
        "Read dataset files (e.g., JSON, CSV) upon initialization.",
        " For each entry:",
        "   - Parse protein sequence and mutation info.",
        "   - Tokenize protein sequences with a shared tokenizer (from transformers).",
        "   - Tokenize textual annotations; handle low-quality descriptions.",
        "   - For data augmentation: apply 'reverse' samples by swapping wild-type and mutant info, generating 'opposite' mutation explanations.",
        "   - Enrich textual annotations for samples with short or low-information descriptions:",
        "       * Use literature references to retrieve relevant abstracts.",
        "       * Generate prompts based on templates (A5-A6).",
        "       * Use GPT APIs or a placeholder function to expand descriptions.",
        "   - Store enriched text, protein tokens, mutation labels, and associated metadata.",
        "Handle data splits: during loading, filter or partition data into train/validation/test sets.",
        "Require flexibility for different modes:",
        "   - Pretraining: focus on protein-text pairs with sequence masking (no mutation labels).",
        "   - Finetuning: include mutation annotations and mutation explanation labels.",
        "   - Testing: include evaluation samples with full info for explanation and proposal tasks."
      ],
      "special considerations": [
        "Ensure consistent tokenization: use same tokenizer for sequences and texts, as per the model (e.g., ESM or BioMedGPT tokenizer).",
        "When performing text enrichment, process literature abstracts with GPT-3.5 (via API or mocked function).",
        "Store enriched data for reuse to avoid recomputation.",
        "Support different data splits: train, validation, test, based on internal logic or external homology/temporal splits (per B.2).",
        "Manage data augmentation with reversed samples, balancing benign/malignant annotations.",
        "Include handling for missing data or incomplete annotations.",
        "Ensure dataset outputs are compatible with collate functions for data loaders."
      ],
      "dependencies": [
        "datasets", "numpy", "json", "csv", "torch", "transformers",
        "GPT API wrapper or local generator for text enrichment",
        "homology computation tools if filtering by sequence similarity"
      ],
      "unclear_parts": [
        "Exact dataset file formats and structure (JSON/CSV schemas).",
        " Whether the literature abstract retrieval is via an external API or pre-downloaded data.",
        "How to handle variable sequence lengths and text lengths during batching.",
        "Specific tokenization details for the textual and protein sequences.",
        "Exact format of enrichment prompts templates."
      ],
      "summary": "Design 'ProteinTextDataset' to facilitate multi-phase data loading. Implement internal methods for raw data parsing, tokenization, enrichment (via literature), and data augmentation. Provide configuration-sensitive processing so that for pretraining, only sequences and texts are included; for finetuning, mutation labels and explanations are integrated; for test, evaluation samples are prepared with full info. Data enrichment leverages literature retrieval and GPT-based expansion, with fallbacks if necessary. Properly batch and collate samples for efficient DataLoader usage."
    }
  ],
  "notes": "This detailed logic analysis ensures the dataset loader can handle our extensive dataset with enriched textual annotations, balancing data quality, and supporting diverse training workflows. Implementation should tightly follow the described steps and validate each component with unit tests for data integrity."
}

## evaluation.py

{
  "evaluation.py": {
    "Purpose": "Implement evaluation routines for both mutation explanation (text generation) and mutation proposal tasks, alongside visualization functions for mutation ranking and performance analysis.",
    "Dependencies": [
      "predictions from trainer.py",
      "dataset information (e.g., ground-truth annotations, mutation details)",
      "standard NLP evaluation libraries (e.g., ROUGE, BLEU, METEOR from nltk or rouge-score packages)",
      "scikit-learn for metrics like Spearman correlation",
      "matplotlib/ seaborn for visualization"
    ],
    "Key Components": {
      "1. evaluate_explanation": {
        "Input": "Predicted explanations and ground-truth annotations for mutation explanations.",
        "Outputs": {
          "ROUGE-L": "Using a standard ROUGE implementation, compute ROUGE-L scores between predicted and reference explanations.",
          "BLEU-2": "Compute BLEU-2 scores to quantify n-gram consistency.",
          "METEOR": "Calculate METEOR scores for semantic and lexical quality assessment."
        },
        "Process": [
          "Ensure predictions and references are tokenized consistently.",
          "Apply rouge-score, nltk bleu_score, nltk METEOR or equivalent libraries.",
          "Aggregate scores over the dataset samples, producing mean and possibly standard deviation metrics for reporting."
        ],
        "Reference": "Use batch processing for all samples, with optional computation of confidence intervals if multiple runs are performed."
      },
      "2. evaluate_mutations": {
        "Input": "Model's ranked candidate mutations, ground-truth mutated positions and amino acids, and potentially the relevance scores.",
        "Outputs": {
          "Recall@50": "Calculate the proportion of true mutation sites present within top-50 ranked proposals.",
          "Position accuracy": "Measure the fraction of proposals correctly placing mutation at the true position.",
          "Amino acid accuracy": "Compute how often the predicted mutated amino acid matches the ground-truth.",
          "Spearman correlation": "Compute the correlation between predicted mutation scores (e.g., mutation likelihood logits) and ground-truth fitness or mutation effects."
        },
        "Process": [
          "For each test protein, compare the top-K proposals with ground-truth mutations.",
          "Count how many of the true mutation positions are within the proposals, for each sample.",
          "Compute the proportion over the dataset for Recall@50.",
          "For amino acid accuracy, check if the top predicted amino acid (or top K) matches ground-truth.",
          "For correlation, align the model's mutation scores with experimentally measured or reference scores.",
          "Aggregate results globally (mean, std) and justify variance with dataset splits."
        ],
        "Note": "Handle potential imbalance in mutation types by stratifying metrics if needed."
      },
      "3. visualization functions": {
        "Purpose": "Visualize mutation proposal rankings, fitness trajectories across multiple rounds, and correlations.",
        "Visualizations": [
          "Fitness landscape curves across multi-round optimization (line plots with mean ± std shading).",
          "Bar plots or histograms for proposal distributions.",
          "Scatter plots for correlation analysis (proposals vs. ground truth).",
          "Heatmaps for mutation scores and proposal rankings."
        ],
        "Process": [
          "Input: mutation proposal scores, sequence, fitness data, ground truth.",
          "Use matplotlib or seaborn to plot curves, distributions, or heatmaps.",
          "Annotate with relevant dataset info (e.g., protein name, mutation site).",
          "Save figures with clear labels and captions for reporting."
        ]
      }
    },
    "Implementation notes": {
      "Data handling": "Predictions and ground truths should be standardized in terms of sequence formats, mutation representation, and scoring systems.",
      "Evaluation consistency": "Ensure tokenization, scoring, and comparison protocols align with the training and test pipelines.",
      "Computational efficiency": "Batch evaluation for multiple samples; precompute metrics where possible.",
      "Robustness": "Handle missing or ambiguous data gracefully; provide error logs for debugging."
    },
    "Unclear points": [
      "Exact schema of predictions: e.g., dictionary formats, ranking scores used for proposals.",
      "Whether to include multi-metric scoring or flexibility for different datasets.",
      "Details on visualization aesthetics (labels, plot types) and output formats."
    ],
    "Summary": "Design evaluation.py to include functions for calculating standard NLP metrics for explanation quality and specialized metrics (recall, accuracy, correlation) for mutation proposal tasks, along with visualization utilities for analysis and presentation of model performances."
  }
}

## main.py

# Main.py Logic Analysis for "MutaPLM" Implementation

---

## Purpose
`main.py` serves as the orchestration script that coordinates the entire training and evaluation pipeline of the MutaPLM framework. It handles configuration parsing, dataset loading and preprocessing, model initialization, training, evaluation, logging, and saving artifacts. It depends on modules such as `dataset_loader.py`, `model.py`, `trainer.py`, `evaluation.py`, `prompt_templates.py`, and `utils.py`.

---

## Key Components & Responsibilities

### 1. Configuration Parsing
- Load hyperparameters and paths from `config.yaml`.
- Support potential command-line overwrite or argument parsing for flexibility (optional).
- Extract key configurations:
  - `training`: learning rate, batch size, steps, optimizer, etc.
  - `model`: model names, delta hidden size, number of layers/heads.
  - `dataset`: paths for pretraining, finetuning, test, and literature/enrichment data.
  - `prompt_templates`: paths or embedded templates.
  - `evaluation`: metrics and evaluation interval.
  - `hardware`: GPU info, training schedule.

### 2. Logging & Reproducibility Setup
- Initialize logging, print banner, and record environment info (PyTorch, transformers version, GPU info).
- Set random seed for reproducibility (using `torch.manual_seed`).

### 3. Dataset Loading & Preparation
- Instantiate `DatasetLoader` class:
  - Load pretraining dataset (`load_pretraining_data`) from specified path.
  - Load and enrich finetuning dataset (`load_finetuning_data`) with literature abstraction info (`data_enrichment`) if specified.
  - Load test data (`load_test_data`) for evaluation.
  - Ensure datasets contain:
    - Protein sequences
    - Mutation annotations
    - Textual descriptions (for finetuning and test)
    - Enriched literature info
- Tokenize datasets:
  - Convert sequences into token IDs using the PLM tokenizer.
  - Tokenize textual descriptions with the text encoder tokenizer.
  - Prepare masked tokens for sequence masking objectives.
  - Generate prompt inputs based on specified prompt templates.

### 4. Model Initialization
- Instantiate `ProteinPLM` with pre-trained PLM name (e.g., "facebook/esm2_t6_8a_14B").
- Instantiate `TextEncoder` (e.g., BioMedGPT or other) with specified pretrained weights.
- Instantiate `ProteinDeltaNetwork`:
  - Pass in PLM instance
  - Set `delta_hidden_dim` from configuration
- For chain-of-thought prompts:
  - Prepare prompt templates via `prompt_templates.py`.
  - Initialize learnable soft prompts (`PromptEmbedding`) if used.
- Wrap these into a training `Model` object or keep separately if using custom logic.

### 5. Optimizer & Scheduler Setup
- Define parameters:
  - For PLMs and text encoders: possibly freeze or fine-tune.
  - For delta network, prompt embeddings: typically trainable.
- Use `AdamW` optimizer:
  - Include all trainable parameters.
  - Set learning rate, epsilon as per config.
- Setup learning rate Scheduler:
  - `linear_warmup` over `warmup_steps`.
  - Linear decay or cosine schedule over total training steps (`max_steps`).

### 6. Training Loop
- Loop over `max_steps` or epoch equivalents.
- For each batch in training:
  - Load batch data:
    - Sequence tokens
    - Text tokens
    - Enrichment info (if any)
  - Forward pass:
    - Obtain protein representations from PLM
    - Compute mutation delta features via delta network
    - For pretraining objectives:
      - Sequence masking & reconstruction
      - Text generation conditioned on protein features
    - For finetuning with CoT:
      - Generate function description (first round)
      - Generate mutational effects and mutation proposals (second round)
  - Compute combined losses:
    - Sequence MLM loss
    - Text generation loss (cross entropy)
    - Mutation position and amino acid prediction heads
  - Backpropagation:
    - Clip gradients (per config)
    - Update parameters with optimizer
  - Periodically:
    - Save checkpoints
    - Log loss metrics and maybe validation performance

### 7. Validation & Evaluation
- Run evaluation at intervals (e.g., every certain number of steps or epochs):
  - Mutation explanation task:
    - Generate explanations for held-out samples
    - Compute ROUGE-L, BLEU-2, METEOR
    - Optionally, perform human-AI candidate assessment (if integrated)
  - Mutation engineering task:
    - Generate top mutation proposals
    - Measure recall@50 and accuracy
    - Correlate with mutation fitness scores (if available)
- Store evaluation metrics and model checkpoints.

### 8. Final Testing
- Run on test datasets:
  - Easy, Medium, Hard splits for explanation
  - Exhaustive proposals for mutation engineering
- Record and output metrics.

### 9. Final Artifacts & Results Handling
- Save trained model weights.
- Save or dump evaluation results (JSON, CSV).
- Log hyperparameters and environment info.
- Save prompts/templates for inference reproducibility.

### 10. Clean-Up & Exit
- Close files, loggers.
- Optional: generate plots or visualization scripts for mutation proposal ranking, fitness trajectories.

---

## Additional Considerations
- Support for GPU training:
  - Use `torch.cuda.is_available`, `torch.device`.
  - Utilize DataParallel or DistributedDataParallel if needed.
- Handle multiple GPUs for large-scale models.
- Enable resuming training from checkpoint.
- Maintain modularity for substituting models, dataloaders, prompts, etc.
- Implement debugging hooks/logging for tracing intermediate tensors.

---

## Summary
`main.py` must:
- Parse configurations.
- Load and preprocess datasets.
- Initialize models and components.
- Set up optimizer & scheduler.
- Run training with multi-task objectives, support CoT strategies.
- Periodically evaluate and log metrics.
- Save models and logs.
- Be flexible for debugging, resuming, and hyperparameter adjustment.

---

This detailed logic analysis ensures a clear, complete blueprint for implementing `main.py` as a robust entrypoint for the MutaPLM framework.

## model.py

{
  "Overview": "The `model.py` module defines all core neural network components necessary for the MutaPLM framework. It encompasses classes that wrap pretrained protein sequence models (ProteinPLM), language models for textual data (TextEncoder), the explicit mutation modeling architecture (ProteinDeltaNetwork), and auxiliary classes for soft prompt token management (PromptEmbedding). The design enables modular composition of the holistic mutation explanation and engineering pipeline, integrating cross-modal features with explicit mutation representations.",
  "Class: ProteinPLM": {
    "Purpose": "To encapsulate a pretrained protein language model (e.g., ESM-2) capable of producing dense per-residue embeddings from input sequences.",
    "Key Responsibilities": [
      "Load pretrained weights from 'transformers' library or local checkpoint.",
      "Tokenize input amino acid sequences into model-expected format.",
      "Forward pass: obtain hidden states (embeddings) for the sequence.",
      "Provide interfaces to extract sequence-level, residue-level representations.",
      "Optionally, perform sequence reconstruction or mutation probability predictions as per downstream heads."
    ],
    "Implementation Notes": {
      "Input": "Sequence string or token IDs.",
      "Output": "Sequence of hidden states (Tensor of shape [L, D]) or pooled sequence embedding.",
      "Initialization": "From transformers.AutoModelForSequenceClassification or similar, loaded with specified checkpoint.",
      "Tokenizer": "Use the corresponding pretrained tokenizer for consistent tokenization.",
      "Device": "Ensure model and inputs are on GPU/CPU as per environment."
    }
  },
  "Class: TextEncoder": {
    "Purpose": "To encode biomedical texts or prompts into dense latent representations aligned with protein features for cross-modal interaction.",
    "Key Responsibilities": [
      "Load pretrained language model (e.g., BioMedGPT, SciBERT).",
      "Tokenize input textual descriptions or prompts.",
      "Perform forward pass to produce token embeddings, pooled sentence embedding, or special prompt embeddings.",
      "Output feature vectors suitable for prompt embeddings or sequence alignment."
    ],
    "Implementation Notes": {
      "Input": "Raw text string or list of strings.",
      "Output": "Tensor of shape [N, D], representing text features.",
      "Integration": "Will supply embeddings for prompts, chain-of-thought, mutation explanations, and soft prompts."
    }
  },
  "Class: PromptEmbedding": {
    "Purpose": "Manage trainable soft prompt tokens used to align textual and protein feature spaces, and to facilitate chain-of-thought prompting.",
    "Key Responsibilities": [
      "Initialize a set of soft tokens as learnable parameters (e.g., `K=32`, dim=D).",
      "Concatenate soft tokens with prompt embeddings during forward passes.",
      "Support gradient updates during training (pretraining and fine-tuning).",
      "Provide interfaces for retrieval of prompt embeddings for use in other models."
    ],
    "Implementation Notes": {
      "Initialization": "Random initialization, requires gradient tracking.",
      "Shape": "Tensor of shape [K, D], where K is the number of soft tokens, D is feature dimension.",
      "Usage": "Concatenate with textual token embeddings before passing into transformer layers."
    }
  },
  "Class: ProteinDeltaNetwork": {
    "Purpose": "To explicitly model mutation effects via delta representations, capturing the difference between wild-type and mutant sequences, and decoding mutation proposals.",
    "Structure": "Encoder-Decoder architecture with dedicated sub-modules:",
    "Components": {
      "WT encoder": "Encodes wild-type sequence features, produces fixed latent features (`z_wt`).",
      "Delta encoder": "Encodes mutation-induced differences (`h_delta`), input from `h_mut - h_wt`, producing `z_delta`.",
      "Delta decoder": "Reconstructs mutation effects (`h_delta`) from `z_delta` and WT features.",
      "Heads": {
        "Position head": "Predicts mutation site or whether position should mutate.",
        "Amino acid head": "Predicts the type of amino acid mutation at each position."
      }
    },
    "Functionality": [
      "Input: `h_wt` (wild-type features), `h_mut` (mutant features).",
      "Process: Compute `h_delta = h_mut - h_wt`, encode with delta encoder, reconstruct with delta decoder.",
      "Output: Mutation representation `h_delta`, mutation proposal distributions (positions & amino acids)."
    ],
    "Implementation Details": {
      "Architecture": "Cross-attention modules for encoder & decoder, with residual connections.",
      "Dimension": "Input/output latent dimensions match PLM hidden size (e.g., 768).",
      "Attention": "Multi-head, with `num_attention_heads`, possibly 12 in default.",
      "Loss": "To be used during training: mutation prediction loss (classification), reconstruction loss."
    }
  },
  "Class: Heads for Mutation Prediction": {
    "Purpose": "Provide task-specific classifiers for mutation site and amino acid type prediction.",
    "Components": {
      "PositionHead": "Fully connected layer predicting mutation position/mutability (binary or categorical).",
      "AminoAcidHead": "Fully connected layer projecting to amino acid classes (e.g., 20)."
    },
    "Implementation Notes": {
      "Input": "Reconstructed `h_mut` features.",
      "Output": "Probability distribution over positions or amino acids (via softmax).",
      "Initialization": "Typically initialized with Kaiming uniform; can be trained with cross-entropy."
    }
  },
  "Integration & Data Flow": "The overall pipeline orchestrates as follows:\n- Input raw protein sequence -> `ProteinPLM` -> `h_wt`\n- Sequence of textual prompts -> `TextEncoder` -> text features\n- Mutation site & sequence info -> `ProteinPLM` -> `h_mut`\n- Compute `h_delta = h_mut - h_wt`\n- `ProteinDeltaNetwork.Encode_delta` -> `z_delta`\n- `ProteinDeltaNetwork.Decode_delta` + `h_wt` -> reconstruct `h_delta`\n- Heads on `h_mut` for mutation proposals\n- During training, compute losses and backpropagate.\n- During inference, generate mutation proposals, explanations, and predictions as per prompts.",
  "Extension & Variants": "Design allows swapping in different pretrained models for proteins and text, supporting multiple modalities. The architecture supports both self-supervised pretraining (sequence masking, textual generation) and task-specific fine-tuning with chain-of-thought prompts.",
  "Implementation points": [
    "Inherit from torch.nn.Module for each class.",
    "Use transformers.AutoModel or AutoModelForSeqClassification for pretrained modules.",
    "Laid out with dependency on `transformers`, `torch`.",
    "Ensure that data passed into `forward()` is already tokenized and on correct device.",
    "Support batch processing with padding masks.",
    "Design modular: separate config parameters for hidden sizes, number of heads, dropout, etc. as per configuration."
  ],
  "Summary": "The structured design in `model.py` should provide a flexible, modular implementation of all core classes, with clear interface methods for encoding sequences, generating prompts, modeling mutation effects explicitly via delta representations, and predicting mutation proposals. These classes will facilitate both pretraining objectives and task-specific fine-tuning for explanation and engineering tasks, ensuring fidelity to the described architecture and training strategies."
}

## prompt_templates.py

{
  "prompt_templates.py": "The primary purpose of this module is to define and organize string templates for prompts used during pretraining, finetuning, and inference phases of the MutaPLM framework. These prompts support sequence modeling, cross-modal alignment, and chain-of-thought reasoning, ensuring consistent and effective input construction for both internal models and external APIs like GPT-4.\n\nKey categories and templates:\n\n1. **Pretraining Prompts**:\n   - Objective: To generate rich protein-text pairs and align their embeddings.\n   - Templates should include placeholders for embedding protein sequences, associated texts (e.g., titles, abstracts), and special tokens as needed.\n   - Example:\n     ```\n     \"Protein: {sequence}\n     Description: {text}\"\n     ```\n   - Usage: For masked LM and sequence-to-text generation during pretraining.\n\n2. **Finetuning Prompts for Chain-of-Thought (CoT) Tasks**:\n   - Objective: To perform reasoning about functions and mutational effects via multi-round dialogs.\n   - **Round 1 (Function Description)**:\n     - System prompt: Sets the role of the model.\n     - Input: Protein sequence and prompts to describe functions.\n     - Template example:\n       ```\n       \"You are an expert in biology. Given this protein sequence:\n       {protein_sequence}\n       Please describe its functions in a few sentences.\"\n       ```\n   - **Round 2 (Mutational Explanation)**:\n     - System prompt: Explains the task (description of mutation effects).\n     - Inputs: Output from round 1, mutated sequence info, and mutation site.\n     - Template example:\n       ```\n       \"Based on the previous function description:\n       {function_description}\n       and the mutation at position {pos} ({original_AA} to {mutant_AA}), explain the effect of this mutation.\"\n       ```\n   - **Round 2 (Mutation Engineering Proposal)**:\n     - System prompt: Requests mutation suggestions fitting the mutational effect description.\n     - Inputs: Function and mutational effect descriptions.\n     - Template example:\n       ```\n       \"Given the functional effect: {effect_description}\n       propose a single amino acid mutation in sequence:\n       {protein_sequence}\"\n       ```\n\n3. **Inference Prompts for External APIs (e.g., GPT-4)**:\n   - Purpose: To generate explanations or mutation proposals based on model outputs.\n   - Define clear instruction templates with placeholders, e.g.:\n     ```\n     \"You are an expert in protein biochemistry. Given the mutation at position {pos} ({original_AA} to {mutant_AA}) in the protein sequence:\n     {sequence}\n     Explain the potential functional impact.\"\n     ```\n     or for proposals:\n     ```\n     \"Based on the following description of mutational effects: {effect_description}\n     suggest a mutation proposal that could enhance or impair the function as described.\"\n     ```\n\n4. **Special Tokens and Placeholders**:\n   - Use tokens such as `<BOP>`, `<EOP>`, `<BOM>`, `<EOM>` for boundary demarcation of different dialog parts, as specified in Appendix A6.\n   - Placeholders for variables: `{protein_sequence}`, `{text}`, `{function_description}`, `{mutation_site}`, `{original_AA}`, `{mutant_AA}`, `{pos}`, `{effect_description}`.\n   - Consistent use of prompts to enforce the chain-of-thought reasoning in multi-turn dialogs.\n\n5. **General Principles for Templates**:\n   - Clear, unambiguous prompts, explicitly describing expected outputs.\n   - Prompts should include instructions to produce only the relevant output without additional commentary.\n   - Use of highlighted special tokens for marking dialog turns and parts, facilitating parsing during inference.\n\n6. **Implementation Considerations**:\n   - Store templates as string literals with placeholders.\n   - Ensure templates are compatible with Python string formatting (`.format()`) or f-strings.\n   - Have separate templates for:\n     - Protein description generation\n     - Mutation effect explanation\n     - Mutation proposal generation\n     - Inference prompts for GPT API calls, embedding context and instructions.\n\n**Summary**:\nDesign prompt_templates.py with string constants for each dialogue context, including placeholders for dynamic content (sequence, effects, mutation site, etc.), boundary tokens, and task-specific instructions. Use consistent formatting to facilitate prompt filling during training, inference, and interaction with external APIs. This organization supports flexible, clear, and effective prompt management crucial for chain-of-thought operations and cross-modal alignment, aligning tightly with the detailed explanations, prompts, and templates described in the paper’s Appendix A6 and A7."
}

## trainer.py

{
  "trainer.py": {
    "Overview": "This module orchestrates the training, validation, and testing procedures for both pretraining and finetuning phases of the MutaPLM system. It manages data iteration, forward passes through models, loss computation for multiple objectives, backpropagation, optimizer updates, learning rate scheduling, checkpointing, and evaluation. It supports multi-task objectives, including sequence masking, text generation, cross-modal alignment, mutation explanation, and mutation proposal tasks, aligned with the paper's methodology.",
    "Main Components": {
      "Initialization": {
        "Inputs": [
          "model (from model.py)",
          "datasets (from dataset_loader.py)",
          "prompt templates (from prompt_templates.py)",
          "training hyperparameters (from config.yaml)",
          "optimizer, scheduler, loss functions"
        ],
        "Tasks": [
          "Instantiate neural network components: PLMs, text encoders, delta networks",
          "Wrap datasets with proper batching and tokenization",
          "Initialize optimizer, scheduler with provided hyperparameters",
          "Set up checkpointing, logging, and evaluation hooks"
        ]
      },
      "Training Loop": {
        "Stages": [
          "Pretraining phase:",
          "Finetuning phase:"
        ],
        "Per-Epoch Process": {
          "Data Loading": "Iterate over datasets in batches",
          "Forward Pass": {
            "common": "For each batch, compute model outputs",
            "pretraining": [
              "Sequence masking prediction",
              "Text generation prediction",
              "Cross-modal embedding alignment"
            ],
            "finetuning": [
              "Function description generation (first round)",
              "Mutational effect explanation (second round)",
              "Mutation proposal prediction (second round)"
            ]
          },
          "Loss Calculation": {
            "Objectives": [
              "Sequence masking loss (cross-entropy with masked amino acids)",
              "Text generation loss (cross-entropy over generated tokens)",
              "Cross-modal embedding loss (alignment between text and protein features)",
              "Mutation explanation loss (ROUGE, BLEU, METEOR approximations or token-level CE)",
              "Mutation proposal loss (classification of mutations, positional and amino acid heads)"
            ],
            "Implementation": "Combine all or sub-objectives with appropriate weights (from hyperparameters)",
            "Notes": "Use functions from utils.py for loss computation; follow the paper's detailed formulas"
          },
          "Backward & Optimization": {
            "Gradient Calculation": "Backpropagate total loss",
            "Gradient Clipping": "Apply gradient clipping with value from config to stabilize training",
            "Optimizer Step": "Update model parameters",
            "Learning Rate Scheduler": "Adjust learning rate as scheduled"
          },
          "Checkpointing & Logging": {
            "Frequency": "Every N steps or epochs, as configured",
            "Content": "Model weights, optimizer states, training metrics, loss history"
          }
        }
      },
      "Validation & Evaluation": {
        "Frequency": "At regular intervals (every epoch or specified steps)",
        "Procedures": {
          "Evaluation on validation set": "Compute defined metrics (ROUGE-L, BLEU, METEOR for explanation; recall@50, accuracy for proposals)",
          "Model selection": "Save best checkpoints based on validation performance",
          "Evaluation on test set": "At final stage, similarly profile and report metrics"
        }
      },
      "Supporting functions": {
        "Batch Preparation": "Tokenize sequences, texts, prepare input tensors",
        "Loss functions": "Implement per-objective loss functions with masking and weighting",
        "Model Saving": "Save checkpoints and logs in structured directories",
        "Learning Rate Scheduling": "Implement warmup + cosine decay or linear warmup schedule",
        "Logging": "Track metrics with tensorboard or simple logging for reproducibility"
      },
      "Special Considerations": {
        "Multi-task Handling": "Combine multiple losses with weights; ensure gradients are properly accumulated",
        "Chain-of-Thought Prompts": "Generate specific prompts for first and second round dialogues during training, possibly with teacher forcing",
        "Data Handling": "Ensure data batches preserve sequence order, alignment, and enriched textual info",
        "Model Components": "Maintain modularity to allow swapping between different PLMs or text encoders",
        "Reproducibility": "Set random seeds, document hyperparameters, maintain code version & environment info"
      },
      "Unclear / Needs Confirmation": {
        "Loss balancing": "Exact weights for multi-objective losses (e.g., sequence CE vs. text CE) not specified, approximate from paper",
        "Evaluation metrics implementations": "Use established packages (e.g., nltk, rouge-score, sacrebleu) for BLEU, ROUGE",
        "Training schedule details": "Exact schedule (warmup steps, decay) to match paper preferences",
        "Handling special tokens": "Ensure prompt tokens, <BOP>, <EOP>, <BOM>, <EOM> are correctly tokenized and integrated"
      }
    },
    "Summary": "The trainer.py module’s core logic encompasses dataset iteration, model forward passes, multi-task loss calculation, backpropagation, and evaluation, supporting both pretraining and finetuning. It integrates the components from model.py, dataset_loader.py, and prompt_templates.py, following the described multi-task objectives. Proper modular design, hyperparameter consistency, and detailed logging are essential to faithfully reproduce the experiments as per the paper’s methodology.",
    "Next steps": "Once the logic is validated, implementation can proceed with ontology-compliant code, modular abstraction, and careful hyperparameter tuning to match the paper’s experimental settings."
  }
}

## utils.py

# utils.py - Logic Analysis

This module will contain auxiliary functions that facilitate key operations including tokenization, model checkpoint management, hyperparameter tuning, logging, and API interactions for inference or enrichment tasks. The logic must be consistent with the provided paper, plan, and configuration, ensuring reproducibility, modularity, and clarity.

---

## 1. Tokenization and Data Processing

### 1.1 Protein Sequence Tokenization
- **Purpose**: Convert raw amino acid sequences into token IDs suitable for PLM input.
- **Implementation**:
  - Use the tokenizer associated with the selected pre-trained protein language model (`plm_name`), e.g., `facebook/esm2_t6_8a_14B`.
  - Function: `tokenize_protein(sequence: str) -> dict`:
    - Input: raw amino acid sequence string.
    - Output: tokenized dictionary containing `input_ids` and `attention_mask`.
  - Maintain consistent tokenization for training, evaluation, and inference.
- **Considerations**:
  - Handle special tokens if required (e.g., start/end, padding).
  - Support batch tokenization for efficiency.

### 1.2 Text Tokenization
- **Purpose**: Tokenize textual descriptions for either input to models or prompt generation.
- **Implementation**:
  - Use tokenizer of `text_encoder_name`, e.g., a SciBERT or BioMedGPT tokenizer.
  - Function: `tokenize_text(text: str, max_length: int) -> dict`:
    - Support truncation/padding to a predefined max sequence length.
    - Return `input_ids` and `attention_mask`.
- **Note**:
  - Ensure compatibility between protein sequence tokens and text tokens during prompt assembly.

---

## 2. Checkpoint Management
- **Purpose**: Save and load model states during training/evaluation for reproducibility.
- **Implementation**:
  - `save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str)`.
  - `load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str)`.
  - Save epoch/step info along with model weights.
- **Structure**:
  - Save in standard `.pt` or `.bin` format.
  - Ensure consistent naming conventions, e.g., `checkpoint_<step>.pth`.
- **Additional**:
  - Support partial reloads if fine-tuning continues from specific stages.

---

## 3. Hyperparameter Tuning Support
- **Purpose**: Provide utilities for dynamic hyperparameter adjustment, scheduling, and experiment tracking.
- **Implementation**:
  - Learning rate schedules:
    - `get_lr_scheduler(optimizer, schedule_type='linear_warmup', total_steps, warmup_steps)`.
      - Implement linear warmup plus cosine annealing (match with config).
  - Gradient clipping:
    - Function: `clip_gradients(model, clip_value)` to prevent exploding gradients.
  - Batch size and accumulation:
    - Set according to `config['training']['batch_size']`, adapt if needed.
- **Experiment Tracking**:
  - Logging functions:
    - `log_metrics(metrics_dict, step, log_dir)`.
    - Integrate with TensorBoard or stdout logging.
  - Save hyperparameters in a JSON or YAML format for reproducibility.

---

## 4. Result Logging & Visualization
- **Purpose**:
  - Record training/validation metrics per epoch/step.
  - Save sample outputs (mutational explanations, proposals).
- **Implementation**:
  - `save_training_progress(losses, metrics, path)`.
  - `log_sample_predictions(samples, path)`:
    - Save model outputs for inspection.
  - Use `SummaryWriter` for TensorBoard logs for live visualization.

---

## 5. API Call Wrappers (for external GPT APIs, if applicable)
- **Purpose**:
  - Efficiently interact with OpenAI API for GPT-4 or GPT-3.5-turbo during data enrichment or evaluation.
- **Implementation**:
  - `call_openai_api(prompt: str, model: str='gpt-4', max_tokens=512) -> str`.
  - Support batching if multiple prompts.
  - Handle rate limiting, retries.
- **Notes**:
  - Implement optional offline mode if API unavailable.
  - Cache previous calls for reproducibility and cost-efficiency.

---

## 6. Utility Functions
### 6.1 String/Sequence Utilities
- e.g., `strip_whitespace(text: str) -> str`.
- e.g., `pad_sequence(sequence: str, length: int, pad_token: int) -> list`.
  
### 6.2 Data Loading and Conversion
- Loading raw datasets, converting CSV/JSON to proper input formats.
- Extract info such as mutation site, textual annotations, etc.

### 6.3 Random Seeds & Reproducibility
- `set_seed(seed: int)` to fix random seeds for `torch`, `numpy`, and Python `random`.
- Ensures consistent training, inference, and data shuffling.

### 6.4 Evaluation Metrics Calculations
- Functions for BLEU, ROUGE, METEOR (via nltk, rouge-score, etc.).
- Functions for recall@50, accuracy, Spearman correlation, matching the described metrics.

---

## 7. Miscellaneous
- **Config parsing**: Load parameters from YAML (`pyyaml`).
- **Device management**: Detect CUDA or CPU and ensure models and tensors are accordingly assigned.
- **Environment configuration**: Set environment variables, e.g., `CUDA_VISIBLE_DEVICES`.
- **Logging utility**: Standardized logger setup with timestamps, levels, and output files.

---

## Summary of Key Functions to Implement:

```python
def set_seed(seed: int): ...
def tokenize_protein(sequence: str) -> dict: ...
def tokenize_text(text: str, max_length: int) -> dict: ...
def save_checkpoint(model, optimizer, step, path): ...
def load_checkpoint(model, optimizer, path): ...
def get_lr_scheduler(optimizer, schedule_type, total_steps, warmup_steps): ...
def clip_gradients(model, clip_value): ...
def log_metrics(metrics: dict, step: int, log_dir: str): ...
def call_openai_api(prompt: str, model='gpt-4', max_tokens=512) -> str: ...
def save_sample_predictions(samples: list, path: str): ...
def load_dataset(path: str): ...
```

All functions should be designed to support flexible integration, consistent data formats, and reproducibility as emphasized in the paper’s methodology and experimental plan.

---

This detailed logic analysis provides a standard, modular foundation for implementing the utility functions required to train, evaluate, and manage this complex multi-modal model system aligned with the paper.

