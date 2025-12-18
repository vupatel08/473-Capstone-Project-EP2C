# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a comprehensive, step-by-step plan to reproduce "MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering," structured into key components: methodology, experimental design, datasets, models, training procedures, hyperparameters, and evaluation metrics.

---

## 1. Overview & Objectives
- **Primary goal**: Develop an **explainable and engineerable protein mutation model** based on PLMs and cross-modal protein-text data.
- **Main contributions to reproduce**:
  - **Protein delta network** capturing explicit mutation effects.
  - Chain-of-thought (CoT)-style transfer learning pipeline with pretraining (on large corpora) and fine-tuning (on annotated mutation data).
  - Large-scale mutation dataset with rich textual annotations (MutaDescribe).
  - Evaluation on mutation explanation (text generation metrics) and mutation engineering (mutation proposal ranking, recall, accuracy).

---

## 2. Data Collection & Preparation

### 2.1 Pretraining Dataset
- **Source**: UniProtKB/SwissProt (~1.1M protein-text pairs).
- **Process**:
  - Download the latest SwissProt from UniProt.
  - Extract sequences and textual annotations (titles, descriptions, publications).
  - For text: clean, tokenize, and align with protein sequences.
- **Annotations available**:
  - Protein sequences + rich textual info (publications, descriptions).

### 2.2 Fine-tuning & Test Dataset: MutaDescribe
- **Source**:
  - UniProtKB Phenotypes & Variants entry (~107K mutants across ~21K proteins).
  - Enrich with literature abstracts (using PubMed references).
- **Enrichment process**:
  - Use provided mutation info to retrieve related publications.
  - Use GPT-3.5-Turbo prompts to extract detailed textual explanations.
  - Generate reverse (benign/malignant polarity flip) samples for data augmentation (~64.5K).
- **Annotations**:
  - Sequence, mutation site, original and mutated amino acids, textual mutational effects.

### 2.3 Additional Data
- **Species distribution**, **cellular localization**, **mutations per protein** for analysis.
- **Metadata & licensing info**: Collect license info for all datasets and code.

---

## 3. Model Architecture

### 3.1 Protein PLMs
- **PLMs to use**:
  - Baseline: ESM-2, ProtTrans, ProtGPT2 (initialize with pre-trained weights).
  - Large model: BioMedGPT-LM (if allowed), or open-source equivalents (LLaMA2 fine-tuned on biomedical data).
- **Implementation details**:
  - Use HuggingFace or official codebases for pre-trained models.
  - Maintain the same tokenizer/token embedding setup.
  - For the text encoder: align GAT-based transformers or BioBERT-style encoders.

### 3.2 Protein Delta Network
- **Input**:
  - Wild-type sequence features: `h_wt`.
  - Mutant sequence features: `h_mut`.
- **Architecture**:
  - Separate wild-type encoder: query-based cross-attention with trainable query vectors (`Q_wt`).
  - Delta encoder: similarly structured, independent parameters, producing `z_delta`.
  - Delta decoder:
    - Cross-attention layer (from WT features aligned with delta features).
    - Feed-forward network (with residual connection).
- **Output**:
  - Mutational features: `h_delta`.
  - Representation of mutation effect (`z_delta`) in a shared feature space.

### 3.3 Cross-modal bridging
- **Text-to-protein**:
  - Use trainable soft tokens appended to textual input.
  - The text encoder produces latent features which are aligned with protein features.
- **Protein-to-text**:
  - Use `z_wt`` and `z_delta` as inputs.
  - Decoder generates textual explanations (via transformer transformer).
- **Prompts & soft tokens**:
  - Based on the paper, implement prompts for CoT reasoning:
    - Function description (first round).
    - Mutational effects (second round).
  - Use prompt templates from Appendix A2.

---

## 4. Pretraining Procedure

### 4.1 Objectives
- **Protein Text Generation**:
  - Conditional autoregressive LM to generate textual descriptions, conditioned on `z_wt`+text embeddings.
- **Sequence Masked Language Modeling**:
  - Mask 15% of amino acids, predict masked tokens via PLM.
- **Cross-modal training**:
  - Align protein features with textual semantics (using soft tokens).

### 4.2 Training Steps
- **Protein Model Pretraining**:
  - Initialize with pre-trained PLMs (ESM-2, BioMedGPT-LM, etc.).
  - For large models: Use gradient checkpointing.
  - Use a batch size (e.g., 16-128) depending on GPU memory.
  - Adopt AdamW optimizer with learning rate warm-up (e.g., linear for first 10K steps).
  - Use the objective from Equations A5 and A8 (Eq. A5 for pretraining, Eq. A8 for cross-modal alignment).

### 4.3 Implementation details
- Train on large datasets (~1 million protein-text pairs).
- Train for ∼200K steps; adjust based on validation.
- Use mixed precision (FP16) with gradient accumulation to handle memory.
- Save model checkpoints periodically.

---

## 5. Fine-tuning Strategy

### 5.1 Chain-of-thought (CoT) Fine-tuning
- **Input**:
  - Trailing prompts from Appendix A6.
  - Rich textual annotations (mutational effects, functions).
- **Workflow**:
  - First round: Generate function description from the WT sequence and prompt.
  - Second round: Generate mutational effect explanations and mutation proposals.
- **Training loss**:
  - Sum of:
    - Next-token prediction (cross entropy) for function descriptions.
    - Next-token prediction for mutational effects.
    - Mutation proposal decoding (pos and amino acid) via classification heads.
  - Use the objectives in Eq. A10-A12.

### 5.2 Hyperparameters
- Learning rate: ~1e-5 to 5e-5.
- Batch size: 8–32 depending on GPU.
- Sequence masking: 15% amino acids.
- Number of rounds: 2.
- Use hyperparameters from Appendix A3, A12.
- Use gradient clipping (e.g., norm 1 or 2) for stability.
- Dropout rates: follow PLM standards (~0.1).

### 5.3 Multi-round finetuning
- Implement iterative inference with prompts, updating text features.
- Use teacher forcing during training.
- If possible, add regularization via model ensembling or soft prompt tuning.

---

## 6. Mutation Proposal & Engineering Experiments

### 6.1 Mutation proposals:
- Generate candidate mutations via mutation likelihood heads (`f_pos`, `f_aa`).
- Use greedy or beam search strategies for top mutations:
  - Sequence proposals at multiple loci.
  - Top-k proposals with highest mutation probability.

### 6.2 Evaluation Metrics
- **Explanation**: ROUGE-L, BLEU-2/4, METEOR.
- **Engineering**:
  - Recall@50: whether top-50 proposals include ground-truth mutation.
  - Accuracy/precision of proposal mutation positions and amino acids.
- **Fitness prediction**: Spearman correlation (Section 4.4, Appendix A11).

---

## 7. Experimental Setup & Hardware
- **Hardware**:
  - GPUs: Preferably A100/4x V100s for training large models.
  - Memory: 80–120 GB VRAM if training large PLMs.
- **Training Time**:
  - Pretraining: ~1-2 weeks (~200K steps).
  - Fine-tuning: days to weeks depending on dataset size.
- **Reproducibility**:
  - Use random seed (e.g., 42).
  - Document hyperparameters.
  - Log intermediate models and evaluation results.
  - Save inference prompts/templates for evaluation.

---

## 8. Additional Notes & Unresolved Aspects
- **Architectural details**:
  - Exact layer sizes, number of attention heads, dropout rates—use defaults matching PLMs or follow Appendix A references.
- **Cross-modal alignment**:
  - Key to implementation is aligning `z_delta` (protein mutation feature space) with textual features; plan to use the soft-token method as detailed.
- **Evaluation & Human Study**:
  - Incorporate human expert evaluation for mutational explanations (as in Section 4.2.3).
  - Compare GPT-4, GPT-3.5, and model-generated explanations.
- **Data Cleaning & Quality Assurance**:
  - Follow prompts/templates from Appendix A5-A6 for literature extraction.
  - Empirically verify textual annotation quality and mutation polarity correctness.

---

## Summary
This roadmap provides:
- The architecture blueprint, including delta network, prompts, and cross-modal mechanisms.
- The datasets needed, how to prepare/enrich them.
- The training pipeline: pretraining objectives, transfer learning, prompts for chain-of-thought reasoning.
- Hyperparameters based on paper references.
- Evaluation metrics and experimental considerations.

Once this plan is executed, implementation at code level can proceed systematically, layer by layer, ensuring fidelity to the original methodology and experimental setup.

---

Would you like me to proceed with detailed step-by-step code architecture, or clarify any specific segment?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular system utilizing HuggingFace Transformers for PLM and text encoders, PyTorch Lightning for training workflows, and custom modules for delta network, prompt management, and evaluation. The system will load pre-trained protein and text models, construct the protein delta network, and implement the chain-of-thought prompting pipeline for pretraining and finetuning. Data enrichment and management will be handled via dedicated data loader classes. The training loop will incorporate multi-task objectives, including sequence masking, text generation, and mutation proposal prediction, following the paper’s optimization strategy.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "prompt_templates.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict)\n        +load_pretraining_data() -> Dataset\n        +load_finetuning_data() -> Dataset\n        +load_test_data() -> Dataset\n    }\n    class ProteinTextEncoder {\n        +__init__(pretrained_model_name: str)\n        +encode_protein(sequence: str) -> Tensor\n        +encode_text(text: str) -> Tensor\n    }\n    class ProteinPLM {\n        +__init__(pretrained_model_name: str)\n        +forward(sequence_ids: Tensor) -> Tensor\n    }\n    class TextLanguageModel {\n        +__init__(pretrained_model_name: str)\n        +forward(token_ids: Tensor, attention_mask: Tensor) -> Tensor\n        +generate(prompt_embeddings: Tensor) -> str\n    }\n    class ProteinDeltaNetwork {\n        +__init__(plm: ProteinPLM, delta_hidden_dim: int)\n        +encode_delta(wt_feats: Tensor, mutant_feats: Tensor) -> Tensor\n        +decode_delta(delta_feats: Tensor) -> Tensor\n        +predict_mutation(h_wt: Tensor, h_delta: Tensor) -> dict\n    }\n    class PromptManager {\n        +get_pretraining_prompts() -> dict\n        +get_finetuning_prompts() -> dict\n    }\n    class Trainer {\n        +__init__(model: object, data: Dataset)\n        +train()/evaluate()\n        +save_checkpoint(path: str)\n    }\n    class Evaluator {\n        +evaluate_text_explanation(model, data) -> dict\n        +evaluate_mutation_proposals(model, data) -> dict\n    }\n\nRelationships:\nMain --> DatasetLoader\nMain --> Trainer\nTrainer --> ProteinPLM\nTrainer --> ProteinDeltaNetwork\nMain --> PromptManager\nTrainer --> TextLanguageModel\nProteinDeltaNetwork --> ProteinPLM \nProteinPLM --> Main\nTextLanguageModel --> Main\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant PLM as ProteinPLM\n    participant DeltaNet as ProteinDeltaNetwork\n    participant LLM as TextLanguageModel\n    participant PR as PromptManager\n    participant TR as Trainer\n    participant EV as Evaluator\n    M->>DL: load all datasets\n    DL-->>M: datasets loaded\n    M->>PLM: initialize with pre-trained weights\n    M->>PLM: initialize text encoder\n    M->>DeltaNet: initialize network with PLM\n    M->>PR: load prompt templates\n    activate training pipeline\n    loop from 1 to max_steps\n        Trainer->>PLM: encode protein sequence\n        Trainer->>DeltaNet: predict mutations, compute delta features\n        Trainer->>LLM: generate explanations with prompts\n        compute total loss (sequence, text, mutation prediction)\n        Trainer->>Trainer: optimize parameters\n        note right of Trainer: periodically save checkpoint\n    end\n    M->>EV: evaluate explanation quality\n    EV-->>M: store metrics\n    M->>EV: evaluate mutation proposals\n    EV-->>M: store proposal metrics\n    M->>Main: run complete\n",
    "Anything UNCLEAR": "Details on the exact open-source model versions to use, and specific hyperparameters for optimizer settings and training schedule, need clarification. Also, the precise format of the enriched textual annotations, including tokenization and special tokens, should be specified for full implementation fidelity."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "transformers==4.26.1",
        "datasets==2.13.0",
        "scipy==1.10.1",
        "numpy==1.22.4",
        "scikit-learn==0.24.2",
        "pyyaml==5.4.1",
        "pytorch_lightning==1.7.7"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains classes and functions for loading, preprocessing, and splitting the protein-text datasets: including 'load_pretraining_data', 'load_finetuning_data', and 'data_enrichment'. Dependencies: requires datasets and numpy; should integrate dataset formats for protein sequences and textual annotations, including enriched mutation descriptions."
        ],
        [
            "model.py",
            "Contains model class definitions:
                - ProteinPLM: wraps pre-trained PLMs (ESM-2, BioMedGPT, etc.)
                - TextEncoder: wraps language models (BioMedGPT, GPT-4, etc.)
                - ProteinDeltaNetwork: implements the separate protein delta encoder, decoder, and mutation prediction heads
                - PromptEmbedding: manages soft prompt tokens for cross-modal alignment
            Dependencies: requires transformers and torch; depends on pretrained models for initial weights."
        ],
        [
            "trainer.py",
            "Implements training routines:
                - Pretraining: sequence masking, text generation, cross-modal embedding alignment
                - Finetuning: chain-of-thought prompting, mutation explanation prediction, mutation proposal prediction
                - Handles multi-task objectives as specified in the paper (sequence + text + mutation heads)
            Dependencies: depends on model.py, dataset_loader.py, and torch Lightning; integrates loss functions, optimizer, scheduler."
        ],
        [
            "evaluation.py",
            "Contains evaluation classes and functions:
                - evaluate_explanation: ROUGE, BLEU, METEOR
                - evaluate_mutations: Recall@50, accuracy, correlation metrics
                - visualizations for mutation proposals
            Dependencies: depends on predictions from trainer.py and dataset info"
        ],
        [
            "prompt_templates.py",
            "Defines prompt templates for pretraining and finetuning:
                - Prompts for protein sequences, textual annotations
                - Chain-of-thought prompts for explanation and engineering
                - Prompts used by GPT APIs during inference if needed."
        ],
        [
            "main.py",
            "The entry point orchestrating dataset loading, model setup, training, evaluation, and logging. Responsible for configuration setup, command-line argument parsing, and logging results. Depends on all other modules."
        ],
        [
            "utils.py",
            "Utility functions for tokenization, API calls (if using GPT API), result logging, checkpoint management, and hyperparameter tuning support."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "prompt_templates.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "The system's core shares dataset formats (protein sequences with textual annotations, enriched mutation data), tokenization logic, and a common logging/stats module for tracking training progress and metrics.",
    "Anything UNCLEAR": "Clarify whether to integrate external GPT APIs or only fine-tune the models internally; also, details on the exact dataset schema, especially for the enriched textual mutation annotations, need confirmation."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001       # Based on typical transformer fine-tuning, not explicitly specified in paper
  batch_size: 16             # Adaptive depending on GPU memory; recommended as starting point
  warmup_steps: 10000        # Linear warm-up for learning rate scheduling
  max_steps: 200000          # Approximate total steps from pretraining schedule
  gradient_clip_value: 1.0   # Prevent exploding gradients
  optimizer: AdamW           # Common choice for transformers
  weight_decay: 0.01         # Typical weight decay
  dropout_rate: 0.1          # Follow transformer defaults
  
model:
  plm_name: "facebook/esm2_t6_8a_14B"   # Example; replace with actual pre-trained PLM
  text_encoder_name: "allenai/scibert_scivocab_uncased" # Example; can be BioMedGPT if available
  delta_hidden_dim: 768    # To match PLM hidden size; adjust as needed
  num_attention_heads: 12  # Default for base-sized transformer
  num_layers: 12           # Standard transformer depth
  
dataset:
  pretraining_dataset_path: "path/to/pretraining/data"    # To be prepared as per code
  finetuning_dataset_path: "path/to/finetuning/data"      # Annotated mutation data
  test_dataset_path: "path/to/test/data"                   # MutaDescribe dataset
  enrichment_data_path: "path/to/literature/abstracts"   # Literature abstracts for data enrichment
  
prompt_templates:
  pretraining_prompt_template: "standard_pretrain_prompt_template.md"
  finetuning_prompt_template: "cot_prompt_template.md"
  mutation_proposal_prompt_template: "mutation_proposal_template.md"

optimization:
  learning_rate_schedule: linear_warmup  # Schedule for training
  optimizer_params:
    eps: 1e-8
  scheduler_params:
    warmup_steps: 10000
    total_steps: 200000

evaluation:
  explanation_metrics: [ROUGE-L, BLEU-2, METEOR]
  mutation_proposal_metrics: [Recall@50, Top-k accuracy, Spearman correlation]
  evaluation_epochs: 10

hardware:
  gpus: 4                    # For large-scale training, as suggested (~200K steps)
  total_training_days: "1-2 weeks" # Approximate; depends on hardware

notes:
  # Hyperparameters are adapted from paper references, as explicit values are not provided.
  # Dataset paths should be customized as per user environment.
```

---

**Note:** Full configuration is available in `planning_config.yaml`
