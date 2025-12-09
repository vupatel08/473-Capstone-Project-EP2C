# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, detailed plan to replicate the methodology and experiments of the "Drug Discovery with Dynamic Goal-aware Fragments (GEAM)" paper, based entirely on the information provided. It covers core components, datasets, model architecture, hyperparameters, and evaluation strategies.

---

# 1. Overview and Workflow Architecture

**Key Concept:**  
GEAM is a cyclic, goal-aware molecular generation framework composed of three core modules, interacting iteratively:

- **FGIB (Goal-aware Fragment Extraction):** Extracts goal-relevant fragments using graph information bottleneck (GIB) principles.
- **SAC (Fragment Assembly):** Uses reinforcement learning to assemble fragments into molecules with desired properties.
- **GA (Fragment Modification):** Employs genetic algorithms to introduce novel fragments and diversify molecules.

**Workflow Cycle Summary:**
- Given initial molecules, extract a goal-aware fragment vocabulary.
- Use SAC to assemble molecules based on this vocabulary.
- Optionally apply GA to modify molecules, adding novel fragments.
- Extract new goal fragments from the generated molecules dynamically, updating the vocabulary.
- Repeat to rapidly explore the chemical space with goal-oriented molecules.

---

# 2. Core Components & Methodology Outline

### 2.1. Goal-Aware Fragment Extraction (FGIB)

**Objectives:**
- Predict target property \(Y\) from molecular graphs.
- Identify important subgraphs (fragments) contributing to \(Y\), via a graph information bottleneck (GIB).

**Implementation Details:**
- **Input Data:** Molecular graphs \(G = (V, E)\) with atom features, bonds.
- **Model:**
   - Use a Message Passing Neural Network (MPNN) with 3 message passing steps (parameters: \(\text{num\_msg\_passes} = 3\)).
   - **Node embeddings \(h_i\):** obtained via MPNN (Gilmer et al., 2017).
   - **Fragment embeddings \(e_j\):** average over node embeddings \(h_i\) belonging to fragment \(F_j = (V_j, E_j)\) (nodes contained in the subgraph).
   - **Fragment importance \(w_j\):** computed via an MLP \( \text{MLP}(e_j) \in [0,1] \) with sigmoid activation.
- **Noise Injection for Goal-Awareness:**
   - Inject noise into fragment embeddings based on \(w_j\):
     \[
     \tilde{e}_j = w_j e_j + (1 - w_j) \hat{\mu} + \epsilon,
     \]
     where \(\epsilon \sim \mathcal{N}(0, (1-w_j)\hat{\Sigma})\).
   - \(\hat{\mu}, \hat{\Sigma}\): empirical mean and covariance of fragment embeddings in training.
- **Loss Function:**
  - Variational IB loss \(\mathcal{L}_{IB}\):
    \[
    \min_\theta -I(Z, Y;\theta) + \beta I(Z, G;\theta),
    \]
    bounded variationally via Eq. (4): \(\mathcal{L}(\theta,\phi)\) in Eq. (5).
  - Practical approximation involves optimizing classifier \(q_\phi(Y|Z)\) and regularization via KL divergence between \(p_\theta(Z|G)\) and a prior distribution \(u(Z)\).
- **Fragment Scoring:**
  - Compute \( \text{score}(F_j) \) in Eq. (6):
    \[
    \text{score}(F_j) = \frac{1}{|S(F_j)|} \sum_{(G,Y) \in S(F_j)} \frac{w_j(G, F_j)}{\sqrt{|V_j|}} Y,
    \]
  - Select top-\(K\) fragments based on these scores as the goal-aware vocabulary.

### 2.2. Fragment Assembly Module (RL via SAC)

**Objectives:**
- Generate molecules that satisfy goal properties using fragment vocabulary \(s\).
- Formulate as a Markov Decision Process (MDP):

  - **State:** partially generated molecule \(g_t\).
  - **Actions:**
    1. Attachment site \(a_1\) on \(g_t\),
    2. Fragment \(F \in s\),
    3. Attachment site \(a_3\) on \(F\).

- **Policy Networks:**
  - \(\pi_1\): selects attachment site for \(g_t\),
  - \(\pi_2\): selects fragment \(F\),
  - \(\pi_3\): selects attachment site on \(F\).
- Use graph embeddings:
  - Encode \(g_t\) via GCN, sum pooling to get \(\mathbf{h}_{g_t}\).
  - Fragment embedding via GCN + sum.
- **Actions Distribution:**
  - Categorical distributions (via Gumbel-Softmax) for all discrete choices.
- **Reward:**
  - Evaluated via oracle (e.g., docking scores, property calculators).
- **Training:**
  - Use SAC (Haarnoja et al., 2018), enable exploration.
  - Allow 4000 initial random steps with molecule collection.
  - Termination based on molecule size \(\leq 40\) atoms.
  - **Hyperparameters:**
    - \(\text{num\_msg\_passes} = 3\),
    - SAC parameters: learning rate, temperature, entropy coefficients as per Yang et al. (2021).

### 2.3. Fragment Modification & Dynamic Vocabulary Update (GA + FGIB)

**Objectives:**
- Generate novel molecules by recombination/crossover/mutation (GA),
- Extract new goal-relevant fragments from these molecules via FGIB,
- Update the goal fragment vocabulary dynamically.

**Implementation:**
- **GA setup:**
  - Population size \(P = 100\),
  - Reproduction: 3 offspring per cycle,
  - Crossover/mutation rules: as per Jensen (2019),
- **Post-GA Extract:**
  - Use FGIB to compute scores \(\text{score}(F_j)\) for current molecules.
  - Select top-\(L\) fragments (max vocab size), based on scores.
- **Vocabulary update:**
  - Merge new fragments into current goal vocabulary.
  - If size exceeds max \(L=1000\), prune lowest-scoring fragments.
- **Cycle:**
  - Alternate between assembly (SAC) and modification (GA + FGIB).  
  - Continue until convergence or max cycles.

---

# 3. Dataset & Data Preparation

- **Training Data for FGIB:**
  - Use **ZINC250k** dataset (Irwin et al., 2012).
  - Use the same train/test split as Kusner et al. (2017).
  - For goal-aware extraction, require:
    - Target property \(Y\):
      - For ligand generation: docking scores, drug-likeness, etc.,
      - For multi-property tasks: specific property functions.
  - Extract subgraphs (fragments) from molecules with heuristic rules (filter out invalid molecules, sanitization in RDKit).

### Fragment Extraction:
- Generate initial fragment vocabulary via FGIB scores on the training set.
- Filter out chemically invalid fragments (e.g., invalid valence or sanitization exceptions).

---

# 4. Hyperparameters & Settings

| Parameter | Suggested Values / Usage |
|--------------|---------------------------|
| Message passing steps (MPNN) | 3 (Gilmer et al., 2017) |
| MLP layers | 2 (in fragment importance and predictor) |
| Batch size | 32–128 (based on GPU memory) |
| \(\beta\) in IB loss | \(1 \times 10^{-5}\) to \(1 \times 10^{-4}\) |
| Initial vocab size \(K\) | 300 fragments (fixed) |
| Max vocab size \(L\) | 1000 fragments |
| Vocabulary update batch | Up to 50 fragments per cycle |
| SAC learning rate | \(1 \times 10^{-4}\) (typical) |
| SAC episodes | Continue until molecules of size ≥ 40 atoms or convergence |
| Reproduction in GA | 3 per cycle |
| Mutation rate | 0.1 (10%) |
| Number of molecules sampled per cycle | 3000 for evaluation |

---

# 5. Evaluation Metrics & Protocols

**Property Calculation:**
- Use RDKit for QED and SA evaluation.
- Docking scores via QuickVina2 (exhaustiveness=1).
- Normalize docking and SA as per Eq. (9).

**Generated Molecules:**
- For each task, generate 3000 molecules per run.
- Compute:
  - **Hit Ratio**: molecules meeting criteria (hit threshold, novelty < 0.4 similarity).
  - **Top 5% docking scores**.
  - **Novelty**: fraction with similarity < 0.4 to training molecules.
  - **Diversity metrics**: number of unique molecules, #Circles (using publicly available scripts).
  - **Molecular validity**: chemical validity via RDKit.

**Qualitative Analysis:**
- Visualize molecules, interaction diagrams (PLIP).
- Show examples of novel molecules with properties.

---

# 6. Reproducibility and Implementation Notes

- **Code Development:**
  - Modularize into: FGIB, SAC for assembly, GA + fragment extraction, evaluation toolkit.
  - Encode molecules as graphs (node features + bonds).
  - Use PyTorch Geometric or DGL for GNNs.
  - Use RL libraries or custom SAC implementation.
  - Gumbel-softmax for discrete action sampling.
  - Maintain reproducibility with fixed seeds.

- **Special Considerations:**
  - Filter out invalid molecules at each step.
  - During dynamic vocabulary update, prune to top ones by score.
  - Carefully tune \(\beta\) for IB and learning rates for RL.
  - Use a robust oracle evaluation pipeline.

---

# 7. Summary Roadmap

- **Step 1:** Data loading and preprocessing of ZINC250k.
- **Step 2:** Implement FGIB:
  - GNN encoder,
  - Fragment extraction,
  - Noise injection,
  - IB loss training,
  - Scoring fragments.
- **Step 3:** Select top-\(K\) fragments as initial vocabulary.
- **Step 4:** Train SAC-based fragment assembly policy:
  - State encoding,
  - Action sampling,
  - Reward calculation,
  - RL training.
- **Step 5:** Run genetic algorithm:
  - Crossover/mutate molecules,
  - Filter valid molecules,
  - Evaluate and keep top molecules.
- **Step 6:** Extract new fragments via FGIB from offsprings.
- **Step 7:** Update vocabulary \(s\), prune if size exceeds max.
- **Step 8:** Repeat cycle until convergence/iterations.
- **Step 9:** Generate final molecules, evaluate on metrics.
- **Step 10:** Analyze and visualize results.

---

This plan encompasses detailed methodology, dataset, model architecture, hyperparameters, and evaluation strategies to faithfully reproduce the paper's approach, without the official code. It ensures clarity for subsequent code implementation stages.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular Python system leveraging PyTorch Geometric for GNN-based fragment encoding, RLlib (Ray) for reinforcement learning of molecule assembly, RDKit for molecule manipulation, and a custom IB loss module for FGIB. The core modules are: (1) FGIB for goal-aware fragment extraction, (2) SAC-based fragment assembly, (3) Genetic Algorithm for molecule modification, with a control loop orchestrating repeated cycles and dynamic vocabulary updates. Data flow: load datasets with RDKit, preprocess into graphs, train FGIB to score fragments, select top fragments, train SAC policy, generate molecules, apply GA, extract new fragments, update vocabulary, repeat.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "fgib.py",
        "sac_policy.py",
        "ga_optimizer.py",
        "utils.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__()\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str)\n        +load() -> Tuple[Dataset, Dataset]\n    }\n    class MolecularGraph {\n        +nodes: List[Node]\n        +edges: List[Edge]\n        +attributes: Dict\n    }\n    class Fragment {\n        +graph: MolecularGraph\n        +score: float\n    }\n    class FGIB {\n        +train_on(data: List[MolecularGraph], properties: List[float]) -> None\n        +score_fragments(fragments: List[MolecularGraph], data: List[MolecularGraph], properties: List[float]) -> List[Fragment]\n        +save()\n        +load()\n    }\n    class SACPolicy {\n        +__init__(params: dict)\n        +train(experience: List[Experience]) -> None\n        +sample(state: MolecularGraph) -> (int, Fragment, int)\n        +save()\n        +load()\n    }\n    class GAOptimizer {\n        +__init__(population_size: int, mutation_rate: float)\n        +reproduce(population: List[MolecularGraph]) -> List[MolecularGraph]\n        +mutate(molecule: MolecularGraph) -> MolecularGraph\n        +crossover(parent1: MolecularGraph, parent2: MolecularGraph) -> MolecularGraph\n        +save()\n        +load()\n    }\n    class Experience {\n        +state: MolecularGraph\n        +action: Tuple[int, Fragment, int]\n        +reward: float\n        +next_state: MolecularGraph\n    }\n    Main --> DatasetLoader\n    Main --> FGIB\n    Main --> SACPolicy\n    Main --> GAOptimizer\n    FGIB --> MolecularGraph\n    SACPolicy --> Experience\n    GAOptimizer --> MolecularGraph\n    Main --> evaluation\n    classEvaluation --> Main\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant FGIB as FGIB\n    participant RL as SACPolicy\n    participant GA as GAOptimizer\n    participant EV as evaluation\n    M->>DL: initialize and load datasets\n    DL-->>M: datasets\n    M->>FGIB: train_on(molecule graphs, properties)\n    FGIB-->>M: scoring functions\n    M->>FGIB: score_fragments(current fragments, dataset, properties)\n    FGIB-->>M: fragments with scores\n    M->>RL: initialize policy (from previous cycle or new)\n    loop for each cycle\n        RL->>RL: train on experiences\n        RL-->>M: trained policy\n        loop for each molecule to generate\n            RL->>RL: sample actions (attach site, fragment, attach site)\n            RL-->>M: generated molecule\n            EV: evaluate molecule in oracle, get reward\n            RL: store experience\n        end\n        GA->>GA: select top molecules, apply crossover/mutation\n        GA-->>M: new molecules\n        FGIB->>FGIB: score new molecules, extract fragments\n        update vocabulary with new fragments\n    end\n    M->>evaluation: evaluate final molecules on metrics\n    evaluation-->>M: results\n",
    "Anything UNCLEAR": "Clarify the availability and specifics of the dataset formats required, especially the initial molecule dataset and target properties. Details on the training details (e.g., hyperparameters for GNN, SAC, GA) should be confirmed. Also, the exact API for the oracle scoring functions and fragment extraction methods need to be specified."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.9",
        "torch_geometric",
        "rdkit-pypi",
        "ray[rllib]",
        "numpy",
        "scipy"
    ],
    "Required Other language third-party packages": [
        "QuickVina2 (either compiled as a binary) or subprocess calls to docking software",
        "Open-source molecule chemistry utility programs (RDKit)"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Handles loading ZINC250k dataset, molecule parsing into graph structures, and splitting into train/test sets. Uses RDKit to convert SMILES to graphs, filters invalid molecules, and standardizes input format for FGIB and RL modules."
        ],
        [
            "fgib.py",
            "Implements the Goal-aware Fragment Information Bottleneck (FGIB). Defines a class for training the GIB model: includes GNN encoder (message passing layers using torch_geometric), fragment extraction, scoring fragments, and IB loss calculation. Depends on dataset_loader.py for data input; outputs fragment scores and selected top-K fragments."
        ],
        [
            "model.py",
            "Contains GNN model class (e.g., MessagePassing layers) used by FGIB for encoding molecular graphs and fragment graphs."
        ],
        [
            "sac_policy.py",
            "Defines SAC-based reinforcement learning agent for molecule assembly. Uses graph neural encoders (torch_geometric) for current molecule states, policy networks for action sampling (attachment site, fragment choice), and value estimation. Contains training loop with experience replay, policy update steps, and interaction with environment defined by molecule graphs."
        ],
        [
            "ga_optimizer.py",
            "Implements genetic algorithm: selection, crossover, mutation operations on molecule graphs, with RDKit validation, aimed at generating diversified molecules and providing offsprings for subsequent updates. Depends on molecule graph manipulation APIs from model.py and dataset_loader.py."
        ],
        [
            "evaluation.py",
            "Includes property and similarity calculation functions: docking scores via external tools, QED/SA via RDKit, novelty via fingerprint similarity, diversity metrics, and PLIP interaction visualization. Used after each generation cycle to quantify progress."
        ],
        [
            "main.py",
            "Main orchestration script: initializes dataset loader, trains FGIB, performs fragment scoring, initializes RL policy, runs loop of molecule generation, GA modifications, fragment extraction, vocabulary updates, until stopping conditions. Calls files above in sequence; controls overall flow."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "model.py",
        "fgib.py",
        "sac_policy.py",
        "ga_optimizer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "OpenAPI spec not required as system is backend pipeline; interface between modules is through function calls and class methods, with JSON or Python objects passing data structures.",
    "Shared Knowledge": "Common data formats: RDKit molecule objects, torch_geometric Data objects, fragment graph representations. Hyperparameters for GNN (e.g., node features, message passing steps), IB regularization coefficient, RL training parameters, and vocabulary size constraints are shared constants.",
    "Anything UNCLEAR": "Clarification needed on the exact external docking tool (QuickVina2 or other), expected molecule data format at each stage, and detailed hyperparameters settings (learning rates, batch sizes, graph sizes, etc.)."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 10
  gamma: 0.99  # Discount factor for RL
  ib_beta: 1e-5  # IB regularization coefficient
  message_passes: 3  # Number of message passing steps in GNN
  fc_layers: 2  # Number of layers in MLPs
  max_vocabulary_size: 1000
  initial_vocabulary_size: 300
  vocab_update_size_per_cycle: 50
  molecule_max_size: 40  # Maximum size of molecules in the RL environment
  num_reproduction: 3  # Offspring per GA cycle
  mutation_rate: 0.1  # Mutation probability in GA
  num_molecules_sample: 3000  # Number of molecules generated for evaluation
  rl_learning_rate: 1e-4
  rl_buffer_size: 10000
  rl_batch_size: 64
  rl_training_epochs: 10

dataset:
  dataset_path: "data/ZINC250k.smi"
  train_split_ratio: 0.8
  val_split_ratio: 0.1
  test_split_ratio: 0.1

properties:
  docking_score_range: [-20, 0]
  normalization:
    docking_score: [-20, 0]
    sa: [0, 9]
    qed: [0, 1]
    combined: [0, 1]  # For multiplying and evaluation

evaluation:
  top_k_fraction: 0.05
  num_generated_molecules: 3000
  similarity_threshold: 0.4
  max_similarity: 0.4
  # Docking software: QuickVina2 (exhaustiveness=1)
docking:
  tool: "QuickVina2"
  exhaustiveness: 1
  docking_range: [-20, 0]

hyperparameters:
  ib_noise_mu: 0
  ib_noise_sigma: 1
  ib_loss_coeff: 1e-5
  molecule_size_limit: 40
  vocab_size_fixed: 300
  vocab_size_max: 1000
  vocab_update_per_cycle: 50

special_notes:
  # Filter out molecules invalid after RDKit sanitization.
  # Use RDKit for QED, SA, diversity, and similarity scores.
  # Use GNN and GCN: torch_geometric with 3 message passes.
```

---

**Note:** Full configuration is available in `planning_config.yaml`
