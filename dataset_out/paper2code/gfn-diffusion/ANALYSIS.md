# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## buffer.py

{
  "file": "buffer.py",
  "content": "Buffer management for off-policy exploration and low-energy sample storage is a critical component of the experimental pipeline described in the paper. The key functions include storing samples with their unnormalized energies, sampling stored samples for off-policy updates, maintaining buffer size, and prioritizing samples based on their energies.\n\n### Core Requirements and Constraints:\n\n1. **Buffer Capacity**:\n   - Must support large capacities (600,000 to 900,000 samples) as specified in the config.\n   - Use FIFO (first-in-first-out) replacement: when buffer is full, remove oldest samples.\n   - For the VAE conditional tasks, buffer size can be 90000 as per the provided hyperparameters.\n\n2. **Data Storage**:\n   - Store samples as tensors (`Tensor`) representing states `x`.\n   - Store their associated unnormalized target density or energy `R(x)` (raw energy value or unnormalized density value), as a float.\n   - For conditional models, store the associated condition `x_cond` (e.g., image inputs) alongside the samples.\n\n3. **Sampling from Buffer**:\n   - Provide a method `sample(batch_size)` that returns a list of samples (and optionally their energies), drawn with uniform or prioritized sampling.\n   - For prioritized sampling, use rank-based priority with parameter `k=0.01` from config.\n   - Should support sampling with or without replacement for efficient use in training.\n\n4. **Adding Samples**:\n   - Implement an `add(sample, energy)` method to insert new samples into the buffer.\n   - When full, evict oldest samples.\n   - For conditional samples, also store associated conditions.\n\n5. **Maintain Buffer**:\n   - A method `maintain()` or periodically invoked logic to prune, update, or rebalance (if needed).\n   - In FIFO buffers, maintenance mainly handles eviction; in prioritized buffers, updates importance measures.\n\n6. **Prioritization Scheme**:\n   - Use rank-based priorities: samples with higher energies (or lower energies, depending on target) get higher priority.\n   - The priority function: \( p(x) \propto (\mathrm{k} \times |\mathcal{D}| + \operatorname{rank}_{\mathcal{D}}(x))^{-1} \), with \(k=0.01\).\n   - Implement internal ranking (e.g., sorted `R(x)` values) for priority calculation.\n\n7. **Interface and Internal Data Structures**:\n   - For efficiency, store samples and energies as `Tensor`s within a dedicated data structure.\n   - Maintain auxiliary data for priority, e.g., a sorted array or heap.\n   - When sampling, select based on priorities (probabilistic, via `pdf`) or uniformly.\n\n8. **Implementation Details**:\n   - Use numpy or torch data structures; prefer `Tensor` for compatibility.\n   - For large buffers, consider chunked storage or memory-mapped arrays if necessary.\n   - Use thread-safe or lock-free mechanisms if asynchronous; otherwise, assume sequential access.\n\n9. **Summary of Methods and Attributes**:\n   - `__init__(capacity)` — initialize buffer with maximum capacity.\n   - `add(sample: Tensor, energy: float or Tensor)` — insert new sample.\n   - `sample(batch_size: int) -> List[Tuple[Tensor, float]]` — draw samples, possibly based on priority.\n   - Internal attributes:\n     - `self.samples: Tensor` — shape: (capacity, input_dim)\n     - `self.energies: Tensor` — shape: (capacity,)\n     - `self.indices: list` or `Tensor` — for FIFO eviction.\n     - `self.priority_scores: Tensor` — for prioritized sampling.\n\n10. **Additional Notes**:\n    - Ensure that when buffer is full, the oldest samples are evicted.\n    - Update priority scores efficiently after adding new samples.\n    - When sampling via priority, use a discrete probability distribution over the stored samples.\n    - Support optional sampling strategies (uniform, prioritized, epsilon-greedy, etc.) if needed for future extensions.\n\nBy faithfully implementing this logic, the `buffer.py` module will enable efficient off-policy exploration, enable replay of promising low-energy samples, and support the experimental findings of the paper regarding buffer size, prioritization, and sampling strategies."
}

## dataset_loader.py

{
  "dataset_loader.py - Logic Analysis": [
    {
      "Purpose": "Implement a flexible data loading module to provide training, validation, and test data for synthetic energy-based tasks and optionally for real datasets like MNIST, supporting both unconditional and conditional sampling scenarios.",
      "Key Responsibilities": [
        "Load and generate datasets appropriate for the specified energy or data type.",
        "For synthetic energy tasks (e.g., Manywell, Funnel, GMM), generate samples directly via the energy functions defined in the experiment setup.",
        "For high-dimensional synthetic distributions, implement sampling routines that can produce representative data (e.g., samples from the mixture models, or from the specified potential function).",
        "For conditional models (e.g., VAE on MNIST), load real dataset images, process them into suitable formats, and prepare conditioning variables.",
        "Ensure datasets are correctly split into training and testing sets, or provide fixed test sets for evaluation consistency."
      ],
      "Inputs": [
        "Configuration parameters (from config.yaml), such as dataset type, name, dimensionality, energy function, conditional flags.",
        "Predefined or synthetic energy functions (as Python functions or callable objects).",
        "Paths to datasets (e.g., MNIST), or routines to generate synthetic data."
      ],
      "Outputs": [
        "A dataset object or generator (e.g., PyTorch Dataset, DataLoader) providing batches of data samples.",
        "For synthetic energy functions, a sampling function that generates samples on demand.",
        "For conditional datasets, pairs of (condition, data) samples in appropriate tensor formats."
      ],
      "Implementation Details": [
        "Create a class or functions that can instantiate either synthetic data or load real datasets based on configuration flags.",
        "For synthetic tasks: define the energy function (either inline or imported), and implement a 'sample_data' routine that samples from the unnormalized density R(x) = exp(-E(x)).",
        "Use rejection sampling or MCMC in low-dimensional cases if needed to obtain samples from complex distributions, but prefer direct generation methods for high-dimensional models.",
        "Normalize data to appropriate ranges (e.g., [0, 1] for images).",
        "Provide functions for precomputing or caching samples for reproducibility if needed.",
        "Support datasets being generated 'on-the-fly' (for synthetic) or loaded from disk (for real data)."
      ],
      "Considerations for Reproducibility": [
        "Set random seeds (e.g., torch.manual_seed, numpy.random.seed).",
        "Document data generation procedures (parameters, energy functions).",
        "Ensure deterministic sampling if possible, especially for synthetic data."
      ],
      "Potential Challenges": [
        "Correctly implementing and verifying synthetic energy functions predefined in the paper.",
        "Ensuring the generated samples match the distributions used in experiments (matching parameters, distributions).",
        "Handling conditional datasets: aligning conditioning variables with model input expectations.",
        "Balancing computational efficiency with sample correctness (e.g., avoiding expensive MCMC when unnecessary)."
      ],
      "Assumptions": [
        "Energy functions for synthetic tasks are defined as Python functions consistent with those used during training (e.g., 2D GMM, funnel).",
        "For MNIST, datasets are preprocessed appropriately, normalized, and split into training/test sets.",
        "Reproducibility is maintained via fixed seeds and consistent function implementations."
      ],
      "In summary": "The dataset_loader.py module should be capable of (a) generating or loading high-dimensional synthetic datasets defined by energy functions (e.g., Manywell, Funnel), (b) providing batches of samples for training using the specific sampling methods described, (c) supporting conditional datasets such as MNIST with VAEs, and (d) ensuring data reproducibility and compatibility with training and evaluation procedures. The implementation must respect constraints related to energy function definitions, data formats, and reproducibility as per the experimental setup.",
      "Next steps": "Develop classes/functions for dataset generation/loading, ensure energy functions are correctly implemented, and integrate seed management for reproducibility."
    }
  ]
}

## evaluation.py

{
  "evaluation.py": {
    "Purpose": "Implement functions to evaluate trained diffusion-structured off-policy samplers, including estimation of the log-partition function \(\log Z\), computation of sample quality metrics such as Wasserstein distance, and visualization of sample distributions over energy landscapes.",
    "Dependencies": [
      "torch",
      "numpy",
      "scipy.stats",
      "matplotlib.pyplot",
      "evaluation_utils.py" (containing functions for metrics calculations: importance sampling estimators, Wasserstein distance)
    ],
    "Inputs": [
      "trained_model": the neural SDE or diffusion model object implementing sampling (from 'sampler.py')",
      "energy_fn": callable that computes the unnormalized energy \(\mathcal{E}(x)\) or log density \(\log R(x)\), as per the tasks",
      "samples": Tensor of generated samples (from the model's sampling process)",
      "target_samples": Tensor of true samples if available (for Wasserstein comparison)",
      "evaluation_config": parameters such as number of Monte Carlo samples, whether to perform importance sampling, estimate \(\log Z\), etc.
    ],
    "Outputs": [
      "Estimated \(\log Z\)",
      "Wasserstein distance value",
      "Visual plots: energy contour overlays, sample scatter plots",
      "Optional: diagnostics such as importance weights"
    ],
    "Core Components": [
      "1. Estimation of \(\log Z\):",
      "   - Use importance sampling with trajectories generated by the model. For each trajectory \(\tau\), compute importance weight \(w_\tau = R(x_1) p_B(\tau|x_1) / p_F(\tau)\).",
      "   - Monte Carlo estimate of \(\log Z\):",
      "     \(\log \hat{Z} = \log \left( \frac{1}{K} \sum_{i=1}^K w_{\tau_i} \right)\)",
      "   - Variance-based estimator (VarGrad):",
      "     Use multiple trajectories sharing the same target, compute individual importance weights, and estimate variance to refine \(\log Z\).",
      "   - For high-dimensional energies, consider the effect of the variance of importance weights to ensure stability.",
      "2. Wasserstein Distance:",
      "   - Compute 2-Wasserstein between generated samples and true target samples, using scipy or custom function.",
      "   - Handle high-dimensional case carefully, possibly via sampling subsets or using fast approximations.",
      "3. Visualization:",
      "   - Generate 2D contour plots over energy landscape:",
      "     * Evaluate \(\mathcal{E}(x)\) on a grid over the relevant domain.",
      "     * Plot energy contours, overlay generated samples, true samples if available.",
      "   - Scatter plots of generated samples in 2D slices or principal component space.",
      "   - Save figures with informative titles/annotations.",
      "4. Utility functions:",
      "   - Function to run importance sampling for \(\log Z\) estimation.",
      "   - Function to compute Wasserstein distance.",
      "   - Function to generate energy contours and overlay samples.",
      "   - Functions to handle batch processing, normalization, and logging."
    ],
    "Implementation details": {
      "est_logZ": {
        "parameters": {
          "method": "importance_sampling or VarGrad",
          "num_samples": 2000,
          "trajectory_samples": "From trained model p_F",
          "importance_weights": "Compute for each trajectory based on energy and reverse process probabilities"
        },
        "steps": [
          "Sample a set of trajectories \(\{\tau_i\}\) from the model p_F using 'sampler.py'.",
          "For each trajectory, evaluate the importance weight:",
          "   \(w_{\tau_i} = R(x_1^{(i)}) p_B(\tau_i | x_1^{(i)}) / p_F(\tau_i)\).",
          "Estimate \(\hat{Z}\) as the average importance weight: \(\hat{Z} = \frac{1}{K} \sum w_{\tau_i}\).",
          "Compute \(\log \hat{Z}\) and its confidence intervals if needed.",
          "If using VarGrad, group trajectories sharing the same \(x_1\), compute variances, and estimate \(\log Z\) accordingly.",
          "Return the estimate."
        ]
      },
      "compute_wasserstein": {
        "parameters": {
          "samples": generated samples Tensor,
          "target_samples": reference samples Tensor or None,
          "dim": dimensionality,
          "method": "scipy.stats.wasserstein_distance" or custom implementation
        },
        "steps": [
          "If target_samples are available, compute pairwise cost matrix (Euclidean distances).",
          "Use scipy's wasserstein_distance for 1D slices, or implement multivariate approximations (e.g., sliced Wasserstein, Sinkhorn).",
          "Return the Wasserstein distance value."
        ]
      },
      "generate_energy_contours": {
        "parameters": {
          "energy_fn": callable,
          "xlim": tuple for axis limits,
          "ylim": tuple,
          "resolution": grid size,
          "samples": generated samples,
          "target_samples": optional for comparison,
          "save_path": path to save plots
        },
        "steps": [
          "Create a meshgrid over the domain.",
          "Compute energy \(\mathcal{E}(x)\) over grid points.",
          "Plot energy contours using matplotlib.",
          "Overlay generated samples (scatter) in the same plot.",
          "Optionally overlay true samples if available.",
          "Save figure with descriptive filename."
        ]
      },
      "visualize_samples": {
        "parameters": {
          "samples": generated samples,
          "projection": e.g., PCA or select 2D slice,
          "energy_fn": for coloring,
          "save_path": path
        },
        "steps": [
          "Apply PCA or select dimensions for 2D visualization.",
          "Scatter plot with color mapped to \(\mathcal{E}(x)\) or density.",
          "Add axis labels, colorbars, annotations.",
          "Save visualization."
        ]
      }
    },
    "Notes": {
      "1": "Ensure that samples generated from the model are consistent with the training method: e.g., from trajectories produced by the Euler sampler, possibly using the same step sizes and energy functions.",
      "2": "For high-dimensional tasks like LGCP, only sample summary metrics or lower-dimensional projections are feasible for visualization.",
      "3": "The importance sampling estimator's variance and stability are crucial; use logs of weights if needed to avoid numerical underflow.",
      "4": "In case of unavailable target samples, rely on the Monte Carlo importance sampling estimates to assess \(\log Z\).",
      "5": "Plot energy landscapes over the same domain grid used in the experiments to ensure comparability."
    }
  }
}

## main.py

# Logic Analysis for main.py

## Purpose:
- The main.py script serves as the primary orchestrator to initialize settings, manage the training loop, and coordinate evaluation and visualization operations. It should be structured to:
  - Load and parse configuration parameters.
  - Instantiate model components (neural SDE networks, reverse models).
  - Initialize buffers and exploration mechanisms.
  - Set up training procedures with appropriate objectives and optimizer.
  - Run the main training loop with periodic evaluations.
  - Save checkpoints and results at specified intervals.

---

## Core Responsibilities and Sequence:

### 1. Load Configuration
- Read 'config.yaml' to extract all hyperparameters, architectural choices, and experimental settings.
- Establish deterministic seed for reproducibility (`torch.manual_seed()` and `np.random.seed()`).
- Set device ('cuda:0' or CPU).

### 2. Instantiate Data and Energy Functions
- For synthetic energy tasks (Manywell, Funnel, 25GMM, LGCP):
  - Define the energy function \(\mathcal{E}(x)\) as per the task.
- For conditional tasks (VAE):
  - Load pretrained VAE decoder.
  - (Optional) Load or define the conditional energy function that incorporates the condition \(x\).
- If dataset is real (e.g., MNIST), prepare data loader accordingly.

### 3. Model Initialization
- Instantiate neural networks for drift \(u(x, t)\) and diffusion \(g(x,t)\):
  - Use provided architecture parameters (input_dim, hidden_dim, network_type).
  - For fixed or learned diffusion coefficient (check 'diffusion_value' from config).
  - Initialize parameters with appropriate schemes (e.g., last-layer zeros).
- Instantiate the reverse process network structure similarly.
- Place models on device.

### 4. Buffer Initialization
- Create replay buffer with capacity as specified ('buffer_capacity').
- For off-policy exploration:
  - Instantiate local search buffer (\(\mathcal{D}_{LS}\)) with same or different capacity.
- Initialize data structures (list or custom class) for buffer management.
- Initialize prioritized sampling scheme if used.

### 5. Optimizer and Scheduler Setup
- Initialize Adam optimizers for:
  - Neural SDE parameters (\(u, g\))
  - Log-partition estimate \(Z_\theta\)
  - Reverse process parameters, if applicable
- Set hyperparameters per config (learning rate, clip norms).

### 6. Training Loop
**For** iteration = 1 to total_iterations:
- **Sampling Step:**
  - Decide whether to do on-policy (trajectory) sampling or off-policy (buffer, local search).
  - With probability (e.g., 50%), sample trajectories:
    - From initial distribution (\(\mu_0\)) using Euler-Maruyama, via 'sampler.py'.
    - Compute associated energies and log likelihoods.
  - Else:
    - Sample states from replay buffer or local search buffer (\(\mathcal{D}_{LS}\)).
    - Optionally perform local MH (parallel MALA) steps:
      - Update acceptance rates.
      - Propose new samples using gradient of energy.
      - Accept or reject via MH criterion with adaptive step size.
    - Generate trajectories from these states using reverse process or policy model.
- **Compute Losses:**
  - From trajectories, calculate trajectory balance (TB) loss.
  - Optionally compute variance estimator loss (VarGrad).
  - Incorporate exploration loss if applicable.
  - Weight losses appropriately.
- **Backpropagation & Optimization:**
  - Zero gradients.
  - Perform backward pass.
  - Clip gradients if needed.
  - Step optimizers.
- **Buffer Update:**
  - Add promising samples (low energy, high probability) to replay buffer.
  - Maintain buffer size and prioritization.
- **Adaptive Step Size:**
  - If using local MH, update step size \(\eta\) based on acceptance rate feedback.
- **Logging:**
  - Record current loss values, step size, acceptance rate.
  - Save current model parameters periodically.
  - Store trajectories or sample sets for visualization and metrics.

### 7. Periodic Evaluation:
- At specified evaluation frequency:
  - Use 'evaluation.py' routines:
    - Estimate \(\log Z\) via importance sampling, importance-weighted methods.
    - Compute Wasserstein distance against known ground truth or benchmark samples.
    - Collect samples for visualization (contour plots, scatter plots).
  - Save evaluation metrics and plots.

### 8. Checkpointing:
- Save model states (parameters, optimizer states) every 'save_checkpoint_every' iterations.
- Save current buffer contents, model logs, and hyperparameters.

### 9. Finalization:
- After training convergence or max iterations:
  - Perform a final evaluation.
  - Save final models.
  - Generate comprehensive visualizations and metric summaries.
  - Summarize training statistics.

---

## Additional Details:
- **Error handling:** Check for GPU availability; fallback to CPU if needed.
- **Reproducibility:** Ensure consistent seeding and deterministic algorithms if possible.
- **Configuration overrides:** Allow command-line arguments to modify key hyperparameters (if necessary).
- **Logging:** Use logging library or print statements for progress updates.
- **Visualization:** Compile energy contour plots, sample scatter plots, and energy histograms for qualitative assessment.

---

## Uncertainties and Clarifications:
- Exact schedule of \(\eta\) adaptation (feedback-based or pre-scheduled).
- Whether diffusion coefficient \(g\) is learned or fixed during training.
- Precise buffer prioritization scheme (rank-based, uniform, or hybrid).
- Frequency of buffer updates and local search MH steps.
- Handling of conditional energies versus unconditional energies.

---

This comprehensive analysis provides a detailed, step-by-step plan to implement main.py systematically, ensuring alignment with the paper's methodology and experimental setup. Implementing according to this plan will support reproducible and faithful reproduction of original results.

## model.py

{
  "Purpose": "Implement neural network modules for the core components of diffusion-structured samplers, including neural SDE drifts and diffusion functions, as well as reverse process models needed for sampling. The modules are used across 'sampler.py' and 'trainer.py' to parameterize stochastic differential equations and their approximations, ensuring consistency with the experimental outline and design.",
  
  "Design Principles": [
    "Modularity: Design each component as a class for flexibility and clarity.",
    "Compatibility: Networks should accept inputs as tensors; conditioning on time and possibly energy-related information",
    "Parameter encapsulation: Include methods to retrieve parameters for optimizer steps.",
    "Scalability: Simple architectures (MLPs) that can handle varying input dimensions.",
    "Hyperparameter control: Enable configuration of input/output dimensions, hidden layer size, and optional activation functions.",
    "Extensibility: Allow registration of learned diffusion coefficient g or fix as constant, with easy modifications.",
    "Consistency: Follow input/output signatures and initialization schemes aligned with the overall training pipeline."
  ],
  
  "Class Breakdown": [
    {
      "Class Name": "NeuralSDE",
      "Purpose": "Parameters and forward pass for drift u and diffusion g functions modeling neural SDEs.",
      "Attributes": {
        "input_dim": "Dimensionality of x (e.g., 2 or 32)",
        "hidden_dim": "Number of units in hidden layers (e.g., 400)",
        "network_type": "Default 'MLP', possible extension for other architectures"
      },
      "Methods": {
        "forward(self, x: torch.Tensor, t: float) -> torch.Tensor": "Returns the drift u(x, t; θ) or diffusion g(x, t; θ) at input state x and time t.",
        "get_parameters(self) -> dict": "Returns all neural network parameters for optimizer assignment.",
        "initialize(self)": "Constructs the network layers with proper initialization, e.g., Xavier or Kaiming, per positional conventions."
      },
      "Notes": "Implement as a flexible class that can be instantiated for either drift or diffusion functions, or instantiate separate classes if needed. For the diffusion coefficient g, either fix to a constant or parameterize as a neural network if learning g."
    },
    {
      "Class Name": "DiffusionCoefficient",
      "Purpose": "Encapsulate the handling of the diffusion coefficient g, either fixed or learned.",
      "Attributes": {
        "fixed": "Boolean, whether g is fixed at a scalar value or modeled",
        "value": "If fixed, scalar value like 1.0 or configured from 'diffusion_value', else learnable parameter"
      },
      "Methods": {
        "forward(self, x: torch.Tensor, t: float) -> torch.Tensor": "Return scalar or tensor representing g(x, t); fixed value or learned."
      },
      "Notes": "If fixed, implement as a constant tensor; if learned, as a small neural network similar to 'NeuralSDE' but returns scalar."
    }
  ],
  
  "Input configurations": [
    "The network will accept standard inputs: x (tensor of shape [batch_size, input_dim]) and scalar or tensor t (preferably float or tensor).",
    "Time t may be broadcasted or concatenated with x as additional input features for their networks, or embedded as a scalar extension.",
    "Conditioning on energy or other auxiliary signals: Not specified as necessary; focus on core x and t inputs."
  ],
  
  "Initialization & Variants": [
    "Use Xavier or Kaiming initialization for hidden layers.",
    "Implement activation functions like ReLU.",
    "Inherit from 'nn.Module' for all network classes.",
    "Provide methods for parameter access for optimizer integration."
  ],
  
  "Configuration controls": [
    "Use 'network_type' parameter for potential architecture variations.",
    "Allow passing layer sizes, activation functions via constructor."
  ],
  
  "Validation & Testing": [
    "Ensure 'forward' outputs are compatible with the numerical schemes in 'sampler.py'.",
    "Validate that the output shape matches input batch size and target dimension.",
    "Create unit tests to verify parameter retrieval and forward pass consistency."
  ],
  
  "Summary": "This module enables defining all neural network parameterizations needed for the diffusion process and its reverse. By following flexible, simple architecture schemes and adhering to interface conventions, it supports the primary training and sampling workflows described in the overarching plan, with hyperparameters and network details controlled via configuration. It also accommodates extensions like learned diffusion coefficients or alternate network types if future modifications are desired."
}

## sampler.py

{
  "sampler.py": {
    "Overview": "This module implements the core sampling routines based on the Euler-Maruyama discretization of neural SDEs, including the forward diffusion process, the generation of samples from the learned model, and auxiliary operations like mixture proposals and adaptive step size control. It interfaces heavily with the neural network definitions from 'model.py' and supports both training and evaluation modes. It must support flexible configurations for the number of steps, the diffusion coefficients, and the potential for learned or fixed diffusion parameters.",
    "Main Components": [
      {
        "Neural SDE Class": {
          "Purpose": "Encapsulates the neural network modules modeling the drift 'u' and diffusion 'g' (if learned). Provides methods for forward evaluation and parameter access.",
          "Dependencies": "Depends on 'model.py' for network architecture and parameter initialization.",
          "Key Methods": [
            "forward(x, t)": "Compute drift (u) and diffusion (g) at state x and time t.",
            "get_parameters()": "Return current network weights for optimizer updates."
          ],
          "Configuration": "Configured with input_dim (e.g., 2 or 32), hidden_dim=400, network_type='MLP'."
        }
      },
      {
        "Euler-Maruyama Sampler Class": {
          "Purpose": "Implements the discretized simulation of the forward neural SDE to generate trajectories (samples). Supports adjustable number of steps, step size, and batch processing.",
          "Dependencies": [
            "NeuralSDE for network outputs",
            "energy_fn for potential or target energy function",
            "configuration parameters: T (number of steps), delta_t (step size)"
          ],
          "Key Methods": [
            "sample(x0, energy_fn, steps)": "Generate a trajectory starting from x0, applying the Euler-Maruyama steps, returning the list of states or the final state.",
            "discretize(x, t, delta_t, noise)": "Compute one step of Euler-Maruyama: x_{t+1} = x_t + u(x_t,t)*delta_t + g(x_t,t)*sqrt(delta_t)*noise."
          ],
          "Implementation details": "For each step, sample Gaussian noise, compute drift and diffusion from neural networks, and update state accordingly."
        }
      },
      {
        "Sampling Procedure": {
          "Workflow": [
            "Initialize at x0 (could be a delta at zero or sampled from prior).",
            "Iteratively apply 'discretize' for each of T steps, updating state at each iteration.",
            "In training, often sample multiple trajectories to compute loss; in evaluation, generate a batch of samples."
          ],
          "Input parameters": [
            "x0": initial state, either a point (e.g., zeros) or from a distribution.",
            "energy_fn": unnormalized energy function R(x) = exp(-E(x)), used if energy-dependent modifications are performed.",
            "steps": number of integration steps (T=100 by default, delta_t=0.01)."
          ],
          "Outputs": [
            "Trajectory of states: list of T+1 states for possible use in trajectory-based loss calculations.",
            "Final sample(s): used for evaluation metrics."
          ]
        }
      },
      {
        "Adaptive Step Size & MH Local Search": {
          "Purpose": "Implement optional adaptive \(\eta\) for MH proposals, critical when using local search (parallel MALA).",
          "Method": "Adjust \(\eta\) based on the acceptance rate, striving towards target (~0.574).",
          "Implementation": "After each batch of MH steps, compute acceptance ratio; adapt \(\eta\) accordingly with increase or decrease factors (1.1 or 0.9)."
        }
      },
      {
        "Handling Diffusion Coefficients": {
          "Fixed": "Use constant diffusion g = diffusion_value (1.0 or specified), directly applied at each step.",
          "Learned": "Neural networks output the diffusion term g(x,t), to be trained jointly with drift u."
        }
      },
      {
        "Energy Function Integration": {
          "Usage": "For methods incorporating energy-based loss or evaluation, 'energy_fn' passed to 'sample' supports energy and gradient evaluation.",
          "Gradient Computation": "If needed, compute \(\nabla \mathcal{E}(x)\) via autograd for gradient-based proposals or inductive biases."
        }
      },
      {
        "Outputs & Data Storage": {
          "Trajectory Storage": "List of states for loss computation: trajectory-based KL, Variance estimators, visualization.",
          "Sample Storage": "Batch of final states for evaluation, metrics, or buffer insertion."
        }
      },
      {
        "Numerical & Implementation Details": {
          "Discretization": "Euler scheme with noise sampled from standard normal, scaled appropriately.",
          "Batch Processing": "Support for batch size > 1; vectorize 'discretize' over batch states.",
          "Device": "Ensure tensors are on GPU ('cuda') if specified, for efficiency."
        }
      },
      {
        "Error Handling and Validation": {
          "Sanity Checks": "Check for NaNs or infinities during simulation; clip gradients or states if necessary.",
          "Accept/Reject MH Proposals": "Implement MH acceptance ratio with logs, ensuring numerical stability."
        }
      }
    ],
    "Additional Considerations": [
      "Implementation must be consistent with the 'config.yaml' settings, especially for T, delta_t, diffusion coefficient (fixed vs learned), and batch size.",
      "Provide utility functions for energy gradient computation, initial sample generation, and optional visualization of trajectories.",
      "Modular design: 'NeuralSDE' class for the network, 'EulerSampler' for simulation, with interfaces supporting flexible experimentation.",
      "Ensure reproducibility via seed settings and deterministic mode if needed.",
      "Offer hooks for logging acceptance rates, step sizes, and sample statistics for monitoring."
    ],
    "Summary": "This module forms the core simulation engine connecting neural network models to stochastic trajectories, supporting both training (trajectory-based loss computations) and inference/sampling. It must adhere tightly to the specified configuration and allow flexible enhancements like adaptive MH step size, learned diffusion, and multi-step trajectories. Proper implementation will enable accurate reproduction of the paper’s diffusion sampler experiments."
  }
}

## train.py

{
  "train.py - Logic Analysis": "Purpose and Objectives: \n  - Implement the main training loop to optimize the neural SDE parameters (drift u(x,t;θ), diffusion g(x,t;θ)), the backward process parameters (if learned), and the partition function estimate (log Z_θ).\n  - Incorporate multiple loss functions: trajectory balance (TB), variance-based estimator (VarGrad), as well as auxiliary exploration terms.\n  - Manage off-policy exploration using replay buffers, local search via MALA proposals, and adaptive step size tuning.\n\nKey Components and Workflow:\n  1. Initialization:\n     - Load configuration parameters: learning rates, batch sizes, total iterations, network architectures, buffer sizes, MH settings.\n     - Instantiate models: neural SDE (drift u, diffusion g), backward process (if applicable), and log Z estimator.\n     - Initialize optimizer(s) (Adam) with specified hyperparameters.\n     - Set up buffers: main buffer for samples (e.g., promising low-energy states), optional off-policy buffer for additional exploration.\n     - Set up exploration parameters: number of MH steps per buffer update, burn-in steps, initial step size, target acceptance rate, step size adaptation factors.\n  2. Main Training Loop (for each iteration i = 1 to total_iterations):\n     a. Decide sampling strategy:\n        - With probability 0.5 (or as configured), perform on-policy (trajectory-based) sampling:\n          - Sample initial states from the prior or initial distribution.\n          - Forward simulate using Euler-Maruyama for T steps, with current models.\n          - Store trajectories for loss computation.\n        - Else, perform off-policy sampling:\n          - Sample states from the replay buffer.\n          - Use local search / MH proposals (parallel MALA) to improve samples:\n            - For each sampled state, run K MH steps with adaptive step size \(\eta\).\n            - Accept/reject based on MH ratio involving energy difference and gradient.\n            - Update \(\eta\) periodically to target acceptance rate;\n            - Store accepted low-energy samples into the buffer.\n     b. Generate trajectories for gradient estimation:\n        - From on-policy or off-policy samples, generate complete trajectories (for TB loss) or partial trajectories (for VarGrad).\n        - Compute the model's transition densities: Gaussian for forward, backward model if learnable.\n        - Calculate the TB loss:\n          \(\left(\log \frac{Z_\theta P_F(\tau)}{R(x_1) P_B(\tau|x_1)}\right)^2\), ensuring the diagonal or normalization constants are properly handled.\n        - For VarGrad, compute the variance-based loss: \(\text{Var} \left( \log \frac{R(x_1) P_B(\tau|x_1)}{P_F(\tau)} \right)\) over a minibatch.\n     c. Compute auxiliary exploration loss:\n        - For models with exploration, include the exploration factor (e.g., entropy regularization, buffer-based reward maximization).\n        - Incorporate local search guidance or other exploration terms (e.g., trajectory reweighting, reward shaping). \n     d. Aggregate loss:\n        - Combine TB (or variants), VarGrad, exploration, and any additional regularization into a total loss.\n        - Apply gradient clipping if specified.\n        - Use automatic mixed precision if desired.\n     e. Optimization step:\n        - Zero gradients.\n        - Backpropagate total loss.\n        - Update model parameters via Adam optimizer.\n        - Update log Z parameter(s) if modeled explicitly.\n     f. Buffer management:\n        - Add accepted low-energy states or trajectories to the replay buffer.\n        - Maintain buffer size (FIFO or prioritized sampling).\n        - Update or sample from buffer in subsequent iterations.\n     g. Step size adaptation for MH:\n        - After every burn-in interval, compute acceptance rate \(\alpha_k\) over MH proposals.\n        - If acceptance > target, increase \(\eta\) by factor \(\mathcal{I}\).\n        - Else, decrease \(\eta\) by factor \(f_d\).\n        - Clamp \(\eta\) within reasonable bounds to prevent instability.\n  3. Monitoring and Logging:\n     - Record metrics: total loss, TB loss, variance estimate, acceptance ratios.\n     - Log the current step size \(\eta\), buffer occupancy, and energy statistics.\n     - Save model checkpoints at specified intervals.\n  4. Post-Training Evaluation:\n     - Use evaluation.py routines: estimate \(\log Z\), compute Wasserstein distance, visualize samples.\n     - Generate energy contour plots and sample distributions for diagnostics.\n  5. Additional considerations:\n     - Ensure reproducibility by setting seed and device specifications.\n     - Modularize code: separate functions for sampling, loss computation, buffer updates, MH proposals, step size updates, and logging.\n     - Support for conditional energies (VAE) should include condition input handling, conditional network forward passes.\n\nSummary:\n- The core of train.py is a flexible training loop that flexibly samples, computes losses, updates models, and explores via MH with adaptive step size.\n- It manages off-policy exploration via buffers and local search.\n- It incorporates multiple loss functions and evaluation routines, configurable via YAML.\n- Emphasis on reproducibility, modularity, and efficiency guides the code structure."
}

## visualization.py

{
  "Visualization.py Logic Analysis": "The purpose of 'visualization.py' is to generate insightful visual representations of the results obtained from the sampling experiments, including energy contours and scatter plots of the sampled points. These visualizations help to qualitatively assess mode coverage, sample diversity, and convergence behavior in various dimensions and energy landscapes.\n\nSince 'visualization.py' depends on 'evaluation.py', it will primarily consume the outputs and metrics computed during evaluations—namely, the samples generated by the trained models and possibly the corresponding energy values. The flow outline and key implementation details are as follows:\n\n1. Input Data and Dependencies:\n   - Load or receive as arguments the generated samples from the sampling process (from 'evaluation.py' or directly from 'sampler.py').\n   - Access the energy function (or energy values) associated with these samples, either stored during evaluation or passed explicitly.\n   - Domain-specific settings such as true energy contours (if available), range bounds, and visualization parameters.\n   - Hyperparameters such as the dimension of the energy landscape (often 2D for illustrative plots but can adapt to higher dims with projections), grid resolution, and plotting styles.\n\n2. Energy Contour Plot ('Energy Landscape Visualization'):\n   - Generate a grid covering the relevant input domain or latent space.\n   - Compute energy values over this grid by applying the energy function \(\mathcal{E}(x)\).\n   - Use 'matplotlib' to create contour plots or heatmaps, depicting regions of high and low energy.\n   - If high-dimensional (e.g., 32 or higher), consider projecting onto 2D slices, principal components, or other meaningful subspaces for visualization.\n   - Overlay sampled points on the energy contours to assess mode coverage and clustering behavior.\n\n3. Sample Scatter Plot(s):\n   - For 2D or projected 2D visualizations, scatter plot the samples generated by the models.\n   - Color points based on their energy value or membership in particular modes.\n   - Include true mode centers or known regions, if relevant, for comparison.\n   - Annotate plots with key metrics, such as Wasserstein distances or estimated \(\log Z\), if these are part of evaluation results.\n\n4. Additional Visualizations:\n   - Density or marginal plots if applicable.\n   - High-dimensional visualizations with techniques like t-SNE or PCA, for comparable analysis.\n   - Collection of plots from different models or training checkpoints, for comparison.\n\n5. Implementation Details:\n   - Use 'matplotlib.pyplot' for plotting.\n   - Maintain consistent axes limits, labels, and color schemes for comparability across different experiments.\n   - Save plots to files with systematic naming conventions, e.g., 'energy_contour_{task}_{step}.png', 'samples_scatter_{task}_{run}.png'.\n   - Ensure plots are clear, with readable labels and colorbars.\n\n6. Integration with 'evaluation.py':\n   - 'evaluation.py' should output the generated samples and energy evaluations.\n   - 'visualization.py' loads these outputs or receives them as function arguments.\n   - It should be called after evaluation at specific iterations or checkpoints.\n\n7. Flexibility & Extensibility:\n   - Support for different energy functions or datasets.\n   - Modular design to allow plotting various subspaces or projections.\n   - Configurable via command-line arguments or a configuration object, e.g., plot resolutions, color schemes.\n\n8. Ensuring Fidelity and Clarity:\n   - Cross-verify that the energy values match the target (synthesized or real energies).\n   - Accurately overlay sample points and true mode centers.\n   - Handle edge cases, such as high-dimensional data or sparse sample coverage, by reducing dimensionality or sampling dense grids.\n\nOverall, 'visualization.py' acts as a post-processing utility to produce qualitative diagnostics of the trained sampling models, leveraging energy evaluation, sample outputs, and dimension reduction techniques, with the goal of illustrating mode coverage, sample diversity, and landscape features clearly and effectively."
}

