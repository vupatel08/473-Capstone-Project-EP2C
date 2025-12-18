# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

To reproduce the method described in the paper, here is a comprehensive, step-by-step plan that systematically covers all components of methodology and experiments, grounded explicitly in the paper's details.

---

# Overview and Core Goals
- **Objective:** Calibrate the noise scale (e.g., Gaussian noise variance) of a differential privacy mechanism directly to the operational attack risk (e.g., attack advantage, FPR/FNR, or benefit) rather than via the standard $(\varepsilon, \delta)$ interpretation.
- **Approach:** Use the paper's *attack-aware calibration* methods based on *privacy loss random variables (PLRVs)* and *trade-off curves* via Theorem 3.3, Algorithm 1, and binary search over noise parameter $\omega$.

---

# Step 1: Understand the Theoretical Foundations
- **Key concepts to implement:**
  - **Privacy loss random variables (PLRVs):** Discrete distributions $(X,Y)$ characterizing the mechanism's privacy profile.
  - **Trade-off curve $T(P,Q)$:** Bounds the FNR for a given FPR, obtained via Neyman–Pearson optimal tests and the corresponding $\beta^*(\tau, \gamma)$.
  - **Trade-off curve $f_\omega(\alpha)$:** Parameterized by mechanism noise $\omega$, computed via PLRVs, as per Theorem 3.3 and Algorithm 1.
  - **Advantage $\eta_\omega$ and benefit measures:** Calculated directly from PLRVs.
- **Core tools:**
  - Exact formulae for mechanisms with finite support (e.g., Gaussian with known $\sigma$).
  - Construction of discrete dominating pairs $(P,Q)$ from privacy curves, using Algorithm 7 (from Doroshenko et al., 2022), adapted here.

---

# Step 2: Data and Mechanism Specification
- **Mechanism details:**
  - **Gaussian Mechanism:**
    - Sensitivity $\Delta_2$ (should be specified or estimated from the task/data).
    - Noise variance $\sigma^2$, which is the calibration target.
    - Analytical formulas for privacy profile: from Proposition G.1.
  - **Other mechanisms:** If available, their privacy profile functions $\varepsilon_\omega(\delta)$ or moments.
- **PLRVs construction:**
  - **For Gaussian:**
    - PLRVs can be computed exactly using the Gaussian density and the formula in Proposition G.1.
  - **For general mechanisms:**
    - Use Algorithm 7 from Doroshenko et al. (2022) to build dominating pairs $(P,Q)$ on a discretized grid of the privacy profile (e.g., over $\varepsilon$, $\delta$).
    - Compute PLRVs from the distributions $(P,Q)$ (see Def. 3.2).

---

# Step 3: Implementing Mechanism and Privacy Profile Computation
- **(a) Exact formulas for Gaussian:**
  - Implement functions to compute:
    - Privacy profile $f(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - \mu)$ with $\mu = \Delta_2 / \sigma$.
    - $(\varepsilon, \delta)$-DP conditions from Proposition G.1.
- **(b) Approximate privacy profile via $\varepsilon_\omega(\delta)$:**
  - For mechanisms with privacy accountant (e.g., DP-SGD), call the accountant function at discretized $\delta$ values.
  - For $\varepsilon(\delta)$:
    - Use a discretization over $\delta \in [0,1]$, say $\delta_i$, with granularity $\Delta_\delta$.
    - Approximate $f_\omega(\alpha)$ via the supremum over this grid per Eq. (20).
- **(c) Constructing dominating pairs $(P,Q)$:**
  - Use Algorithm 7 (from Doroshenko et al.), input: discretized privacy profile curves and parameters $(\varepsilon_1, \Delta, k)$.
  - Once $(P,Q)$ are built, derive PLRVs:
    - For each support point, compute $X = \log Q(o)/P(o)$ and $Y = \log Q(o')/P(o')$ for $o \sim Q$, $o' \sim P$.

---

# Step 4: Computing Trade-off Curves $T(P,Q)$ and $f_\omega(\alpha)$
- **Implement Algorithm 1:**
  - Inputs: $(X,Y)$ (PLRV) supports and probabilities.
  - For a discretized set of $\alpha$ (e.g., via quantiles of $X$), find thresholds $\tau$ and $\gamma$ using Eqs. (42) and (44).
  - Compute $T(P,Q)(\alpha)=\beta^*(\tau,\gamma)$ (see Def. 3.2).
  - Store these pairs; interpolate for arbitrary $\alpha$.
- **Compute $f_\omega(\alpha)$:**
  - For each mechanism noise parameter $\omega$, build the PLRV $(X_\omega,Y_\omega)$.
  - Derive $T_\omega(\alpha)$ via Algorithm 1.
  - Return $f_\omega(\alpha) = T_\omega(\alpha)$.

---

# Step 5: Calculating Operational Privacy Risks
- **Advantage $\eta_\omega$:**
  - Compute from PLRV via Proposition 3.3:
    - $\eta_\omega \leq P_{Y>0} - P_{X>0}$.
- **Attack advantage or benefit measures:**
  - For a target $\eta^*$ (or benefit thresholds), use binary search over $\omega$:
    - For each candidate $\omega$, compute $\eta_\omega$ or $f_\omega(\alpha)$.
    - Check if the constraint (e.g., $\eta_\omega \leq \eta^*$) is satisfied.
    - Use the halving / binary search approach to find the minimal $\omega^*$ satisfying the desired operational risk.

---

# Step 6: Calibration Procedures
- **(a) Advantage calibration "direct":**
  1. Set target advantage $\eta^*$.
  2. Use Algorithm 4 with PLRVs:
     - For each $\omega$ tested, compute $X_\omega, Y_\omega$, then $T_\omega$.
     - Evaluate $\eta_\omega = P_{Y>0} - P_{X>0}$.
  3. Binary search over $\omega$ to find minimal $\omega^*$ with $\eta_\omega \leq \eta^*$.
- **(b) FPR/FNR calibration:**
  1. Specify $(\alpha^*, \beta^*)$ (or equivalently, $\alpha^*, \beta^*$ based on the conversion in Table 2).
  2. Compute the threshold $\tau$ and $\gamma$ as per Algorithm 6:
     - Using PLRVs, find the $\alpha^*$-quantile of $X$,
     - Compute $\gamma$, then evaluate the resulting $f_\omega(\alpha^*)$.
  3. Binary search over $\omega$ for the minimal satisfying the $f_\omega(\alpha^*) \geq \beta^*$ constraint.

---

# Step 7: Algorithmic Implementation Details & Numerical Optimization
- **Binary search inputs:**
  - Search bounds on $\omega$ (e.g., for Gaussian, from a minimal ($\sigma_{\min}$) to maximal ($\sigma_{\max}$)) based on mechanism sensitivities and privacy constraints.
  - Tolerance for approximation error.
- **Precomputations:**
  - For mechanisms with closed-form formulas, directly compute the parameters (e.g., $\mu$, $\sigma$).
  - For complex mechanisms, precompute multiple $(P,Q)$ pairs at different $\omega$ values, possibly via Algorithm 7 on different discretizations.
- **Interpolation:**
  - Use piecewise linear interpolation of $f_\omega(\alpha)$ results.
- **Efficiency:**
  - Implement Algorithm 1 for each $(X,Y)$ with vectorized operations.
  - For black-box mechanisms, cache the results of privacy profiles at various $\delta$ or $\varepsilon$ in lookup tables to avoid repeated expensive accountant calls.

---

# Step 8: Validation and Experimentation
- Validate the implementation:
  - Verify that for Gaussian mechanism, the recalculated $f_\omega(\alpha)$ matches analytical formulas.
  - For complex mechanisms, check that the trade-off curves are monotonic and piecewise linear.
- Experiment:
  - Vary privacy parameters, dataset sizes, and mechanism noise.
  - Plot efficiency vs. tightness of calibration.
  - Confirm that the direct attack-risk calibration yields utility improvements over standard $(\varepsilon, \delta)$-based calibration.

---

# Additional Clarifications Needed
- Exact sensitivity $\Delta_2$ for mechanisms—either fixed or estimated from data/model.
- How to handle data-specific PLRVs—whether to use mechanisms with known formulas or discretize privacy profiles.
- The extent of discretization granularity ($\Delta$, number of points $k$) for Algorithm 7.
- Typical ranges of the mechanism noise parameter $\omega$ for search.

---

**Summary:**
This plan provides precise steps to:
- Compute privacy profiles (Gaussian or via accountant).
- Construct dominating pairs and PLRVs via discretization.
- Calculate trade-off curves $T(P,Q)$ using Algorithm 1 and Theorem 3.3.
- Evaluate operational risks ($\eta$, benefit, FPR/FNR).
- Use binary search to calibrate noise parameters ($\sigma$, $\omega$) directly to these risks.
- All components are modular, enabling implementation with analytical formulas for Gaussian or black-box privacy accountants.

This roadmap ensures a reproducible, scalable, and theoretically sound approach aligned with the paper's methodology.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular Python system that: (1) computes privacy loss distributions (PLRVs) for mechanisms either via analytical formulas (Gaussian) or discretized privacy profile curves (DP accountant); (2) constructs dominating pairs (P,Q) using Algorithm 7; (3) derives trade-off curves T(P,Q) with Algorithm 1; (4) evaluates operational risks (advantage, benefit, FPR/FNR) directly from PLRVs; (5) performs binary search over mechanism noise parameters (e.g., \sigma for Gaussian, \omega for others) to calibrate directly to target risks using Algorithm 4 or 6; and (6) unifies these as a calibration framework supporting mechanisms with known formulas or black-box privacy accountants, with comprehensive logging and validation. The code leverages open-source numerical libraries (numpy, scipy), privacy accounting tools (dp_accounting, if applicable), and plotting (matplotlib, seaborn).",
    "File list": [
        "main.py",  
        "privacy_profile.py",  # functions to compute privacy profiles, exact formulas, or call accountant
        "dominating_pair.py",  # Algorithm 7 implementation for constructing dominating pairs from privacy profiles
        "tradeoff_curve.py",  # Algorithm 1, 3, 4, 6: compute T(P,Q), f_ω(α), advantage, and calibration routines
        "plrv.py",  # functions to generate and handle privacy loss random variables
        "calibration.py",  # binary search routines for direct risk calibration
        "utils.py",  # helper functions, discretization, plotting, validation
        "config.py"    # configuration parameters for mechanisms, discretizations, thresholds
    ],
    "Data structures and interfaces": "classDiagram\n    class MechanismProfile {\n        +compute_profile() -> dict\n        +compute_f_alpha(α: float) -> float\n        +get_privacy_params() -> dict\n    }\n    class DominatingPair {\n        +construct_from_profiles(profile_list: List[dict], params: dict) -> 'DominatingPair'\n        +get_PLRVs() -> tuple\n        +compute_tradeoff_curve() -> dict\n    }\n    class TradeOffCurve {\n        +evaluate(alpha: float) -> float\n        +interpolate(alpha: float) -> float\n    }\n    class Calibration {\n        +calibrate_advantage(target_eta: float) -> float\n        +calibrate_FPR_FNR(target_alpha: float, target_beta: float) -> float\n        +binary_search_noise(target_risk_func: callable, target_value: float) -> float\n    }\n    class Disk discretizer {\n        +discretize_param_range(param_min: float, param_max: float, granularity: float) -> List[float]\n    }\n    MechanismProfile --> DominatingPair: uses\n    DominatingPair --> PLRV: generates\n    DominatingPair --> TradeOffCurve: produces\n    Calibration --> TradeOffCurve: uses\n    main.py --> all classes: orchestration\n    utils.py --> all: helper functions\n    # Methods annotated with detailed signatures, type annotations, and documentation comments for clarity.",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant MP as MechanismProfile\n    participant DP as DominatingPair\n    participant TO as TradeOffCurve\n    participant CAL as Calibration\n    M->>MP: load mechanism parameters (e.g., sigma, sensitivity)\n    MP->>MP: compute privacy profile (exact or accountant)\n    MP-->DP: create dominating pair (discretize profile, Algorithm 7)\n    DP->>DP: derive PLRVs\n    DP-->TO: compute trade-off curve T(P,Q) (Algorithm 1)\n    TO-->CAL: evaluate risks (advantage, benefit, FPR/FNR) over a set of alphas\n    CAL->>CAL: binary search for minimum noise parameter (sigma or omega) satisfying target risk constraints (Algorithm 4, 6)\n    CAL-->Main: return optimal noise level\n    Main-->User: output calibrated noise level and detailed risk assessments",
    "Anything UNCLEAR": "Clarify if the mechanisms are only Gaussian or also other mechanisms with black-box privacy accountant support, and specify the expected discretization granularity and resolution for the privacy profiles. Also, confirm whether specific operational risks (advantage, benefit, FPR/FNR) are prioritized in calibration or both are equally supported."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "scipy==1.7.0",
        "matplotlib==3.4.3",
        "seaborn==0.11.2",
        "torch==1.9.0",
        "opacus==0.13.0",
        "dp_accounting==0.3.4"  # hypothetical or placeholder for privacy accountant library supporting DP profiles
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "privacy_profile.py",
            "Provides classes/functions to compute privacy profiles, either via analytical formulas (Gaussian) or by wrapping privacy accountants (discretization over epsilon/delta), including methods to derive f(alpha) and generate PLRVs."
        ],
        [
            "plrv.py",
            "Defines classes/functions to construct privacy loss random variables (PLRVs) from distributions (support points and probabilities), including methods to compute log ratios and handle discretizations."
        ],
        [
            "dominating_pair.py",
            "Implements Algorithm 7 to construct dominating pairs (P,Q) from privacy profile curves/discretized profiles, supporting both mechanisms with formulas and black-box accountants, with methods to derive PLRVs."
        ],
        [
            "tradeoff_curve.py",
            "Implements Algorithm 1, Algorithm 3, Algorithm 4, and Algorithm 6 to compute trade-off curves T(P,Q), f_omega(alpha), and to evaluate and calibrate mechanism noise directly to operational privacy risks using binary search."
        ],
        [
            "calibration.py",
            "Provides binary search routines to calibrate mechanism noise parameters (sigma or omega) to target operational risk constraints (advantage, FPR/FNR), using computed trade-off curves and PLRV support points."
        ],
        [
            "utils.py",
            "Includes helper functions for discretization, plotting, and validation, such as uniform discretization over parameters, generating alpha/FPR/FNR ranges, and plotting trade-off curves."
        ],
        [
            "main.py",
            "Entry point: orchestrates loading mechanism parameters, computes privacy profiles via relevant modules, constructs dominating pairs, generates PLRVs, derives trade-off curves, performs risk calibration with binary search, and outputs final calibrated noise parameters and operational risk assessments."
        ]
    ],
    "Task list": [
        "privacy_profile.py",
        "plrv.py",
        "dominating_pair.py",
        "tradeoff_curve.py",
        "calibration.py",
        "utils.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (not required, no front-end communication needed)",
    "Shared Knowledge": "Mechanism sensitivity values, discretization parameters, and risk thresholds (eta, alpha, beta) are defined in a config module or at the top of main.py for consistency. PLRV/class methods support flexible, modular incorporation of different mechanisms and privacy profiles.",
    "Anything UNCLEAR": "Need clarification on the expected mechanism types (Gaussian, accountant-based) supported, and whether to assume only finitely supported PLRVs. Also, confirmation on whether external DP accountant APIs (like opacus or custom) will be used, and details on hardware constraints for large experiments."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
mechanism:
  type: "Gaussian"
  sensitivity: 1.0  # sensitivity Δ₂, set as per the experimental setup
  initial_sigma: 0.5  # starting noise level for calibration
  train_noise_levels: [0.5, 0.6, 0.7, 0.8, 1.0]  # sample noise levels for initial profiling

training:
  batch_size: 256
  epochs: 3
  learning_rate: 0.001  # default, to be adjusted based on task
  gradient_clipping_norm: 1.0
  p_poisson: 0.0004  # Poisson sampling probability as per SST-2 setup
  model:
    type: "LoRA"  # or "Standard" depending on experiment
    dimension: 4
    scaling_factor: 32

dataset:
  name: "SST-2"
  split: "train/test"
  size_train: 67348
  size_test: 872

privacy:
  delta: 1e-5  # fixed DP delta
  epsilon_range: [2, 10]  # typical range for calibration

evaluation:
  metrics: ["accuracy"]
  eval_steps: 10000
  validation_split: "default"

experiment:
  mechanism_eval: "Gaussian"  # or "Accountant" if using accountant
  risk_thresholds:
    advantage: 0.5
    fpr: 0.01
    fnr: 0.05
```

---

**Note:** Full configuration is available in `planning_config.yaml`
