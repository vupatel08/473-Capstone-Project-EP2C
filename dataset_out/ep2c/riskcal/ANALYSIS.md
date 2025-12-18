# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## calibration.py

# Logic Analysis for calibration.py

This module encompasses routines to perform binary search-based calibration of mechanism noise parameters (such as Gaussian noise sigma or alternative mechanism parameters) to satisfy specified operational privacy risk constraints, specifically advantage, FPR, and FNR. The implementation operates over precomputed or dynamically computed trade-off curves $f_\omega(\alpha)$, derived via Algorithm 1, based on privacy loss random variables (PLRVs) and dominating pairs (P,Q). The core goal is to identify the minimal noise parameter $\omega^*$ that ensures the mechanism's privacy operation stays within the user-specified risk threshold.

---

## Core Objectives
- **Given:** a target risk level (e.g., advantage $\eta^*$, FPR $\alpha^*$ and FNR $\beta^*$), and a mechanism characterized by its privacy profile and PLRVs.
- **Output:** the mechanism noise parameter ($\sigma$ for Gaussian, or general $\omega$) that achieves the risk constraint with minimal noise (for utility preservation).

---

## Input Parameters & Data Handling
1. **Target Risk Constraint:**
   - For advantage calibration: `target_eta` (float in [0,1]).
   - For FPR/FNR calibration: `target_alpha`, `target_beta`.
2. **Mechanism Parameters:**
   - Discretized support and probabilities of PLRVs (`X`, `Y`) tailored for mechanism at specific noise level $\omega$.
3. **Trade-off Curves:**
   - Function or table providing $f_\omega(\alpha)$ for different $\omega$.
4. **Search bounds:**
   - `omega_min`, `omega_max`: bounds of the noise parameter space (e.g., sigma in Gaussian).
   - `tolerance`: binary search precision (e.g., 1e-3).
5. **Discretization & Interpolation:**
   - For efficiency, support points are discretized (e.g., uniform grid).
   - Interpolation (linear) of $f_\omega(\alpha)$ when querying values between grid points.
   
---

## Step-by-step Logical Workflow

### 1. **Preprocessing & Data Retrieval**
- **Load or accept PLRVs**: For each candidate $\omega$, generate or retrieve PLRVs $(X_\omega,Y_\omega)$.
- **Compute or retrieve trade-off curve $f_\omega(\alpha)$**:
  - Use Algorithm 1, given the support/probabilities of PLRVs $(X_\omega,Y_\omega)$.
  - Store the curve as a piecewise linear function over a range of $\alpha$, with support points from the algorithm.
- **Calculate operational risk metrics**:
  - For advantage calibration:
    - Compute $\eta_\omega = P_{Y>0} - P_{X>0}$.
  - For FPR/FNR calibration:
    - Use Algorithm 6: derive $\tau$ (quantile of X), compute $\gamma$, evaluate $f_\omega(\alpha^*)$.
- **Keep track of the mechanism parameter $\omega$** (sigma or other).

### 2. **Define the Risk Function to Satisfy**
- *Advantage calibration*: risk(target_eta) = $ \eta_\omega $; the goal: find minimal $\omega$ such that $\eta_\omega \leq$ target_eta.
- *FPR/FNR calibration*: risk(target_beta) = $f_\omega(\alpha^*)$; the goal: find minimal $\omega$ such that $f_\omega(\alpha^*) \geq \beta^*$.

### 3. **Binary Search Algorithm Design**
- **Initialize search bounds**:
  - `low = omega_min`, `high = omega_max`.
- **While** the interval $high - low > tolerance$:
  - **Midpoint calculation**: $\omega_{mid} = (low + high)/2$.
  - **Generate/compute PLRVs** for $\omega_{mid}$:
    - Call the appropriate privacy profile functions.
    - Derive corresponding PLRVs (X, Y).
    - Compute $f_{\omega_{mid}}(\alpha)$ or operational metrics:
      - Advantage: directly calculate $\eta_{\omega_{mid}}$.
      - FPR/FNR: evaluate $f_{\omega_{mid}}(\alpha^*)$.
  - **Check constraint**:
    - For advantage: if $\eta_{\omega_{mid}} \leq target_eta$, set `high = $\omega_{mid}$`.
    - For FPR/FNR: if $f_{\omega_{mid}}(\alpha^*) \geq \beta^*$, set `high = $\omega_{mid}$`.
    - Else, set `low = $\omega_{mid}$`.
- **Result**:
  - Return the $\omega^* \approx (low + high)/2$ as the minimal parameter satisfying risk constraints.
   
### 4. **Edge Cases & Validations**
- Confirm that provided bounds are valid for the mechanism:
  - For Gaussian: ensure that $\sigma$ is within reasonable bounds, e.g., [0.01, max noise].
- Check monotonicity:
  - $f_\omega(\alpha)$ should be monotonic with $\omega$, simplifying the binary search.
  
### 5. **Implementation Details & Optimizations**
- **Caching**:
  - Cache $f_\omega(\alpha)$ and risk metrics at each $\omega$ to avoid recomputation.
- **Adaptive discretization**:
  - For high-precision, refine discretization near the target risk threshold.
- **Parallelization**:
  - For mechanisms with heavy PLRV computation, parallelize over multiple $\omega$ values during the binary search.

---

## Additional Considerations
- **Support for multiple mechanisms/formulas**:
  - For Gaussian mechanisms, apply the analytical formula directly, avoiding Algorithm 1 overhead.
  - For complex mechanisms, use privacy accountant APIs, discretized discretizations, and Algorithm 7 as preprocess.
- **Trade-off curve accuracy**:
  - Ensure the piecewise linear interpolation in Algorithm 1 preserves the tightness needed for risk constraints.
- **Sensitivity estimation**:
  - For mechanisms like DP-SGD, sensitivity must be known or set (from config.sensitivity).
- **Calibration strategies**:
  - Prioritize advantage calibration for utility; fallback to FPR/FNR if advantage bounds are too loose.

---

## Summary of Key Functions in 'calibration.py'
- `compute_tradeoff_curve(plrv_X, plrv_Y)`: Calls Algorithm 1, returning a piecewise linear function over $\alpha$.
- `compute_advantage(plrv_X, plrv_Y)`: Extracts advantage via the PLRV probabilities.
- `compute_f_alpha(plrv_X, plrv_Y, alpha)`: Evaluates $f_\omega(\alpha)$ at a specific measure.
- `binary_search_noise(target_risk, risk_type)`: Implements the described binary search over $\omega$ for advantage or FPR/FNR constraints.
- `calibrate_advantage(target_eta, plrv_generator)`: Wrapper for advantage calibration.
- `calibrate_fpr_fnr(target_alpha, target_beta, plrv_generator)`: Wrapper for FPR/FNR calibration.

---

This detailed logical analysis informs the implementation strategies for the routines in 'calibration.py' and ensures precise, efficient, and theoretically sound mechanism noise calibration to operational privacy risks as described in the paper.

## dominating_pair.py

{
  "dominating_pair.py": "This module implements the construction of dominating pairs (P,Q) as specified by Algorithm 7 from Doroshenko et al. (2022), adapted to the paper's methodology. Its core function is to produce distribution pairs that dominate or tightly upper bound the privacy loss distributions of the mechanism under the specified privacy profile, facilitating tight trade-off curve computation (Theorem 3.3).\n\nThe critical inputs are the mechanism's privacy profile curve(s), discretized over a suitable parameter range, along with discretization parameters (grid step size, number of points). It must support mechanisms with closed-form formulas (Gaussian) and mechanisms with privacy accountant output (discretized profile). Use the options provided in the configuration: mechanisms may be Gaussian with known formulas or black-box with privacy profile data.\n\nMain functions and flow:\n\n1. Input handling:\n   - Accept the privacy profile curve(s) as discretized data: arrays of (ε, δ) pairs for each mechanism component, or directly the privacy profile functions.\n   - Accept parameters: discretization granularity, number of points (k), starting ε (or analogous parameter), mechanism sensitivity, or privacy accountant output.\n\n2. Compute the privacy profile curve(s):\n   - For mechanisms with formulas (Gaussian):\n     - Use \(\varepsilon\) range from input, with steps, to generate \(\delta\) values via the analytical formulas.\n   - For mechanisms with privacy accountant:\n     - Call the accountant at a grid of \(\delta\) values (or \(\varepsilon\) values, depending on support), to produce arrays of (ε, δ).\n   - Store these in arrays, ensuring they form a monotonic curve.\n\n3. Construction of the dominating pair (P,Q):\n   - Use Algorithm 7, which computes the discrete dominating pair based on the privacy profile curve(s).\n   - The core steps involve:\n       a. Compute the pmf of P and Q over the discretized support points, using the privacy profile data. This involves:\n          - Deriving support points: from the privacy profile \(\delta(\varepsilon)\), compute corresponding probability mass at each \(\varepsilon\) point, with associated probabilities.\n          - For Gaussian: directly calculate these pmfs based on the Gaussian density and the formulas for the privacy profile.\n          - For accountant: approximate pmfs by discretization over the privacy profile curve.\n       b. Use Algorithm 7's specific method — which involves row-wise calculations of P and Q support points, matching the privacy profile's constraints — to produce these pmfs.\n   - Generate the distributions P and Q support points and their probabilities, which give the discrete distributions.\n\n4. Derive PLRVs (X,Y):\n   - For each support point (o), compute the privacy loss ratios:\n       - \(X = \log \frac{Q(o)}{P(o)}\)\, for \(o \sim Q\), support points \(o \sim P\)\n       - \(Y = \log \frac{Q(o)}{P(o)}\), for \(o \sim P\)\n   - The PLRVs are supported on the points with their respective probabilities, forming the basis for trade-off curve evaluations.\n\n5. Return:\n   - The constructed dominating pair: distributions P and Q with their support points and pmfs.\n   - The PLRVs: support points for X and Y, with support sizes, probability masses.\n\nAdditional considerations:\n- Support discretization: ensure that the support points \(\{\varepsilon_i\}\) (or equivalent parameters) are fine-grained enough to preserve tight bounds — typically using the granularity parameter from config.\n- Numerical stability: handle cases where pmf support points are extremely small (truncation or thresholding), especially for composed mechanisms.\n- Flexibility: support multiple mechanisms or components, by constructing separate pairs and convolving their PLR distributions accordingly.\n- Efficiency: leverage the structure of the privacy profile and PMFs to minimize computational costs, possibly caching repeated calculations.\n\nOverall, the module converts privacy profile data — either analytical or via accountant — into the probabilistic distributions (P,Q) and their PLRVs, foundational for the calibration routines and trade-off analysis in subsequent modules."
}

## main.py

# Logic Analysis for main.py

This script is the orchestrating entry point of the entire calibration framework. Its purpose is to initialize, coordinate, and execute the steps necessary to calibrate the mechanism's noise parameter (e.g., Gaussian sigma) directly to a specified operational privacy risk (e.g., attack advantage or FPR/FNR) level, as described in the paper. It should generate outputs including the calibrated noise level and associated risk metrics, enabling practical deployment.

Below are the detailed logical components, their sequence, key functions, data flow, and inter-module interactions.

---

## 1. **Load Configuration and Initialize Parameters**

- **Purpose:**  
  - Read the provided `config.yaml` for mechanism, training, dataset, privacy, and evaluation parameters; this ensures flexibility and reproducibility.
  - Set up hyperparameters for discretization, starting noise levels, risk thresholds, and any constants.

- **Implementation details:**  
  - Use a YAML parser (e.g., `yaml.safe_load`) to load configurations.
  - Extract parameters such as:
    - `mechanism.type` (Gaussian or alternative).
    - `mechanism.sensitivity`.
    - `mechanism.initial_sigma`.
    - `mechanism.train_noise_levels`.
    - Dataset info (`name`, split sizes).
    - Privacy parameters (`delta`, ranges).
    - Calibration targets (`advantage`, `fpr`, `fnr`).
  - Define key variables for search bounds (`sigma_min`, `sigma_max`), discretization granularity (`Δ`), etc.

---

## 2. **Set Up Environment and Data Structures**

- **Purpose:**  
  - Prepare data structures for storing intermediate results:
    - Mechanism profiles (e.g., `MechanismProfile` objects).
    - Discretized privacy profile points.
    - Dominating pairs (objects of class `DominatingPair`).
    - PLRV objects (`X,Y`) supporting computational routines.
    - Risk thresholds for the binary search (e.g., `eta_star`, `alpha_star`, `beta_star`).

- **Implementation details:**  
  - Initialize lists/dictionaries to hold profiles.
  - Predefine discretization grids based on `config.yaml` (e.g., over epsilon/delta if using accountant, or directly over sigma).

---

## 3. **Compute Privacy Profiles for Candidate Noise Levels**

- **Purpose:**  
  - For each initial noise level in `train_noise_levels`:  
    - Compute the corresponding privacy profile (`f(α)`) or $(\varepsilon, \delta)$-DP profile.
    - For Gaussian mechanisms, compute analytically using formulas (Proposition G.1).
    - For other mechanisms, either call the privacy accountant API (`dp_accounting`) or approximate via discretization (Eq. (20)).

- **Implementation:**  
  - Loop over the list: `sigma` in `train_noise_levels`.
  - For each sigma:
    - Instantiate mechanism object (`MechanismProfile`) with sigma, sensitivity.
    - Compute privacy profile:
      - **Gaussian:** use analytical formula.
      - **Other:** call privacy accountant at discretized $\delta$ values, then derive $f(\alpha)$ via Algorithm 1 or Eq. (20).
    - Store results for later use.

---

## 4. **Construct Discrete Dominating Pairs (Algorithm 7)**

- **Purpose:**  
  - Use privacy profiles (`dec) from step 3 to build dominating pairs `(P,Q)` that bound the mechanism's privacy properties tightly.
  - Support mechanisms with known formulas or approximate profiles.

- **Implementation:**  
  - Input privacy profile data (support points and probabilities).
  - Use Algorithm 7 with discretization parameters (`Δ`, `k`) to generate the pair `(P,Q)` for each mechanism profile.
  - Generate the associated PLRVs `(X,Y)` for the dominating pairs:
    - Use Algorithm 7’s output (support points and pmfs).
    - Compute log ratios at support points to build `(X,Y)` support and probabilities.

---

## 5. **Derive Trade-off Curves T(P,Q) and f_ω(α)**

- **Purpose:**  
  - For each dominating pair `(P,Q)`, compute the trade-off curve using Algorithm 1:
    - Obtain $(X,Y)$ support and pmfs.
    - For each desired FPR level $\alpha$, find the threshold $\tau$ and margin $\gamma$ (support from Eq. (42)) to compute $T(P,Q)(\alpha)$ (= FNR).
    - Store the curve `$f_{\omega}(\alpha)$` for each mechanism.

- **Implementation:**  
  - Loop over each `(P,Q)`.
  - Call Algorithm 1:
    - For each $\alpha$ (discretized, e.g., via quantiles of $X$), compute $T(P,Q)(\alpha)$.
  - Store the resulting curve for each mechanism’s discretization.

---

## 6. **Calculate Operational Privacy Risks (Advantage / Benefit / FPR/FNR)**

- **Purpose:**  
  - From the PLRVs `(X,Y)`:
    - Compute advantage $\eta_\omega \leq P[Y>0] - P[X>0]$.
    - For FPR/FNR: evaluate $f_\omega(\alpha)$ at specified $\alpha^*$ and derive $\beta^*$ via Algorithm 6.

- **Implementation:**  
  - Compute probabilities from `(X,Y)` support pmfs.
  - Use Proposition 3.3 or Algorithm 4:
    - For advantage:
      - `eta_omega = P_Y_gt_0 - P_X_gt_0`.
    - For FPR/FNR:
      - Find threshold $\tau$ for $\alpha^*$ via Eq. (42).
      - Calculate $\beta^*$ using the support pmfs and Eq. (44).
- **Store or record** these risk metrics.

---

## 7. **Perform Binary Search for Noise Level $\omega^*$ (e.g., for Gaussian: sigma)**

- **Purpose:**  
  - Find the minimal noise parameter satisfying the target operational risk constraint:
    - **Advantage calibration:** $\eta_\omega \leq \eta^{*}$
    - **FPR/FNR calibration:** $f_\omega(\alpha^*) \geq \beta^{*}$

- **Implementation:**  
  - Initialize search bounds: `sigma_min`, `sigma_max` (from config or empirical estimates).
  - Utilize the binary search routine (`calibration.py`):
    - At each step:
      - Pick candidate $\omega$.
      - Compute `(X,Y)` PLRVs:
        - For Gaussian, analytically derive.
        - For accountant-based mechanisms, run privacy accountant at discretized points.
      - Compute risk metric:
        - `eta` or $f_\omega(\alpha^*)$.
      - Check if the risk constraint is satisfied.
      - Adjust bounds accordingly.
  - Continue until achieving desired precision (tolerance for $\omega$).

---

## 8. **Output Results**

- **Purpose:**  
  - Final selected mechanism noise parameter (`sigma^*` or `omega^*`).
  - Corresponding operational risk metrics.
  - Optional plots of trade-off curves, risk trajectories, or sensitivity analysis.

- **Implementation:**  
  - Print or save:
    - Calibrated noise level.
    - Achieved advantage FPR/FNR or benefit.
    - The trade-off curve data points.
  - Save plots generated via `matplotlib`/`seaborn` for visualization.

---

## 9. **Validate and Log**

- **Purpose:**  
  - Check that the final risk metric indeed meets the target constraints.
  - Log intermediate steps, parameter choices, discretization details, and final results for reproducibility.
  - Ensure that all operations follow the theoretical guarantees from the paper.

---

## 10. **Optional: Visualization and Summary**

- Generate plots illustrating:
  - Trade-off curves for different $\omega$.
  - The risk vs. noise level relation.
  - The improvement over standard $(\varepsilon,\delta)$ calibration (if comparing).
- Summarize the final mechanism configuration and risks in a report or logs.

---

# Conclusion:
The `main.py` must **coordinate the entire pipeline** from configuration, profile computation, dominating pairs creation, trade-off analysis, operational risk calculation, to binary search for optimal noise parameters. It should be modular, leveraging the classes and functions from the other modules, ensure numerical stability, and preserve reproducibility through well-documented steps and parameter logs.

This detailed logic analysis provides a clear, step-by-step blueprint for implementing `main.py` aligned with the paper, plan, and design.

## plrv.py

{
  "file": "plrv.py",
  "purpose": "Implement classes and functions to construct, handle, and analyze Privacy Loss Random Variables (PLRVs) from mechanisms’ distributions, supporting both exact formulations (Gaussian) and discretized approximations (privacy profiles or accountant outputs). This module facilitates the calculation of log ratios, support points, and probabilities needed for trade-off curve evaluation and operational risk calibration.",
  "core components": [
    {
      "class": "PLRV",
      "description": "Encapsulates a privacy loss random variable with support points and associated probabilities, supporting operations like computing log ratios, support support points, and probability mass functions.",
      "attributes": {
        "support_points": "Array of support values (e.g., log ratios).",
        "probabilities": "Array of probabilities corresponding to support points.",
        "support_support": "Optional; the actual support set, including discrete points and possibly infinities."
      },
      "methods": [
        {
          "name": "from_distributions",
          "purpose": "Construct PLRV support from two distributions P and Q, either via analytical formulas (for Gaussian mechanisms) or discretization of privacy profile curves (for mechanisms with accountant access).",
          "inputs": [
            "distributions P and Q: support points and pmfs (for discrete case) or mechanisms to generate these supports.",
            "mechanism type: e.g., 'Gaussian' or 'Profile' (discretized profile).",
            "discretization parameters: granularity, support bounds, number of points (k).",
            "error tolerance: optional, for controlling discretization accuracy."
          ],
          "outputs": "PLRV instance with support points and probabilities."
        },
        {
          "name": "compute_log_ratios",
          "purpose": "Calculate the log ratio log(Q(o)/P(o)) for each support point o, necessary for deriving privacy loss support support.",
          "inputs": "None; operates on the support points provided.",
          "outputs": "Array of log ratios, supports supports with possible infinities."
        },
        {
          "name": "get_support",
          "purpose": "Return support points, support probabilities, and support support set, essential for constructing dominance pairs and curve evaluation.",
          "inputs": "None.",
          "outputs": [
            "support_points",
            "probabilities",
            "support_support"
          ]
        },
        {
          "name": "probability_mass",
          "purpose": "Retrieve the probability mass function (pmf) over support points, either explicitly or via interpolation if discretized.",
          "inputs": "None.",
          "outputs": [
            "a dictionary or array mapping support points to pmfs."
          ]
        },
        {
          "name": "edge_support",
          "purpose": "Handle potential infinities in support, e.g., support at -∞ or +∞, or support supported at boundary points, as per referenced algorithms and theory.",
          "inputs": "None.",
          "outputs": "Support with infinities properly included."
        },
        {
          "name": "get_cdf_support",
          "purpose": "Return the cumulative distribution over support points, for threshold quantile calculations and Neyman–Pearson tests.",
          "inputs": "None.",
          "outputs": "CDF over support points, supports support as cumulative distribution array."
        }
      ],
      "additional functions": [
        {
          "name": "discretize_privacy_profile",
          "purpose": "Convert a mechanism’s privacy profile curve \(\varepsilon (\delta)\) or \(\delta (\varepsilon)\) into a discretized PMF of PRLV supports, using Algorithm 7 from Doroshenko et al. (2022).",
          "inputs": [
            "privacy profile curve data: arrays of \(\varepsilon\) and \(\delta\).",
            "discretization parameters: bounds, granularity \(\Delta\), number of points k.",
            "support bounds: min and max of support for X and Y.",
            "mechanism type: e.g., Gaussian, or accountant-based"
          ],
          "outputs": "PMFs for \(X\) and \(Y\) support points with associated probabilities."
        },
        {
          "name": "support_support_points",
          "purpose": "Generate and return the support support set for the PLRV, properly handling finite supports, infinities, and possible support truncations.",
          "inputs": "Support points array, optional thresholds.",
          "outputs": "Support support array including notes on infinities."
        }
      ],
      "design notes": [
        "Construct PLRVs from mechanisms: for Gaussian mechanisms, directly compute the log ratio at the support-optimized \(\sigma\); for discretized profiles, generate support points and pmfs according to Algorithm 7.",
        "Support points should be sorted, and probabilities normalized.",
        "Infinities are supported and handled explicitly in support sets.",
        "Methods should validate the support's consistency (non-negativity, sum to 1).",
        "Maintain flexibility to support different mechanism classes depending on usage, enabling build-in analytical formulas or APIs calling privacy accountant modules.",
        "This module's output is used downstream in Algorithm 1 for trade-off curve evaluation, and in mechanism calibration routines."
      ]
    }
  ],
  "additional considerations": [
    "Ensure proper support for entropy and log ratio calculations, including handling \(\log 0\) (support at support support at infinities must be handled carefully).",
    "Provide utility methods for discretization resolution control and support support augmentation if mechanism distributions have support at mixed finite/infinite points.",
    "Design classes/functions to be compatible and easily integrable with the overarching Python package architecture, enabling composition and validation."
  ],
  "summary": "The plrv.py module is central for constructing and manipulating privacy loss random variables in both analytical (Gaussian) and approximate (discretized profile) forms. It provides support support extraction, log ratio calculations, and utilities for threshold-based trade-off curve evaluation, enriching the overall calibration framework with precise operational risk assessments."
}

## privacy_profile.py

# Logic Analysis for privacy_profile.py

This module's primary objective is to provide a flexible and extensible API for deriving privacy profiles, either through analytical formulas (for mechanisms like Gaussian) or via discretized privacy profile curves obtained from off-the-shelf privacy accountants (for mechanisms like DP-SGD). It also includes methods to derive operational privacy risk measures, such as the function f(α), and to create privacy loss random variables (PLRVs) from the distributional information of mechanisms. This forms the foundation for the calibration routines that directly tune mechanism noise to operational attacks.

---

### Core Functional Components

1. **Mechanism Class: `MechanismProfile`**
   - Purpose:
     - Encapsulate mechanism parameters and methods to compute privacy profiles.
     - Support both mechanisms with closed-form formulas (e.g., Gaussian) and mechanisms with privacy accountants (e.g., DP-SGD).
   - Key Methods:
     - `compute_profile()`: Return privacy profile as a curve or data structure. For formulas, direct computation; for accountants, calls to the accountant API.
     - `compute_f_alpha(α)`: Evaluate the lower bound function f(α) over the mechanism's privacy profile.
     - `get_privacy_params()`: Return mechanism-specific attributes, e.g., for Gaussian, `sigma`, `sensitivity`; for accountant-based, the list of discretized `(ε, δ)` pairs.

2. **Class: `PrivacyAccountant` (possibly embedded within `MechanismProfile`)**
   - Purpose:
     - Internal or external component to support mechanisms where privacy profiles are obtained via black-box APIs.
   - Inputs:
     - Discretized pairs of `(ε, δ)` or parameters for DP accountant.
   - Outputs:
     - Privacy profile data over a discretized grid suitable for constructing dominating pairs.

3. **Functionality for Analytical Mechanisms (Gaussian)**
   - Analytical formulas directly based on known relations:
     - Privacy profile: \(f(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - \mu)\), where \(\mu = \Delta_2/\sigma\).
     - (ε, δ) bounds: from Proposition G.1, via Gaussian CDFs.
   - These formulas are fundamental for quick, exact calculations, reducing numerical errors.

4. **Discretized Privacy Profile Construction**
   - Inputs:
     - A sequence of `(ε_i, δ_i)` pairs.
     - Discretization granularity (e.g., `Δ`), profile range, and grid size (`k`).
   - Process:
     - Use Algorithm 7:
       - For each `(ε_i, δ_i)` point, compute the privacy profile at discretized `ε` values on a grid.
       - Derive the corresponding distributions `(P,Q)` (support points and probabilities).
     - Construct the overall DP profile for the mechanism via composition if needed.

5. **PLRV Generation (`PLRVs`)**
   - Inputs:
     - Distributions `(P, Q)` derived from the privacy profile or directly via formulas.
   - Process:
     - Compute the probability mass function (PMF) over support points for `X = log Q(o)/P(o)`:
       - For each support `o`, density-derived or support point `o`, compute `X(o)`.
     - For `Y`, compute similar quantities for the null and alternative distributions.
     - Store these as structured data (arrays, dicts) for later use.

6. **OPERATIONAL RISK FUNCTIONS (`f(α)`, advantage, etc.)**
   - Methods:
     - For a given `α`, evaluate `f(α)` by determining thresholds `τ` (quantiles of `X`) and corresponding `γ`.
     - Use Proposition 3.3 and Algorithm 1 methods:
       - Support supports for `X`, `Y`.
       - Find the `τ`—the `(1 - α)` quantile of `X`.
       - Compute `β^*(τ, γ)` via distribution functions (CDFs).
     - These operations are implemented for each mechanism instance, facilitating calibration to operational targets.

---

### Implementation Details and Considerations

- **Analytical vs. Discretized Profiles:**
  - For mechanisms with closed-form formulas (Gaussian), implement direct functions:
    - `compute_psi()` returning the profile curve `f(α)` (via inverse Gaussian CDFs).
    - `compute_epsilon_delta()` for privacy bounds.
  - For mechanisms with privacy accountants:
    - Wrap calls to the privacy accountant (e.g., `dp_accounting`) to produce `(ε, δ)` pairs discretized over the domain.
    - Provide methods to generate the privacy profile curve numerically (`compute_profile()`), supporting the use of Algorithm 1.

- **Discretization and Support Points:**
  - Use uniform grid points over the privacy parameters (`ε` or `δ`), as dictated by the configuration.
  - Compute corresponding PMFs `(P, Q)` per Algorithm 7:
    - Use the discretized profile and the density functions to derive probabilities.
    - Store PMFs as support points with associated probabilities for PLRV construction.

- **PLRV Construction:**
  - Support point objects:
    - `X`: array of support points and their probabilities.
    - `Y`: similarly, for the distribution of the other distribution.
  - Care:
    - Handle atoms at `-∞` and `+∞` carefully, ensuring the support and discrete support points are aligned.

- **Efficiency and Flexibility:**
  - For exact formulas, directly instantiate the class with parameters and compute functions.
  - For black-box accountant support, batch API calls for multiple `(ε, δ)` pairs; cache results to avoid repetition.
  - Support computations over the entire support of `X` and `Y` to facilitate multiple risk evaluations.

- **Interfaces:**
  - Given future modules (`dominating_pair.py`, `tradeoff_curve.py`), the class should provide:
    - Raw data (`X`, `Y`, support, PMFs).
    - Methods to evaluate `f(α)` at arbitrary α.
    - Methods to compute operational risks (`advantage`, `FPR`, `FNR`).

- **Extensibility:**
  - Abstract mechanisms can be extended:
    - `MechanismProfile` base class with subclasses:
      - `GaussianMechanismProfile`
      - `AccountantMechanismProfile`

---

### Summary of Logic Flow

1. **Initialization:**
   - Instantiate `MechanismProfile` with mechanism parameters from config (e.g., Gaussian sigma, sensitivity).
   - For accountant-based mechanisms, load API/access points.

2. **Compute Privacy Profile:**
   - If Gaussian:
     - Use `compute_gaussian_profile()` directly.
   - If accountant:
     - Generate discrete `(ε, δ)` grid. Call accountant API for each point.
     - Use Algorithm 7 to derive `(P,Q)` and support points.
3. **Construct PLRVs:**
   - Convert `(P,Q)` into support for `X` and `Y`.
4. **Derive `f(α)` curve:**
   - For multiple `α`, evaluate thresholds and `β^*(τ, γ)` per Algorithm 1.

5. **Provide operational risks:**
   - Calculate bounds on advantage, FPR, and FNR from PLRVs:
     - For advantage, `η ≤ P Y>0 - P X>0`.
     - For specific `α`, `β` pairs, compute via Algorithm 6.

6. **Output:**
   - `MechanismProfile` object with all computed curves, risks, and PLRVs.
   - Interfaces for calibration routines to use these computed data in the main calibration module.

---

### Final Notes

- Ensure the class design supports:
  - Adaptive discretization granularity and range.
  - Support for exact analytical formulas and approximate numerics.
  - Efficient support point storage and retrieval.
  - Clear separation between mechanism specifics and operational risk evaluation.

- Use unit tests for:
  - Correct computation of Gaussian formulas.
  - Correct discretization and construction of `(P,Q)`.
  - Correct evaluation of `f(α)` and operational risks.

This detailed logic analysis forms a robust blueprint for implementing `privacy_profile.py` consistent with the theoretical, methodological, and experimental framework of the paper.

## tradeoff_curve.py

# Logic Analysis for tradeoff_curve.py

**Purpose:**  
Implement core algorithms for computing and evaluating trade-off curves between false positive rate (FPR) (α) and false negative rate (FNR) (β) in the context of differential privacy (DP), specifically Algorithm 1, Algorithm 3, Algorithm 4, and Algorithm 6, as detailed in the paper. These are essential for directly calibrating mechanism noise (Gaussian or other mechanisms) to operational privacy risks based on privacy loss random variables (PLRVs).

---

# Key Components and Their Roles:

## 1. **Algorithm 1: Compute Trade-Off Curve $T(P,Q)$ (from PLRVs)**  
- **Inputs:**  
  - PLRV support points and probabilities $(X, Y)$ for a dominating pair $(P,Q)$.  
  - Select a set of $\alpha$ levels, ideally via quantiles of $X$ or uniformly discretized.  
- **Outputs:**  
  - Trade-off curve points: pairs of $(\alpha, \beta)$ where $\beta$ is the FNR of the most powerful attack at FPR $\alpha$.
- **Methodology:**  
  - For each desired FPR level $\alpha$, find the Neyman-Pearson threshold $\tau$ and coin flip probability $\gamma$ that achieve the specified FPR, using support-based quantile search.  
  - Calculate $\beta^*(\tau, \gamma)$ (see Eq. 37), which gives FNR at that FPR, forming the curve $T(P,Q)(\alpha)$.
- **Implementation notes:**  
  - Use the distribution functions of $X$ and $Y$ (CDFs) for efficient inversion.  
  - Support points are sorted; selection of thresholds is via inverted CDFs at support points.  
  - Piecewise linear interpolation across support points for better resolution.

---

## 2. **Algorithm 3: Compute Advantage from PLRVs**  
- **Inputs:**  
  - Discrete distributions (support points and probabilities) for $X$ and $Y$.  
- **Outputs:**  
  - Advantage bound $\eta \le P_{Y > 0} - P_{X > 0}$.  
- **Methodology:**  
  - Calculate the probability that $Y > 0$ (integrate over $Y$ support).  
  - Calculate the probability that $X > 0$ similarly.  
  - The difference bounds the maximum advantage, per Proposition 3.3.  
- **Implementation notes:**  
  - Cross-support points for $X$ and $Y$ support easy support-based calculations.  
  - Support points should be stored with their probabilities in suitable data structures (arrays/dictionaries).

---

## 3. **Algorithm 4: Direct Advantage Calibration**  
- **Purpose:**  
  - Binary search over noise parameter $\omega$ (e.g., Gaussian $\sigma$) to find the minimal that satisfies an operational advantage constraint $\eta^*$.  
- **Methodology:**  
  - For each candidate $\omega$, generate $(X_\omega,Y_\omega)$ via privacy analysis (exact or approximate).  
  - Compute advantage bound using Algorithm 3.  
  - Adjust $\omega$ via binary search to meet the threshold $\eta^*$.  
- **Implementation notes:**  
  - Use precomputed or on-the-fly PLRVs support points.  
  - Store $\omega$ bounds based on mechanism parameter ranges (e.g., $\sigma_{\min}$, $\sigma_{\max}$).  
  - Terminate binary search when desired precision is achieved.

---

## 4. **Algorithm 6: Direct FPR/FNR Calibration**  
- **Purpose:**  
  - Binary search over $\omega$ to find the minimal noise level ensuring the trade-off curve $f_\omega(\alpha^*) \ge \beta^*$, i.e., the mechanism’s attack FPR/FNR is within acceptable bounds at a specific FPR level $\alpha^*$.  
- **Methodology:**  
  - For each candidate $\omega$, use Algorithm 1 to compute the $f_\omega(\alpha^*)$ (via $T(P,Q)$).  
  - Compare with target $\beta^*$.  
  - Adjust $\omega$ via binary search until constraint is met with desired precision.  
- **Implementation notes:**  
  - Support discretizations and thresholds are tailored to achieve the specified FPR/FNR pair.  
  - Use precomputed or dynamically generated $(X_\omega,Y_\omega)$.

---

# Supporting Functions and Data Structures:

- **Support points and probabilities of PLRVs ($X$, $Y$):**  
  - Arrays `x_support`, `x_probs`, `y_support`, `y_probs` storing support points and probabilities.  
  - Use sorted support points for binary search and quantile inversion.  

- **Computation of $T(P,Q)(\alpha)$:**  
  - For each $\alpha$, identify index where the support support exceeds the $(1-\alpha)$-quantile in $X$.  
  - Determine threshold $\tau$ as this quantile.  
  - Set $\gamma$ as per Eq. (42).  
  - Compute $\beta^*(\tau, \gamma)$ using the support probabilities of $Y$ (Eq. 37).  

- **Approximate or exact evaluation of $f_\omega(\alpha)$:**  
  - For mechanisms with closed-form $f(\alpha)$ (Gaussian), implement direct formula.  
  - For others, run Algorithm 1 per candidate $\omega$.

- **Binary search routines:**  
  - Generic over a specified $\omega$ range.  
  - For each step, evaluate constraint (advantage or FPR/FNR) via algorithms above, compare to thresholds, and refine.

---

# Additional considerations:

- **Discretization granularity:**  
  - Choose based on a tradeoff between precision and computational cost, e.g., $\Delta$ in Algorithm 7, or support steps for $X$, $Y$.  
  - Support points span the PLRV support; must be dense enough for precise thresholds.

- **Handling infinite support points:**  
  - Treat as boundary points with appropriates adjustments (e.g., $-\infty$, $\infty$ support points).  
  - Use numerical support approximation; support probabilities at extremal points.

- **Computational complexity:**  
  - Focus on support-based calculation rather than integral evaluations, leveraging the discretization.  
  - Cache results where possible (discretized privacy profiles, PLRVs at different $\omega$).

- **Validation:**  
  - Verify $f_\omega(\alpha)$ behavior for known mechanisms (Gaussian with formulas).  
  - Confirm trade-off curve continuity and monotonicity.

---

# Summary:

Tradeoff_curve.py must provide functions to:
- Construct trade-off curves from PLRVs (Algorithm 1) by identifying $X,Y$ support points, thresholds, and success probabilities.
- Compute mechanism operational risks (advantage, benefit) directly from PLRVs.
- Perform binary search over noise parameters to meet specified operational risk thresholds via algorithms 4 and 6.
- Support mechanisms with analytical or approximate privacy profiles, supporting broad flexibility.
- Maintain support for discretized and support-based operations, ensuring robustness, efficiency, and integration with downstream calibration routines.

This comprehensive understanding enables precise, reproducible implementation aligned with the paper’s methodology.

## utils.py

{
  "utils.py": "The utils.py module serves as a collection of auxiliary functions vital for discretization, plotting, and validation tasks within the calibration pipeline outlined in the paper. Its core functions should facilitate the dynamic generation of discretized parameter ranges, the visualization of trade-off curves, and various validation checks to ensure computational accuracy and consistency.\n\nThe main functionalities to implement include:\n\n1. Discretization of parameters:\n    - Uniform discretization of any mechanism parameter range such as noise levels (σ or ω), privacy parameters (ε, δ), or other relevant parameters.\n    - Implementation of a function that, given min and max bounds and a granularity (delta), returns an evenly spaced list of values for iterative binary search or profile construction.\n    - Support multiple discretization schemes if needed (linear, logarithmic).\n\n2. Generating and facilitating alpha/FPR/FNR ranges:\n    - Functions to generate sequences of alpha (FPR) values uniformly or according to specified grids, covering the entire domain [0, 1].\n    - Functions to generate corresponding FNR (1 - benefit, 1 - β) values, either through direct specification or by evaluation over discretized trade-off curves.\n    - Support for plotting trade-off curves with proper axis labels and legends.\n\n3. Plotting trade-off curves:\n    - Functions that accept trade-off curve data (X and Y supports with corresponding probabilities) and produce clear plots for the calibrated risks.\n    - Titles, axis labels, legends, and optional confidence intervals or error shading are included for comprehensive visualization.\n\n4. Validation and verification helpers:\n    - Check monotonicity and consistency of computed trade-off curves.\n    - Validate that the discretized privacy profiles align with the analytical formulas (for Gaussian) or with the outputs from DP accountant APIs.\n    - Provide comparison plots or summaries between approximate and analytical profiles.\n\n5. Configuration support:\n    - Support reading discretization parameters, thresholds, and ranges from a configuration dictionary or object to ensure consistent, experiment-specific discretization.\n\nImplementation details:\n- Use numpy functions for numerical sequences, e.g., numpy.linspace for uniform discretization.\n- Provide options for logarithmic vs. linear spacing.\n- Plot with matplotlib, ensuring readable font sizes and labels.\n- Return data in list or numpy array formats suitable for downstream functions.\n- Include helper functions to convert between probability support points and alpha/FPR/FNR values.\n\nSample function signatures include:\n- def discretize_param_range(param_min: float, param_max: float, granularity: float, scheme: str = 'linear') -> np.ndarray\n- def generate_alpha_grid(num_points: int) -> np.ndarray\n- def plot_tradeoff_curve(alpha_vals: np.ndarray, fnr_vals: np.ndarray, title: str = '') -> None\n- def validate_tradeoff_curve(curve_data: dict) -> bool\n\nThis utility module should be designed for easy integration with the main pipeline, enabling flexible visualization, validation, and parameter range discretization to support efficient and accurate calibration procedures as described in the paper."
}

