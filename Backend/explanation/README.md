# EP2C Explanation Layer

## Overview

The Explanation Layer is a comprehensive system that provides traceability, documentation, and explainability features for code generated from academic papers. It bridges the gap between paper content and generated implementation.

## Features

### 🔗 Paper-to-Code Traceability
- **Bidirectional Mapping**: Links code components to paper sections and vice versa
- **Coverage Analysis**: Measures how much of the paper is implemented in code
- **Component Extraction**: Automatically identifies classes, functions, and methods in generated code

### 📚 Comprehensive Documentation
- **Auto-Generated README**: Creates detailed README with paper references, structure, and guidance
- **Next-Steps Guidance**: Provides actionable steps for users to work with generated code
- **Missing Information Alerts**: Highlights what's not specified in the paper

### ⚠️ Missing Information Detection
- **Hyperparameter Detection**: Identifies parameters used in code but not in paper
- **Dataset Gap Analysis**: Detects missing dataset specifications
- **Implementation Details**: Flags placeholders and TODOs in code
- **Severity Classification**: Categorizes gaps as high, medium, or low priority

### 📊 Explainability Evaluation
- **Traceability Coverage**: Percentage of paper sections mapped to code
- **Comment Density**: Ratio of comments to code
- **Paper Reference Accuracy**: How well code references paper sections
- **Readability Score**: Code documentation quality
- **Overall Explainability Score**: Weighted composite metric

## Components

### 1. `explanation_generator.py`
Creates bidirectional traceability maps between code and paper sections.

**Key Methods:**
- `generate_traceability_map()`: Main method to create code-paper links
- `_extract_code_components()`: Extracts classes and functions from code
- `_find_related_paper_sections()`: Uses LLM to find relevant paper sections
- `_calculate_coverage_score()`: Computes traceability coverage

### 2. `readme_generator.py`
Generates comprehensive README files with paper references and guidance.

**Key Sections:**
- Paper metadata and abstract
- Repository structure
- Code-to-paper traceability mapping
- Missing information alerts
- Getting started guide
- Next steps recommendations

### 3. `missing_info_detector.py`
Identifies gaps between paper specifications and generated code.

**Detection Categories:**
- **Hyperparameters**: Learning rate, batch size, epochs, etc.
- **Dataset Information**: Missing dataset specifications
- **Implementation Details**: TODOs, placeholders, hardcoded paths
- **Hardware Requirements**: GPU/CPU specifications

### 4. `explanation_evaluator.py`
Evaluates the quality of explanations and traceability.

**Metrics:**
- Traceability coverage (0-1)
- Comment density (0-1)
- Paper reference accuracy (0-1)
- Missing information score (0-1, inverted)
- Readability score (0-1)
- Overall explainability score (weighted average)

### 5. `explainability_pipeline.py`
Orchestrates all explanation layer components.

**Pipeline Steps:**
1. Load paper JSON and generated code
2. Generate traceability map
3. Detect missing information
4. Generate comprehensive README
5. Evaluate explainability metrics
6. Save all outputs

## Usage

### Basic Usage

```bash
python explainability_pipeline.py \
    --paper_json ../outputs/Transformer.json \
    --code_dir ../outputs/Transformer_repo \
    --planning_artifacts ../outputs/Transformer/planning_trajectories.json \
    --output_dir ../outputs/Transformer/explanation_layer \
    --config ../outputs/Transformer_repo/config.yaml
```

### Individual Components

#### Generate Traceability Map
```python
from explanation_generator import ExplanationGenerator

generator = ExplanationGenerator()
traceability_map = generator.generate_traceability_map(
    paper_json, generated_files, planning_artifacts
)
```

#### Detect Missing Information
```python
from missing_info_detector import MissingInfoDetector

detector = MissingInfoDetector()
missing_info = detector.detect_missing_information(
    paper_content, config_data, code_files
)
```

#### Generate README
```python
from readme_generator import READMEGenerator

generator = READMEGenerator()
readme = generator.generate_readme(
    paper_metadata, code_structure, traceability_map, missing_info, config_data
)
```

#### Evaluate Explainability
```python
from explanation_evaluator import ExplanationEvaluator

evaluator = ExplanationEvaluator()
metrics = evaluator.evaluate_explainability(
    generated_code, traceability_map, missing_info, paper_sections
)
```

## Output Files

The pipeline generates the following files in the output directory:

1. **`traceability_map.json`** - Code-to-paper bidirectional mapping
2. **`README.md`** - Comprehensive documentation
3. **`missing_information.json`** - Detected gaps and missing information
4. **`explainability_metrics.json`** - Evaluation metrics
5. **`explainability_report.txt`** - Human-readable evaluation report

## Dependencies

```bash
pip install openai pyyaml
```

## Example Output

```
============================================================
EP2C EXPLANATION LAYER GENERATION
============================================================

Loading inputs...

Step 1: Generating paper-to-code traceability map...
Generating paper-to-code traceability map...

Step 2: Detecting missing information...
Step 3: Generating comprehensive README...
Step 4: Evaluating explainability metrics...

Saving explanation layer outputs...

============================================================
EXPLANATION LAYER GENERATION COMPLETE
============================================================

Overall Explainability Score: 72.50%
Files saved to: ../outputs/Transformer/explanation_layer
============================================================
```

## Integration with EP2C Pipeline

The explanation layer integrates seamlessly with the EP2C pipeline:

```
Paper → Planning → Analysis → Coding → Explanation Layer → UI
                         ↓
              Traceability Map
              Missing Info Report
              README.md
              Explainability Metrics
```

## Future Enhancements

- Interactive traceability UI with clickable links
- Visual diagrams mapping code to paper figures
- Equation-to-code mapping for mathematical formulas
- Automated testing to verify explanations
- Multi-modal explanations (text, figures, equations)

## License

Part of the EP2C project. See main project license.

