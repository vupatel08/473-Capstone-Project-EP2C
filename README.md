# 473-Capstone-Project-EP2C
Explainable Paper-to-Code
CMSC473 – Machine Learning Capstone Project


## Project Summary
Reproducibility has become one of the biggest challenges in machine learning research. Thousands of papers are published every year, but many lack complete code or enough details to replicate results. This slows progress and makes it difficult for students and researchers to build on prior work.

EP2C (Explainable Paper-to-Code) aims to close this gap by:
- Generating runnable code repositories directly from academic papers.
- Adding an explanation layer that links code back to paper sections, highlights missing datasets/hyperparameters, and provides next-step guidance.
- Offering new evaluation metrics that measure not only code correctness, but also explainability and usability.


## System Pipeline
1. Paper Parsing – Extract metadata & structure (title, abstract, methods, equations, figures).
2. Dataset/Code Search – Query HuggingFace API for existing repos/datasets.
3. System Architecture Creation – Build a repository blueprint.
4. Code Generation – Use fine-tuned LLMs (CodeLlama, StarCoder).
5. Iterative Evaluation – Debug, lint, resolve dependencies.
6. **Explanation Layer** – Generate README, traceability maps, missing info detection, and explainability evaluation.
7. UI Integration – Paper/code side-by-side with clickable traceability.
8. Exporting – Download as .zip or push to GitHub.

## Explanation Layer Features

The EP2C Explanation Layer provides comprehensive traceability and explainability:

### 🔗 **Paper-to-Code Traceability**
- Bidirectional mapping between code components and paper sections
- Direct links showing which code implements which paper sections
- Coverage score measuring how much of the paper is implemented

### 📚 **Comprehensive Documentation**
- Auto-generated README with paper references
- Code comments linking back to paper sections, equations, and figures
- Implementation rationale and design decisions

### ⚠️ **Missing Information Detection**
- Identifies hyperparameters not specified in paper
- Highlights missing dataset information
- Alerts for implementation gaps
- Provides suggestions for manual configuration

### 📊 **Explainability Evaluation**
- Traceability coverage metrics
- Comment density analysis
- Paper reference accuracy
- Overall explainability score
- Detailed recommendations for improvement

## Usage

### Generating Explanation Layer

```bash
cd Backend/explanation
python explainability_pipeline.py \
    --paper_json ../outputs/paper.json \
    --code_dir ../outputs/generated_repo \
    --planning_artifacts ../outputs/planning_artifacts.json \
    --output_dir ../outputs/explanation_layer \
    --config ../outputs/config.yaml
```

### Components

- **`explanation_generator.py`** - Creates bidirectional paper-code traceability maps
- **`readme_generator.py`** - Generates comprehensive README with links and documentation
- **`missing_info_detector.py`** - Detects missing information and configuration gaps
- **`explanation_evaluator.py`** - Evaluates explainability quality metrics
- **`explainability_pipeline.py`** - Orchestrates the complete explanation layer generation


## Project Timeline
- Month 1: Initial setup + MVP (Flask prototype, pipeline design, dataset collection).
- Month 2: Complete MVP + evaluation (UI integration, debugging, testing).
- Final Weeks: Case studies, benchmarking, polish UI, and deliverables.
- Planned completion: December 2025
