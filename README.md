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
6. Explanation Layer – Generate README, comments, and highlight links.
7. UI Integration – Paper/code side-by-side with clickable traceability.
8. Exporting – Download as .zip or push to GitHub.


## Project Timeline
- Month 1: Initial setup + MVP (Flask prototype, pipeline design, dataset collection).
- Month 2: Complete MVP + evaluation (UI integration, debugging, testing).
- Final Weeks: Case studies, benchmarking, polish UI, and deliverables.
- Planned completion: December 2025
