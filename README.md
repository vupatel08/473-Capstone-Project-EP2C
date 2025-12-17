# EP2C: Explainable Paper-to-Code

EP2C converts research papers into executable code repositories with an integrated explanation layer that links code back to paper sections.

## What It Does

- **Generates Code**: Creates runnable Python repositories from academic papers
- **Explains Everything**: Links code components to paper sections, highlights missing information, and provides documentation
- **Interactive Viewer**: Side-by-side paper and code with clickable traceability

## Quick Start

cd Backend
python unified_pipeline.py \
    --paper_pdf path/to/paper.pdf \
    --paper_name PaperName \
    --gpt_version gpt-4o
**Note:** If using `o3-mini`, images from the paper will be automatically skipped as o3-mini doesn't support vision inputs.

## Key Features

- **Paper-to-Code Traceability**: See which code implements which paper sections
- **Web Interface**: Upload papers and explore generated code interactively

## Pipeline

1. **Parsing** - Extract paper content using MinerU
2. **Planning** - Design repository structure and architecture
3. **Analysis** - Define file-level logic and responsibilities
4. **Coding** - Generate executable code files
5. **Explanation** - Create traceability maps, documentation, and metrics

## Code

Available at: https://github.com/vupatel08/473-Capstone-Project-EP2C
