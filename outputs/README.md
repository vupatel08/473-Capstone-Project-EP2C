# EP2C Outputs Directory

This directory contains the results and outputs from different paper-to-code systems integrated into EP2C.

## Directory Structure

### `paper2code/`
Contains outputs from the Paper2Code system:
- **GAN/**: Complete pipeline artifacts for GAN paper processing
  - `planning_artifacts/`: Planning phase outputs
  - `analyzing_artifacts/`: Analysis phase outputs  
  - `coding_artifacts/`: Code generation artifacts
  - Various JSON files with trajectories and responses
- **GAN_repo/**: Generated Python codebase for GAN implementation
- **Transformer/**: Complete pipeline artifacts for Transformer paper processing
- **Transformer_repo/**: Generated Python codebase for Transformer implementation
- **results/**: Evaluation results from Paper2Code runs

### `autop2c/`
Contains outputs from the Automated Paper-to-Code system:
- **markdown_files/**: Processed markdown content from papers
- **result_1.jpg**, **result_2_new.jpg**: Result images from AutoP2C runs
- **process.png**: Process visualization diagram

## Usage

These outputs demonstrate the capabilities of different paper-to-code approaches:

1. **Paper2Code**: Uses a structured LLM-based approach with planning → analysis → coding phases
2. **AutoP2C**: Uses content processing → design → code generation with iterative fixing

Both systems produce complete, runnable Python codebases from research papers, showing different strategies for automated code generation from academic literature.

## Generated Codebases

- **Transformer Implementation**: Complete PyTorch implementation of the Transformer model for machine translation
- **GAN Implementation**: Complete PyTorch implementation of Generative Adversarial Networks

Each generated codebase includes:
- Dataset loading and preprocessing
- Model architecture implementation
- Training loops and optimization
- Evaluation metrics and testing
- Configuration files
- Documentation linking back to paper sections
