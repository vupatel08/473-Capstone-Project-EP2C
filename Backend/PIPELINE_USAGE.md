# EP2C Full Pipeline Usage Guide

## Overview

The EP2C pipeline has been integrated with the explanation layer. You can now run the complete pipeline from paper parsing to code generation with explainability features.

## Quick Start

### Prerequisites

1. **OpenAI API Key**: Set environment variable
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Dependencies**: Install required packages
   ```bash
   cd Backend
   pip install -r requirements.txt
   ```

### Running the Full Pipeline

The easiest way to run the complete pipeline is using the orchestrator script:

```bash
cd Backend
python run_full_pipeline.py \
    --paper_pdf path/to/paper.json \
    --paper_name Transformer \
    --gpt_version o3-mini \
    --paper_format JSON \
    --output_dir outputs
```

### Arguments

- `--paper_pdf` (required): Path to paper file (JSON or LaTeX)
- `--paper_name` (required): Identifier for the paper (e.g., "Transformer", "GAN")
- `--gpt_version` (optional): GPT model version (default: "o3-mini")
- `--paper_format` (optional): "JSON" or "LaTeX" (default: "JSON")
- `--output_dir` (optional): Base output directory (default: "outputs")

## Pipeline Phases

The pipeline runs in 3 main phases:

### 1. Planning Phase (`1_planning.py`)
- Creates overall plan
- Designs system architecture
- Generates logic design
- Creates `config.yaml`

### 2. Analysis Phase (`2_analyzing.py`)
- Performs detailed logic analysis for each file
- Generates analysis artifacts

### 3. Coding Phase (`3_coding.py`)
- Generates code files
- **Automatically runs explanation layer** after code generation
  - Creates traceability maps
  - Detects missing information
  - Generates comprehensive README
  - Evaluates explainability metrics

## Output Structure

After running the pipeline, you'll get:

```
outputs/
└── paper2code/
    └── {paper_name}/
        ├── planning_artifacts/          # Planning phase outputs
        ├── analyzing_artifacts/          # Analysis phase outputs
        ├── coding_artifacts/             # Code generation artifacts
        ├── explanation_layer/           # ✨ NEW: Explanation layer outputs
        │   ├── traceability_map.json    # Code-to-paper mappings
        │   ├── README.md                 # Comprehensive documentation
        │   ├── missing_information.json # Missing info alerts
        │   ├── explainability_metrics.json # Quality metrics
        │   └── explainability_report.txt # Human-readable report
        ├── planning_trajectories.json   # Planning conversation history
        ├── planning_config.yaml         # Configuration file
        └── {paper_name}_repo/           # Generated code repository
            ├── main.py
            ├── model.py
            ├── trainer.py
            └── ...
```

## Running Individual Phases

If you need to run phases separately:

### Planning Only
```bash
python models/1_planning.py \
    --paper_name Transformer \
    --pdf_json_path paper.json \
    --output_dir outputs/paper2code/Transformer
```

### Analysis Only
```bash
python models/2_analyzing.py \
    --paper_name Transformer \
    --pdf_json_path paper.json \
    --output_dir outputs/paper2code/Transformer
```

### Coding (with Explanation Layer)
```bash
python models/3_coding.py \
    --paper_name Transformer \
    --pdf_json_path paper.json \
    --output_dir outputs/paper2code/Transformer \
    --output_repo_dir outputs/paper2code/Transformer/Transformer_repo
```

## Explanation Layer Features

The explanation layer (automatically generated in phase 3) provides:

1. **Traceability Maps**: Links code components to paper sections
2. **Missing Information Detection**: Identifies gaps in paper specifications
3. **Comprehensive README**: Auto-generated documentation with paper references
4. **Explainability Metrics**: Quality scores for code explainability

## Troubleshooting

### Explanation Layer Fails

If the explanation layer fails, the pipeline will continue without it. Check:
- OpenAI API key is set correctly
- Paper JSON format is valid
- Generated code directory exists
- Planning artifacts are available

### Import Errors

If you see import errors:
```bash
# Make sure you're in the Backend directory
cd Backend

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Path Issues

Always use absolute paths or ensure you're running from the correct directory:
```bash
# From project root
cd Backend
python run_full_pipeline.py --paper_pdf ../path/to/paper.json ...
```

## Example Workflow

```bash
# 1. Set API key
export OPENAI_API_KEY=sk-...

# 2. Navigate to Backend
cd Backend

# 3. Run full pipeline
python run_full_pipeline.py \
    --paper_pdf ../outputs/paper.json \
    --paper_name MyPaper \
    --gpt_version o3-mini

# 4. Check outputs
ls -la outputs/paper2code/MyPaper/explanation_layer/
cat outputs/paper2code/MyPaper/explanation_layer/README.md
```

## Next Steps

After generating code and explanation layer:
1. Review the generated README in `explanation_layer/README.md`
2. Check missing information alerts
3. Review traceability maps to understand code-paper links
4. Use the generated code repository directly

**Note:** The frontend is not required. All outputs are available as files:
- README.md for documentation
- JSON files for traceability maps
- Generated code repository for execution

See `NO_FRONTEND_GUIDE.md` for details on using outputs without frontend.

