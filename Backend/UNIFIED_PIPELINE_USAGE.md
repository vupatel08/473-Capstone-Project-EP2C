# EP2C Unified Pipeline - Quick Start

## How to Run

```bash
cd Backend
python unified_pipeline.py \
    --paper_pdf path/to/paper.pdf \
    --paper_name PaperName \
    --gpt_version o3-mini
```

## Prerequisites

1. Set OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Arguments

- `--paper_pdf` (required): Path to PDF, JSON, or markdown file
- `--paper_name` (optional): Paper identifier (defaults to filename)
- `--gpt_version` (optional): Model version (default: `o3-mini`)

## Example

```bash
python unified_pipeline.py \
    --paper_pdf main.pdf \
    --paper_name cryptocurrency_trading_mlp \
    --gpt_version o3-mini
```

## Output

Generated code and documentation will be in:
```
Backend/example_driver/outputs/paper2code/{paper_name}/
```

That's it. The pipeline automatically:
- Parses PDFs
- Generates code
- Creates documentation
- Builds explanation layer
