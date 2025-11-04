# Using EP2C Without Frontend

## What Happens Without the Frontend?

The EP2C backend pipeline is **completely self-contained** and works perfectly without any frontend. The frontend was designed for visualization and interactive exploration, but all the core functionality is in the backend.

## What You Get

When you run the pipeline without the frontend, you still get:

### 1. **Generated Code Repository**
- Complete, runnable Python codebase
- All files organized in `{paper_name}_repo/` directory
- Ready to use and execute

### 2. **Comprehensive Documentation**
- **README.md** - Auto-generated documentation with:
  - Paper references and metadata
  - Code-to-paper traceability mapping
  - Missing information alerts
  - Getting started guide
  - Next steps recommendations

### 3. **Traceability Maps** (JSON files)
- `traceability_map.json` - Bidirectional mappings:
  - `code_to_paper`: Which paper sections each code component implements
  - `paper_to_code`: Which code implements each paper section
  - Coverage scores

### 4. **Missing Information Detection**
- `missing_information.json` - Lists all gaps:
  - Hyperparameters not in paper
  - Missing dataset info
  - Implementation placeholders
  - Severity levels (high/medium/low)

### 5. **Explainability Metrics**
- `explainability_metrics.json` - Quality scores:
  - Traceability coverage
  - Comment density
  - Paper reference accuracy
  - Overall explainability score

### 6. **Human-Readable Reports**
- `explainability_report.txt` - Summary report with:
  - Overall score
  - Detailed metrics
  - Recommendations for improvement

## How to Use the Outputs

### View Generated Code
```bash
cd outputs/paper2code/{paper_name}/{paper_name}_repo
ls -la
cat main.py
cat model.py
```

### Read the Documentation
```bash
cat outputs/paper2code/{paper_name}/explanation_layer/README.md
```

### Check Traceability
```bash
# View code-to-paper mappings
cat outputs/paper2code/{paper_name}/explanation_layer/traceability_map.json | jq '.code_to_paper'

# View paper-to-code mappings
cat outputs/paper2code/{paper_name}/explanation_layer/traceability_map.json | jq '.paper_to_code'
```

### Review Missing Information
```bash
cat outputs/paper2code/{paper_name}/explanation_layer/missing_information.json | jq '.'
```

### Check Metrics
```bash
cat outputs/paper2code/{paper_name}/explanation_layer/explainability_report.txt
```

### Run the Generated Code
```bash
cd outputs/paper2code/{paper_name}/{paper_name}_repo
python main.py
```

## What the Frontend Would Have Provided

The frontend (if it existed) would have provided:
- **Visual paper/code side-by-side viewer** - Interactive UI showing paper and code together
- **Clickable traceability links** - Click code to jump to paper sections, and vice versa
- **Interactive exploration** - Browse mappings visually instead of reading JSON

## Current Workflow Without Frontend

1. **Run Pipeline**:
   ```bash
   cd Backend
   python run_full_pipeline.py --paper_pdf paper.json --paper_name MyPaper
   ```

2. **Review Outputs**:
   - Read `explanation_layer/README.md` for overview
   - Check `traceability_map.json` for mappings
   - Review `missing_information.json` for gaps

3. **Use Generated Code**:
   - Navigate to `{paper_name}_repo/`
   - Review code files
   - Run the implementation
   - Modify as needed

4. **Understand Paper-Code Links**:
   - Use `traceability_map.json` to find which code implements which paper sections
   - Reference the README for high-level mappings
   - Check code comments (if added) for paper references

## Example: Finding What Implements a Paper Section

```bash
# Find code that implements "Section 3.2"
cat traceability_map.json | jq '.paper_to_code["Section 3.2"]'

# Output:
# [
#   {
#     "component": "model.py:Transformer",
#     "description": "Implements Transformer architecture",
#     "file": "model.py"
#   }
# ]
```

## Example: Finding Paper Sections for Code

```bash
# Find paper sections for a code component
cat traceability_map.json | jq '.code_to_paper["model.py:Transformer"]'

# Output:
# ["Section 3.2", "Section 3.3"]
```

## Summary

**The backend pipeline is complete and functional without any frontend.** All the core features work:
- ✅ Code generation
- ✅ Traceability mapping
- ✅ Missing info detection
- ✅ Documentation generation
- ✅ Explainability evaluation

The frontend would have been a nice-to-have for visualization, but **you can do everything you need using the generated files and JSON outputs.**

