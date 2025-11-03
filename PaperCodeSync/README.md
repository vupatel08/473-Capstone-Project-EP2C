# PaperCodeSync (PCS)

Synchronization layer between the uploaded research paper and its generated code repoistory. It links textual sections, equations, and figures from the paper to specific symbols (functions, classes, files) in the codebase.

## Pipeline Overview
The PCS pipeline automates three major steps, now integrated through a single `driver.py`:

### Parse the Paper
We extract a structured representation of the uploaded paper using MinerU. MinerU parses the raw PDF and produces a cleaned Markdown (.md) file and auxiliary JSONs. Using this Markdown, PCS builds a unified chunks.json, which is a simplified, section-aware, math-preserving format compatible with our matcher. This produces `chunks.json`, containing section titles, paragraphs, equations (normalized LaTeX), and inline anchors.
```bash
python create_chunks.py path/to/paper.md path/to/output/chunks.json
```

### Parse the Code Repository
We analyze the generated code repository using `Tree-sitter` to extract all identifiable symbols: functions, classes, methods, and module spans, across all supported languages. This produces `symbols.json`, a structured list of code entities and their text spans.
```bash
python create_symbols.py path/to/repo/ --out path/to/output/symbols.json
```

### Generate the Alignment Map
PCS then matches each symbol to its most relevant sections in the paper using both textual overlap and semantic similarity:
- Overlap Scoring (`BM25` or `TF-IDF`): finds literal text matches.
- Semantic Scoring (Sentence Embeddings): captures conceptual meaning.
- Weighted Fusion: combines the two scores with a tunable `α` parameter (found in the config).
This generates `matches.jsonl`, mapping each symbol to its top-K most related paper chunks. This file is the synchronization key between the paper and its code. The PCS frontend will use it to power “click-to-jump” interactions (just like Overleaf's LaTeX editor). For example, clicking a variable or function name in the code jumps to the related equation or paragraph in the paper.
```bash
python create_map.py path/to/symbols.json path/to/chunks.json
```

## Automated Driver
All steps above are orchestrated by:
```bash
python driver.py
```
