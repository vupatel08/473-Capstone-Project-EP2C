# EP2C — Explainable Paper-to-Code
**CMSC473 – Machine Learning Capstone Project**

EP2C converts research papers into executable code repositories with an integrated explanation layer that links code back to paper sections.

### What It Does
- **Generates Code**: Creates runnable Python repositories from academic papers
- **Explains Everything**: Links code components to paper sections, highlights missing information, and provides documentation
- **Interactive Viewer**: Side-by-side paper and code with clickable traceability




## System Pipeline
1. Paper Parsing – Extract metadata & structure (title, abstract, methods, equations, figures).
2. Dataset/Code Search – Query HuggingFace API for existing repos/datasets.
3. Planning - Turn the parsed paper into a clear plan for building the code before writing anything.
4. Analysis - Turn the planning outputs into detailed guidance for each file so the code generation is more accurate.
5. Code Generation - Write the actual code files that make the paper work following the plan and design we already made.
6. **Explanation Layer** – Add an explanation layer that links the generated code back to the paper so users can understand what was built and why.
7. UI Integration – Paper/code side-by-side with clickable traceability.
8. Exporting – Download as .zip




## Usage
Note, Python 3.11.9 or LESS is required.

### Clone the Repoistory
```bash
git clone https://github.com/vupatel08/473-Capstone-Project-EP2C
cd 473-Capstone-Project-EP2C
```


### Create a Virtual Environment and Activate it
#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```


### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


### Environment Variables
**Go into the Backend**
```bash
cd Backend
```
**Create a `.env` file with your OpenAI API key**
```bash
OPENAI_API_KEY=your_key_here
```
**Note:** We have a `.env.example` file you can just copy


### Either Run Web App Run or Run via CLI 

#### Web App
```bash
cd frontend
python app.py
```

#### CLI
```bash
cd Backend
python unified_pipeline.py \
    --paper_pdf path/to/paper.pdf \
    --paper_name PaperName \
    --gpt_version gpt-4o
```
**Note:** If using `o3-mini`, images from the paper will be automatically skipped as o3-mini doesn't support vision inputs.



## Special Thanks To...
We would like to thank the CMSC473 teaching staff for their guidance, as well as the authors of the AutoP2C (Lin et al., 2025) and Paper2Code (Seo et al., 2025), whose framework formed the backbone of our implementation.


