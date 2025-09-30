# EP2C Backend

This is the backend of the EP2C (Explainable Paper-to-Code) project, responsible for processing academic papers, searching for related code repositories, and fine-tuning language models (LLMs) to generate executable code from papers.


## Backend Setup Instructions

This guide provides instructions for setting up the backend environment and running the various tools and models (including Research Tracker, MinerU, CodeBERT, and CodeLlama). Follow the steps below to get started.

### Set up a virtual environment

#### For macOS/Linux: 
```bash 
python3 -m venv .venv
source .venv/bin/activate
``` 
#### For Windows: 
```bash
python -m venv .venv 
.\.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Research Tracker (HuggingFace)
The Research Tracker is a tool that helps you scrape metadata from research papers and track related code and resources.
```bash
cd apis
python research_tracker_api.py
```

### Running the MinerU API
MinerU is a tool for scraping research papers and extracting metadata. It can process PDFs and return relevant information about research papers.

```bash
cd apis
python mineru_api.py
```

#### Running MinerU via CLI
```bash
mineru -p "path_to_your_paper.pdf" -o ./output
```
- `-p`: The path to the research paper (PDF).
- `-o`: The output directory where the results will be stored.

#### Problem with CUDA and PyTorch
While running MinerU, I encountered a CUDA compatibility issue because my GPU (NVIDIA GeForce MX230) does not support the latest CUDA versions required by PyTorch. Here's the warning I saw:
```bash
CUDA error: no kernel image is available for execution on the device
```

### Running the base model of CodeBERT
CodeBERT is a pre-trained model for code-related tasks, such as code completion, code generation, and more.
```bash
cd base
python codebert.py
```

### Running the base model of CodeLlama
CodeLlama is a code generation model that can be used for writing code in response to natural language prompts. Since CodeLlama is a private model, you need to authenticate with HuggingFace to access it. So you have to create a `.env` file in the project's root directory and securely store your HuggingFace token.
```bash
HUGGINGFACE_TOKEN=your_huggingface_token
```
After doing this, you can properly run CodeLlama:
```bash
cd base
python codellama.py
```