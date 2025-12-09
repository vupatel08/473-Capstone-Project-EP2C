from openai import OpenAI
from openai import RateLimitError
import json
from tqdm import tqdm
import argparse
import os
import sys
import base64
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
backend_dir = Path(__file__).parent.parent.resolve()
# Add utils to path for imports
sys.path.insert(0, str(backend_dir))
from utils.papercoder_utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost
project_root = backend_dir.parent
env_paths = [backend_dir / ".env", project_root / ".env"]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str)
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")
parser.add_argument('--parse_output_dir', type=str, default=None, help="Path to parse_output directory containing content_list.json files")

args    = parser.parse_args()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables or .env file")
    print("   Please create a .env file or set OPENAI_API_KEY environment variable")
    sys.exit(1)

client = OpenAI(api_key=api_key)

"""
paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir


if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)
"""

from parsing.parser import codegen_prep

# Extract variables from args
paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = Path(args.output_dir) if args.output_dir else Path("")

# Try to access content_list.json from parsed paper using codegen_prep
paper_content = ""  # For fallback: plain text content
paper_content_items = []  # Structured content with text and images for vision API
parse_output_dir = None
doc_path = None

def encode_image_to_base64(image_path: Path, max_size_mb: float = 20.0) -> str:
    """
    Encode an image file to base64 data URL format for OpenAI API.
    
    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB (default 20MB, OpenAI's limit is ~20MB per image)
    
    Returns:
        Base64-encoded data URL string, or None if encoding fails
    """
    try:
        # Check file size first
        file_size = image_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            print(f"⚠️  Warning: Image {image_path.name} is {file_size_mb:.2f}MB (exceeds {max_size_mb}MB limit). Skipping.")
            return None
        
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image format from file extension
            ext = image_path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/png')  # Default to PNG if unknown
            
            return f"data:{mime_type};base64,{base64_encoded}"
    except FileNotFoundError:
        print(f"⚠️  Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Could not encode image {image_path}: {e}")
        return None

# Determine parse output directory
if args.parse_output_dir:
    # Use explicitly provided parse_output_dir
    parse_output_dir = Path(args.parse_output_dir)
else:
    # Try to find parse_output directory automatically - try common locations
    # Typically parse_output is at work_root/parse_output, where work_root might be example_driver
    possible_parse_dirs = [
        output_dir.parent.parent / "parse_output",  # output_dir/../parse_output
        output_dir.parent / "parse_output",  # output_dir/../parse_output (alternative)
        backend_dir / "example_driver" / "parse_output",  # default location
        backend_dir.parent / "Backend" / "example_driver" / "parse_output",  # alternative structure
    ]

    for parse_dir in possible_parse_dirs:
        if parse_dir.exists():
            parse_output_dir = parse_dir
            break

    # If not found, try to infer from output_dir structure
    if parse_output_dir is None and output_dir:
        # Try going up from output_dir to find parse_output
        current = output_dir.parent
        for _ in range(3):  # Go up max 3 levels
            candidate = current / "parse_output"
            if candidate.exists():
                parse_output_dir = candidate
                break
            current = current.parent

# Get the original document path
if paper_format == "JSON" and pdf_json_path:
    doc_path = Path(pdf_json_path)
elif paper_format == "LaTeX" and pdf_latex_path:
    doc_path = Path(pdf_latex_path)
elif pdf_json_path:  # Fallback
    doc_path = Path(pdf_json_path)
elif pdf_latex_path:  # Fallback
    doc_path = Path(pdf_latex_path)

# Try to use codegen_prep if we have both doc_path and parse_output_dir
if doc_path and parse_output_dir and parse_output_dir.exists():
    try:
        # Check if content_list.json exists for this document
        doc_stem = doc_path.stem
        content_list_path = parse_output_dir / doc_stem / "auto" / f"{doc_stem}_content_list.json"
        
        # Also try with paper_name if different
        if not content_list_path.exists() and paper_name:
            content_list_path = parse_output_dir / paper_name / "auto" / f"{paper_name}_content_list.json"
        
        if content_list_path.exists():
            # Use codegen_prep to get formatted content
            prep_result = codegen_prep([doc_path], parse_output_dir)
            
            if prep_result and len(prep_result) > 0:
                # Format the content for the prompt
                # codegen_prep returns a list with content items (text and images)
                content_parts = []  # For fallback text-only format
                # Use the outer paper_content_items variable
                paper_content_items.clear()  # Clear any existing items
                
                for doc_data in prep_result:
                    for item in doc_data.get("content", []):
                        if item.get("type") == "text":
                            # Use "text" or "content" key depending on what's available
                            text = item.get("text") or item.get("content", "")
                            # Ensure text is a string, not a list or other type
                            if isinstance(text, list):
                                text = "\n".join(str(t) for t in text)
                            elif not isinstance(text, str):
                                text = str(text) if text else ""
                            
                            if text:
                                content_parts.append(text)
                                # Add text to structured content
                                paper_content_items.append({
                                    "type": "text",
                                    "text": text
                                })
                        elif item.get("type") == "image":
                            # Handle image files
                            image_path = item.get("path")
                            if image_path:
                                image_path_obj = Path(image_path) if not isinstance(image_path, Path) else image_path
                                if image_path_obj.exists():
                                    base64_image = encode_image_to_base64(image_path_obj)
                                    if base64_image:
                                        paper_content_items.append({
                                            "type": "image_url",
                                            "image_url": {"url": base64_image}
                                        })
                                        print(f"✓ Added image: {image_path_obj.name}")
                                    else:
                                        print(f"⚠️  Warning: Could not encode image {image_path_obj}")
                                else:
                                    print(f"⚠️  Warning: Image path does not exist: {image_path_obj}")
                            else:
                                print(f"⚠️  Warning: Image path is None or missing")
                
                # Set both formats
                paper_content = "\n".join(content_parts)  # Fallback text-only
                num_images = sum(1 for item in paper_content_items if item.get("type") == "image_url")
                print(f"✓ Loaded paper content from parsed content_list.json ({len(paper_content_items)} items, {num_images} images)")
            else:
                print(f"⚠️  Warning: codegen_prep returned empty result, falling back to direct file read")
        else:
            print(f"⚠️  Warning: content_list.json not found at {content_list_path}, falling back to direct file read")
    except Exception as e:
        print(f"⚠️  Warning: Error using codegen_prep: {e}, falling back to direct file read")
        # Clear any partially populated data
        paper_content_items.clear()
        paper_content = ""

# Fallback: read paper content directly if codegen_prep wasn't used
if not paper_content and not paper_content_items:
    if paper_format == "JSON" and pdf_json_path:
        with open(pdf_json_path, 'r', encoding='utf-8') as f:
            paper_content = json.dumps(json.load(f), ensure_ascii=False, indent=2)
            # Convert to structured format for consistency
            paper_content_items = [{"type": "text", "text": paper_content}]
    elif paper_format == "LaTeX" and pdf_latex_path:
        with open(pdf_latex_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()
            # Convert to structured format for consistency
            paper_content_items = [{"type": "text", "text": paper_content}]
    else:
        print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX'.")
        sys.exit(1)

# If we have structured content items but no plain text, create it
if paper_content_items and not paper_content:
    text_parts = []
    for item in paper_content_items:
        if item.get("type") == "text":
            # Handle both "text" and "content" keys (parser.py uses "content" for final string)
            text = item.get("text") or item.get("content", "")
            # Ensure text is a string, not a list
            if isinstance(text, str):
                text_parts.append(text)
            elif isinstance(text, list):
                # If it's a list, join it
                text_parts.append("\n".join(str(t) for t in text))
            else:
                # Convert to string if it's something else
                text_parts.append(str(text))
    paper_content = "\n".join(text_parts) if text_parts else ""

# Build message content with images if available
def build_message_content_with_images(task_text: str, paper_content_items: list = None) -> list:
    """
    Build message content that can include both text and images.
    If paper_content_items contains images, use structured format with images embedded.
    Otherwise, use plain text format.
    """
    has_images = paper_content_items and any(item.get("type") == "image_url" for item in paper_content_items)
    
    if has_images:
        # Use structured format with images
        message_content = []
        
        # Add task header
        message_content.append({
            "type": "text",
            "text": "## Paper\n\n"
        })
        
        # Add paper content items (text and images) in order
        for item in paper_content_items:
            if item.get("type") == "image_url":
                message_content.append(item)
            elif item.get("type") == "text":
                # Merge consecutive text items
                if message_content and message_content[-1].get("type") == "text":
                    message_content[-1]["text"] += "\n" + item.get("text", "")
                else:
                    message_content.append(item)
        
        # Add task instructions
        if message_content and message_content[-1].get("type") == "text":
            message_content[-1]["text"] += "\n\n" + task_text
        else:
            message_content.append({
                "type": "text",
                "text": task_text
            })
        
        return message_content
    else:
        # Use plain text format (backward compatible)
        return f"""## Paper
{paper_content}

{task_text}"""

# Build the task text
task_text = """## Task
1. We want to reproduce the method described in the attached paper. 
2. The authors did not release any official code, so we have to plan our own implementation.
3. Before writing any Python code, please outline a comprehensive plan that covers:
   - Key details from the paper's **Methodology**.
   - Important aspects of **Experiments**, including dataset requirements, experimental settings, hyperparameters, or evaluation metrics.
4. The plan should be as **detailed and informative** as possible to help us write the final code later.

## Requirements
- You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.
- If something is unclear from the paper, mention it explicitly.

## Instruction
The response should give us a strong roadmap, making it easier to write the code later."""

# Build message content (with images if available)
user_content = build_message_content_with_images(task_text, paper_content_items)

plan_msg = [
        {'role': "system", "content": f"""You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 
You will receive a research paper in read-order. 
Your task is to create a detailed and efficient plan to reproduce the experiments and methodologies described in the paper.
This plan should align precisely with the paper's methodology, experimental setup, and evaluation metrics. 

Instructions:

1. Align with the Paper: Your plan must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present the plan in a well-organized and easy-to-follow format, breaking it down into actionable steps.
3. Prioritize Efficiency: Optimize the plan for clarity and practical implementation while ensuring fidelity to the original experiments."""},
        {"role": "user", "content": user_content}]

file_list_msg = [
        {"role": "user", "content": """Your goal is to create a concise, usable, and complete software system design for reproducing the paper's method. Use appropriate open-source libraries and keep the overall architecture simple.
             
Based on the plan for reproducing the paper’s main method, please design a concise, usable, and complete software system. 
Keep the architecture simple and make effective use of open-source libraries.

-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will ... ,
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "model.py",  
        "trainer.py",
        "evaluation.py" 
    ],
    "Data structures and interfaces": "\nclassDiagram\n    class Main {\n        +__init__()\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Any\n    }\n    class Model {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, data: Any)\n        +train() -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, data: Any)\n        +evaluate() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n",
    "Program call flow": "\nsequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    DL-->>M: return dataset\n    M->>MD: initialize model()\n    M->>TR: train(model, dataset)\n    TR->>MD: forward(x)\n    MD-->>TR: predictions\n    TR-->>M: training complete\n    M->>EV: evaluate(model, dataset)\n    EV->>MD: forward(x)\n    MD-->>EV: predictions\n    EV-->>M: metrics\n",
    "Anything UNCLEAR": "Need clarification on the exact dataset format and any specialized hyperparameters."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Implementation approach: <class 'str'>  # Summarize the chosen solution strategy.
- File list: typing.List[str]  # Only need relative paths. ALWAYS write a main.py or app.py here.
- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax, including classes, method(__init__ etc.) and functions with type annotations, CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design.
- Program call flow: typing.Optional[str] # Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.
- Anything UNCLEAR: <class 'str'>  # Mention ambiguities and ask for clarifications.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the instructions for the nodes, generate the output, and ensure it follows the format example."""}
    ]

task_list_msg = [
        {'role': 'user', 'content': """Your goal is break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies. 
You will break down tasks, analyze dependencies.
             
You outline a clear PRD/technical design for reproducing the paper’s method and experiments. 

Now, let's break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist in writing the code needed to reproduce the paper.

-----

## Format Example
[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0"  
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "data_preprocessing.py",
            "DataPreprocessing class ........"
        ],
        [
            "trainer.py",
            "Trainer ....... "
        ],
        [
            "dataset_loader.py",
            "Handles loading and ........"
        ],
        [
            "model.py",
            "Defines the model ......."
        ],
        [
            "evaluation.py",
            "Evaluation class ........ "
        ],
        [
            "main.py",
            "Entry point  ......."
        ]
    ],
    "Task list": [
        "dataset_loader.py", 
        "model.py",  
        "trainer.py", 
        "evaluation.py",
        "main.py"  
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both data_preprocessing.py and trainer.py share ........",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}

[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format.(e.g., 'numpy==1.21.0').
- Required Other language third-party packages: typing.List[str]  # List down packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible.
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list.
- Full API spec: <class 'str'>  # Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end and back-end communication is not required, leave it blank.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common utility functions or configuration variables.
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions or clarifications needed from the paper or project scope.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example."""}]

# config
config_msg = [
        {'role': 'user', 'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, plan, design specified previously, follow the "Format Example" and generate the code. 
Extract the training details from the above paper (e.g., learning rate, batch size, epochs, etc.), follow the "Format example" and generate the code. 
DO NOT FABRICATE DETAILS — only use what the paper provides.

You must write `config.yaml`.

ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Your output format must follow the example below exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: ...
  batch_size: ...
  epochs: ...
...
```

-----

## Code: config.yaml
"""
    }]

def api_call(msg, gpt_version, max_retries=5, base_delay=1.0):
    """
    Make an API call with automatic retry on rate limit errors.
    
    Args:
        msg: Messages to send to the API
        gpt_version: GPT model version to use
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        API completion response
    """
    for attempt in range(max_retries):
        try:
            if "o3-mini" in gpt_version:
                completion = client.chat.completions.create(
                    model=gpt_version, 
                    reasoning_effort="high",
                    messages=msg
                )
            else:
                completion = client.chat.completions.create(
                    model=gpt_version, 
                    messages=msg
                )
            return completion
        except RateLimitError as e:
            if attempt == max_retries - 1:
                # Last attempt failed, raise the error
                raise
            
            # Try to extract retry-after time from error message
            error_message = str(e)
            retry_after = None
            
            # Look for "try again in X.XXXs" pattern in error message
            match = re.search(r'try again in ([\d.]+)s', error_message, re.IGNORECASE)
            if match:
                retry_after = float(match.group(1))
            
            # If no specific retry-after found, use exponential backoff
            if retry_after is None:
                retry_after = base_delay * (2 ** attempt)
            
            # Add a small jitter to avoid thundering herd
            jitter = retry_after * 0.1 * (0.5 + (hash(str(msg)) % 100) / 100)
            wait_time = retry_after + jitter
            
            print(f"⚠️  Rate limit reached (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.2f}s before retry...")
            time.sleep(wait_time)
        except Exception as e:
            # For other errors, raise immediately
            raise
    
    # Should never reach here, but just in case
    raise Exception("Failed to make API call after all retries") 

responses = []
trajectories = []
total_accumulated_cost = 0

for idx, instruction_msg in enumerate([plan_msg, file_list_msg, task_list_msg, config_msg]):
    current_stage = ""
    if idx == 0 :
        current_stage = f"[Planning] Overall plan"
    elif idx == 1:
        current_stage = f"[Planning] Architecture design"
    elif idx == 2:
        current_stage = f"[Planning] Logic design"
    elif idx == 3:
        current_stage = f"[Planning] Configuration file generation"
    print(current_stage)

    trajectories.extend(instruction_msg)

    completion = api_call(trajectories, gpt_version)
    
    # response
    completion_json = json.loads(completion.model_dump_json())

    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    responses.append(completion_json)

    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})


# save
save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/planning_response.json', 'w') as f:
    json.dump(responses, f)

with open(f'{output_dir}/planning_trajectories.json', 'w') as f:
    json.dump(trajectories, f)

# Extract and save config.yaml from the last response (config generation)
# The config is in the 4th response (index 3)
if len(responses) >= 4:
    config_response = responses[3]
    config_content = config_response['choices'][0]['message']['content']
    
    # Extract YAML from markdown code blocks
    import re
    # Try to extract YAML from ```yaml ... ``` blocks
    yaml_pattern = r'```(?:yaml)?\s*\n(.*?)```'
    yaml_match = re.search(yaml_pattern, config_content, re.DOTALL)
    
    if yaml_match:
        config_yaml = yaml_match.group(1).strip()
        # Remove the "## config.yaml" header if present
        config_yaml = re.sub(r'^##\s*config\.yaml\s*\n', '', config_yaml, flags=re.MULTILINE)
    else:
        # Fallback: try to extract anything between ``` markers
        code_block_pattern = r'```[^\n]*\n(.*?)```'
        code_match = re.search(code_block_pattern, config_content, re.DOTALL)
        if code_match:
            config_yaml = code_match.group(1).strip()
        else:
            # Last resort: use the entire content (might contain extra text)
            config_yaml = config_content
    
    # Save config.yaml
    with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_yaml)
    print(f"✅ Saved planning_config.yaml to {output_dir}/planning_config.yaml")
else:
    print("⚠️  Warning: Config response not found, planning_config.yaml not created")

# Generate PLANNING.md for frontend display
print("\n📝 Generating PLANNING.md...")
planning_md = "# Planning Phase\n\n"
planning_md += "This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.\n\n"

# Add Overall Plan (response 0)
if len(responses) > 0:
    planning_md += "## 1. Overall Plan\n\n"
    plan_content = responses[0]['choices'][0]['message']['content']
    planning_md += plan_content + "\n\n"

# Add Architecture Design (response 1)
if len(responses) > 1:
    planning_md += "## 2. Architecture Design\n\n"
    design_content = responses[1]['choices'][0]['message']['content']
    planning_md += design_content + "\n\n"

# Add Logic Design (response 2)
if len(responses) > 2:
    planning_md += "## 3. Logic Design & Task List\n\n"
    task_content = responses[2]['choices'][0]['message']['content']
    planning_md += task_content + "\n\n"

# Add Config Summary (response 3)
if len(responses) > 3:
    planning_md += "## 4. Configuration\n\n"
    config_content = responses[3]['choices'][0]['message']['content']
    planning_md += config_content + "\n\n"
    planning_md += "---\n\n"
    planning_md += "**Note:** Full configuration is available in `planning_config.yaml`\n"

# Save PLANNING.md
with open(f'{output_dir}/PLANNING.md', 'w', encoding='utf-8') as f:
    f.write(planning_md)
print(f"✅ Saved PLANNING.md to {output_dir}/PLANNING.md")
