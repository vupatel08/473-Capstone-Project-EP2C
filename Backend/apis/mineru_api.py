import subprocess
import json
import os

def scrape_paper(paper_path, output_dir='./'):
    # Escape the path for spaces and special characters
    escaped_paper_path = paper_path.replace(" ", "\ ")

    # Run the mineru scrape command and capture the output
    result = subprocess.run(
        ["mineru", "-p", escaped_paper_path, "-o", output_dir],
        capture_output=True,
        text=True
    )

    # Check if there's any output at all
    if not result.stdout.strip():
        print("No output from MinerU. Please check the paper path or MinerU setup.")
        return None

    # Print the raw output for debugging
    print("MinerU Output:", result.stdout)

    # Try parsing the JSON output
    try:
        # Assuming MinerU outputs JSON; if it's a file, you can load it here
        json_output_file = os.path.join(output_dir, "metadata.json")
        if os.path.exists(json_output_file):
            with open(json_output_file, "r") as f:
                paper_metadata = json.load(f)
            return paper_metadata
        else:
            print("Metadata JSON file not found in output directory.")
            return None
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw output:", result.stdout)
        return None

paper_path = "../../Paper Repo/ImplicitAVE An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction.pdf"
metadata = scrape_paper(paper_path)

if metadata:
    print("Paper Metadata:", metadata)
else:
    print("Failed to scrape or parse paper.")
