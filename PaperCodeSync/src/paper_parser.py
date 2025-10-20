# Parse given research paper into TEI XML using GROBID and return sections as such:
# [{id, heading_path, page_start, page_end, text}]


# Okay so switching to GROBID was a bit of a pain
# First I had to install Docker Desktop on my Windows machine
# Then I had to confirm that it was working for WSL2
# Next I had to pull the GROBID Docker image and run it:
# docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-crf
# After that I had to open a new CLI and run:
# curl -sS -X POST \
#   -F "input=@ExampleResearchPaper.pdf;type=application/pdf" \
#   http://localhost:8070/api/processFulltextDocument \
#   -o ExampleResearchPaper.tei.xml

# IT WORKED!


### Parse research paper into structured JSON using Docling
""" 
from docling.document_converter import DocumentConverter
import json
import sys
from pathlib import Path

def parse_paper(source_path: str):
    converter = DocumentConverter()
    result = converter.convert(source_path)
    doc = result.document

    # let's serialize the full structured document
    data = doc.model_dump(mode="json")  # Pydantic model into a dict
    out_path = Path(source_path).stem + "_docling.json"

    # now we can just save this as formatted JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Parsed successfully. JSON saved to: {out_path}")


parse_paper("https://arxiv.org/pdf/2206.01062")
"""

