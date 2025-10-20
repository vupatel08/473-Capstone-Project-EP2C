# Parse given research paper into TEI XML using GROBID and return sections as such:
# [{id, heading_path, page_start, page_end, text}]


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


# okay so this sort of worked? not really sure how to get the sections out of this
# need to look at the docling docs more closely or just switch to GROBID directly