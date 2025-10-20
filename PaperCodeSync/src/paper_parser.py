# Parse given research paper into TEI XML using GROBID and return sections as such:
# [{id, heading_path, page_start, page_end, text}]


# let's try out Docling first 
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2206.01062" 
converter = DocumentConverter()
doc = converter.convert(source).document
print(doc.export_to_markdown())


# this outputs an .md file, but not really what we want.