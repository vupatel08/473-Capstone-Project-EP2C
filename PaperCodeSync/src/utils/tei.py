from lxml import etree as ET
from typing import Optional, List, Dict

TEI_NS: Dict[str, str] = {"tei": "http://www.tei-c.org/ns/1.0"}

def text_of(el: Optional[ET._Element]) -> str:
    if el is None: return ""
    return "".join(el.itertext()).strip()

def first(el: ET._Element, xpath: str) -> Optional[ET._Element]:
    arr = el.xpath(xpath, namespaces=TEI_NS) if hasattr(el, "xpath") else el.findall(xpath)
    return arr[0] if arr else None

def all_nodes(el: ET._Element, xpath: str) -> List[ET._Element]:
    return el.xpath(xpath, namespaces=TEI_NS) if hasattr(el, "xpath") else el.findall(xpath)

def get_attr(el: Optional[ET._Element], name: str) -> Optional[str]:
    if el is None: return None
    if name == "xml:id":
        return el.get("{http://www.w3.org/XML/1998/namespace}id")
    return el.get(name)
