# Setup Requirements:
# -----------------
# 1. GROBID server running in Docker:
#    ```
#    docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-crf
#    ```

# 2. Convert PDF to TEI XML using GROBID:
#    ```
#    curl -sS -X POST \
#      -F "input=@Paper.pdf;type=application/pdf" \
#      http://localhost:8070/api/processFulltextDocument \
#      -o Paper.tei.xml
#    ```


import json
import os
import re
import hashlib
from lxml import etree as ET

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# make a simple url-ish id from text
def slugify(text, maxlen=80):
    if not text:
        return ""
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    s = s[:maxlen]
    if s == "":
        s = "untitled"
    return s

# get all text from an xml element
def text_of(el):
    if el is None:
        return ""
    t = []
    for piece in el.itertext():
        t.append(piece)
    return "".join(t).strip()

# make a short deterministic id from parts
def sha1_id(*parts, prefix=""):
    h = hashlib.sha1()
    for p in parts:
        if p:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
    out = h.hexdigest()[:12]
    if prefix:
        return prefix + out
    return out

# read an attribute safely (handles xml:id)
def get_attr(el, name):
    if el is None:
        return None
    if name == "xml:id":
        return el.get("{http://www.w3.org/XML/1998/namespace}id")
    return el.get(name)

# return first node matching xpath
def first(el, xpath):
    arr = el.xpath(xpath, namespaces=TEI_NS) if hasattr(el, "xpath") else el.findall(xpath)
    if arr:
        return arr[0]
    return None

# return all nodes matching xpath
def all_nodes(el, xpath):
    if hasattr(el, "xpath"):
        return el.xpath(xpath, namespaces=TEI_NS)
    return el.findall(xpath)

# ensure element has an id or create one
def ensure_id(el, fallback_parts, prefix):
    existing = get_attr(el, "xml:id")
    if not existing:
        existing = el.get("id")
    if existing:
        return existing
    joined = []
    for p in fallback_parts:
        if p:
            joined.append(p)
    joined = "||".join(joined)
    return prefix + "_" + sha1_id(joined)

# pull paper metadata out of tei
def parse_metadata(root):
    title_el = first(root, ".//tei:teiHeader//tei:titleStmt/tei:title")
    title = text_of(title_el)

    authors = []
    auth_paths = [
        ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:author",
        ".//tei:teiHeader//tei:titleStmt//tei:author",
    ]
    auth_nodes = []
    for path in auth_paths:
        for a in all_nodes(root, path):
            auth_nodes.append(a)
    for a in auth_nodes:
        forename = text_of(first(a, ".//tei:persName/tei:forename"))
        surname = text_of(first(a, ".//tei:persName/tei:surname"))
        full = (forename + " " + surname).strip()
        if full == "":
            full = text_of(a)
        email = text_of(first(a, ".//tei:email"))
        aff = text_of(first(a, ".//tei:affiliation"))
        if aff == "":
            aff = text_of(first(a, ".//tei:orgName"))
        authors.append({
            "name": full if full else None,
            "forename": forename if forename else None,
            "surname": surname if surname else None,
            "email": email if email else None,
            "affiliation": aff if aff else None
        })

    date_el = first(root, ".//tei:teiHeader//tei:publicationStmt//tei:date")
    date = None
    if date_el is not None:
        date = get_attr(date_el, "when")
        if not date:
            date = text_of(date_el)

    doi = text_of(first(root, ".//tei:teiHeader//tei:idno[@type='DOI']"))
    if doi == "":
        doi = text_of(first(root, ".//tei:teiHeader//tei:idno[@type='doi']"))

    journal = text_of(first(root, ".//tei:teiHeader//tei:monogr/tei:title"))
    if journal == "":
        journal = text_of(first(root, ".//tei:teiHeader//tei:series/tei:title"))

    abs_ps = all_nodes(root, ".//tei:profileDesc//tei:abstract//tei:p")
    if not abs_ps:
        abs_ps = all_nodes(root, ".//tei:abstract//tei:p")
    abs_txt_parts = []
    for p in abs_ps:
        abs_txt_parts.append(text_of(p))
    abstract_text = " ".join(abs_txt_parts) if abs_txt_parts else None

    keywords = []
    for kw in all_nodes(root, ".//tei:profileDesc//tei:textClass//tei:keywords//tei:term"):
        k = text_of(kw)
        if k:
            keywords.append(k)

    return {
        "title": title if title else None,
        "doi": doi if doi else None,
        "journal": journal if journal else None,
        "date": date if date else None,
        "authors": authors,
        "abstract": abstract_text if abstract_text else None,
        "keywords": keywords
    }

# read figures inside a scope element
def parse_figures(scope, scope_id):
    out = []
    figs = all_nodes(scope, ".//tei:figure")
    for fig in figs:
        fig_id = ensure_id(fig, [scope_id, text_of(first(fig, "./tei:figDesc"))], "fig")
        label = text_of(first(fig, "./tei:label"))
        caption = text_of(first(fig, "./tei:figDesc"))
        if caption == "":
            caption = text_of(first(fig, "./tei:head"))
        g = first(fig, ".//tei:graphic")
        url = None
        if g is not None:
            url = get_attr(g, "url")
        out.append({
            "id": fig_id,
            "label": label if label else None,
            "caption": caption if caption else None,
            "graphic": url if url else None
        })
    return out

# read tables inside a scope element
def parse_tables(scope, scope_id):
    out = []
    tbs = all_nodes(scope, ".//tei:table")
    for tb in tbs:
        tb_id = ensure_id(tb, [scope_id, text_of(first(tb, "./tei:head"))], "tbl")
        head = text_of(first(tb, "./tei:head"))
        rows = []
        for row in all_nodes(tb, ".//tei:row"):
            cells = []
            for c in all_nodes(row, "./tei:cell"):
                cells.append(text_of(c))
            if len(cells) > 0:
                rows.append(cells)
        out.append({"id": tb_id, "title": head if head else None, "rows": rows})
    return out

# read equations inside a scope element
def parse_equations(scope, scope_id):
    out = []
    for fm in all_nodes(scope, ".//tei:formula"):
        fm_id = ensure_id(fm, [scope_id, text_of(fm)], "eq")
        out.append({"id": fm_id, "text": text_of(fm)})
    return out

# pull citation ids from a paragraph-like node
def parse_citations_in_p(p):
    cites = []
    for ref in all_nodes(p, ".//tei:ref[@type='bibr']"):
        tgt = get_attr(ref, "target")
        if tgt:
            cites.append(tgt.lstrip("#"))
    # dedupe and sort without set comprehension
    uniq = {}
    for c in cites:
        uniq[c] = True
    out = list(uniq.keys())
    out.sort()
    return out

# turn <p> and list items into paragraph objects
def parse_paragraphs(div, sec_id):
    paras = []
    i = 0
    for p in all_nodes(div, "./tei:p"):
        pid = ensure_id(p, [sec_id, str(i), text_of(p)[:80]], "p")
        paras.append({"id": pid, "text": text_of(p), "citations": parse_citations_in_p(p)})
        i += 1
    for ul in all_nodes(div, "./tei:list"):
        j = 0
        for item in all_nodes(ul, "./tei:item"):
            pid = ensure_id(item, [sec_id, "li", str(j), text_of(item)[:80]], "p")
            paras.append({"id": pid, "text": text_of(item), "citations": parse_citations_in_p(item)})
            j += 1
    return paras

# recursively parse a <div> section and its children
def parse_section(div, parent_id, depth=1):
    head = text_of(first(div, "./tei:head"))
    div_type = get_attr(div, "type")
    if div_type is None:
        div_type = None
    sec_id = ensure_id(div, [parent_id, head or "", div_type or ""], "sec" + str(depth))

    paragraphs = parse_paragraphs(div, sec_id)
    figures = parse_figures(div, sec_id)
    tables = parse_tables(div, sec_id)
    equations = parse_equations(div, sec_id)

    subs = []
    for sub_div in all_nodes(div, "./tei:div"):
        subs.append(parse_section(sub_div, parent_id=sec_id, depth=depth + 1))

    sec_obj = {
        "id": sec_id,
        "title": head if head else None,
        "type": div_type,
        "paragraphs": paragraphs,
        "figures": figures,
        "tables": tables,
        "equations": equations
    }
    if len(subs) > 0:
        sec_obj["subsections"] = subs
    return sec_obj

# gather the top-level body sections
def parse_body(root):
    body = first(root, ".//tei:text/tei:body")
    if body is None:
        body = first(root, ".//tei:body")
    sections = []
    if body is not None:
        for div in all_nodes(body, "./tei:div"):
            sections.append(parse_section(div, parent_id="body", depth=1))
    return sections

# read bibliography entries
def parse_references(root):
    refs = []
    candidates = all_nodes(root, ".//tei:text/tei:back//tei:listBibl//tei:biblStruct")
    if not candidates:
        candidates = all_nodes(root, ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct")
    for b in candidates:
        rid = ensure_id(b, [get_attr(b, "xml:id") or "", text_of(b)], "ref")
        title = text_of(first(b, "./tei:analytic/tei:title"))
        if title == "":
            title = text_of(first(b, "./tei:monogr/tei:title"))
        year = text_of(first(b, ".//tei:date"))
        if year == "":
            year = get_attr(first(b, ".//tei:date"), "when")
        doi = text_of(first(b, ".//tei:idno[@type='DOI']"))
        if doi == "":
            doi = text_of(first(b, ".//tei:idno[@type='doi']"))
        auth_nodes = all_nodes(b, "./tei:analytic/tei:author")
        if not auth_nodes:
            auth_nodes = all_nodes(b, "./tei:monogr/tei:author")
        authors = []
        for a in auth_nodes:
            forename = text_of(first(a, ".//tei:forename"))
            surname = text_of(first(a, ".//tei:surname"))
            full = (forename + " " + surname).strip()
            if full == "":
                full = text_of(a)
            authors.append(full)
        out_id = get_attr(b, "xml:id")
        if not out_id:
            out_id = rid
        refs.append({
            "id": out_id,
            "title": title if title else None,
            "authors": authors,
            "year": year if year else None,
            "doi": doi if doi else None
        })
    return refs

# build a paper id from title or filename
def build_paper_id(meta, input_path):
    title = ""
    if meta.get("title"):
        title = meta.get("title").strip()
    if title:
        return slugify(title, maxlen=96)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return slugify(base, maxlen=96)

# main converter from tei xml path to dict
def tei_to_json(input_path):
    parser = ET.XMLParser(remove_comments=True, recover=True)
    tree = ET.parse(input_path, parser=parser)
    root = tree.getroot()

    metadata = parse_metadata(root)
    sections = parse_body(root)

    txt = first(root, ".//tei:text")
    if txt is None:
        txt = root
    top_figs = parse_figures(txt, "body")
    top_tbls = parse_tables(txt, "body")
    top_eqs = parse_equations(txt, "body")

    references = parse_references(root)
    paper_id = build_paper_id(metadata, input_path)

    out = {
        "paper_id": paper_id,
        "metadata": metadata,
        "sections": sections,
        "figures": top_figs,
        "tables": top_tbls,
        "equations": top_eqs,
        "references": references,
        "source": {
            "tei_path": os.path.abspath(input_path),
            "tei_flavor": "GROBID-TEI"
        },
        "schema_version": "ep2c.paper.v1"
    }
    return out

# tiny cli to run conversion and write json
if __name__ == "__main__":
    INPUT_PATH = "../Paper.tei.xml"
    OUTPUT_PATH = "../Paper.json"
    result = tei_to_json(INPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[OK] Wrote", OUTPUT_PATH)
