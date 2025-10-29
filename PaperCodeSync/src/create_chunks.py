# Setup Requirements:
# -----------------
# 1. Run GROBID in Docker:
#    docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-crf
#
# 2. Convert PDF to TEI XML:
#    curl -sS -X POST -F "input=@Paper.pdf;type=application/pdf" \
#      http://localhost:8070/api/processFulltextDocument -o Paper.tei.xml

import json
import os
import re
from lxml import etree as ET
from utils.parse_config import load_config
from utils.common import slugify, sha1_id
from utils.tei import TEI_NS, text_of, first, all_nodes, get_attr

# Pull the sub-config
config = load_config("../config.yaml")
chunks = config["chunks"]

SECTION_DEPTH_LIMIT = chunks["section_depth_limit"]
EXCL_TITLES_RE = re.compile(chunks["excluded_section_titles_regex"], re.I)
EXCL_TYPES = {t.lower() for t in chunks["excluded_section_types"]}

INCLUDE_LISTS_AS_PARAGRAPHS = chunks["include_lists_as_paragraphs"]
PARAGRAPH_MIN_CHARS = chunks["paragraph_min_chars"]
INCLUDE_CITATIONS = chunks["include_citations"]
PREFER_BACK_LIST_FOR_REFS = chunks["prefer_back_list_for_refs"]

INCLUDE_FIGURES = chunks["include_figures"]
INCLUDE_TABLES = chunks["include_tables"]
INCLUDE_EQUATIONS = chunks["include_equations"]
FIGURE_INCLUDE_GRAPHIC_URL = chunks["figure_include_graphic_url"]
TABLE_MAX_ROWS = chunks["table_max_rows"]

FORCE_GENERATE_IDS = chunks["force_generate_ids"]
PREFER_FILENAME_FOR_PAPER_ID = chunks["prefer_filename_for_paper_id"]

def ensure_id(el, fallback_parts, prefix):
    if not FORCE_GENERATE_IDS:
        existing = get_attr(el, "xml:id") or el.get("id")
        if existing:
            return existing
    joined = "||".join([p for p in fallback_parts if p])
    return f"{prefix}_{sha1_id(joined)}"

def _include_section(title: str | None, div_type: str | None, depth: int) -> bool:
    if depth > SECTION_DEPTH_LIMIT:
        return False
    if div_type and div_type.lower() in EXCL_TYPES:
        return False
    if EXCL_TITLES_RE and title and EXCL_TITLES_RE.search(title):
        return False
    return True

def _keep_text(txt: str) -> bool:
    if not txt:
        return False
    t = txt.strip()
    return bool(t) and len(t) >= PARAGRAPH_MIN_CHARS

def parse_metadata(root):
    title_el = first(root, ".//tei:teiHeader//tei:titleStmt/tei:title")
    title = text_of(title_el)

    authors = []
    for path in (
        ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:author",
        ".//tei:teiHeader//tei:titleStmt//tei:author",
    ):
        for a in all_nodes(root, path):
            forename = text_of(first(a, ".//tei:persName/tei:forename"))
            surname = text_of(first(a, ".//tei:persName/tei:surname"))
            full = (forename + " " + surname).strip() or text_of(a)
            email = text_of(first(a, ".//tei:email"))
            aff = text_of(first(a, ".//tei:affiliation")) or text_of(first(a, ".//tei:orgName"))
            authors.append({
                "name": full or None,
                "forename": forename or None,
                "surname": surname or None,
                "email": email or None,
                "affiliation": aff or None,
            })

    date_el = first(root, ".//tei:teiHeader//tei:publicationStmt//tei:date")
    date = (get_attr(date_el, "when") if date_el is not None else None) or (text_of(date_el) if date_el is not None else None)

    doi = text_of(first(root, ".//tei:teiHeader//tei:idno[@type='DOI']")) or \
          text_of(first(root, ".//tei:teiHeader//tei:idno[@type='doi']"))

    journal = text_of(first(root, ".//tei:teiHeader//tei:monogr/tei:title")) or \
              text_of(first(root, ".//tei:teiHeader//tei:series/tei:title"))

    abs_ps = all_nodes(root, ".//tei:profileDesc//tei:abstract//tei:p") or \
             all_nodes(root, ".//tei:abstract//tei:p")
    abstract_text = " ".join([text_of(p) for p in abs_ps]) if abs_ps else None

    keywords = [text_of(kw) for kw in all_nodes(root, ".//tei:profileDesc//tei:textClass//tei:keywords//tei:term") if text_of(kw)]

    return {
        "title": title or None,
        "doi": doi or None,
        "journal": journal or None,
        "date": date or None,
        "authors": authors,
        "abstract": abstract_text or None,
        "keywords": keywords,
    }


def parse_figures(scope, scope_id):
    if not INCLUDE_FIGURES:
        return []
    out = []
    for fig in all_nodes(scope, ".//tei:figure"):
        fig_id = ensure_id(fig, [scope_id, text_of(first(fig, "./tei:figDesc"))], "fig")
        label = text_of(first(fig, "./tei:label"))
        caption = text_of(first(fig, "./tei:figDesc")) or text_of(first(fig, "./tei:head"))
        g = first(fig, ".//tei:graphic")
        url = get_attr(g, "url") if (g is not None and FIGURE_INCLUDE_GRAPHIC_URL) else None
        out.append({
            "id": fig_id,
            "label": label or None,
            "caption": caption or None,
            "graphic": url or None,
        })
    return out


def parse_tables(scope, scope_id):
    if not INCLUDE_TABLES:
        return []
    out = []
    for tb in all_nodes(scope, ".//tei:table"):
        tb_id = ensure_id(tb, [scope_id, text_of(first(tb, "./tei:head"))], "tbl")
        head = text_of(first(tb, "./tei:head"))
        rows = []
        for r_i, row in enumerate(all_nodes(tb, ".//tei:row")):
            if r_i >= TABLE_MAX_ROWS:
                break
            cells = [text_of(c) for c in all_nodes(row, "./tei:cell")]
            if cells:
                rows.append(cells)
        out.append({"id": tb_id, "title": head or None, "rows": rows})
    return out


def parse_equations(scope, scope_id):
    if not INCLUDE_EQUATIONS:
        return []
    out = []
    for fm in all_nodes(scope, ".//tei:formula"):
        fm_id = ensure_id(fm, [scope_id, text_of(fm)], "eq")
        out.append({"id": fm_id, "text": text_of(fm)})
    return out


def parse_citations_in_p(p):
    if not INCLUDE_CITATIONS:
        return []
    cites = []
    for ref in all_nodes(p, ".//tei:ref[@type='bibr']"):
        tgt = get_attr(ref, "target")
        if tgt:
            cites.append(tgt.lstrip("#"))
    return sorted({c for c in cites})


def parse_paragraphs(div, sec_id):
    paras = []
    # normal paragraphs
    for i, p in enumerate(all_nodes(div, "./tei:p")):
        txt = text_of(p)
        if not _keep_text(txt):
            continue
        pid = ensure_id(p, [sec_id, str(i), txt[:80]], "p")
        paras.append({"id": pid, "text": txt, "citations": parse_citations_in_p(p)})

    # lists as paragraphs 
    if INCLUDE_LISTS_AS_PARAGRAPHS:
        for ul in all_nodes(div, "./tei:list"):
            for j, item in enumerate(all_nodes(ul, "./tei:item")):
                txt = text_of(item)
                if not _keep_text(txt):
                    continue
                pid = ensure_id(item, [sec_id, "li", str(j), txt[:80]], "p")
                paras.append({"id": pid, "text": txt, "citations": parse_citations_in_p(item)})
    return paras


def parse_section(div, parent_id, depth=1):
    head = text_of(first(div, "./tei:head"))
    div_type = get_attr(div, "type") or None
    if not _include_section(head, div_type, depth):
        return None

    sec_id = ensure_id(div, [parent_id, head or "", div_type or ""], f"sec{depth}")

    paragraphs = parse_paragraphs(div, sec_id)
    figures = parse_figures(div, sec_id)
    tables = parse_tables(div, sec_id)
    equations = parse_equations(div, sec_id)

    subsections = []
    for sub_div in all_nodes(div, "./tei:div"):
        sub = parse_section(sub_div, parent_id=sec_id, depth=depth + 1)
        if sub:
            subsections.append(sub)

    sec_obj = {
        "id": sec_id,
        "title": head or None,
        "type": div_type,
        "paragraphs": paragraphs,
        "figures": figures,
        "tables": tables,
        "equations": equations,
    }
    if subsections:
        sec_obj["subsections"] = subsections
    return sec_obj


def parse_body(root):
    body = first(root, ".//tei:text/tei:body") or first(root, ".//tei:body")
    if not body:
        return []
    sections = []
    for div in all_nodes(body, "./tei:div"):
        sec = parse_section(div, parent_id="body", depth=1)
        if sec:
            sections.append(sec)
    return sections


def parse_references(root):
    refs = []
    if PREFER_BACK_LIST_FOR_REFS:
        candidates = all_nodes(root, ".//tei:text/tei:back//tei:listBibl//tei:biblStruct") or \
                     all_nodes(root, ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct")
    else:
        candidates = all_nodes(root, ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct") or \
                     all_nodes(root, ".//tei:text/tei:back//tei:listBibl//tei:biblStruct")
    for b in candidates:
        rid = ensure_id(b, [get_attr(b, "xml:id") or "", text_of(b)], "ref")
        title = text_of(first(b, "./tei:analytic/tei:title")) or text_of(first(b, "./tei:monogr/tei:title"))
        year = text_of(first(b, ".//tei:date")) or get_attr(first(b, ".//tei:date"), "when")
        doi = text_of(first(b, ".//tei:idno[@type='DOI']")) or text_of(first(b, ".//tei:idno[@type='doi']"))

        auth_nodes = all_nodes(b, "./tei:analytic/tei:author") or all_nodes(b, "./tei:monogr/tei:author")
        authors = []
        for a in auth_nodes:
            forename = text_of(first(a, ".//tei:forename"))
            surname = text_of(first(a, ".//tei:surname"))
            full = (forename + " " + surname).strip() or text_of(a)
            authors.append(full)

        out_id = get_attr(b, "xml:id") or rid
        refs.append({
            "id": out_id,
            "title": title or None,
            "authors": authors,
            "year": year or None,
            "doi": doi or None,
        })
    return refs


def build_paper_id(meta, input_path):
    if not PREFER_FILENAME_FOR_PAPER_ID:
        title = (meta.get("title") or "").strip()
        if title:
            return slugify(title, maxlen=96)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return slugify(base, maxlen=96)


def tei_to_json(input_path):
    parser = ET.XMLParser(remove_comments=True, recover=True)
    tree = ET.parse(input_path, parser=parser)
    root = tree.getroot()

    metadata = parse_metadata(root)
    sections = parse_body(root)

    txt = first(root, ".//tei:text") or root
    top_figs = parse_figures(txt, "body")
    top_tbls = parse_tables(txt, "body")
    top_eqs = parse_equations(txt, "body")

    references = parse_references(root)
    paper_id = build_paper_id(metadata, input_path)

    out = {
        "paper_id": paper_id,
        "metadata": metadata,
        "sections": sections,
        "figures": top_figs if INCLUDE_FIGURES else [],
        "tables": top_tbls if INCLUDE_TABLES else [],
        "equations": top_eqs if INCLUDE_EQUATIONS else [],
        "references": references,
        "source": {
            "tei_path": os.path.abspath(input_path),
            "tei_flavor": "GROBID-TEI",
        },
        "schema_version": "ep2c.paper.v1",
    }
    return out


if __name__ == "__main__":
    INPUT_PATH = "../Paper.tei.xml"
    OUTPUT_PATH = "../Paper.json"
    result = tei_to_json(INPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[OK] Wrote", OUTPUT_PATH)
