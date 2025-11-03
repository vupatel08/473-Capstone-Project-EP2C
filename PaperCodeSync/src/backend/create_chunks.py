import sys, re, json
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.parse_config import load_config
from utils.common import sha1_prefix, keep_text, split_into_chunks
from utils.latex import normalize_latex
from utils.markdown import HEADING_RE, IMG_RE, is_eq_fence, is_reference_heading, FOLD_IMAGE_ALT

_config   = load_config("../config.yaml")
_chunks    = _config.get("chunks", {})

TITLE_SOURCE                   = _chunks.get("title_source", "h1_first")  
MIN_HLEVEL                     = int(_chunks.get("min_heading_level", 1))
MAX_HLEVEL                     = int(_chunks.get("max_heading_level", 6))
EXCLUDE_SECT_RE                = re.compile(_chunks.get("exclude_section_titles_regex", ""), re.I) if _chunks.get("exclude_section_titles_regex") else None
KEEP_EMPTY_SECTIONS            = bool(_chunks.get("keep_empty_sections", False))
COLLECT_REFERENCES             = bool(_chunks.get("collect_references", True))

INCLUDE_EQUATIONS              = bool(_chunks.get("include_equations", True))
INLINE_EQUATION_ANCHORS        = bool(_chunks.get("inline_equation_anchors", True))

INCLUDE_IMAGES                 = bool(_chunks.get("include_images", True))
FIGURE_CAPTION_PREFIX          = str(_chunks.get("figure_caption_prefix", "[figure] "))
FIGURE_INCLUDE_SRC             = bool(_chunks.get("figure_include_src", False))

PARAGRAPH_MIN_CHARS            = int(_chunks.get("paragraph_min_chars", 20))
JOIN_PARAGRAPHS_ACROSS_BLOCKS  = bool(_chunks.get("join_paragraphs_across_blocks", True))

def parse_markdown(md: str) -> Dict[str, Any]:
    lines = md.splitlines()
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    in_eq = False
    eq_buf: List[str] = []
    buf: List[str] = []

    def wants_section(level: int, text: str | None) -> bool:
        if level < MIN_HLEVEL or level > MAX_HLEVEL:
            return False
        if not text:
            return True
        if EXCLUDE_SECT_RE and EXCLUDE_SECT_RE.search(text.strip()):
            return False
        return True

    def push_section(level: int, text: str | None):
        nonlocal cur
        cur = {"level": level, "title": (text.strip() if text else None), "blocks": []}
        sections.append(cur)

    def push_paragraph(text: str):
        nonlocal cur
        if not cur:
            push_section(MIN_HLEVEL, None)
        cur["blocks"].append({"type": "p", "text": text})

    def push_equation(latex: str):
        if not INCLUDE_EQUATIONS:
            return
        nonlocal cur
        if not cur:
            push_section(MIN_HLEVEL, None)
        cur["blocks"].append({"type": "eq", "latex": normalize_latex(latex)})

    def push_image(alt: str, src: str):
        if not INCLUDE_IMAGES:
            return
        nonlocal cur
        if not cur:
            push_section(MIN_HLEVEL, None)
        img_block = {"type": "img", "alt": (alt or "").strip()}
        if FIGURE_INCLUDE_SRC and src:
            img_block["src"] = src
        cur["blocks"].append(img_block)

    def flush_buf():
        nonlocal buf
        text = "\n".join(buf).strip()
        if keep_text(text, PARAGRAPH_MIN_CHARS):
            push_paragraph(text)
        buf = []

    for line in lines:
        if is_eq_fence(line):
            if not in_eq:
                in_eq = True
                eq_buf = []
            else:
                in_eq = False
                latex = "\n".join(eq_buf).strip()
                if not JOIN_PARAGRAPHS_ACROSS_BLOCKS:
                    flush_buf()
                if latex:
                    push_equation(latex)
                eq_buf = []
            continue

        if in_eq:
            eq_buf.append(line)
            continue

        m = HEADING_RE.match(line)
        if m:
            if not JOIN_PARAGRAPHS_ACROSS_BLOCKS:
                flush_buf()
            level = len(m.group(1))
            text = m.group(2).strip()

            if TITLE_SOURCE == "h1_first" and level == 1 and title is None:
                title = text

            if wants_section(level, text):
                push_section(level, text)
            else:
                buf.append(line)
            continue

        mimg = IMG_RE.search(line)
        if mimg:
            if not JOIN_PARAGRAPHS_ACROSS_BLOCKS:
                flush_buf()
            alt, src = mimg.group(1), mimg.group(2)
            push_image(alt, src)
            continue

        if line.strip() == "":
            flush_buf()
            continue

        # Normal text
        buf.append(line)

    flush_buf()

    references: List[str] = []
    if COLLECT_REFERENCES:
        refs_text_blocks: List[str] = []
        for s in sections:
            if is_reference_heading(s.get("title")):
                for b in s.get("blocks", []):
                    if b["type"] == "p":
                        refs_text_blocks.append(b["text"])
        refs_blob = "\n".join(refs_text_blocks)
        references = [
            r.strip()
            for r in re.split(r'\n\s*\n|(?:\n\s*(?:\[\d+\]|\(\d+\)|\d+\.)\s*)', refs_blob)
            if keep_text(r.strip(), 6)
        ]

    if not KEEP_EMPTY_SECTIONS:
        filtered = []
        for s in sections:
            has_blocks = any(
                b["type"] in ("p", "eq", "img")
                for b in s.get("blocks", [])
            )
            if has_blocks:
                filtered.append(s)
        sections = filtered

    return {"title": title, "sections": sections, "references": references}

def build_paper(md_path: str, out_path: str, chunk_max_chars: int = 1400) -> Dict[str, Any]:
    md = Path(md_path).read_text(encoding="utf-8")
    parsed = parse_markdown(md)

    if TITLE_SOURCE == "filename" or not parsed.get("title"):
        title = Path(md_path).stem
    else:
        title = parsed["title"]

    sections_src = parsed["sections"]

    sections: List[Dict[str, Any]] = []
    flat_chunks: List[Dict[str, Any]] = []
    eq_index: List[Dict[str, Any]] = []
    figs_top: List[Dict[str, Any]] = []

    for s in sections_src:
        sec_title = s.get("title")
        paras: List[Dict[str, Any]] = []
        equations: List[Dict[str, Any]] = []
        figures: List[Dict[str, Any]] = []
        buf: List[str] = []

        def flush_para():
            nonlocal buf
            text = "\n".join(buf).strip()
            if keep_text(text, PARAGRAPH_MIN_CHARS):
                paras.append({"id": f"p-{sha1_prefix(text[:96])}", "text": text, "citations": []})
            buf = []

        eq_count = 0
        for b in s.get("blocks", []):
            if b["type"] == "p":
                buf.append(b["text"])

            elif b["type"] == "eq" and INCLUDE_EQUATIONS:
                flush_para()
                eq_count += 1
                latex = b.get("latex") or ""
                eq_id = f"eq-{sha1_prefix(latex[:160])}-{eq_count}"

                if INLINE_EQUATION_ANCHORS:
                    buf.append(f"[eq:{eq_id}]\n$$\n{normalize_latex(latex)}\n$$")

                equations.append({
                    "id": eq_id,
                    "page": None,
                    "latex": normalize_latex(latex),
                    "has_latex": bool(latex)
                })
                eq_index.append(equations[-1])

            elif b["type"] == "img" and INCLUDE_IMAGES:
                alt = (b.get("alt") or "").strip()
                if alt and FOLD_IMAGE_ALT:
                    # fold caption into prose
                    buf.append(f"{FIGURE_CAPTION_PREFIX}{alt}")
                fid = f"fig-{sha1_prefix((alt or '')[:96])}"
                fig_obj = {"id": fid, "caption": alt}
                if FIGURE_INCLUDE_SRC and b.get("src"):
                    fig_obj["src"] = b["src"]
                figures.append(fig_obj)
                figs_top.append(figures[-1])

        flush_para()

        if not KEEP_EMPTY_SECTIONS and not (paras or equations or figures):
            continue

        sections.append({
            "id": f"sec{s['level']}-{sha1_prefix((sec_title or '') + (paras[0]['id'] if paras else ''))}",
            "title": sec_title,
            "type": None,
            "paragraphs": paras,
            "figures": figures,
            "tables": [],
            "equations": equations
        })

        for p in paras:
            for chunk_text in split_into_chunks(p["text"], chunk_max_chars):
                flat_chunks.append({
                    "id": f"chk-{sha1_prefix(chunk_text[:160])}",
                    "page": None,
                    "section": sec_title,
                    "title": None,
                    "text": chunk_text
                })

    refs_out: List[Dict[str, Any]] = []
    if COLLECT_REFERENCES:
        for r in parsed["references"]:
            yr = None
            m = re.search(r'(19|20)\d{2}', r)
            if m: yr = m.group(0)
            refs_out.append({"id": f"ref-{sha1_prefix(r[:96])}", "title": r, "authors": [], "year": yr, "doi": None})

    paper_id = Path(md_path).stem.lower()

    out = {
        "paper_id": paper_id,
        "metadata": {
            "title": title,
            "doi": None,
            "journal": None,
            "date": None,
            "authors": [],     # i decided to leave this empty in the end since mineru was not reliable 
            "abstract": None,
            "keywords": [],
        },
        "sections": sections,
        "chunks": flat_chunks,
        "figures": figs_top,
        "tables": [],
        "equations": eq_index,
        "references": refs_out,
        "source": {
            "mineru_markdown_path": str(Path(md_path).resolve()),
            "mineru_source": "markdown"
        },
        "schema_version": "ep2c.paper.v1",
    }

    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_chunks.py <Paper.md> <Paper.json>")
        sys.exit(2)
    md_path = sys.argv[1]
    out_path = sys.argv[2]
    res = build_paper(md_path, out_path)
    print(f"[OK] Wrote {out_path} | sections={len(res['sections'])} chunks={len(res['chunks'])} eqs={len(res['equations'])}")

if __name__ == "__main__":
    main()
