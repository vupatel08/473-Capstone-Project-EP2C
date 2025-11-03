import sys, re, json
from pathlib import Path
from typing import List, Dict, Any
from utils.common import sha1_prefix, keep_text, split_into_chunks
from utils.latex import normalize_latex
from utils.markdown import HEADING_RE, IMG_RE, FENCED_EQ_RE

def parse_markdown(md: str) -> Dict[str, Any]:
    lines = md.splitlines()
    title = None
    sections = []
    cur = None
    refs_collecting = False

    in_eq = False
    eq_buf: List[str] = []
    buf: List[str] = []

    def push_section(level: int, text: str|None):
        nonlocal cur
        cur = {"level": level, "title": (text.strip() if text else None), "blocks": []}
        sections.append(cur)

    def push_paragraph(text: str):
        nonlocal cur
        if not cur:
            push_section(1, None)
        cur["blocks"].append({"type": "p", "text": text})

    def push_equation(latex: str):
        nonlocal cur
        if not cur:
            push_section(1, None)
        cur["blocks"].append({"type": "eq", "latex": normalize_latex(latex)})

    def push_image(alt: str, src: str):
        nonlocal cur
        if not cur:
            push_section(1, None)
        cur["blocks"].append({"type": "img", "alt": (alt or "").strip(), "src": src})

    def flush_buf():
        nonlocal buf
        text = "\n".join(buf).strip()
        if keep_text(text):
            push_paragraph(text)
        buf = []

    for line in lines:
        if FENCED_EQ_RE.match(line):
            if not in_eq:
                in_eq = True
                eq_buf = []
            else:
                in_eq = False
                latex = "\n".join(eq_buf).strip()
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
            flush_buf()
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1 and title is None:
                title = text
            push_section(level, text)
            refs_collecting = text.lower().startswith("references")
            continue

        mimg = IMG_RE.search(line)
        if mimg:
            flush_buf()
            alt, src = mimg.group(1), mimg.group(2)
            push_image(alt, src)
            continue

        if line.strip() == "":
            flush_buf()
            continue

        buf.append(line)


    flush_buf()

    refs_text_blocks = []
    for s in sections:
        if (s.get("title") or "").lower().startswith("references"):
            for b in s.get("blocks", []):
                if b["type"] == "p":
                    refs_text_blocks.append(b["text"])
    refs_blob = "\n".join(refs_text_blocks)
    raw_refs = [r.strip() for r in re.split(r'\n\s*\n|(?:\n\s*(?:\[\d+\]|\(\d+\)|\d+\.)\s*)', refs_blob) if keep_text(r.strip(), 6)]

    return {"title": title, "sections": sections, "references": raw_refs}


def build_paper(md_path: str, out_path: str, chunk_max_chars: int = 1400) -> Dict[str, Any]:
    md = Path(md_path).read_text(encoding="utf-8")
    parsed = parse_markdown(md)

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
            if keep_text(text):
                paras.append({"id": f"p-{sha1_prefix(text[:96])}", "text": text, "citations": []})
            buf = []

        eq_count = 0
        for b in s.get("blocks", []):
            if b["type"] == "p":
                buf.append(b["text"])
            elif b["type"] == "eq":
                flush_para()
                eq_count += 1
                latex = b.get("latex") or ""
                eq_id = f"eq-{sha1_prefix(latex[:160])}-{eq_count}"
                buf.append(f"[eq:{eq_id}]\n$$\n{normalize_latex(latex)}\n$$")
                equations.append({"id": eq_id, "page": None, "latex": normalize_latex(latex), "has_latex": bool(latex)})
                eq_index.append(equations[-1])
            elif b["type"] == "img":
                alt = (b.get("alt") or "").strip()
                if alt:
                    buf.append(f"[figure] {alt}")
                    fid = f"fig-{sha1_prefix(alt[:96])}"
                    figures.append({"id": fid, "caption": alt})
                    figs_top.append(figures[-1])

        flush_para()

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

    refs_out = []
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
            "authors": [], # keep authors empty, i tried to parse this too but it just wasn't reliable
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_chunks.py <Paper.md> <Paper.json>")
        sys.exit(2)
    md_path = sys.argv[1]
    out_path = sys.argv[2]
    res = build_paper(md_path, out_path)
    print(f"[OK] Wrote {out_path} | sections={len(res['sections'])} chunks={len(res['chunks'])} eqs={len(res['equations'])}")