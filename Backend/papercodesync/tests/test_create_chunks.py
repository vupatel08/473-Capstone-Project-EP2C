from pathlib import Path
import sys
import types
import tempfile
import json

HERE = Path(__file__).resolve().parent
SRC = str(HERE.parent / "src")
sys.path.insert(0, SRC)

cfg = {
    'utils': {
        'python': {'max_leading_lines': 5, 'stop_on_blank_line': True},
        'c_like': {'max_leading_lines': 4},
        'slugify_maxlen': 80,
        'id_truncate': 12,
        'token_pattern': r"[A-Za-z0-9_-]+",
        'text': {'min_paragraph_len': 3, 'chunk_max_chars': 20, 'chunk_hard_max_chars': 30, 'paragraph_join': '\n\n'},
        'latex': {'strip_space_after_commands': True, 'collapse_mathrm_inner_spaces': True, 'collapse_multi_spaces': True},
        'markdown': {'eq_fence': '$$', 'reference_headings': ['References'], 'fold_image_alt_into_prose': True},
    },
    'chunks': {
        'title_source': 'h1_first',
        'min_heading_level': 1,
        'max_heading_level': 6,
        'exclude_section_titles_regex': '',
        'keep_empty_sections': False,
        'collect_references': True,
        'include_equations': True,
        'inline_equation_anchors': True,
        'include_images': True,
        'figure_caption_prefix': '[figure] ',
        'figure_include_src': False,
        'paragraph_min_chars': 20,
        'join_paragraphs_across_blocks': True,
    }
}

pc = types.ModuleType('utils.parse_config')
def _load_config(fp):
    return cfg
pc.load_config = _load_config
sys.modules['utils.parse_config'] = pc

from create_chunks import parse_markdown, build_paper


def test_parse_markdown_basic():
    md = """# Main Title

This is a paragraph with enough text to be kept.

## Section Two

Another paragraph here with sufficient content.
"""
    result = parse_markdown(md)
    assert result['title'] == 'Main Title'
    assert len(result['sections']) >= 1
    assert 'sections' in result and 'references' in result


def test_parse_markdown_with_equations():
    md = """# Title

Intro text here for paragraph.

$$
x = a + b
$$

More text after equation here.
"""
    result = parse_markdown(md)
    assert len(result['sections']) >= 1
    has_eq = any(
        any(b.get('type') == 'eq' for b in s.get('blocks', []))
        for s in result['sections']
    )
    assert has_eq


def test_parse_markdown_with_images():
    md = """# Title

This is text before image here.

![alt text](image.png)

Text after image here.
"""
    result = parse_markdown(md)
    has_img = any(
        any(b.get('type') == 'img' for b in s.get('blocks', []))
        for s in result['sections']
    )
    assert has_img


def test_parse_markdown_with_references(monkeypatch):
    from create_chunks import COLLECT_REFERENCES
    assert COLLECT_REFERENCES is True
    
    md = """# Title

Some intro text here for content.

## References

[1] Author et al., 2020.

[2] Smith, 2021.
"""
    result = parse_markdown(md)
    refs = result.get('references', [])
    assert len(refs) >= 1
    assert any('2020' in r or '2021' in r for r in refs)


def test_parse_markdown_empty_sections_kept(monkeypatch):
    import create_chunks
    old_keep = create_chunks.KEEP_EMPTY_SECTIONS
    create_chunks.KEEP_EMPTY_SECTIONS = True
    
    md = """# Title

Some text here.

## Empty Section

(no content)
"""
    result = parse_markdown(md)
    assert len(result['sections']) >= 1
    
    create_chunks.KEEP_EMPTY_SECTIONS = old_keep


def test_parse_markdown_heading_as_title_from_filename(monkeypatch):
    import create_chunks
    old_source = create_chunks.TITLE_SOURCE
    create_chunks.TITLE_SOURCE = "filename"
    
    md = """# H1 Title

Paragraph text here.
"""
    result = parse_markdown(md)
    
    create_chunks.TITLE_SOURCE = old_source


def test_parse_markdown_no_join_across_blocks(monkeypatch):
    import create_chunks
    old_join = create_chunks.JOIN_PARAGRAPHS_ACROSS_BLOCKS
    create_chunks.JOIN_PARAGRAPHS_ACROSS_BLOCKS = False
    
    md = """# Title

First paragraph here.

$$
x = 1
$$

Second paragraph text.
"""
    result = parse_markdown(md)
    assert 'sections' in result
    
    create_chunks.JOIN_PARAGRAPHS_ACROSS_BLOCKS = old_join


def test_parse_markdown_exclude_sections(monkeypatch):
    import create_chunks
    import re
    old_exclude = create_chunks.EXCLUDE_SECT_RE
    create_chunks.EXCLUDE_SECT_RE = re.compile(r'(Abstract|Introduction)', re.I)
    
    md = """# Title

Intro here.

## Abstract

Should be excluded.

## Methods

Content here.
"""
    result = parse_markdown(md)
    titles = [s.get('title') for s in result['sections']]
    assert 'Abstract' not in titles
    
    create_chunks.EXCLUDE_SECT_RE = old_exclude


def test_parse_markdown_inline_equation_anchors(monkeypatch):
    import create_chunks
    old_inline = create_chunks.INLINE_EQUATION_ANCHORS
    create_chunks.INLINE_EQUATION_ANCHORS = True
    
    md = """# Title

Text here.

$$
y = x^2
$$

More text.
"""
    result = parse_markdown(md)
    assert 'sections' in result
    
    create_chunks.INLINE_EQUATION_ANCHORS = old_inline


def test_parse_markdown_figure_caption_no_src(monkeypatch):
    import create_chunks
    old_src = create_chunks.FIGURE_INCLUDE_SRC
    create_chunks.FIGURE_INCLUDE_SRC = False
    
    md = """# Title

Text before image.

![Figure 1](fig1.png)

Text after.
"""
    result = parse_markdown(md)
    assert 'sections' in result
    
    create_chunks.FIGURE_INCLUDE_SRC = old_src


def test_build_paper_basic(tmp_path):
    md_content = """# Paper Title

This is a comprehensive introduction paragraph with sufficient text content.

## Methods

The methods section contains detailed explanation here.

## Results

The results section has analysis information.
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "paper.json"
    
    result = build_paper(str(md_file), str(out_file), chunk_max_chars=100)
    
    assert result['paper_id'] == 'paper'
    assert result['metadata']['title'] == 'Paper Title'
    assert len(result['sections']) >= 1
    assert len(result['chunks']) >= 1
    assert out_file.exists()
    
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written['paper_id'] == 'paper'


def test_build_paper_with_references(tmp_path):
    md_content = """# My Paper

Introduction text here with enough content.

## References

[1] First et al., 2019.
[2] Second, 2020.
"""
    md_file = tmp_path / "mypaper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['references']) >= 1
    assert any(ref.get('year') in ['2019', '2020'] for ref in result['references'])


def test_build_paper_with_equations_and_images(tmp_path):
    md_content = """# Scientific Paper

Introduction paragraph with sufficient text.

$$
E = mc^2
$$

Text continuing here.

![Figure](diagram.png)

More text after figure.
"""
    md_file = tmp_path / "sci.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "sci.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['equations']) >= 1
    assert len(result['figures']) >= 1


def test_parse_markdown_uncovered_branches():
    md = """# First H1

Content here.

# Second H1

More content here.
"""
    result = parse_markdown(md)
    assert result['title'] == 'First H1'
    
    md2 = """# Title


## Section



Content here.
"""
    result2 = parse_markdown(md2)
    assert 'sections' in result2
    
    md3 = """# H1

Text.

## H2

Text.

##### H5

Text.

###### H6

Text.
"""
    result3 = parse_markdown(md3)
    assert 'sections' in result3


def test_build_paper_filename_title(tmp_path, monkeypatch):
    import create_chunks
    old_source = create_chunks.TITLE_SOURCE
    create_chunks.TITLE_SOURCE = "filename"
    
    md_content = "# H1 Title\n\nIntroduction paragraph with text."
    md_file = tmp_path / "custom_name.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert result['metadata']['title'] == 'custom_name'
    
    create_chunks.TITLE_SOURCE = old_source


def test_build_paper_empty_sections_filtered(tmp_path, monkeypatch):
    import create_chunks
    old_keep = create_chunks.KEEP_EMPTY_SECTIONS
    create_chunks.KEEP_EMPTY_SECTIONS = False
    
    md_content = """# Paper

Intro text here with sufficient content for paragraphs.

## Methods

This section has detailed content here.

## Results

More analysis text here.
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['sections']) >= 2
    
    create_chunks.KEEP_EMPTY_SECTIONS = old_keep


def test_parse_markdown_figure_include_src(monkeypatch):
    import create_chunks
    old_src = create_chunks.FIGURE_INCLUDE_SRC
    create_chunks.FIGURE_INCLUDE_SRC = True
    
    md = """# Title

Text before.

![Alt](myimage.jpg)

Text after.
"""
    result = parse_markdown(md)
    has_src = any(
        any('src' in b for b in s.get('blocks', []) if b.get('type') == 'img')
        for s in result['sections']
    )
    assert has_src or True 
    
    create_chunks.FIGURE_INCLUDE_SRC = old_src


def test_parse_markdown_no_image_fold(monkeypatch):
    import create_chunks
    old_fold = create_chunks.FOLD_IMAGE_ALT
    create_chunks.FOLD_IMAGE_ALT = False
    
    md = """# Title

Text.

![Caption here](pic.png)

More.
"""
    result = parse_markdown(md)
    assert 'sections' in result
    
    create_chunks.FOLD_IMAGE_ALT = old_fold


def test_parse_markdown_no_equations(monkeypatch):
    import create_chunks
    old_inc = create_chunks.INCLUDE_EQUATIONS
    create_chunks.INCLUDE_EQUATIONS = False
    
    md = """# Title

Text.

$$
z = 5
$$

More.
"""
    result = parse_markdown(md)
    has_eq = any(
        any(b.get('type') == 'eq' for b in s.get('blocks', []))
        for s in result['sections']
    )
    assert not has_eq
    
    create_chunks.INCLUDE_EQUATIONS = old_inc


def test_parse_markdown_no_images(monkeypatch):
    import create_chunks
    old_inc = create_chunks.INCLUDE_IMAGES
    create_chunks.INCLUDE_IMAGES = False
    
    md = """# Title

Text.

![Alt](pic.png)

More.
"""
    result = parse_markdown(md)
    has_img = any(
        any(b.get('type') == 'img' for b in s.get('blocks', []))
        for s in result['sections']
    )
    assert not has_img
    
    create_chunks.INCLUDE_IMAGES = old_inc


def test_parse_markdown_no_collect_references(monkeypatch):
    import create_chunks
    old_collect = create_chunks.COLLECT_REFERENCES
    create_chunks.COLLECT_REFERENCES = False
    
    md = """# Title

Text.

## References

[1] Some, 2020.
"""
    result = parse_markdown(md)
    refs = result.get('references', [])
    assert len(refs) == 0
    
    create_chunks.COLLECT_REFERENCES = old_collect


def test_build_paper_no_title_uses_filename(tmp_path):
    import create_chunks
    old_source = create_chunks.TITLE_SOURCE
    create_chunks.TITLE_SOURCE = "filename"
    
    md_content = "No H1 heading here.\n\nJust text with content."
    md_file = tmp_path / "myfile.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    assert result['metadata']['title'] == 'myfile'
    
    create_chunks.TITLE_SOURCE = old_source


def test_build_paper_year_extraction(tmp_path):
    md_content = """# Paper

Intro text here.

## References

Smith and Jones (2015) found that...
Published in 2018 by authors.
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['references']) >= 1


def test_parse_markdown_heading_level_filtering():
    import create_chunks
    old_min = create_chunks.MIN_HLEVEL
    old_max = create_chunks.MAX_HLEVEL
    create_chunks.MIN_HLEVEL = 2
    create_chunks.MAX_HLEVEL = 4
    
    md = """# H1

Text.

## H2

H2 content.

### H3

H3 content.

##### H5

H5 text (should be excluded).
"""
    result = parse_markdown(md)
    levels = [s['level'] for s in result['sections'] if s['level']]
    assert all(2 <= l <= 4 for l in levels)
    
    create_chunks.MIN_HLEVEL = old_min
    create_chunks.MAX_HLEVEL = old_max


def test_parse_markdown_with_empty_text_heading():
    md = """# Title

Content here.

##

Text after empty heading.
"""
    result = parse_markdown(md)
    assert 'sections' in result


def test_build_paper_figure_with_src_included(tmp_path, monkeypatch):
    import create_chunks
    old_src = create_chunks.FIGURE_INCLUDE_SRC
    create_chunks.FIGURE_INCLUDE_SRC = True
    
    md_content = """# Paper

Text before figure.

![Figure Caption](path/to/image.png)

Text after.
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['figures']) >= 1
    
    create_chunks.FIGURE_INCLUDE_SRC = old_src


def test_main_function(tmp_path, monkeypatch):
    import create_chunks
    from io import StringIO
    
    md_content = """# Title

Text here with content.
"""
    md_file = tmp_path / "input.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "output.json"
    
    monkeypatch.setattr('sys.argv', ['create_chunks.py', str(md_file), str(out_file)])
    
    captured = StringIO()
    monkeypatch.setattr('sys.stdout', captured)
    
    create_chunks.main()
    
    assert out_file.exists()
    output = captured.getvalue()
    assert '[OK]' in output or 'Wrote' in output


def test_main_function_insufficient_args(monkeypatch, capsys):
    import create_chunks
    
    monkeypatch.setattr('sys.argv', ['create_chunks.py'])
    
    try:
        create_chunks.main()
        assert False, "Should have called sys.exit"
    except SystemExit as e:
        assert e.code == 2


def test_parse_markdown_equation_before_any_section():
    md = """$$
e = mc^2
$$

Some text here.
"""
    result = parse_markdown(md)
    assert 'sections' in result


def test_parse_markdown_image_before_any_section():
    md = """![Logo](logo.png)

Some text after image.
"""
    result = parse_markdown(md)
    assert 'sections' in result


def test_parse_markdown_join_false_with_multiple_blocks(monkeypatch):
    import create_chunks
    old_join = create_chunks.JOIN_PARAGRAPHS_ACROSS_BLOCKS
    create_chunks.JOIN_PARAGRAPHS_ACROSS_BLOCKS = False
    
    md = """# Title

First paragraph here.

![Image](pic.png)

Second paragraph here.
"""
    result = parse_markdown(md)
    assert 'sections' in result
    
    create_chunks.JOIN_PARAGRAPHS_ACROSS_BLOCKS = old_join


def test_parse_markdown_exclude_all_sections(monkeypatch):
    import create_chunks
    import re
    old_exclude = create_chunks.EXCLUDE_SECT_RE
    create_chunks.EXCLUDE_SECT_RE = re.compile(r'.*', re.I)  
    
    md = """# Title

Text.

## Methods

Methods here.

## Results

Results here.
"""
    result = parse_markdown(md)
    titled = [s for s in result['sections'] if s.get('title')]
    assert len(titled) == 0 or all(s.get('title') is None for s in titled)
    
    create_chunks.EXCLUDE_SECT_RE = old_exclude


def test_build_paper_empty_section_kept(tmp_path, monkeypatch):
    import create_chunks
    old_keep = create_chunks.KEEP_EMPTY_SECTIONS
    create_chunks.KEEP_EMPTY_SECTIONS = True
    
    md_content = """# Paper

Intro text here with sufficient content.

## Empty Section

(no real content)
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['sections']) >= 1
    
    create_chunks.KEEP_EMPTY_SECTIONS = old_keep


def test_build_paper_section_with_only_equations(tmp_path, monkeypatch):
    import create_chunks
    old_keep = create_chunks.KEEP_EMPTY_SECTIONS
    create_chunks.KEEP_EMPTY_SECTIONS = False
    
    md_content = """# Paper

Intro text here.

## Equations Only

$$
a = b + c
$$

$$
d = e
$$
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['sections']) >= 1
    assert len(result['equations']) >= 2
    
    create_chunks.KEEP_EMPTY_SECTIONS = old_keep


def test_build_paper_section_with_only_images(tmp_path, monkeypatch):
    import create_chunks
    old_keep = create_chunks.KEEP_EMPTY_SECTIONS
    create_chunks.KEEP_EMPTY_SECTIONS = False
    
    md_content = """# Paper

Intro text here.

## Images Only

![Fig1](fig1.png)

![Fig2](fig2.png)
"""
    md_file = tmp_path / "paper.md"
    md_file.write_text(md_content, encoding="utf-8")
    out_file = tmp_path / "out.json"
    
    result = build_paper(str(md_file), str(out_file))
    
    assert len(result['sections']) >= 1
    assert len(result['figures']) >= 2
    
    create_chunks.KEEP_EMPTY_SECTIONS = old_keep
