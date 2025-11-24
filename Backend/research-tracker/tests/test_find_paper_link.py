from pathlib import Path
import sys
import importlib
import tempfile

HERE = Path(__file__).resolve().parent
PKG_ROOT = str(HERE.parent)
sys.path.insert(0, PKG_ROOT)

import find_paper_link


def test__normalize_arxiv_id_and_extract():
    assert find_paper_link._normalize_arxiv_id('1234.56789v2') == 'https://arxiv.org/abs/1234.56789'

    txt = 'See this paper: https://arxiv.org/abs/1234.56789v3 for details.'
    assert find_paper_link._extract_arxiv_from_text(txt) == 'https://arxiv.org/abs/1234.56789'

    txt2 = 'This is available at arxiv: 1234.56789v1'
    assert find_paper_link._extract_arxiv_from_text(txt2) == 'https://arxiv.org/abs/1234.56789'

    txt3 = 'identifier 1234.56789 appears'
    assert find_paper_link._extract_arxiv_from_text(txt3) == 'https://arxiv.org/abs/1234.56789'

    assert find_paper_link._extract_arxiv_from_text('nothing here') is None


def test__pdf_to_arxiv_reads_pages_and_extracts(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text
        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage('first page'), FakePage('see https://arxiv.org/abs/1234.56789v5 in second')]

    monkeypatch.setattr(find_paper_link, 'PdfReader', FakeReader)

    with tempfile.NamedTemporaryFile() as tf:
        res = find_paper_link._pdf_to_arxiv(tf.name, max_pages=2)
    assert res == 'https://arxiv.org/abs/1234.56789'

    class EmptyReader:
        def __init__(self, path):
            self.pages = [FakePage('nothing relevant'), FakePage('still nothing')]
    monkeypatch.setattr(find_paper_link, 'PdfReader', EmptyReader)
    with tempfile.NamedTemporaryFile() as tf:
        res2 = find_paper_link._pdf_to_arxiv(tf.name, max_pages=2)
    assert res2 is None
