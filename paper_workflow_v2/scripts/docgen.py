"""
Small document builder on top of python-docx (system python3), shared by the four
paper_workflow_v2 documents. Content is given as a list of blocks:

  ('h1', text) ('h2', text) ('h3', text)
  ('p', text)                     inline **bold** and *italic* supported
  ('bullets', [text, ...])        ('numbered', [text, ...])
  ('table', header, rows, widths_cm)
  ('fig', filename, width_cm, caption)
  ('eq', text)                    centred display line (Unicode maths)
  ('code', text)                  monospaced shaded block
  ('note', kind, title, text)     kind in {'key', 'warn', 'info'}
  ('pagebreak',)
"""

import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

V2 = Path(__file__).resolve().parent.parent
FIG = V2 / 'figures'
NUMBERS_TEX = V2.parent / 'Paper' / 'manuscript' / 'tables' / 'numbers.tex'
STATS_JSON = FIG / 'stats.json'


def numbers():
    """Every number of the manuscript (tables/numbers.tex, regenerated from the study results) plus the
    document-specific statistics computed by make_figures.py (figures/stats.json)."""
    import json
    N = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', NUMBERS_TEX.read_text()))
    N = {k: v.replace('--', '–') for k, v in N.items()}
    if STATS_JSON.exists():
        N.update(json.loads(STATS_JSON.read_text()))
    return N
NAVY = RGBColor(0x14, 0x3D, 0x6B)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
INK2 = RGBColor(0x52, 0x51, 0x4E)
FONT = 'Calibri'
NOTE_FILL = {'key': 'E8F1FC', 'warn': 'FDF0E8', 'info': 'EEF6F1'}
NOTE_BAR = {'key': '2A78D6', 'warn': 'EB6834', 'info': '1BAF7A'}


def _shade(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexfill)
    tcPr.append(shd)


def _cell_borders(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        spec = kw.get(edge)
        el = OxmlElement(f'w:{edge}')
        if spec:
            el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), str(spec[0])); el.set(qn('w:color'), spec[1])
        else:
            el.set(qn('w:val'), 'nil')
        b.append(el)
    tcPr.append(b)


def _fix_widths(t, widths_cm):
    """Fixed layout with explicit grid widths (respected by Word and LibreOffice)."""
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    tblW = tblPr.find(qn('w:tblW'))
    if tblW is None:
        tblW = OxmlElement('w:tblW'); tblPr.append(tblW)
    tblW.set(qn('w:w'), str(int(sum(widths_cm) * 567))); tblW.set(qn('w:type'), 'dxa')
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn('w:gridCol')), widths_cm):
        gc.set(qn('w:w'), str(int(w * 567)))
    for row in t.rows:
        for c, w in zip(row.cells, widths_cm):
            c.width = Cm(w)


def _field(run, instr):
    for kind, text in (('begin', None), (None, instr), ('separate', None), ('end', None)):
        if kind:
            el = OxmlElement('w:fldChar'); el.set(qn('w:fldCharType'), kind); run._r.append(el)
        else:
            el = OxmlElement('w:instrText'); el.set(qn('xml:space'), 'preserve'); el.text = text; run._r.append(el)


def _runs(par, text, size=None, color=None, italic=False):
    """Add text with **bold** / *italic* markup."""
    for tok in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text):
        if not tok:
            continue
        if tok.startswith('**'):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith('*'):
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        if italic:
            r.italic = True


class Doc:
    def __init__(self, title, subtitle, header_text, version_note):
        self.d = Document()
        self.header_text = header_text
        self.fig_no = 0
        self.tab_no = 0
        st = self.d.styles
        st['Normal'].font.name = FONT; st['Normal'].font.size = Pt(10.5)
        st['Normal'].element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
        pf = st['Normal'].paragraph_format
        pf.space_after = Pt(6); pf.line_spacing = 1.12
        for name, size, before, after, col in (('Heading 1', 15, 16, 6, NAVY), ('Heading 2', 12.5, 12, 4, NAVY),
                                               ('Heading 3', 11, 9, 3, BLUE)):
            s = st[name]
            s.font.name = FONT; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = col
            s.element.rPr.rFonts.set(qn('w:asciiTheme'), '') if False else None
            rf = s.element.rPr.rFonts
            for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
                rf.set(qn(a), FONT)
            s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after)
            s.paragraph_format.keep_with_next = True
        sec = self.d.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        for side in ('left_margin', 'right_margin'):
            setattr(sec, side, Cm(2.0))
        sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(1.9)
        sec.different_first_page_header_footer = True
        self._title_page(title, subtitle, version_note)
        self._header_footer(sec)
        self.toc_entries = []

    # ── page furniture ────────────────────────────────────────────────────
    def _header_footer(self, sec):
        hp = sec.header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _runs(hp, self.header_text, size=8, color=INK2)
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run('QIRAIL Laboratory · Department of Radiation Oncology · CMC Vellore    |    Page ')
        r.font.size = Pt(8); r.font.color.rgb = INK2
        r2 = fp.add_run(); r2.font.size = Pt(8); r2.font.color.rgb = INK2
        _field(r2, 'PAGE')

    def _title_page(self, title, subtitle, version_note):
        d = self.d
        for _ in range(5):
            d.add_paragraph()
        p = d.add_paragraph(); r = p.add_run('QIRAIL LABORATORY · CMC VELLORE'); r.font.size = Pt(10)
        r.font.color.rgb = BLUE; r.bold = True
        p = d.add_paragraph(); r = p.add_run(title); r.font.size = Pt(24); r.bold = True; r.font.color.rgb = NAVY
        p.paragraph_format.space_after = Pt(10)
        p = d.add_paragraph(); _runs(p, subtitle, size=13, color=INK2)
        p.paragraph_format.space_after = Pt(28)
        rows = [('Author', 'Hasan Shaikh'),
                ('Affiliation', 'QIRAIL Laboratory, Department of Radiation Oncology, Christian Medical College, Vellore'),
                ('Version', '2.0 (November 2026)'),
                ('Status', version_note)]
        t = d.add_table(rows=len(rows), cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.LEFT
        _fix_widths(t, [3.2, 13.8])
        for i, (k, v) in enumerate(rows):
            for j, (txt, w) in enumerate(((k, 3.2), (v, 13.8))):
                c = t.cell(i, j); c.width = Cm(w)
                c.paragraphs[0].text = ''
                _runs(c.paragraphs[0], txt, size=9.5, color=NAVY if j == 0 else None)
                if j == 0:
                    c.paragraphs[0].runs[0].bold = True
                _cell_borders(c, bottom=(4, 'D9D9D9'))
        p = d.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE)

    # ── content ───────────────────────────────────────────────────────────
    def contents(self, headings):
        self.d.add_heading('Contents', level=1)
        for i, h in enumerate(headings, 1):
            p = self.d.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            _runs(p, f'{i}.  {h}', size=10.5)
        p = self.d.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE)

    def add(self, blocks):
        h1 = 0
        for b in blocks:
            kind = b[0]
            if kind == 'h1':
                h1 += 1
                self.d.add_heading(f'{h1}  {b[1]}', level=1)
            elif kind == 'h2':
                self.d.add_heading(b[1], level=2)
            elif kind == 'h3':
                self.d.add_heading(b[1], level=3)
            elif kind == 'p':
                p = self.d.add_paragraph(); _runs(p, b[1])
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            elif kind == 'bullets':
                for item in b[1]:
                    p = self.d.add_paragraph(style='List Bullet'); _runs(p, item)
                    p.paragraph_format.space_after = Pt(3)
            elif kind == 'numbered':                       # explicit numbers: every list restarts at 1
                for k, item in enumerate(b[1], 1):
                    p = self.d.add_paragraph()
                    p.paragraph_format.left_indent = Cm(0.75)
                    p.paragraph_format.first_line_indent = Cm(-0.6)
                    p.paragraph_format.space_after = Pt(3)
                    p.paragraph_format.tab_stops.add_tab_stop(Cm(0.75))
                    _runs(p, f'{k}.\t' + item)
            elif kind == 'table':
                self._table(*b[1:])
            elif kind == 'fig':
                self._figure(*b[1:])
            elif kind == 'eq':
                p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(b[1]); r.font.name = 'Cambria Math'; r.font.size = Pt(11); r.italic = True
                p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(8)
            elif kind == 'code':
                t = self.d.add_table(rows=1, cols=1); _fix_widths(t, [17]); c = t.cell(0, 0)
                _shade(c, 'F4F4F2'); _cell_borders(c, left=(12, 'C9C8C2'))
                c.paragraphs[0].text = ''
                for k, line in enumerate(b[1].split('\n')):
                    p = c.paragraphs[0] if k == 0 else c.add_paragraph()
                    p.paragraph_format.space_after = Pt(0)
                    r = p.add_run(line); r.font.name = 'Consolas'; r.font.size = Pt(8.5)
                self.d.add_paragraph().paragraph_format.space_after = Pt(2)
            elif kind == 'note':
                _, nk, title, text = b
                t = self.d.add_table(rows=1, cols=1); _fix_widths(t, [17]); c = t.cell(0, 0)
                _shade(c, NOTE_FILL[nk]); _cell_borders(c, left=(24, NOTE_BAR[nk]))
                c.paragraphs[0].text = ''
                r = c.paragraphs[0].add_run(title); r.bold = True; r.font.size = Pt(10)
                r.font.color.rgb = RGBColor.from_string(NOTE_BAR[nk])
                for para in text.split('\n'):
                    p = c.add_paragraph(); _runs(p, para, size=10)
                    p.paragraph_format.space_after = Pt(2)
                self.d.add_paragraph().paragraph_format.space_after = Pt(2)
            elif kind == 'pagebreak':
                self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            else:
                raise ValueError(kind)

    def _table(self, header, rows, widths, caption=None):
        if caption:
            self.tab_no += 1
            p = self.d.add_paragraph(); p.paragraph_format.keep_with_next = True
            _runs(p, f'**Table {self.tab_no}.** {caption}', size=9.5)
        t = self.d.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        _fix_widths(t, widths)
        for j, (h, w) in enumerate(zip(header, widths)):
            c = t.cell(0, j); c.width = Cm(w); _shade(c, '143D6B')
            c.paragraphs[0].text = ''
            r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(255, 255, 255)
            _cell_borders(c, bottom=(6, '143D6B'))
        for i, row in enumerate(rows, 1):
            for j, (v, w) in enumerate(zip(row, widths)):
                c = t.cell(i, j); c.width = Cm(w)
                if i % 2 == 0:
                    _shade(c, 'F3F6FA')
                c.paragraphs[0].text = ''
                _runs(c.paragraphs[0], str(v), size=9)
                c.paragraphs[0].paragraph_format.space_after = Pt(0)
                _cell_borders(c, bottom=(2, 'D9D9D9'))
        n = len(t.rows)
        for i, row in enumerate(t.rows):                 # keep the table (and its header) on one page
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement('w:cantSplit'); trPr.append(cs)
            if i == 0:
                th = OxmlElement('w:tblHeader'); trPr.append(th)
            if i < n - 1 and n <= 14:
                for c in row.cells:
                    for par in c.paragraphs:
                        par.paragraph_format.keep_with_next = True
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def _figure(self, fname, width_cm, caption):
        self.fig_no += 1
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(FIG / fname), width=Cm(width_cm))
        c = self.d.add_paragraph(); _runs(c, f'**Figure {self.fig_no}.** {caption}', size=9, color=INK2)
        c.paragraph_format.space_after = Pt(10)

    def save(self, name):
        out = V2 / f'{name}.docx'
        self.d.save(out)
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(V2), str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
