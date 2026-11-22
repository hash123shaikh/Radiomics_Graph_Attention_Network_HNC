"""
Study report (Markdown + Word + PDF) for the whole-tumour population-graph analysis.

Every number is read from tables/numbers.tex and every table from tables/*.tex, both written by
make_assets.py, so the report, the manuscript and the results cannot disagree. Run with the system
python (python-docx); the PDF is converted with LibreOffice.

  /usr/bin/python3 Paper/manuscript_popgraph/scripts/make_report.py
-> grafs-gat/popgraph/REPORT.md, Paper/manuscript_popgraph/report/popgraph_report.{docx,pdf}
"""
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
TAB, FIG = HERE / 'tables', HERE / 'figures'
OUT = HERE / 'report'
MD = ROOT / 'grafs-gat' / 'popgraph' / 'REPORT.md'

N = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', (TAB / 'numbers.tex').read_text()))
N = {k: v.replace('--', '–') for k, v in N.items()}


def T(s):
    """Fill {macro} placeholders."""
    return s.format(**N)


def latex_rows(name):
    """Rows of a generated LaTeX tabular as lists of plain-text cells."""
    rows = []
    for line in (TAB / f'{name}.tex').read_text().splitlines():
        line = line.strip()
        if not line.endswith('\\\\') or line.startswith('\\begin'):
            continue
        if '\\multicolumn' in line or '\\cmidrule' in line:
            continue
        cells = [c.strip() for c in line[:-2].split('&')]
        clean = []
        for c in cells:
            c = re.sub(r'\\textbf\{([^}]*)\}', r'\1', c)
            c = c.replace('$^{a}$', 'ᵃ').replace('$^{b}$', 'ᵇ').replace('$^{c}$', 'ᶜ')
            c = c.replace('cm$^3$', 'cm³').replace('$\\Delta$', 'Δ').replace('$p$', 'p').replace('$k$', 'k')
            c = c.replace('\\%', '%').replace('--', '–').replace('\\_', '_').replace('\\ ', ' ')
            c = re.sub(r'\\[a-zA-Z]+', '', c).replace('{', '').replace('}', '').replace('$', '')
            clean.append(c)
        rows.append(clean)
    return rows


SECTIONS = [
    ('Summary', [
        'This report summarises the population-graph analysis carried out on the protocol of Mateus et al., '
        'in which a published convolutional neural network (CNN) was retrained on the Canadian Head-Neck-PET-CT '
        'centres and tested on MAASTRO (HN1). Population graph neural networks (GCN, GraphSAGE and GAT; one node '
        'per patient, edges to the most similar training patients) were compared with the identical network '
        'without edges (MLP), with conventional classifiers and with the CNN, on radiomic features of the primary '
        'gross tumour volume (GTV) or of the whole tumour, that is, the union of the primary and nodal GTVs.',
        'With whole-tumour features the graph networks chosen on validation reached external AUCs of '
        '{mainLRMin}–{mainLRMax} for locoregional recurrence (LR), {mainDMMin}–{mainDMMax} for distant metastasis (DM) '
        'and {mainOSMin}–{mainOSMax} for overall survival (OS), and exceeded the CNN in {winsGraph} of 12 '
        'endpoint–design–input settings, against {winsPrimVol} of 12 with primary-GTV features. The gain came from '
        'the tumour definition and not from tumour-burden clinical variables ({winsTotVol} and {winsTotBur} of 12 '
        'with and without them). The patient graph added nothing: the MLP without edges exceeded the CNN in '
        '{winsMLP} of 12 settings, only {pairedNsig} of {pairedN} paired graph–MLP differences was nominally '
        'significant and it favoured the MLP, and with identical features the graph network was better on MAASTRO '
        'in only {matchedTestFracAll}% of {matchedN} matched comparisons. All remaining shortfalls against the CNN '
        'were for DM, for which MAASTRO contained only {eTestDM} events.',
    ]),
    ('Data and design', [
        'Development used {nPatCan} Canadian patients and external testing {nPatMAA} MAASTRO patients, with the '
        'endpoints, horizons, exclusions and the two development designs (stratified 5-fold cross-validation and '
        'the HGJ+CHUS versus HMR+CHUM cohort split) of Mateus et al. After exclusions MAASTRO contributed '
        '{nTestLR} patients with {eTestLR} LR events, {nTestDM} with {eTestDM} DM events and {nTestOS} with '
        '{eTestOS} deaths. Whole-tumour features (479 per patient: shape, original and Laplacian-of-Gaussian '
        'first-order and texture features) were extracted with PyRadiomics after 1 mm resampling. The grid '
        'crossed two feature sets, three selectors, three subset sizes, three neighbourhood sizes and the four '
        'networks, every configuration was a ten-seed ensemble, and the configuration with the highest '
        'validation AUC was chosen separately for the graph networks and for the MLP; MAASTRO labels were never '
        'used for training or choosing.',
        'Nodal GTVs were delineated separately in {nodalPctCan}% of Canadian but {nodalPctMAA}% of MAASTRO patients. '
        'In one Canadian centre (HGJ) separate nodal GTVs existed for only 12% of patients although 89% were node '
        'positive, and its primary GTVs were the largest, which suggests that nodes were included in the structure '
        'designated as the primary tumour. The whole-tumour definition describes the same anatomy in both cohorts.',
    ]),
    ('External discrimination', [
        'Table 1 compares the chosen models with the CNN. For LR and OS the graph networks exceeded the CNN in all '
        'four settings. For DM they exceeded it in cross-validation with imaging only ({mainDMCVImg} versus '
        '{cnnDMCVImg}), were within 0.02 in two settings, and remained below it in the cohort split with clinical '
        'data ({mainDMCohClin}, 95% CI {mainCIDMCohClin}, versus {cnnDMCohClin}). A graph network on clinical '
        'variables alone reached {clinOnlyLRMin}–{clinOnlyLRMax} for LR, {clinOnlyDMMin}–{clinOnlyDMMax} for DM and '
        '{clinOnlyOSMin}–{clinOnlyOSMax} for OS.',
    ]),
    ('Tumour definition versus tumour burden', [
        'Table 2 shows the 2×2 ablation. Replacing primary-GTV by whole-tumour features raised the DM AUC from '
        '{ablPrimVolDMMin}–{ablPrimVolDMMax} to {ablTotVolDMMin}–{ablTotVolDMMax} and the OS AUC from '
        '{ablPrimVolOSMin}–{ablPrimVolOSMax} to {ablTotVolOSMin}–{ablTotVolOSMax}, and left LR unchanged. '
        'Tumour-burden clinical variables helped the primary-GTV models for DM but added nothing once the nodes '
        'were part of the radiomic region.',
    ]),
    ('Does the graph help?', [
        'Table 3 compares the chosen graph network with the chosen MLP on the same MAASTRO patients. The '
        'differences ranged from {pairedDeltaMin} to {pairedDeltaMax} AUC and only one reached nominal '
        'significance (p = {pairedMinP}), in favour of the MLP. Across all {matchedN} graph configurations, each '
        'compared with the MLP on identical features, the graph network had the higher validation AUC in '
        '{matchedValFracAll}% and the higher external AUC in {matchedTestFracAll}% (median difference '
        '{matchedTestMedAll}).',
    ]),
    ('Training randomness and conventional classifiers', [
        'Retraining every chosen configuration with {seedRepeats} disjoint ten-seed ensembles gave a median '
        'standard deviation of the external AUC of {seedSDmed} (maximum {seedSDmax}); the graph network exceeded '
        'the CNN in every re-run in {seedStableWins} of 12 settings and in none in {seedStableLoss}. Conventional '
        'classifiers on the same whole-tumour features exceeded the CNN in {winsCls} of 12 settings (LR '
        '{clsLRMin}–{clsLRMax}, DM {clsDMMin}–{clsDMMax}, OS {clsOSMin}–{clsOSMax}). For LR, logistic regression on '
        'the clinical variables of the earlier analysis alone, which include age and HPV status, reached '
        '{clsClinLRMin}–{clsClinLRMax} and outperformed every imaging model; the graph networks used the clinical '
        'variables of the CNN study (T, N, volume), which contain neither, so the CNN comparison does not show '
        'that imaging adds to a complete clinical model for LR.',
    ]),
    ('Interpretation and limitations', [
        'Radiomic models of the whole tumour match or exceed the published CNN for LR and OS and approach it for '
        'DM, and the improvement over the earlier primary-GTV models is explained by the tumour definition, which '
        'captures nodal disease and harmonises inconsistent primary delineations, and not by the patient graph. '
        'The whole-tumour representation was introduced after the primary-GTV results were known, so MAASTRO was '
        'used more than once; this is mitigated by its a-priori motivation (the CNN sees the nodes), by '
        'validation-only model choice and by reporting the full ablation, but it should be confirmed on an '
        'independent cohort. The DM test set has only {eTestDM} events, the CNN comparison is descriptive because '
        'its predictions are unavailable, and nodal delineations in public collections are incomplete.',
    ]),
]

TABLES = {
    'External discrimination': ('table_main', 'Table 1. External AUC on MAASTRO of the whole-tumour models chosen on '
                                'validation versus the CNN (CV: mean (range) of per-fold AUCs; cohort split: AUC '
                                '[95% CI]). ᵃ Mateus et al.; ᵇ whole tumour with tumour-burden variables; '
                                'ᶜ logistic regression or random forest on the same features.', 'fig_auc_vs_cnn'),
    'Tumour definition versus tumour burden': ('table_ablation', 'Table 2. Representation ablation (external AUC of '
                                               'the chosen graph network; columns: LR, DM, OS × CV/cohort × '
                                               'imaging/+clinical).', 'fig_ablation'),
    'Does the graph help?': ('table_paired', 'Table 3. Chosen graph network versus chosen MLP on MAASTRO (paired '
                             'bootstrap).', 'fig_graph_vs_mlp'),
    'Training randomness and conventional classifiers': (None, None, 'fig_seed'),
    'Data and design': ('table_burden', 'Tumour burden by cohort (median, IQR).', 'fig_ct_tumour'),
}
ABL_HEAD = ['Region', 'Clinical'] + [f'{t} {d} {i}' for t in ('LR', 'DM', 'OS') for d in ('CV', 'Coh')
                                     for i in ('Img', '+Cl')] + ['Wins']


def build_docx():
    doc = Document()
    st = doc.styles['Normal']
    st.font.name, st.font.size = 'Calibri', Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(1.8)
        s.top_margin = s.bottom_margin = Cm(1.8)
    h = doc.add_heading('Whole-tumour population graph analysis: report', 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph('GRAFS-GAT project, population-graph study under the protocol of Mateus et al. '
                      'All values are aggregate results; no patient-level data are included.').runs[0].italic = True
    md = ['# Whole-tumour population graph analysis: report', '',
          '*Population-graph study under the protocol of Mateus et al. Aggregate results only; generated by '
          '`Paper/manuscript_popgraph/scripts/make_report.py` from the manuscript tables.*', '']
    for title, paras in SECTIONS:
        doc.add_heading(title, 1)
        md += [f'## {title}', '']
        for p in paras:
            q = doc.add_paragraph(T(p))
            q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            md += [T(p), '']
        if title in TABLES:
            name, cap, fig = TABLES[title]
            if name:
                rows = latex_rows(name)
                if name == 'table_ablation':
                    rows = [ABL_HEAD] + [r for r in rows if r and r[0] not in ('', 'Tumour region')]
                doc.add_paragraph(cap).runs[0].bold = True
                t = doc.add_table(rows=len(rows), cols=len(rows[0]))
                t.style = 'Light Grid Accent 1'
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                for i, r in enumerate(rows):
                    for j, c in enumerate(r[:len(rows[0])]):
                        cell = t.cell(i, j)
                        cell.text = c
                        for par in cell.paragraphs:
                            for run in par.runs:
                                run.font.size = Pt(7 if len(rows[0]) > 8 else 8)
                                run.bold = i == 0
                md += [f'**{cap}**', '', '| ' + ' | '.join(rows[0]) + ' |',
                       '|' + '---|' * len(rows[0])]
                md += ['| ' + ' | '.join(r) + ' |' for r in rows[1:]] + ['']
            if fig and (FIG / f'{fig}.png').exists():
                doc.add_picture(str(FIG / f'{fig}.png'), width=Cm(17.4))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                md += [f'![{fig}](../../Paper/manuscript_popgraph/figures/{fig}.png)', '']
    OUT.mkdir(exist_ok=True)
    f = OUT / 'popgraph_report.docx'
    doc.save(f)
    MD.write_text('\n'.join(md))
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(OUT), str(f)],
                   check=True, capture_output=True, timeout=300)
    print('wrote', MD, f, OUT / 'popgraph_report.pdf')


if __name__ == '__main__':
    build_docx()
