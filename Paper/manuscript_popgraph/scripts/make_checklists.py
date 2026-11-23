"""
Completed reporting checklists (TRIPOD+AI, CLAIM 2024, CLEAR) and quality scores (METRICS, RQS) for the
Supplementary Information. Locations refer to the sections, tables and figures of main.tex.

  python Paper/manuscript_popgraph/scripts/make_checklists.py  ->  tables/tabS_checklists.tex
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'tables' / 'tabS_checklists.tex'

# (item, topic, reported, location / comment)
TRIPOD = [
    ('Title', None),
    ('1', 'Title', 'Yes', 'Title (population graph neural networks, outcome prediction, head and neck cancer)'),
    ('Abstract', None),
    ('2', 'Abstract', 'Yes', 'Structured abstract'),
    ('Introduction', None),
    ('3a', 'Healthcare context and rationale', 'Yes', 'Introduction, paragraphs 1--2'),
    ('3b', 'Target population, intended purpose and users', 'Partly', 'Introduction, paragraph 1; research-stage model, intended users not specified'),
    ('3c', 'Known health inequalities', 'No', 'Not addressed'),
    ('4', 'Objectives', 'Yes', 'Introduction, last paragraph'),
    ('Methods', None),
    ('5a', 'Data sources and representativeness', 'Yes', 'Section 2.1; Data availability'),
    ('5b', 'Dates of data collection and follow-up', 'No', 'Given in the collection reports, not in the manuscript'),
    ('6a', 'Setting, number and location of centres', 'Yes', 'Section 2.1'),
    ('6b', 'Eligibility criteria', 'Yes', 'Section 2.1; Table 1, note d'),
    ('6c', 'Treatments received and their handling', 'Partly', 'Table 1 (concurrent chemotherapy); extended clinical set, Section 2.6'),
    ('7', 'Data pre-processing and quality checking', 'Yes', 'Sections 2.2 and 2.4'),
    ('8a', 'Outcome definition and time horizon', 'Yes', 'Section 2.1'),
    ('8b', 'Qualifications of outcome assessors', 'n/a', 'Outcomes taken from the clinical records of the collections'),
    ('8c', 'Blinding of outcome assessment', 'Partly', 'Outcomes recorded in the collections independently of this study'),
    ('9a', 'Choice of initial predictors', 'Yes', 'Sections 2.2, 2.3 and 2.6'),
    ('9b', 'Definition of predictors', 'Yes', 'Sections 2.2, 2.3 and 2.6'),
    ('9c', 'Qualifications of predictor assessors', 'Partly', 'Section 2.2 (clinical GTVs by radiation oncologists); experience not documented'),
    ('10', 'Sample size', 'Yes', 'Section 2.1 (fixed by the public collections, no calculation); events in Table 1'),
    ('11', 'Missing data', 'Yes', 'Sections 2.3 and 2.4'),
    ('12a', 'Use and partitioning of data', 'Yes', 'Sections 2.1 and 2.5'),
    ('12b', 'Handling of predictors', 'Yes', 'Section 2.4'),
    ('12c', 'Model type, building steps, tuning, internal validation', 'Yes', 'Sections 2.4--2.6'),
    ('12d', 'Heterogeneity across clusters', 'Partly', 'Centre-based cohort split; per-fold ranges (Table 2)'),
    ('12e', 'Performance measures and model comparison', 'Partly', 'Section 2.8 (AUC, paired bootstrap); calibration and clinical utility not assessed'),
    ('12f', 'Model updating', 'n/a', 'No updating performed'),
    ('12g', 'Calculation of predictions in evaluation', 'Yes', 'Section 2.4 (mean probability of the ten-seed ensemble); released code'),
    ('13', 'Class imbalance', 'Yes', 'Weighted loss (Section 2.4); balanced class weights (Section 2.6)'),
    ('14', 'Fairness', 'No', 'Not assessed (Discussion, limitations)'),
    ('15', 'Model output and thresholds', 'Yes', 'Predicted probability, no threshold (Sections 2.4 and 2.8)'),
    ('16', 'Differences between development and evaluation data', 'Yes', 'Table 1; Section 3.1; Discussion'),
    ('Open science', None),
    ('17', 'Ethical approval', 'Yes', 'Declarations'),
    ('18a', 'Funding', 'Yes', 'Declarations'),
    ('18b', 'Conflicts of interest', 'Yes', 'Declarations'),
    ('18c', 'Protocol', 'Yes', 'Section 2.1 (no protocol prepared)'),
    ('18d', 'Registration', 'Yes', 'Section 2.1 (not registered)'),
    ('18e', 'Data sharing', 'Yes', 'Data availability'),
    ('18f', 'Code sharing', 'Yes', 'Code availability'),
    ('19', 'Patient and public involvement', 'Yes', 'Declarations (none)'),
    ('Results', None),
    ('20a', 'Flow of participants', 'Partly', 'Table 1 and Section 3.1; no diagram, follow-up time not summarised'),
    ('20b', 'Characteristics of participants', 'Yes', 'Table 1, including missing HPV status'),
    ('20c', 'Comparison of development and evaluation data', 'Yes', 'Table 1'),
    ('21', 'Participants and events in each analysis', 'Yes', 'Table 1 (outcome cohorts); Section 3.1'),
    ('22', 'Full model specification', 'Partly', 'Section 2.4; Supplementary Table S1; released code (trained weights not released)'),
    ('23a', 'Performance with confidence intervals', 'Yes', 'Tables 2--4; Figs. 3--6; subgroups not analysed'),
    ('23b', 'Heterogeneity in performance across clusters', 'Partly', 'Per-fold ranges (Table 2)'),
    ('24', 'Model updating results', 'n/a', 'No updating performed'),
    ('Discussion', None),
    ('25', 'Interpretation', 'Yes', 'Discussion, paragraphs 1--5'),
    ('26', 'Limitations', 'Yes', 'Discussion, last paragraph'),
    ('27a', 'Handling of poor-quality or missing input data', 'Partly', 'Discussion (delineation conventions across cohorts)'),
    ('27b', 'User interaction and required expertise', 'Partly', 'Model requires radiotherapy GTV delineations (Section 2.2)'),
    ('27c', 'Next steps for future research', 'Yes', 'Discussion, last paragraph; Conclusion'),
]

CLAIM = [
    ('Title and abstract', None),
    ('1', 'Identification as an AI study, with category of technology', 'Yes', 'Title; Abstract'),
    ('2', 'Summary of design, methods, results and conclusions', 'Yes', 'Abstract'),
    ('Introduction', None),
    ('3', 'Background, intended use and role of the AI approach', 'Yes', 'Introduction, paragraphs 1--3'),
    ('4', 'Aims, objectives and hypotheses', 'Yes', 'Introduction, last paragraph'),
    ('Methods', None),
    ('5', 'Prospective or retrospective study', 'Yes', 'Section 2.1 (retrospective)'),
    ('6', 'Study goal', 'Yes', 'Introduction, last paragraph; Section 2.1'),
    ('7', 'Data sources', 'Yes', 'Section 2.1; Data availability'),
    ('8', 'Inclusion and exclusion criteria', 'Yes', 'Section 2.1; Table 1, note d'),
    ('9', 'Data pre-processing', 'Yes', 'Sections 2.2 and 2.4'),
    ('10', 'Selection of data subsets', 'Yes', 'Section 2.2 (radiomic regions)'),
    ('11', 'De-identification methods', 'Yes', 'De-identified by The Cancer Imaging Archive (Declarations)'),
    ('12', 'Handling of missing data', 'Yes', 'Sections 2.3 and 2.4'),
    ('13', 'Image acquisition protocol', 'Partly', 'Section 2.2 (image source; parameters in the collection reports)'),
    ('14', 'Definition of the reference standard', 'Yes', 'Section 2.1'),
    ('15', 'Rationale for the reference standard', 'Yes', 'Section 2.1 (endpoints of the CNN protocol)'),
    ('16', 'Source of reference standard annotations', 'Yes', 'Section 2.1 (clinical follow-up of the collections)'),
    ('17', 'Annotation of the test set', 'Yes', 'Sections 2.1 and 2.2'),
    ('18', 'Inter- and intrarater variability', 'No', 'One delineation per patient (Section 2.2)'),
    ('19', 'Assignment of data to partitions', 'Yes', 'Section 2.1'),
    ('20', 'Level at which partitions are disjoint', 'Yes', 'Section 2.1 (patient and centre level)'),
    ('21', 'Intended sample size', 'Partly', 'Section 2.1 (fixed by the collections)'),
    ('22', 'Detailed description of the model', 'Yes', 'Section 2.4; Fig. 2'),
    ('23', 'Software libraries, frameworks and packages', 'Yes', 'Section 2.8 (with versions)'),
    ('24', 'Initialisation of model parameters', 'Yes', 'Section 2.4'),
    ('25', 'Details of the training approach', 'Yes', 'Section 2.4'),
    ('26', 'Method of selecting the final model', 'Yes', 'Section 2.5'),
    ('27', 'Ensembling techniques', 'Yes', 'Section 2.4 (ten-seed averaging)'),
    ('28', 'Metrics of model performance', 'Yes', 'Section 2.8'),
    ('29', 'Statistical measures of significance and uncertainty', 'Yes', 'Section 2.8'),
    ('30', 'Robustness or sensitivity analysis', 'Yes', 'Sections 2.7 and 2.8 (ablation, seed re-runs)'),
    ('31', 'Methods for explainability or interpretability', 'No', 'Not performed'),
    ('32', 'Evaluation on internal data', 'Yes', 'Validation AUCs (Supplementary Table S1)'),
    ('33', 'Testing on external data', 'Yes', 'MAASTRO (HN1), Sections 2.1 and 3'),
    ('34', 'Clinical trial registration', 'n/a', 'Not a clinical trial'),
    ('Results', None),
    ('35', 'Numbers of patients included and excluded', 'Partly', 'Table 1; Section 3.1; exclusions as in the CNN study'),
    ('36', 'Characteristics of cases in each partition', 'Partly', 'Table 1 (development and test cohorts)'),
    ('37', 'Performance metrics with statistical uncertainty', 'Yes', 'Tables 2--4; Figs. 3--6'),
    ('38', 'Estimates of diagnostic performance', 'No', 'Threshold-free AUC only'),
    ('39', 'Failure analysis of incorrectly classified cases', 'No', 'Not performed'),
    ('Discussion', None),
    ('40', 'Study limitations', 'Yes', 'Discussion, last paragraph'),
    ('41', 'Implications for practice', 'Yes', 'Discussion; Conclusion'),
    ('Other information', None),
    ('42', 'Protocol or additional technical details', 'Yes', 'Supplementary Information; code repository'),
    ('43', 'Availability of software, model and data', 'Yes', 'Data and Code availability (trained models not released)'),
    ('44', 'Sources of funding and role of funders', 'Yes', 'Declarations (no funding)'),
]

CLEAR = [
    ('Title, abstract and keywords', None),
    ('1', 'Relevant title, specifying the radiomic methodology', 'Partly', 'Title names the model and task but not radiomics'),
    ('2', 'Structured summary with relevant information', 'Yes', 'Abstract'),
    ('3', 'Relevant keywords for radiomics', 'Yes', 'Keywords'),
    ('Introduction', None),
    ('4', 'Scientific or clinical background', 'Yes', 'Introduction, paragraph 1'),
    ('5', 'Rationale for using a radiomic approach', 'Yes', 'Introduction, paragraphs 2--3'),
    ('6', 'Study objective(s)', 'Yes', 'Introduction, last paragraph'),
    ('Method: study design', None),
    ('7', 'Adherence to guidelines or checklists', 'Yes', 'Section 2.1; Supplementary Tables S3--S7'),
    ('8', 'Ethical details', 'Yes', 'Declarations'),
    ('9', 'Sample size calculation', 'Yes', 'Section 2.1 (based on availability)'),
    ('10', 'Study nature', 'Yes', 'Section 2.1 (retrospective cohorts)'),
    ('11', 'Eligibility criteria', 'Yes', 'Section 2.1; Table 1, note d'),
    ('12', 'Flowchart for technical pipeline', 'No', 'Not provided'),
    ('Method: data', None),
    ('13', 'Data source', 'Yes', 'Section 2.1; Data availability (public)'),
    ('14', 'Data overlap', 'Yes', 'Section 2.1'),
    ('15', 'Data split methodology', 'Yes', 'Sections 2.1 and 2.5'),
    ('16', 'Imaging protocol', 'Partly', 'Section 2.2 (parameters in the collection reports)'),
    ('17', 'Definition of non-radiomic predictor variables', 'Yes', 'Sections 2.3 and 2.6'),
    ('18', 'Definition of the reference standard', 'Yes', 'Section 2.1'),
    ('Method: segmentation', None),
    ('19', 'Segmentation strategy', 'Yes', 'Section 2.2; Fig. 1'),
    ('20', 'Details of operators performing segmentation', 'Partly', 'Section 2.2 (radiation oncologists at the treating centres)'),
    ('Method: pre-processing', None),
    ('21', 'Image pre-processing details', 'Yes', 'Section 2.2'),
    ('22', 'Resampling method and its parameters', 'Yes', 'Section 2.2 (1 mm, linear / nearest neighbour)'),
    ('23', 'Discretization method and its parameters', 'Yes', 'Section 2.2 (fixed bin width 25 HU)'),
    ('24', 'Image types', 'Yes', 'Section 2.2 (original; LoG, $\\sigma=1$--5 mm)'),
    ('Method: feature extraction', None),
    ('25', 'Feature extraction method', 'Yes', 'Section 2.2 (PyRadiomics 3.0.1, 3D)'),
    ('26', 'Feature classes', 'Yes', 'Section 2.2'),
    ('27', 'Number of features', 'Yes', 'Section 2.2 (479)'),
    ('28', 'Default configuration statement', 'Yes', 'Section 2.2'),
    ('Method: data preparation', None),
    ('29', 'Handling of missing data', 'Yes', 'Sections 2.3 and 2.4'),
    ('30', 'Details of class imbalance', 'Yes', 'Table 1 (event rates); Sections 2.4 and 2.6'),
    ('31', 'Details of segmentation reliability analysis', 'No', 'Not possible, one delineation per patient (Section 2.2)'),
    ('32', 'Feature scaling details', 'Yes', 'Section 2.4 (training statistics only)'),
    ('33', 'Dimension reduction details', 'Yes', 'Section 2.5'),
    ('Method: modelling', None),
    ('34', 'Algorithm details', 'Yes', 'Sections 2.4 and 2.6'),
    ('35', 'Training and tuning details', 'Yes', 'Sections 2.4 and 2.5'),
    ('36', 'Handling of confounders', 'No', 'Cohort differences reported (Table 1) but not adjusted for'),
    ('37', 'Model selection strategy', 'Yes', 'Section 2.5 (validation AUC only)'),
    ('Method: evaluation', None),
    ('38', 'Testing technique', 'Yes', 'Section 2.1 (external, one institution)'),
    ('39', 'Performance metrics and rationale', 'Yes', 'Section 2.8 (AUC, as in the CNN study)'),
    ('40', 'Uncertainty evaluation and measures', 'Yes', 'Section 2.8 (bootstrap intervals, seed re-runs)'),
    ('41', 'Statistical performance comparison', 'Yes', 'Section 2.8 (paired bootstrap; no multiplicity correction)'),
    ('42', 'Comparison with non-radiomic and combined methods', 'Yes', 'Sections 2.5 and 2.6 (clinical-only models)'),
    ('43', 'Interpretability and explainability methods', 'No', 'Not performed'),
    ('Results', None),
    ('44', 'Baseline demographic and clinical characteristics', 'Yes', 'Table 1 (no statistical tests between cohorts)'),
    ('45', 'Flowchart for eligibility criteria', 'No', 'Numbers in Table 1; no flowchart'),
    ('46', 'Feature statistics', 'Partly', 'Selector and subset size in Supplementary Table S1; feature names not listed'),
    ('47', 'Model performance evaluation', 'Yes', 'Validation (Supplementary Table S1) and test (Tables 2--4)'),
    ('48', 'Comparison with non-radiomic and combined approaches', 'Yes', 'Section 3.5; Table 2'),
    ('Discussion', None),
    ('49', 'Overview of important findings', 'Yes', 'Discussion, paragraph 1'),
    ('50', 'Previous works with differences from the current study', 'Yes', 'Discussion, paragraphs 3--4'),
    ('51', 'Practical implications', 'Yes', 'Discussion; Conclusion'),
    ('52', 'Strengths and limitations', 'Yes', 'Discussion, last paragraph'),
    ('Open science', None),
    ('53', 'Sharing images along with segmentation data (n/e)', 'Yes', 'Public TCIA collections (Data availability)'),
    ('54', 'Sharing radiomic feature data', 'No', 'Regenerable from the public data with the released code'),
    ('55', 'Sharing pre-processing scripts or settings', 'Yes', 'Code availability'),
    ('56', 'Sharing source code for modelling', 'Yes', 'Code availability'),
    ('57', 'Sharing final model files', 'No', 'Not released; reproducible with fixed seeds'),
    ('58', 'Sharing a ready-to-use system (n/e)', 'No', 'Not provided'),
]

# METRICS (Kocak et al. 2024): item, category, weight, present (True / False / None = condition not met)
METRICS = [
    ('1', 'Study design', 'Adherence to radiomics and/or machine learning-specific checklists or guidelines', 0.0368, True, 'Section 2.1'),
    ('2', '', 'Eligibility criteria that describe a representative study population', 0.0735, True, 'Section 2.1'),
    ('3', '', 'High-quality reference standard with a clear definition', 0.0919, True, 'Section 2.1'),
    ('4', 'Imaging data', 'Multi-center', 0.0438, True, 'Five centres'),
    ('5', '', 'Clinical translatability of the imaging data source', 0.0292, True, 'Routine pretreatment CT'),
    ('6', '', 'Imaging protocol with acquisition parameters', 0.0438, False, 'Referred to collection reports'),
    ('7', '', 'Interval between imaging used and reference standard', 0.0292, True, 'Pretreatment CT; fixed horizons'),
    ('8', 'Segmentation', 'Transparent description of segmentation methodology', 0.0337, True, 'Section 2.2; Fig. 1'),
    ('9', '', 'Formal evaluation of fully automated segmentation', 0.0225, None, 'Condition not met (manual)'),
    ('10', '', 'Test set masks produced by a single reader or automated tool', 0.0112, False, 'Not documented'),
    ('11', 'Image processing and', 'Appropriate image pre-processing with transparent description', 0.0622, True, 'Section 2.2'),
    ('12', '\\quad feature extraction', 'Use of standardized feature extraction software', 0.0311, True, 'PyRadiomics'),
    ('13', '', 'Transparent extraction parameters or default configuration statement', 0.0415, True, 'Section 2.2'),
    ('14', 'Feature processing', 'Removal of non-robust features', 0.0200, False, 'No robustness analysis'),
    ('15', '', 'Removal of redundant features', 0.0200, True, 'mRMR; correlation filter'),
    ('16', '', 'Appropriateness of dimensionality compared to data size', 0.0300, True, '2--15 selected features'),
    ('17', '', 'Robustness assessment of end-to-end deep learning pipelines', 0.0200, None, 'Condition not met'),
    ('18', 'Preparation for', 'Proper data partitioning process', 0.0599, True, 'Section 2.1'),
    ('19', '\\quad modelling', 'Handling of confounding factors', 0.0300, False, 'Not adjusted'),
    ('20', 'Metrics and', 'Appropriate performance evaluation metrics for task', 0.0352, True, 'AUC'),
    ('21', '\\quad comparison', 'Consideration of uncertainty', 0.0234, True, 'Bootstrap; seed re-runs'),
    ('22', '', 'Calibration assessment', 0.0176, False, 'Not assessed'),
    ('23', '', 'Use of uni-parametric imaging or proof of its inferiority', 0.0117, True, 'CT only'),
    ('24', '', 'Comparison with a non-radiomic approach or proof of added clinical value', 0.0293, True, 'Clinical-only models'),
    ('25', '', 'Comparison with simple or classical statistical models', 0.0176, True, 'Logistic regression'),
    ('26', 'Testing', 'Internal testing', 0.0375, False, 'Validation used for selection only'),
    ('27', '', 'External testing', 0.0749, True, 'MAASTRO (HN1)'),
    ('28', 'Open science', 'Data availability', 0.0075, True, 'Public TCIA data'),
    ('29', '', 'Code availability', 0.0075, True, 'Public repository'),
    ('30', '', 'Model availability', 0.0075, False, 'Not released'),
]

# RQS (Lambin et al. 2017): item, criterion, points awarded, possible range, justification
RQS = [
    ('1', 'Image protocol quality', 0, '0 to 2', 'Protocols not documented in the manuscript'),
    ('2', 'Multiple segmentations', 0, '0 to 1', 'One delineation per patient'),
    ('3', 'Phantom study on all scanners', 0, '0 to 1', 'Not performed'),
    ('4', 'Imaging at multiple time points', 0, '0 to 1', 'Not performed'),
    ('5', 'Feature reduction or adjustment for multiple testing', 3, '$-$3 or 3', 'Selection on training data (Section 2.5)'),
    ('6', 'Multivariable analysis with non-radiomic features', 1, '0 to 1', 'Imaging with clinical inputs'),
    ('7', 'Detection and discussion of biological correlates', 0, '0 to 1', 'Not measured'),
    ('8', 'Cut-off analyses', 0, '0 to 1', 'No risk groups'),
    ('9', 'Discrimination statistics', 2, '0 to 2', 'AUC with bootstrap intervals'),
    ('10', 'Calibration statistics', 0, '0 to 2', 'Not assessed'),
    ('11', 'Prospective study registered in a trial database', 0, '0 to 7', 'Retrospective'),
    ('12', 'Validation', 3, '$-$5 to 5', 'External, one other institution'),
    ('13', 'Comparison to gold standard', 2, '0 to 2', 'T and N category models'),
    ('14', 'Potential clinical utility', 0, '0 to 2', 'No decision curve analysis'),
    ('15', 'Cost-effectiveness analysis', 0, '0 to 1', 'Not performed'),
    ('16', 'Open science and data', 3, '0 to 4', 'Open images, segmentations and code'),
]


def esc(t):
    return t.replace('&', '\\&').replace('%', '\\%').replace('~', '\\textasciitilde{}')


def checklist(rows, caption, label, intro):
    out = [f'\\begin{{longtable}}{{@{{}}L{{0.06\\linewidth}}L{{0.36\\linewidth}}L{{0.08\\linewidth}}L{{0.415\\linewidth}}@{{}}}}',
           f'\\caption{{{caption}}}\\label{{{label}}}\\\\',
           '\\toprule', 'Item & Topic & Reported & Location in the manuscript / comment \\\\', '\\midrule', '\\endfirsthead',
           f'\\multicolumn{{4}}{{@{{}}l}}{{\\small\\textit{{{intro} (continued)}}}}\\\\',
           '\\toprule', 'Item & Topic & Reported & Location in the manuscript / comment \\\\', '\\midrule', '\\endhead',
           '\\bottomrule', '\\endlastfoot']
    for r in rows:
        if r[1] is None:
            out.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{r[0]}}}}} \\\\')
        else:
            out.append(' & '.join(esc(x) for x in r) + ' \\\\')
    out.append('\\end{longtable}')
    counts = {k: sum(1 for r in rows if r[1] is not None and r[2] == k) for k in ('Yes', 'Partly', 'No', 'n/a')}
    return '\n'.join(out), counts


def main():
    parts = []
    tri, ct = checklist(TRIPOD, 'TRIPOD+AI checklist (Collins et al., 2024). Reported: Yes, Partly, No or n/a (not applicable).',
                        'tab:tripod', 'TRIPOD+AI')
    cla, cc = checklist(CLAIM, 'CLAIM 2024 checklist (Tejani et al., 2024).', 'tab:claim', 'CLAIM 2024')
    cle, cl = checklist(CLEAR, 'CLEAR checklist (Kocak et al., 2023). n/e: not essential (recommended) item.', 'tab:clear', 'CLEAR')

    # METRICS score: sum of weights of items present over the sum of weights of applicable items
    appl = [m for m in METRICS if m[4] is not None]
    got = sum(m[3] for m in appl if m[4])
    tot = sum(m[3] for m in appl)
    score = 100 * got / tot
    cat = ['very low', 'low', 'moderate', 'good', 'excellent'][min(int(score // 20), 4)]
    met = ['\\begin{longtable}{@{}L{0.04\\linewidth}L{0.17\\linewidth}L{0.37\\linewidth}rcL{0.21\\linewidth}@{}}',
           f'\\caption{{METRICS self-assessment (Kocak et al., 2024). Score: weight if the item is met and 0 otherwise; '
           f'conditional items whose condition does not apply (n/a) are excluded from the denominator. '
           f'Total {got:.4f} of {tot:.4f} applicable, METRICS score {score:.1f}\\% ({cat} quality).}}\\label{{tab:metrics}}\\\\',
           '\\toprule', 'No. & Category & Item & Weight & Met & Comment \\\\', '\\midrule', '\\endfirsthead',
           '\\toprule', 'No. & Category & Item & Weight & Met & Comment \\\\', '\\midrule', '\\endhead',
           '\\bottomrule', '\\endlastfoot']
    for no, catg, item, w, ok, com in METRICS:
        mk = 'n/a' if ok is None else ('Yes' if ok else 'No')
        met.append(f'{no} & {catg} & {esc(item)} & {w:.4f} & {mk} & {esc(com)} \\\\')
    met.append('\\midrule')
    met.append(f' & & Total score & {got:.4f} & & {score:.1f}\\% of {tot:.4f} \\\\')
    met.append('\\end{longtable}')

    rqs_total = sum(r[2] for r in RQS)
    rq = ['\\begin{longtable}{@{}L{0.04\\linewidth}L{0.40\\linewidth}ccL{0.34\\linewidth}@{}}',
          f'\\caption{{Radiomics Quality Score (Lambin et al., 2017) self-assessment. Total {rqs_total} of 36 points '
          f'({100 * rqs_total / 36:.1f}\\%).}}\\label{{tab:rqs}}\\\\',
          '\\toprule', 'No. & Criterion & Points & Range & Justification \\\\', '\\midrule', '\\endfirsthead',
          '\\toprule', 'No. & Criterion & Points & Range & Justification \\\\', '\\midrule', '\\endhead',
          '\\bottomrule', '\\endlastfoot']
    for no, crit, pts, rng, why in RQS:
        rq.append(f'{no} & {esc(crit)} & {pts} & {rng} & {esc(why)} \\\\')
    rq.append('\\midrule')
    rq.append(f' & Total & {rqs_total} & $-$8 to 36 & {100 * rqs_total / 36:.1f}\\% of the maximum \\\\')
    rq.append('\\end{longtable}')

    def summ(c):
        return f"{c['Yes']} reported, {c['Partly']} partly reported, {c['No']} not reported and {c['n/a']} not applicable"

    note = ('\\section*{Supplementary Note 1: Reporting guidelines and quality assessment}\n'
            'The study was reported following TRIPOD+AI, CLAIM 2024 and CLEAR, and its methodological quality was '
            'self-assessed with METRICS and RQS by the author. Locations refer to the sections, tables and figures of the main '
            'text. Of the TRIPOD+AI items, ' + summ(ct) + '; of the CLAIM 2024 items, ' + summ(cc) +
            '; and of the CLEAR items, ' + summ(cl) + f'. The METRICS score was {score:.1f}\\% ({cat} quality) and the RQS '
            f'{rqs_total} of 36 points ({100 * rqs_total / 36:.1f}\\%). The items not met concern mainly calibration and clinical '
            'utility, which were outside the aim of a discrimination benchmark, segmentation reliability and imaging protocol '
            'details, which are limited by the public collections, and the release of feature data and trained models, which '
            'can be regenerated from the public data with the released code.\n')
    parts = [note, tri, cla, cle, '\n'.join(met), '\n'.join(rq)]
    OUT.write_text('\n\n'.join(parts) + '\n')
    print(f'TRIPOD+AI {ct}\nCLAIM {cc}\nCLEAR {cl}\nMETRICS {score:.1f}% ({cat})  RQS {rqs_total}/36')


if __name__ == '__main__':
    main()
