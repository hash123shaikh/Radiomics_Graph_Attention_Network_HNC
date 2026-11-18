"""Document 3: implementation history, methodological audit and roadmap (upgrade of the original bug log in paper_workflow/)."""
from docgen import Doc

H = ['Overview', 'Phase 1: engineering defects (version 1)', 'Phase 2: methodological audit',
     'Current state of the pipeline', 'Open issues', 'Roadmap', 'Lessons learned', 'Code map']

bugs = [
    ['1', 'install_deps.sh, requirements.txt', 'Python 3.9 required; non-existent package versions',
     'Python 3.8, scipy 1.10.1, opencv-python 4.8.0.76'],
    ['2', 'data_loader.py, config.py', 'DICOM folder structure assumed incorrectly', 'detect nested CT/ and RTSTRUCT/ folders'],
    ['3', 'data_loader.py', 'rt-utils mask in (X, Y, Z) order, SimpleITK expects (Z, Y, X)', 'transpose before GetImageFromArray'],
    ['4', 'data_loader.py, config.py', 'follow-up filter removed most event patients (4 instead of 34)',
     'keep events; superseded by the horizon rule (Section 3)'],
    ['5', 'dataset.py', 'string clinical values crashed float conversion', 'explicit encoding of every clinical variable'],
    ['6', 'config.py', 'input layer built for 93 features, graphs had 4', 'input size taken from the graphs'],
    ['7', 'dataset.py, model.py', 'PyG concatenated per-graph clinical vectors into one flat vector',
     'store clinical data with shape (1, d)'],
    ['8', 'train.py, main.py', 'plateau scheduler was None', 'pass and unpack both schedulers'],
    ['9', 'dataset.py', 'missing clinical values produced NaN loss', 'fill missing values; explicit "unknown" categories'],
    ['10', 'config.py', 'clinical file path wrong', 'path corrected'],
    ['11', 'dataset.py, config.py', 'ordinal encoding of categorical variables', 'one-hot encoding'],
    ['12', 'config.py, train.py', 'learning rate decayed to zero within 80 epochs', 'scheduler patience 20 and floor 10⁻⁵'],
    ['13', 'config.py', 'no early stopping; best model at epoch 1', 'early stopping (now on validation loss, patience 50)'],
]

audit = [
    ['Cohort identity', 'The data were described as 120 CMC Vellore patients; the files were the public HN1 (MAASTRO) collection.',
     'Data described correctly: HN1 is the external test cohort; development uses Head-Neck-PET-CT.'],
    ['Outcome definition', 'Bug 4 fix counted events at any time as positive, inflating the event rate.',
     'Horizon rule of Mateus et al.: event within 2 years (LR, DM) or 4 years (OS); later events negative; censored before the horizon excluded.'],
    ['Validation design', 'Single random 70/15/15 split; 19 test patients; no external test.',
     '5-fold stratified CV and cohort split on the four Canadian centres; external test on MAASTRO, used once.'],
    ['Node scaling', 'GTV-relative scaling made every GTV node [1, 1, 1, 1], removing tumour information.',
     'z-scoring with the training GTV statistics (authors\' ScalerGeneration.py), clipping at ±5.'],
    ['Supervoxel selection', 'Cosine similarity to the GTV.', 'Euclidean distance in z-scored feature space (authors\' GraphGeneration.py).'],
    ['Network', 'Extra input projection, ELU, batch-normalised clinical projection.',
     'Authors\' architecture: GATConv → LayerNorm → ReLU, GTV read-out, dropout 0.5, clinical concatenation, linear output.'],
    ['Checkpoint', 'Early stopping on validation AUC (patience 15–30); best epoch often epoch 1.',
     'Lowest validation loss (patience 50) as in Bae et al. (2025); Mateus rule as sensitivity analysis.'],
    ['Clinical variables', 'ECOG and tumour volume (MAASTRO only) used; Head-Neck-PET-CT does not record them.',
     '16 variables recorded identically in both cohorts (age, sex, HPV, T, N, subsite, chemotherapy).'],
    ['Feature count', 'Documents stated ~83 PyRadiomics features.', '93 features (18 first order and 75 texture).'],
    ['Interpolation', 'Drafts described B-spline CT resampling.', 'Code uses linear interpolation; manuscript corrected.'],
    ['Development images', 'HECKTOR release: 199 of 298 patients, all oropharyngeal, more than half of the events missing.',
     'Complete TCIA collection: CT of the PET/CT with RTSTRUCT for every patient; per-patient primary-GTV map; events identical to Mateus et al.'],
    ['Supervoxels', 'SLIC on the whole 50 mm box with min–max scaling: about one fifth of supervoxels mostly air.',
     'Preprocessing v4: body mask and soft-tissue window (−160 to 240 HU) before SLIC.'],
]

blocks = [
    ('h1', H[0]),
    ('p', 'This document records how the radiomic graph pipeline evolved from a first working implementation to the current '
          'study code. Version 1 (May 2026) documented the engineering defects that prevented the pipeline from running. A '
          'subsequent review against the original publication (Bae et al. 2025), the authors\' code and the validation design of Mateus et al. '
          'identified methodological problems that did not cause errors but would have invalidated the results. Both are '
          'listed here, followed by the open issues identified in the current data and a prioritised roadmap.'),
    ('table', ['Period', 'Milestone'], [
        ['May 2026', 'version 1: five-stage pipeline running on 121 MAASTRO patients; validation AUC 0.64 on 19 patients'],
        ['Nov 2026', 'restructured package grafs-gat; corrected outcome rule, scaling, network and validation'],
        ['Nov 2026', 'Mateus et al. design adopted; LR, DM and OS; nine feature-selection strategies × four topologies'],
        ['Nov 2026', 'first full study run on the 199-patient HECKTOR image set (preprocessing v3), manuscript draft'],
        ['Nov 2026', 'repository released (MIT); reproducibility tooling; v3 model release v0.1.0-models'],
        ['Nov 2026', 'Stage 2 refinements (body mask, soft-tissue window) as preprocessing v4'],
        ['17–18 Nov 2026', 'complete TCIA Head-Neck-PET-CT (298 patients, DICOM) loaded; full study re-run with v4; manuscript updated'],
    ], [2.6, 14.4], 'Timeline.'),

    ('h1', H[1]),
    ('p', 'The thirteen defects below were found while making version 1 run end to end. They are retained because the same '
          'problems recur in most radiomics projects.'),
    ('table', ['#', 'Files', 'Problem', 'Resolution'], bugs, [0.8, 4.0, 6.4, 5.8], 'Engineering defects of version 1.'),
    ('p', 'Bugs 4, 12 and 13 deserve emphasis because they produced plausible output without any error message: the follow-up '
          'filter silently reduced the events from 34 to 4, the scheduler silently stopped learning, and the absence of early '
          'stopping silently selected an untrained model.'),

    ('h1', H[2]),
    ('p', 'The audit compared every stage with Bae et al. (2025), its supplementary material, the authors\' released code '
          '(CodeRelease) and the protocol of Mateus et al. The table lists each problem, why it mattered and how it is handled '
          'in the current code.'),
    ('table', ['Topic', 'Version 1', 'Current implementation'], audit, [3.0, 6.6, 7.4], 'Methodological corrections.'),
    ('note', 'key', 'Why these corrections matter',
     'Several of these issues would have biased the results in the optimistic direction: counting late events as positives '
     'changes the endpoint, a single small random split gives highly variable AUCs, and choices made with knowledge of the '
     'test patients inflate performance. The current results are lower than the version 1 estimates but are reproducible and '
     'externally validated.'),

    ('h1', H[3]),
    ('table', ['Component', 'Status'], [
        ['Preprocessing and supervoxels', '435 patients (298 Canadian, 137 MAASTRO), preprocessing v4; 100 supervoxels target'],
        ['Radiomic features', '93 per region, cached'],
        ['Feature selection and stability', 'nine strategies; stability across folds, bootstrap and centres'],
        ['Graphs and models', 'four topologies; LR, DM, OS; both designs; both checkpoint rules; 3 seeds'],
        ['Results export', 'study tables, ROC figures, per-patient predictions (outputs/results)'],
        ['Inference', 'self-contained checkpoints; predict.py reproduces stored predictions to 10⁻⁷'],
        ['Manuscript', 'complete draft updated to the full-cohort v4 results; regenerated from the results by one script'],
    ], [5.0, 12.0], 'Pipeline status.'),
    ('p', 'The main results of the full study: external AUCs of the configurations chosen on validation 0.54–0.66 for LR, '
          '0.66–0.73 for DM and 0.62–0.66 for OS; no configuration significantly better than the reference configuration; '
          'clinical-only random forest strongest for DM (0.82); GAT similar to the published CNN for LR and OS, inferior for DM; '
          'spectral Graph-FS the most stable selector but among the least discriminative (stability–AUC ρ −0.92 for LR); '
          'random forest on the same GTV features as good as the GAT; validation AUC weakly predictive of external AUC; '
          'near-uniform attention. The earlier 199-patient result for LR (0.71 in the cohort split) did not replicate.'),
    ('fig', 'fig5_auc_overview.png', 16.0, 'External AUC with 95% CI for all endpoints, designs and input sets.'),

    ('h1', H[4]),
    ('table', ['Issue', 'Evidence', 'Consequence', 'Action'], [
        ['Poor LR transfer', 'LR AUC 0.54–0.66 for the GAT and 0.53–0.59 for the published CNN',
         'locoregional signal does not transfer to MAASTRO', 'study cohort differences (subsite, chemotherapy, PET/CT vs planning CT)'],
        ['Graph adds no information', 'random forest on the GTV features as good as the GAT for every endpoint',
         'peritumoral nodes not yet useful', 'more events; nodal GTV nodes; finer supervoxels'],
        ['Near-uniform attention', 'attention 0.047–0.048 against a uniform 0.048', 'no interpretable attention maps',
         'more events; lower attention dropout; attention entropy regularisation'],
        ['Few external DM events', '7 MAASTRO events', 'DM confidence intervals span 0.4–0.9', 'additional external cohort; time-to-event models'],
        ['Unreliable model selection', 'validation–external AUC correlation 0.07 to 0.59', 'chosen configuration close to a random draw',
         'pre-specified reference; nested CV; report the grid'],
        ['Stability vs discrimination', 'most stable selector (spectral Graph-FS) among the least discriminative',
         'stability is not a surrogate for prognostic value', 'report both; hybrid Graph-FS as compromise'],
    ], [3.4, 4.6, 4.2, 4.8], 'Open issues identified in the current study.'),

    ('h1', H[5]),
    ('h2', 'Priority 1: before submission'),
    ('numbered', [
        'Done: complete Head-Neck-PET-CT collection loaded, study re-run with preprocessing v4, manuscript regenerated and '
        'Results and Discussion rewritten.',
        'Done: v4 trained models published as release v1.0.0-models (24/24 ensembles verified); repository renamed GRAFS-GAT.',
        'Fill in the author details, funding and ethics statement, and verify every reference.',
    ]),
    ('h2', 'Priority 2: method improvements'),
    ('numbered', [
        'Stage 2: sensitivity analysis of the number of supervoxels (body mask and soft-tissue window are implemented in v4).',
        'Time-to-event formulation (Cox or discrete-time survival loss) so that censored patients contribute.',
        'Nested cross-validation for hyper-parameters instead of adopting the published values; a dedicated OS configuration.',
        'Stability across radiomic extraction settings (bin width, resampling), as in the original Graph-FS study.',
        'Lymph-node nodes: add nodal GTVs as graph nodes connected to the primary tumour.',
    ]),
    ('h2', 'Priority 3: validation and translation'),
    ('numbered', [
        'External validation on the CMC Vellore cohort once ethics approval and curated data are available.',
        'Multimodal graphs with PET node features (requires re-downloading the PET images, deleted locally to save space).',
        'Clinical review of attention maps once attention becomes informative.',
        'Prospective evaluation at radiotherapy planning.',
    ]),

    ('h1', H[6]),
    ('numbered', [
        '**Verify cohort counts after every filtering step.** Silent clinical-logic errors (Bug 4) are more dangerous than crashes.',
        '**Read the authors\' code, not only the paper.** Scaling, supervoxel selection and the network architecture all '
        'differed between the first implementation and the reference code.',
        '**Validation design determines the credibility of the result.** A single small random split gives an AUC that can '
        'change by 0.1 when two patients change.',
        '**Count events, not patients.** Thirty events cannot support a 15,000-parameter model or reliable model selection.',
        '**Report the reference and the whole grid.** The best validation configuration is an optimistic and unstable summary.',
        '**Stability is not discrimination.** Graph-FS made the features reproducible but, in its label-free form, less predictive.',
        '**Re-run before claiming.** The 199-patient LR advantage disappeared on the complete collection.',
        '**Look at the images.** The air supervoxels and the compressed intensity range were only noticed when the supervoxels '
        'were drawn on the CT.',
        '**Make everything regenerable.** Cached stages, self-contained checkpoints and a figure script turned each re-run '
        'into one command.',
    ]),

    ('h1', H[7]),
    ('table', ['Path (grafs-gat/)', 'Content'], [
        ['main.py', 'command-line pipeline (all stages, --mode, --task, --fold, --epoch_rule)'],
        ['run_mateus_study.sh', 'the complete study'],
        ['export_results.py', 'study tables, figures and final models'],
        ['predict.py', 'inference with saved models'],
        ['grafsgat/config.py', 'cohorts, endpoints, designs, grid, hyper-parameters'],
        ['grafsgat/data/', 'loading, labels and clinical encoding, preprocessing, supervoxels, features, splits'],
        ['grafsgat/selection/', 'nine feature-selection strategies and stability measures'],
        ['grafsgat/graphs/', 'four topologies; datasets and class-balanced sampling'],
        ['grafsgat/models/gat.py', 'radiomic graph attention network'],
        ['grafsgat/engine/', 'training, evaluation, attention extraction, random-forest baselines'],
        ['grafsgat/experiments/', 'grid, cross-fold summary, export'],
        ['grafsgat/viz/', 'figures and HTML reports'],
        ['outputs/', 'cache, runs, results, final_models of the current (v4, full-cohort) study'],
    ], [5.4, 11.6], 'Current code layout.'),
]

if __name__ == '__main__':
    d = Doc('Implementation History, Audit and Roadmap',
            'From the first working pipeline to a reproducible, externally validated study',
            'Implementation history, audit and roadmap',
            'Supersedes the original bug log in paper_workflow/ (May 2026).')
    d.contents(H)
    d.add(blocks)
    print(d.save('03_Implementation_History_Audit_and_Roadmap_v2'))
