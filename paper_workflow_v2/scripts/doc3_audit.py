"""Document 3: implementation history, methodological audit and roadmap (upgrade of RadGraph_Bug_Log_and_Future_Work)."""
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
     'Lowest validation loss (patience 50) as in RadGraph; Mateus rule as sensitivity analysis.'],
    ['Clinical variables', 'ECOG and tumour volume (MAASTRO only) used; Head-Neck-PET-CT does not record them.',
     '16 variables recorded identically in both cohorts (age, sex, HPV, T, N, subsite, chemotherapy).'],
    ['Feature count', 'Documents stated ~83 PyRadiomics features.', '93 features (18 first order and 75 texture).'],
    ['Interpolation', 'Drafts described B-spline CT resampling.', 'Code uses linear interpolation; manuscript corrected.'],
]

blocks = [
    ('h1', H[0]),
    ('p', 'This document records how the radiomic graph pipeline evolved from a first working implementation to the current '
          'study code. Version 1 (May 2026) documented the engineering defects that prevented the pipeline from running. A '
          'subsequent review against the RadGraph publication, the authors\' code and the validation design of Mateus et al. '
          'identified methodological problems that did not cause errors but would have invalidated the results. Both are '
          'listed here, followed by the open issues identified in the current data and a prioritised roadmap.'),
    ('table', ['Period', 'Milestone'], [
        ['May 2026', 'version 1: five-stage pipeline running on 121 MAASTRO patients; validation AUC 0.64 on 19 patients'],
        ['Nov 2026', 'restructured package radgraph-hnscc; corrected outcome rule, scaling, network and validation'],
        ['Nov 2026', 'Mateus et al. design adopted; LR, DM and OS; nine feature-selection strategies × four topologies'],
        ['Nov 2026', 'full study run (both designs, both checkpoint rules), exported results, manuscript draft'],
    ], [2.6, 14.4], 'Timeline.'),

    ('h1', H[1]),
    ('p', 'The thirteen defects below were found while making version 1 run end to end. They are retained because the same '
          'problems recur in most radiomics projects.'),
    ('table', ['#', 'Files', 'Problem', 'Resolution'], bugs, [0.8, 4.0, 6.4, 5.8], 'Engineering defects of version 1.'),
    ('p', 'Bugs 4, 12 and 13 deserve emphasis because they produced plausible output without any error message: the follow-up '
          'filter silently reduced the events from 34 to 4, the scheduler silently stopped learning, and the absence of early '
          'stopping silently selected an untrained model.'),

    ('h1', H[2]),
    ('p', 'The audit compared every stage with the RadGraph paper, its supplementary material, the authors\' released code '
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
        ['Preprocessing and supervoxels', '336 patients (199 Canadian, 137 MAASTRO); 100 supervoxels each'],
        ['Radiomic features', '93 per region, cached'],
        ['Feature selection and stability', 'nine strategies; stability across folds, bootstrap and centres'],
        ['Graphs and models', 'four topologies; LR, DM, OS; both designs; both checkpoint rules; 3 seeds'],
        ['Results export', 'study tables, ROC figures, per-patient predictions'],
        ['Inference', 'self-contained checkpoints; predict.py reproduces stored predictions to 10⁻⁷'],
        ['Manuscript', 'complete draft with supplementary material; regenerated from the results by one script'],
    ], [5.0, 12.0], 'Pipeline status.'),
    ('p', 'The main results on the 199-patient image set: for LR, external AUC 0.71 (95% CI 0.62–0.81) with clinical variables '
          'in the cohort split, above the published CNN (0.59); no configuration significantly better than the RadGraph '
          'reference; DM and OS not predicted better than chance by the configurations chosen on validation; spectral Graph-FS '
          'the most stable selector; validation AUC not predictive of external AUC; near-uniform attention.'),
    ('fig', 'fig5_auc_overview.png', 16.0, 'External AUC with 95% CI for all endpoints, designs and input sets.'),

    ('h1', H[4]),
    ('table', ['Issue', 'Evidence', 'Consequence', 'Action'], [
        ['Incomplete development images', '199 of 298 patients; all oropharyngeal; more than half of events missing',
         'low power, subsite shift to MAASTRO (36% larynx)', 'obtain full TCIA collection and re-run'],
        ['Air in supervoxels', 'median 22% of supervoxels > 50% air', 'graph nodes that describe air', 'body mask in Stage 2'],
        ['Compressed soft-tissue contrast', 'soft tissue mapped to 0.26–0.28 of the SLIC intensity range',
         'supervoxels follow air and bone boundaries', 'soft-tissue window before SLIC'],
        ['Near-uniform attention', '98% of weights within ±1.2% of the mean', 'no interpretable attention maps',
         'more events; lower attention dropout; attention entropy regularisation'],
        ['Few DM events', '12 development and 7 external events', 'DM not estimable', 'full collection; time-to-event models'],
        ['Unreliable model selection', 'validation–external AUC correlation −0.46 to 0.20', 'chosen configuration ≈ random',
         'pre-specified reference; nested CV; report the grid'],
    ], [3.4, 4.6, 4.2, 4.8], 'Open issues identified in the current study.'),

    ('h1', H[5]),
    ('h2', 'Priority 1: before submission'),
    ('numbered', [
        'Obtain the complete Head-Neck-PET-CT collection (planning CT and RTSTRUCT for all 298 patients), re-run the study and '
        'regenerate the manuscript; re-read Results and Discussion afterwards.',
        'Confirm the provenance of the current Canadian images (HECKTOR release) and state it in the manuscript.',
        'Fill in the author details, funding, ethics statement and repository URL, and verify every reference.',
    ]),
    ('h2', 'Priority 2: method improvements'),
    ('numbered', [
        'Stage 2: body mask and soft-tissue window before SLIC; sensitivity analysis of the number of supervoxels.',
        'Time-to-event formulation (Cox or discrete-time survival loss) so that censored patients contribute.',
        'Nested cross-validation for hyper-parameters instead of adopting RadGraph values; a dedicated OS configuration.',
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
        '**Count events, not patients.** Thirteen events cannot support a 15,000-parameter model or reliable model selection.',
        '**Report the reference and the whole grid.** The best validation configuration is an optimistic and unstable summary.',
        '**Stability is not discrimination.** Graph-FS made the features reproducible but not more predictive.',
        '**Look at the images.** The air supervoxels and the compressed intensity range were only noticed when the supervoxels '
        'were drawn on the CT.',
        '**Make everything regenerable.** Cached stages, self-contained checkpoints and a figure script turned each re-run '
        'into one command.',
    ]),

    ('h1', H[7]),
    ('table', ['Path (radgraph-hnscc/)', 'Content'], [
        ['main.py', 'command-line pipeline (all stages, --mode, --task, --fold, --epoch_rule)'],
        ['run_mateus_study.sh', 'the complete study'],
        ['export_results.py', 'study tables, figures and final models'],
        ['predict.py', 'inference with saved models'],
        ['radgraph/config.py', 'cohorts, endpoints, designs, grid, hyper-parameters'],
        ['radgraph/data/', 'loading, labels and clinical encoding, preprocessing, supervoxels, features, splits'],
        ['radgraph/selection/', 'nine feature-selection strategies and stability measures'],
        ['radgraph/graphs/', 'four topologies; datasets and class-balanced sampling'],
        ['radgraph/models/gat.py', 'radiomic graph attention network'],
        ['radgraph/engine/', 'training, evaluation, attention extraction, random-forest baselines'],
        ['radgraph/experiments/', 'grid, cross-fold summary, export'],
        ['radgraph/viz/', 'figures and HTML reports'],
        ['outputs/', 'cache, runs, results, final_models'],
    ], [5.4, 11.6], 'Current code layout.'),
]

if __name__ == '__main__':
    d = Doc('Implementation History, Audit and Roadmap',
            'From the first working pipeline to a reproducible, externally validated study',
            'Implementation history, audit and roadmap',
            'Supersedes RadGraph_Bug_Log_and_Future_Work (May 2026).')
    d.contents(H)
    d.add(blocks)
    print(d.save('03_Implementation_History_Audit_and_Roadmap_v2'))
