"""Document 1: study rationale, design and methods (upgrade of 'GNN_Paper -- Introduction_and_Methods')."""
from docgen import Doc

H = ['Background and rationale', 'Aims and research questions', 'Data sources', 'Endpoints and eligibility',
     'Validation design', 'Image preprocessing and supervoxels', 'Radiomic features', 'Feature selection',
     'Graph construction', 'Graph attention network', 'Comparator models', 'Evaluation and statistics',
     'Software and reproducibility', 'Findings to date', 'References']

blocks = [
    ('h1', H[0]),
    ('p', 'Head and neck squamous cell carcinoma (HNSCC) is among the ten most common cancers worldwide. Most patients with '
          'locally advanced disease are treated with radiotherapy, often with concurrent chemotherapy, and locoregional '
          'recurrence remains the dominant pattern of failure. Patients with the same stage, subsite and treatment nevertheless '
          'have very different outcomes, which motivates pretreatment models that could guide treatment intensification, '
          'de-escalation or follow-up.'),
    ('p', 'Radiomics extracts quantitative descriptors of intensity and texture from routine CT. Conventional radiomic models '
          'summarise the gross tumour volume (GTV) with one feature vector, discarding both the internal heterogeneity of the '
          'tumour and the surrounding tissue through which microscopic extension and nodal spread occur. RadGraph (Bae et al., '
          '2025) addressed this by representing each patient as a graph: the GTV is the central node, surrounding '
          'supervoxels are peripheral nodes, all described by radiomic features, and a graph attention network (GAT) reads '
          'the outcome from the GTV node after exchanging information with the surrounding tissue.'),
    ('p', 'Two choices in RadGraph were fixed rather than studied: a small endpoint-specific set of node features and a '
          '"radiomic star" graph that connects the GTV to the supervoxels most similar to it. Both choices matter for '
          'reproducibility. Radiomic features are highly inter-correlated, so conventional feature selection changes with '
          'small perturbations of the training data. Graph-based feature selection (Graph-FS; Moradmand et al., 2025) '
          'addresses this by selecting one representative per community of correlated features and produced markedly more '
          'stable selections across institutions. Finally, Mateus et al. (2023) showed that reported performance in this '
          'field depends strongly on the validation design and proposed a reproducible protocol for the public '
          'Head-Neck-PET-CT and HN1 (MAASTRO) collections.'),

    ('h1', H[1]),
    ('p', 'The study asks which feature-selection strategy yields node features that are both predictive and reproducible '
          'when used in a spatial radiomic GAT, and whether the graph topology matters. Four research questions follow:'),
    ('numbered', [
        '**Discrimination.** Does any combination of feature selection and graph topology improve external discrimination '
        'over the published RadGraph configuration for locoregional recurrence (LR), distant metastasis (DM) and overall '
        'survival (OS)?',
        '**Reproducibility.** How stable are the selected features across cross-validation folds, bootstrap resamples and '
        'centres, and is Graph-FS more stable than conventional selectors?',
        '**Stability and generalisation.** Are more stable feature sets also more discriminative externally?',
        '**Benchmark.** How does the radiomic GAT compare with the convolutional neural network of Mateus et al. evaluated on '
        'the same external cohort with the same designs?',
    ]),

    ('h1', H[2]),
    ('table', ['Cohort', 'Role', 'Centres', 'Patients', 'Imaging used'], [
        ['Head-Neck-PET-CT (TCIA)', 'development (training and validation)', 'CHUM, CHUS, HGJ, HMR (Québec, Canada)',
         '298 in the clinical file; 199 with CT and primary GTV (CHUM 56, CHUS 72, HGJ 53, HMR 18)', 'CT and primary GTV (HECKTOR release)'],
        ['HN1 (TCIA)', 'external test only', 'MAASTRO Clinic, Maastricht', '137', 'planning CT and RTSTRUCT'],
    ], [3.4, 3.3, 3.6, 4.2, 2.5], 'Data sources.'),
    ('p', 'Both collections are distributed by The Cancer Imaging Archive. The Canadian images available locally are those '
          'of the HECKTOR challenge release, which contains only oropharyngeal tumours; 199 of the 298 patients therefore have '
          'images, and the patients without images carry more than half of the outcome events (Section 4). The complete '
          'collection, including laryngeal, nasopharyngeal and hypopharyngeal cases, is being obtained from TCIA. Clinical '
          'variables recorded identically in both cohorts were used: age, sex, HPV status, T and N category, subsite and '
          'concurrent chemotherapy.'),

    ('h1', H[3]),
    ('p', 'Endpoints follow Mateus et al.: LR and DM within 2 years of diagnosis and death from any cause within 4 years (OS; '
          '4 years is the median follow-up of the development data). For each endpoint, a patient is labelled positive if the '
          'event occurred within the horizon and negative if the patient was followed event-free for at least the horizon; '
          'events after the horizon count as negative, and patients censored before the horizon without an event are '
          'excluded. The rule avoids the two errors of the earlier implementation: excluding early events because their '
          'event time is shorter than the horizon, and counting late events as positives.'),
    ('table', ['Endpoint', 'Canadian, all eligible (events)', 'Canadian with images (events)', 'MAASTRO (events)'], [
        ['LR, 2 years', '276 (30)', '190 (13)', '121 (30)'],
        ['DM, 2 years', '276 (27)', '191 (12)', '109 (7)'],
        ['OS, 4 years', '164 (44)', '92 (19)', '135 (45)'],
    ], [3.4, 4.6, 4.6, 4.4], 'Patients and events per endpoint.'),
    ('note', 'warn', 'Event scarcity',
     'With 12–19 events in the development data, each cross-validation validation fold contains only two or three LR or DM '
     'events, and the external DM cohort contains seven. Confidence intervals are therefore wide, and validation-based model '
     'selection is unreliable (Section 14).'),

    ('h1', H[4]),
    ('p', 'Two development designs are used, both with MAASTRO as the external test set (Mateus et al., 2023). In **5-fold '
          'stratified cross-validation**, the Canadian patients are divided into five folds with equal event proportions; the '
          'entire pipeline, from feature selection to model training, is refitted on four folds and validated on the fifth. '
          'Configurations are compared on the pooled out-of-fold validation AUC, and external predictions are averaged over '
          'the five fold models. In the **cohort split** of Diamant et al. (2019), models are trained on HGJ and CHUS and '
          'validated on HMR and CHUM, which also measures transfer between institutions. In both designs the external cohort '
          'is used exactly once, to evaluate the configuration chosen on validation.'),
    ('fig', 'fig1_study_design.png', 16.5, 'Study design: data, preprocessing, radiomics, the feature-selection × topology grid, '
     'models and validation.'),

    ('h1', H[5]),
    ('p', 'CT and GTV are resampled to 1 mm isotropic voxels (linear interpolation for CT, nearest neighbour for the mask). The '
          'peritumoral region is the GTV bounding box enlarged by 50 mm, to which both arrays are cropped. The region minus the '
          'GTV is divided into 100 supervoxels with zero-parameter SLIC; the GTV itself is a separate node. Stage 2 is '
          'documented in detail, including two recommended improvements (body mask and soft-tissue window), in the companion '
          'document *Stage 2: Peritumoral Supervoxel Generation*.'),

    ('h1', H[6]),
    ('p', 'PyRadiomics extracts 93 features for the GTV and every supervoxel with a fixed bin width of 25 HU and no intensity '
          'normalisation: first order (18), GLCM (24), GLRLM (16), GLSZM (16), GLDM (14) and NGTDM (5). Shape features are '
          'excluded because they describe how supervoxels were constructed rather than the tissue. Features and region '
          'centroids are cached per patient.'),

    ('h1', H[7]),
    ('p', 'All selectors operate on the GTV features of the training patients only, after removing near-constant features, '
          'applying a signed logarithm to heavy-tailed features and z-scoring. Every data-driven selector keeps k = 6 '
          'features, so that strategies are compared at equal model capacity.'),
    ('table', ['Strategy', 'Principle', 'Uses the outcome'], [
        ['Table S3 (RadGraph)', 'fixed published features (4 for LR, 6 for DM); reference', 'no (fixed)'],
        ['All features', 'no selection (93 node features); control', 'no'],
        ['mRMR', 'maximum relevance (ANOVA F) / minimum redundancy (mean |r|)', 'yes'],
        ['LASSO', 'L1 logistic regression; ranking by entry on the regularisation path', 'yes'],
        ['Boruta', 'random-forest importance tested against shadow features', 'yes'],
        ['RFE-RF', 'recursive elimination with a random forest', 'yes'],
        ['Graph-FS (CC)', 'feature graph; one representative per connected component (degree centrality)', 'no'],
        ['Graph-FS (spectral)', 'feature graph; spectral clusters; feature closest to each centroid', 'no'],
        ['Graph-FS (hybrid)', 'feature graph; spectral communities; most relevant feature per community (introduced here)', 'yes'],
    ], [3.6, 10.6, 2.8], 'Feature-selection strategies.'),
    ('p', 'Graph-FS builds a feature graph whose nodes are features and whose edges connect features with an absolute '
          'Pearson correlation above the mean plus 0.3 standard deviations of all pairwise correlations. The connected-component '
          'variant keeps one representative per component; because the 93 features form a single component at this threshold, '
          'the threshold is raised in steps of 0.1 until at least six components exist. The spectral variant clusters the '
          'features on the eigenvectors of the graph Laplacian. Both are label-free. The hybrid variant keeps the redundancy '
          'control of the graph but takes the most outcome-relevant feature of each community, so that stability and '
          'relevance are combined (Figure 2).'),
    ('fig', 'fig_graphfs_workflow.png', 17.0, 'Graph-based feature selection on the training data of the first fold (LR): feature '
     'matrix, similarity matrix, edge rule, feature graph, spectral Graph-FS and hybrid Graph-FS.'),

    ('h1', H[8]),
    ('p', 'Each patient graph has the GTV as read-out node and 20 supervoxel nodes. Node features are the selected features, '
          'z-scored with the means and standard deviations of the GTV features of the training patients (as in the authors\' '
          'ScalerGeneration.py) and clipped at ±5. Four topologies are compared (Figure 3): the **radiomic star** connects the '
          'GTV with the 20 supervoxels nearest to it in z-scored feature space (Euclidean distance); the **spatial star** with '
          'the 20 supervoxels nearest in physical space; the **radiomic kNN** and **spatial kNN** graphs add undirected edges '
          'between every node and its four nearest neighbours in feature or physical space.'),
    ('fig', 'fig2_topologies.png', 17.0, 'The four topologies on the CT of an external test patient (HN1260). Orange: GTV and '
     'read-out node; blue: selected supervoxels; yellow: kNN edges.'),

    ('h1', H[9]),
    ('p', 'The network reproduces the RadGraph architecture (authors\' GAT_modeling.py) and its endpoint-specific '
          'hyper-parameters (Figure 4). For LR, four GAT layers with four heads of 16 units, each followed by graph-wise layer '
          'normalisation and ReLU, precede a single-head final layer. The 16-dimensional GTV embedding is read out, passed '
          'through dropout, optionally concatenated with 16 encoded clinical variables, and mapped by a linear layer and '
          'sigmoid to the event probability.'),
    ('fig', 'fig_architecture_author.png', 17.0, 'Radiomic graph attention network (left) and multi-head attention at the GTV '
     'node (right).'),
    ('table', ['Setting', 'LR and OS', 'DM'], [
        ['GAT layers / heads / units per head', '5 / 4 / 16', '3 / 4 / 8'],
        ['Attention dropout', '0.67', '0.89'],
        ['Learning rate / weight decay', '4×10⁻⁴ / 7×10⁻⁴', '3×10⁻⁴ / 8×10⁻⁴'],
        ['Class balancing (per epoch)', 'both classes resampled to the mean size', 'minority oversampling'],
        ['Optimiser / loss', 'Adam, linear warm-up, plateau reduction / binary cross-entropy', 'same'],
        ['Stopping and checkpoint', 'early stopping on validation loss (patience 50, max 500 epochs); lowest validation loss', 'same'],
        ['Sensitivity analysis', 'Mateus rule: best validation AUC with train–validation gap ≤ 0.05', 'same'],
        ['Seeds / parameters', '3 seeds averaged / 15,057 (with clinical input)', '3 / 1,841'],
    ], [5.2, 7.0, 4.8], 'Network and training settings. OS uses the LR settings because RadGraph specifies no OS model.'),

    ('h1', H[10]),
    ('p', 'Two random forests are trained on the same splits: one on the GTV features chosen by each selector, which isolates '
          'what the peritumoral graph adds, and one on the clinical variables alone. Their sampling strategy is chosen on '
          'validation AUC.'),

    ('h1', H[11]),
    ('p', 'Discrimination is measured by the AUC with 95% percentile bootstrap confidence intervals (1000 resamples). Each '
          'configuration is compared with the reference configuration and with the random forest on the same features by '
          'paired DeLong tests with Holm correction. Selection stability is measured by the Kuncheva index (overlap corrected '
          'for chance), the Jaccard index and Kendall\'s W, across the five folds, across 30 bootstrap resamples per fold and '
          'between the two largest centres. Spearman correlations relate stability to external AUC and validation AUC to '
          'external AUC. Mateus et al. report the mean and range of per-fold test AUCs in cross-validation; these are recorded '
          'for direct comparison.'),

    ('h1', H[12]),
    ('p', 'The pipeline (radgraph-hnscc) is a Python package with a single command-line entry point. Every stage is cached '
          'and deterministic for a given seed, every model is stored as a self-contained checkpoint (weights, '
          'hyper-parameters, selected features, scaler, clinical encoder, threshold and topology), and an inference tool '
          'applies the stored ensembles to new patients without retraining. The manuscript figures, tables and every number '
          'quoted in the text are regenerated from the results by one script.'),
    ('code', 'cd radgraph-hnscc\n'
             './run_mateus_study.sh            # full study\n'
             'python export_results.py         # study tables and figures\n'
             'python predict.py --task LR --dataset MAASTRO   # inference'),

    ('h1', H[13]),
    ('p', 'Results on the 199-patient image set are preliminary (Figure 5). For LR, the GAT chosen on validation reached an '
          'external AUC of 0.71 (95% CI 0.62–0.81) with clinical variables in the cohort split, above the 0.59 of the Mateus '
          'et al. CNN; in cross-validation the mean per-fold AUC matched the CNN (0.51–0.56 versus 0.53–0.57). No configuration '
          'was significantly better than the RadGraph reference after Holm correction. DM and OS were not predicted better '
          'than chance by the configurations chosen on validation, consistent with the small number of development events. '
          'Spectral Graph-FS gave the most stable selections (Kuncheva 0.64–0.73 across folds, against 0.11–0.34 for LASSO), but '
          'stability was not associated with external discrimination, and validation AUC did not predict external AUC '
          '(Spearman ρ between −0.46 and 0.20). The final-layer attention was nearly uniform (within ±1.2% of 1/21 for 98% of '
          'supervoxels), so attention maps should not be interpreted as localising risk.'),
    ('fig', 'fig_roc_all.png', 16.0, 'External ROC curves for LR, DM and OS under both designs, with and without clinical data.'),
    ('fig', 'fig6_stability.png', 16.0, 'Feature-selection stability (Kuncheva index) across folds, bootstrap resamples and centres.'),

    ('h1', H[14]),
    ('numbered', [
        'Bae J, Mani K, Czerwonka L, Vanison C, Ryu S, Prasanna P. Spatial radiomic graphs for outcome prediction in radiation '
        'therapy-treated head and neck squamous cell carcinoma using pretreatment CT. Radiol Imaging Cancer. 2025;7(2):e240161.',
        'Moradmand H, Molitoris J, Ling X, et al. Graph feature selection for enhancing radiomic stability and reproducibility '
        'across multiple institutions in head and neck cancer. Sci Rep. 2025;15:27995.',
        'Mateus P, Volmer L, Wee L, et al. Image based prognosis in head and neck cancer using convolutional neural networks: '
        'a case study in reproducibility and optimization. Sci Rep. 2023;13:18176.',
        'Diamant A, Chatterjee A, Vallières M, Shenouda G, Seuntjens J. Deep learning in head & neck cancer outcome prediction. '
        'Sci Rep. 2019;9:2764.',
        'Vallières M, Kay-Rivest E, Perrin LJ, et al. Radiomics strategies for risk assessment of tumour failure in '
        'head-and-neck cancer. Sci Rep. 2017;7:10117.',
        'Aerts HJWL, Velazquez ER, Leijenaar RTH, et al. Decoding tumour phenotype by noninvasive imaging using a quantitative '
        'radiomics approach. Nat Commun. 2014;5:4006.',
        'Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. ICLR. 2018.',
        'Achanta R, Shaji A, Smith K, et al. SLIC superpixels compared to state-of-the-art superpixel methods. IEEE TPAMI. '
        '2012;34(11):2274–2282.',
        'van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. '
        'Cancer Res. 2017;77(21):e104–e107.',
        'Kuncheva LI. A stability index for feature selection. Proc. IASTED Artificial Intelligence and Applications. 2007:390–395.',
        'DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated ROC curves. Biometrics. '
        '1988;44:837–845.',
    ]),
]

if __name__ == '__main__':
    d = Doc('Spatial Radiomic Graph Attention Networks for Head and Neck Cancer',
            'Study rationale, design and methods: feature selection, graph topology and external validation',
            'Study rationale, design and methods',
            'Supersedes "GNN_Paper -- Introduction_and_Methods" (May 2026), which described a single-cohort design.')
    d.contents(H)
    d.add(blocks)
    print(d.save('01_Study_Rationale_Design_and_Methods_v2'))
