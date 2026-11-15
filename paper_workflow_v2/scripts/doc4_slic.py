"""Document 4: Stage 2 - peritumoral supervoxel generation (upgrade of step2_slic_supervoxel_documentation)."""
from docgen import Doc

H = ['Purpose and position in the pipeline', 'Why supervoxels', 'Inputs from Stage 1',
     'Step 1: peritumoral mask', 'Step 2: intensity normalisation', 'Step 3: the SLIC-zero algorithm',
     'Step 4: post-processing of the label map', 'Output and observed behaviour in this study',
     'Use of supervoxels downstream', 'Known limitations and recommended improvements',
     'Parameter reference', 'Troubleshooting', 'References']

blocks = [
    ('h1', H[0]),
    ('p', 'Stage 2 partitions the tissue surrounding the primary gross tumour volume (GTV) into compact, spatially contiguous '
          'sub-volumes called supervoxels. Each supervoxel later becomes a candidate node of the patient graph that the graph '
          'attention network (GAT) analyses. The partition is computed with simple linear iterative clustering in its '
          'zero-parameter form (SLIC-zero, also known as SLICO), exactly as in the original RadGraph implementation '
          '(Bae et al., 2025; authors\' RadiomicSupervoxels.py).'),
    ('table', ['Stage', 'Module', 'Operation', 'Output'], [
        ['1', 'radgraph/data/preprocessing.py', 'resample CT and GTV to 1 mm, define the 50 mm box, crop', 'CT, GTV and box arrays'],
        ['**2**', '**radgraph/data/supervoxels.py**', '**SLIC-zero inside the box, GTV excluded**', '**supervoxel label map**'],
        ['3', 'radgraph/data/features.py', 'PyRadiomics features of the GTV and of every supervoxel', '93 features per region, centroids'],
        ['4', 'radgraph/graphs/builder.py', 'select 20 supervoxels, connect them to the GTV (four topologies)', 'one graph per patient'],
        ['5', 'radgraph/engine/train.py', 'train and evaluate the GAT', 'models, predictions, attention'],
    ], [1.2, 5.0, 6.3, 4.5], 'Pipeline stages; this document covers Stage 2.'),
    ('note', 'key', 'Key design principle',
     'The GTV is never split into supervoxels. Its voxels are removed before SLIC runs, and the GTV enters the graph as a '
     'single node whose features are extracted from the GTV mask itself. Supervoxels describe only the peritumoral tissue.'),

    ('h1', H[1]),
    ('p', 'Conventional radiomics summarises the tumour with a single feature vector and ignores the surrounding tissue, through '
          'which microscopic extension and nodal spread occur. Analysing every voxel separately is impractical (around 10⁷ '
          'voxels per patient) and statistically unstable. Supervoxels are an intermediate representation: about one hundred '
          'locally homogeneous regions per patient, each large enough for stable texture statistics yet small enough to '
          'preserve spatial heterogeneity.'),
    ('table', ['Representation', 'Spatial detail', 'Regions per patient', 'Use'], [
        ['Whole-GTV radiomics', 'none', '1', 'random-forest baselines'],
        ['Individual voxels', 'maximal', '~10⁷', 'computationally infeasible'],
        ['SLIC-zero supervoxels', 'intermediate', '100 (≈ 22 cm³ each)', 'graph nodes in this study'],
    ], [4.5, 3.0, 4.0, 5.5], 'Levels of spatial representation.'),

    ('h1', H[2]),
    ('p', 'Stage 1 resamples the CT with linear interpolation and the GTV mask with nearest-neighbour interpolation to '
          '1 mm isotropic voxels, defines the peritumoral region as the GTV bounding box enlarged by 50 mm in every direction '
          '(clipped to the image), and crops both arrays to this box. Stage 2 therefore receives:'),
    ('table', ['Array', 'Shape', 'Type', 'Content'], [
        ['ct_array', '(Z, Y, X)', 'float32', 'CT in Hounsfield units, 1 mm isotropic, cropped to the box'],
        ['gtv_array', '(Z, Y, X)', 'uint8', 'primary GTV mask (label 1 of the segmentation)'],
        ['region_mask', '(Z, Y, X)', 'uint8', 'the 50 mm box (all ones after cropping)'],
        ['n_segments', 'scalar', 'int', 'target number of supervoxels, config.N_SUPERVOXELS_TARGET = 100'],
    ], [3.0, 2.5, 2.0, 9.5], 'Inputs to Stage 2.'),

    ('h1', H[3]),
    ('p', 'The peritumoral mask is the box with the GTV removed. SLIC never sees or labels a GTV voxel, so the graph keeps '
          'exactly one tumour node and the two feature-extraction paths (GTV and supervoxels) cannot overlap.'),
    ('code', 'peri = (region_mask > 0) & (gtv_array == 0)      # box minus GTV\n'
             'if not peri.any():\n'
             '    raise ValueError("Peritumoral region is empty")'),

    ('h1', H[4]),
    ('p', 'SLIC clusters voxels by intensity and position, so the intensities are rescaled to [0, 1] using the minimum and '
          'maximum inside the peritumoral mask:'),
    ('eq', 'I_norm(v) = clip( (I(v) − min_R) / (max_R − min_R + 10⁻⁶), 0, 1 )'),
    ('code', 'vals = ct_array[peri].astype(np.float32)\n'
             'lo, hi = float(vals.min()), float(vals.max())\n'
             'img = np.clip((ct_array - lo) / (hi - lo + 1e-6), 0.0, 1.0)'),
    ('note', 'warn', 'Correction to version 1 of this document',
     'Version 1 stated that region-restricted normalisation gives soft tissue the full [0, 1] range. In this study it does '
     'not: the 50 mm box usually extends beyond the skin and contains air (median minimum −1024 HU) as well as bone or '
     'dental metal (median maximum 2976 HU). Soft tissue between 0 and 100 HU is therefore mapped to approximately '
     '0.26–0.28, only 2.5% of the available range, and SLIC is driven mainly by air and bone boundaries. A soft-tissue '
     'window before normalisation (for example −160 to 240 HU) combined with a body mask is recommended (Section 10).'),

    ('h1', H[5]),
    ('h2', 'Seeding on a regular grid'),
    ('p', 'SLIC places approximately K seeds on a regular grid inside the mask, with spacing'),
    ('eq', 'S = (V / K)^(1/3)'),
    ('p', 'where V is the number of voxels in the mask and K the target number of supervoxels. With the median box of '
          '2.2 litres and K = 100, S ≈ 28 mm, which already indicates that supervoxels in this study are large.'),
    ('h2', 'Distance in the joint intensity–spatial space'),
    ('p', 'Each voxel is assigned to the nearest cluster centre according to a distance that combines intensity and position:'),
    ('eq', 'D = √( d_I² + (m / S)² · d_s² )'),
    ('p', 'with d_I the intensity difference, d_s the Euclidean distance in voxels and m a compactness constant. A large m gives '
          'regular, compact regions; a small m lets boundaries follow intensity edges.'),
    ('h2', 'Zero-parameter compactness'),
    ('p', 'A single global m cannot suit fat, muscle and tumour margins at the same time. SLIC-zero replaces it, for each '
          'cluster k and iteration, by the largest intensity distance observed in that cluster during the previous iteration:'),
    ('eq', 'm_k = max_{v ∈ k} d_I(v, c_k),     D_k = √( d_I² + (m_k / S)² · d_s² )'),
    ('p', 'Homogeneous regions obtain a small m_k and compact supervoxels; heterogeneous regions a large m_k and boundaries '
          'that follow tissue interfaces. In scikit-image (version 0.21 in this project) the compactness argument then serves only '
          'as a starting value.'),
    ('h2', 'Local search, update and convergence'),
    ('p', 'Each centre only examines voxels within a 2S window, which makes an iteration linear in the number of voxels rather '
          'than proportional to voxels × clusters. Centres are then moved to the mean position and intensity of their members. '
          'scikit-image performs a fixed maximum of 10 iterations (max_num_iter), which is sufficient for CT.'),
    ('h2', 'Connectivity enforcement'),
    ('p', 'With enforce_connectivity=True, fragments that are disconnected from their cluster are merged into an adjacent '
          'region, so every supervoxel is a single connected volume, which is required for meaningful texture features.'),
    ('code', 'labels = slic(img, mask=peri, n_segments=100, slic_zero=True, channel_axis=None,\n'
             '              enforce_connectivity=True, start_label=1)'),

    ('h1', H[6]),
    ('p', 'Labels outside the peritumoral mask are reset to zero and the remaining labels are renumbered to the contiguous '
          'range 1…n with a single vectorised call, so downstream code can iterate over range(1, n + 1) without gaps. A warning '
          'is printed if fewer than 20 supervoxels remain, because the graph requires 20.'),
    ('code', 'labels = np.where(peri, labels, 0)\n'
             'uniq, inv = np.unique(labels, return_inverse=True)\n'
             'labels = inv.reshape(labels.shape).astype(np.int32)   # 0 stays 0, others become 1..n'),

    ('h1', H[7]),
    ('p', 'The label map is stored with the cropped CT and GTV in the preprocessing cache '
          '(outputs/cache/preprocessed/<patient>.npz, key supervoxel_labels). Across all 336 patients processed in this study '
          '(199 Canadian, 137 MAASTRO), the behaviour was highly regular:'),
    ('table', ['Quantity', 'Observed value'], [
        ['Supervoxels per patient', '100 in every patient'],
        ['Supervoxels with fewer than 8 voxels', 'none'],
        ['Median supervoxel volume', '22.3 cm³ (≈ 2.8 cm cube)'],
        ['Median peritumoral box volume', '2.2 litres'],
        ['Median GTV volume', '10.1 cm³'],
        ['Supervoxels that are mostly air (>50% of voxels below −500 HU)', 'median 22% per patient (IQR 18–29%)'],
    ], [9.0, 8.0], 'Observed supervoxel statistics.'),
    ('fig', 'slic_steps.png', 17.0,
     'Stage 2 on an external test patient (HN1260, oropharyngeal cancer; axial slice through the GTV). **a** Cropped CT with the '
     'GTV (orange). **b** Peritumoral mask: the whole 50 mm box minus the GTV, including air outside the neck. **c** SLIC-zero '
     'supervoxels (random colours). **d** Supervoxels in which more than half of the voxels are air (red).'),
    ('fig', 'supervoxel_sizes.png', 10.5, 'Distribution of supervoxel volumes over all 33,600 supervoxels of the study.'),

    ('h1', H[8]),
    ('p', '**Feature extraction (Stage 3).** For every supervoxel and for the GTV, PyRadiomics computes 93 features '
          '(first order 18, GLCM 24, GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5) with a fixed bin width of 25 HU; shape features '
          'are excluded. Supervoxels with fewer than 8 voxels are skipped, and the centroid of every region is stored.'),
    ('p', '**Graph construction (Stage 4).** Twenty supervoxels are selected per patient. Node features are the selected '
          'radiomic features, z-scored with a scaler fitted on the GTV features of the training patients and clipped at ±5. '
          'In the radiomic topologies the 20 supervoxels with the smallest **Euclidean distance** to the GTV in this z-scored '
          'feature space are chosen (as in the authors\' GraphGeneration.py); in the spatial topologies the 20 closest '
          'centroids in millimetres are chosen.'),
    ('note', 'warn', 'Correction to version 1',
     'Version 1 described selection by cosine similarity, GTV-relative normalisation and edge distances as model inputs. '
     'The authors\' code and the current implementation use Euclidean distance on z-scored features, the GTV is scaled like '
     'every other node, and edge attributes are stored but not used by the network.'),

    ('h1', H[9]),
    ('p', 'The analysis of the study data identified three weaknesses of Stage 2 that are worth correcting in the next version '
          'of the pipeline. Each would require re-running Stages 2 to 5.'),
    ('numbered', [
        '**Body mask.** Restrict the peritumoral mask to the patient body (for example CT > −500 HU followed by hole filling), '
        'so that no supervoxel describes air. At present about one fifth of the supervoxels are mostly air and can be selected '
        'as graph nodes.',
        '**Soft-tissue window.** Clip the CT to a soft-tissue window (for example −160 to 240 HU) before normalisation, so that '
        'SLIC responds to soft-tissue interfaces rather than to air and bone.',
        '**Supervoxel scale.** With a 50 mm margin and 100 supervoxels, each supervoxel is about 22 cm³, comparable to twice the '
        'median GTV. A sensitivity analysis with 200–400 supervoxels, or a fixed target volume, would test whether finer '
        'regions carry more prognostic information.',
    ]),

    ('h1', H[10]),
    ('table', ['Parameter', 'Value', 'Effect', 'When to change'], [
        ['N_SUPERVOXELS_TARGET', '100', 'target number of supervoxels', 'sensitivity analysis of supervoxel scale'],
        ['slic_zero', 'True', 'adaptive compactness', 'never; defines the method'],
        ['enforce_connectivity', 'True', 'single connected regions', 'only for debugging'],
        ['channel_axis', 'None', 'grey-level 3-D input', 'never'],
        ['start_label', '1', '0 is reserved for background and GTV', 'never'],
        ['PERITUMORAL_MARGIN_MM', '50', 'size of the peritumoral box', 'margin sensitivity analysis'],
        ['MIN_SUPERVOXEL_VOXELS', '8', 'minimum size for feature extraction', 'rarely; no supervoxel is this small'],
        ['N_SUPERVOXELS_SELECTED', '20', 'supervoxels per graph', 'graph-size sensitivity analysis'],
    ], [4.6, 1.6, 5.2, 5.6], 'Stage 2 parameters (radgraph/config.py).'),

    ('h1', H[11]),
    ('table', ['Symptom', 'Likely cause', 'Remedy'], [
        ['Warning: fewer than 20 supervoxels', 'very large GTV or very small box', 'inspect the box; lower n_segments for that patient'],
        ['ValueError: peritumoral region is empty', 'GTV fills the box or the mask is empty', 'check the GTV label and Stage 1 output'],
        ['GTV voxels carry supervoxel labels', 'GTV not removed from the mask', 'pass gtv_array; check the peri mask'],
        ['Blocky supervoxels following air or bone', 'normalisation range dominated by air and bone', 'apply the soft-tissue window (Section 10)'],
        ['TypeError on multichannel', 'scikit-image older than 0.19', 'use scikit-image 0.21 (channel_axis argument)'],
    ], [5.0, 5.5, 6.5], 'Troubleshooting.'),

    ('h1', H[12]),
    ('numbered', [
        'Achanta R, Shaji A, Smith K, Lucchi A, Fua P, Süsstrunk S. SLIC superpixels compared to state-of-the-art superpixel '
        'methods. IEEE Trans Pattern Anal Mach Intell. 2012;34(11):2274–2282.',
        'Bae J, Mani K, Czerwonka L, Vanison C, Ryu S, Prasanna P. Spatial radiomic graphs for outcome prediction in radiation '
        'therapy-treated head and neck squamous cell carcinoma using pretreatment CT. Radiol Imaging Cancer. 2025;7(2):e240161.',
        'van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. '
        'Cancer Res. 2017;77(21):e104–e107.',
        'van der Walt S, Schönberger JL, Nunez-Iglesias J, et al. scikit-image: image processing in Python. PeerJ. 2014;2:e453.',
    ]),
]

if __name__ == '__main__':
    d = Doc('Stage 2: Peritumoral Supervoxel Generation',
            'Zero-parameter SLIC in the radiomic graph pipeline: method, implementation, observed behaviour and improvements',
            'Stage 2 · Peritumoral supervoxel generation',
            'Supersedes step2_slic_supervoxel_documentation (May 2026). Describes the current code (radgraph/data/supervoxels.py).')
    d.contents(H)
    d.add(blocks)
    print(d.save('04_Stage2_Supervoxel_Generation_v2'))
