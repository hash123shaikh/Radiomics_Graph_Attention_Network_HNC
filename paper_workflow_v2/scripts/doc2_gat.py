"""Document 2: graph attention networks explained (upgrade of GAT_Questions_and_Answers)."""
from docgen import Doc

H = ['Building blocks: graphs, neural networks and attention', 'How a graph attention layer works',
     'The radiomic graph attention network of this study', 'How the network is trained',
     'Three core questions', 'Frequently asked questions', 'Quick reference']

blocks = [
    ('note', 'info', 'How to read this document',
     'Sections 1 and 2 explain the concepts without assuming a machine-learning background. Section 3 describes exactly what '
     'the network of this study computes, Section 4 how it is trained, and Sections 5 and 6 answer the questions that '
     'clinicians, students and reviewers most often ask. Every number refers to the current implementation and to the '
     'preliminary results on the 199-patient development image set.'),

    ('h1', H[0]),
    ('h2', 'What is a graph?'),
    ('p', 'In mathematics a graph is a set of objects, called nodes, connected by links, called edges. A road map is a graph '
          '(cities and roads), and so is a molecule (atoms and bonds). In this study every patient is one graph. The primary '
          'tumour (GTV) is one node, twenty regions of the surrounding tissue (supervoxels) are further nodes, and edges '
          'describe which regions exchange information. Each node carries a short list of numbers, its feature vector: six '
          'radiomic texture and intensity measurements of that region.'),
    ('p', 'A graph is used instead of a table because a table treats each region independently, whereas a graph also encodes '
          'how the regions are arranged around the tumour. A model operating on graphs can therefore learn from combinations '
          'of regions and their relation to the tumour, not only from each region in isolation.'),
    ('h2', 'What is a neural network?'),
    ('p', 'A neural network is a function with adjustable numbers, called parameters or weights. Input features are '
          'multiplied by weights, summed and passed through simple non-linear functions, layer after layer, to produce an '
          'output such as the probability of recurrence. During training the weights are adjusted step by step so that the '
          'predictions on the training patients become more accurate. The size of these steps is set by the learning rate, '
          'and the direction of each step is obtained by backpropagation of the prediction error.'),
    ('h2', 'What is attention?'),
    ('p', 'Attention lets a model weight its inputs according to their relevance instead of treating them equally. At a busy '
          'meeting one listens closely to the person addressing the room and barely to side conversations; the importance '
          'assigned to each speaker is an attention weight. In a neural network the weights are computed from the data '
          'themselves and can therefore differ for every patient.'),

    ('h1', H[1]),
    ('h2', 'Message passing'),
    ('p', 'All graph neural networks share one operation, message passing. In every layer each node collects the feature '
          'vectors of its neighbours, combines them with its own, and updates its representation. Stacking layers lets '
          'information travel further: after two layers a node has indirect knowledge of its neighbours\' neighbours.'),
    ('h2', 'From averaging to attention'),
    ('p', 'A basic graph network combines neighbours by averaging them, so a region at the invasive margin and a region of '
          'distant muscle contribute equally. A graph attention network (GAT; Veličković et al., 2018) replaces the average by '
          'a weighted sum whose weights are learned:'),
    ('eq', 'h′_i = Σ_{j ∈ N(i) ∪ {i}} α_ij · W h_j ,       with   Σ_j α_ij = 1'),
    ('p', 'Note that the node also attends to itself: PyTorch Geometric adds a self-loop to every node by default. The GTV '
          'node in a star graph therefore distributes its attention over 21 nodes, 20 supervoxels and itself, and a '
          'perfectly uniform attention would give every node 1/21 ≈ 0.048.'),
    ('h2', 'Computing one attention weight'),
    ('p', 'The weight that the GTV (node i) gives to supervoxel j is computed in four steps:'),
    ('numbered', [
        '**Projection.** Both feature vectors are multiplied by a learned matrix W, which maps the six radiomic features to '
        '16 numbers per attention head.',
        '**Scoring.** The two projected vectors are concatenated and multiplied by a learned vector a, giving one raw score; '
        'a LeakyReLU (slope 0.2 for negative values) is applied.',
        '**Normalisation.** The raw scores of all neighbours of node i are passed through a softmax, so the weights are '
        'positive and sum to one.',
        '**Regularisation.** During training, attention dropout randomly removes neighbours from this sum (probability 0.67 for '
        'LR and OS, 0.89 for DM), which prevents the network from relying on a few regions.',
    ]),
    ('eq', 'α_ij = softmax_j ( LeakyReLU( aᵀ [ W h_i ‖ W h_j ] ) )'),
    ('h2', 'Several heads in parallel'),
    ('p', 'Each layer runs four independent attention heads with their own W and a, so that different notions of relevance '
          'can be learned. In the first four layers the four 16-dimensional outputs are concatenated into 64 numbers per node; '
          'the final layer uses a single head and returns 16 numbers per node.'),
    ('h2', 'How information travels in each topology'),
    ('p', 'In a **star** graph supervoxels are connected only to the GTV. In layer 1 the GTV receives a weighted summary of '
          'all supervoxels and each supervoxel receives the GTV; from layer 2 onwards each supervoxel indirectly receives the '
          'other supervoxels through the GTV. In the **kNN** topologies supervoxels are also connected to their nearest '
          'neighbours in feature or physical space, so information can travel between regions without passing through the '
          'tumour. Because the prediction is read from the GTV, the supervoxels always influence the output through the GTV.'),
    ('fig', 'fig2_topologies.png', 17.0, 'The four topologies drawn on the CT of an external patient (HN1260). Orange: GTV '
     'and read-out node; blue: the 20 selected supervoxels; yellow: kNN edges.'),

    ('h1', H[2]),
    ('h2', 'Input'),
    ('p', 'Each patient graph has 21 nodes: the GTV (node 0) and 20 supervoxels. Every node carries the same six radiomic '
          'features, chosen on the training patients by one of nine feature-selection strategies (Section 6, Q7). Features are '
          'z-scored with the mean and standard deviation of the GTV features of the training patients and clipped at ±5, so '
          'that the GTV and the supervoxels are on the same scale and no information from validation or test patients enters '
          'the scaling.'),
    ('h2', 'Layers'),
    ('table', ['Block', 'Operation', 'Output per node'], [
        ['GAT layers 1–4', 'GATConv (4 heads × 16), graph-wise LayerNorm, ReLU', '64'],
        ['GAT layer 5', 'GATConv (1 head × 16)', '16'],
        ['Read-out', 'keep the GTV node only; dropout 0.5', '16 (one vector per patient)'],
        ['Clinical fusion (optional)', 'concatenate 16 clinical variables', '32'],
        ['Output', 'linear layer to one logit; sigmoid', 'probability of the event'],
    ], [4.2, 8.6, 4.2], 'Architecture of the LR and OS network (the DM network has 3 layers and 8 units per head).'),
    ('fig', 'fig_architecture_author.png', 17.0, 'Radiomic graph attention network (left) and multi-head attention at the GTV '
     'node (right).'),
    ('h2', 'Parameters'),
    ('table', ['Component', 'Calculation', 'Parameters'], [
        ['GAT layer 1', '6×64 weights + 64 + 64 attention + 64 bias', '576'],
        ['GAT layers 2–4', '3 × (64×64 + 3×64)', '12,864'],
        ['LayerNorm ×4', '4 × (64 scale + 64 shift)', '512'],
        ['GAT layer 5', '64×16 + 3×16', '1,072'],
        ['Output layer', '32 weights + 1 bias', '33'],
        ['**Total**', '', '**15,057**'],
    ], [4.2, 8.6, 4.2], 'Trainable parameters of the LR network with clinical input (DM network: 1,841).'),
    ('p', 'Fifteen thousand parameters is small by deep-learning standards but large relative to 13 development events, which '
          'is why strong dropout, weight decay and early stopping are essential.'),

    ('h1', H[3]),
    ('p', 'The network minimises binary cross-entropy with the Adam optimiser. To counter class imbalance, the training set is '
          'resampled at every epoch: for LR and OS both classes are resampled to the mean class size, for DM the minority class '
          'is oversampled. The learning rate rises linearly during a warm-up (10 epochs for LR and OS, 5 for DM) and is halved '
          'when the validation loss stops improving for 20 epochs, with a floor of 10⁻⁵. Gradients are clipped at norm 1.'),
    ('p', 'Training stops when the validation loss has not improved for 50 epochs (maximum 500), and the checkpoint with the '
          'lowest validation loss is kept, as in RadGraph. As a sensitivity analysis, the checkpoint rule of Mateus et al. is '
          'also applied: the epoch with the highest validation AUC among epochs whose training–validation AUC gap is at most '
          '0.05. Each configuration is trained with three random seeds and the predictions are averaged.'),
    ('note', 'warn', 'Corrections to version 1',
     'Version 1 described an input projection layer, ELU activations, a batch-normalised clinical projection, GTV-relative '
     'scaling, cosine similarity for supervoxel selection and early stopping on validation AUC after 15 epochs. None of these is '
     'part of the RadGraph code or of the current implementation: the network starts directly with a GAT layer, uses ReLU and '
     'layer normalisation, concatenates the clinical variables unchanged, scales all nodes with the training GTV statistics, '
     'selects supervoxels by Euclidean distance and stops on the validation loss.'),

    ('h1', H[4]),
    ('h2', 'Q1. What is the read-out, and why only the GTV?'),
    ('p', 'After five layers every node has a 16-dimensional representation, but one prediction per patient is needed. Many '
          'graph networks average all nodes. RadGraph instead keeps only the GTV node, because the GTV has received attention-'
          'weighted messages from the surrounding tissue at every layer and is the clinically meaningful anchor. The supervoxel '
          'representations are therefore not discarded information: their content has already been passed into the GTV.'),
    ('code', '21 node vectors after layer 5\n'
             '  -> keep node 0 (GTV): 16 numbers\n'
             '  -> dropout 0.5\n'
             '  -> [+ 16 clinical variables] = 32 numbers\n'
             '  -> linear layer -> one logit -> sigmoid -> P(event)'),
    ('h2', 'Q2. How does the network assign weights to the supervoxels?'),
    ('p', 'At every layer and in every head, each edge receives a weight from the projection–scoring–softmax sequence of '
          'Section 2. The weights depend on the features of both ends of the edge and are recomputed for every patient. In '
          'principle, regions that resemble high-risk tissue can therefore receive more weight in one patient than in another.'),
    ('h2', 'Q3. How do supervoxels exchange information in a star graph?'),
    ('p', 'Supervoxels are not directly connected in a star graph, but the GTV relays information. In layer 1 the GTV '
          'summarises all supervoxels; in layer 2 each supervoxel receives that summary; from then on every supervoxel '
          'indirectly carries information from all others. In the kNN topologies information can also move directly between '
          'neighbouring supervoxels.'),

    ('h1', H[5]),
    ('h2', 'Q4. How does this differ from conventional radiomics?'),
    ('table', ['Aspect', 'Conventional radiomics', 'Radiomic graph network'], [
        ['Regions', 'whole GTV only', 'GTV and 20 peritumoral supervoxels'],
        ['Inputs per patient', 'one vector of selected features', '21 nodes × 6 features, plus edges'],
        ['Spatial information', 'none', 'encoded by the graph'],
        ['Model', 'random forest, logistic regression', 'graph attention network'],
        ['Interaction between regions', 'not modelled', 'learned through message passing'],
    ], [4.6, 5.6, 6.8], 'Conventional radiomics versus the radiomic graph network.'),
    ('p', 'In this study the GAT outperformed a random forest trained on exactly the same GTV features for 7 of 9 feature sets '
          'in cross-validation for LR (mean external AUC 0.59 versus 0.57), which suggests that the peritumoral tissue adds '
          'some information, although none of these differences was significant.'),
    ('h2', 'Q5. What is the AUC, and why are confidence intervals essential?'),
    ('p', 'The area under the ROC curve (AUC) is the probability that a randomly chosen patient with the event receives a '
          'higher predicted risk than a randomly chosen patient without it: 0.5 is chance and 1.0 perfect separation. Unlike '
          'accuracy it is not inflated by class imbalance. With 30 external LR events, a 95% bootstrap confidence interval '
          'around an AUC of 0.65 spans roughly ±0.11, so differences of a few hundredths between configurations are not '
          'meaningful; formal comparisons use the DeLong test with Holm correction.'),
    ('h2', 'Q6. What do the attention maps show?'),
    ('p', 'Attention weights of the final layer can be mapped back onto the CT, giving a map of the regions the GTV attended '
          'to. In this study, however, attention was almost uniform: across all 121 external LR patients, 98% of the '
          'supervoxel weights lay within ±1.2% of the patient mean, and the ratio between the largest and smallest weight in a '
          'patient was typically 1.015. The network therefore behaved essentially as an averaging aggregator, and the maps '
          'cannot be interpreted as localising risk. Likely reasons are the small number of training events, the strong '
          'attention dropout and the similarity of the selected supervoxels to each other.'),
    ('fig', 'fig_attention_ct.png', 16.0, 'Attention for an external patient with locoregional recurrence (HN1792). The weights '
     'range only from 0.047 to 0.048, against a uniform value of 0.048.'),
    ('h2', 'Q7. Why six features, and how are they selected?'),
    ('p', 'With 13 development events, a network with many input features overfits, so a small feature set is necessary. '
          'RadGraph used fixed endpoint-specific sets (four features for LR). This study compares nine strategies at an equal '
          'size of six features: the RadGraph set, all features, four supervised selectors (mRMR, LASSO, Boruta, RFE with a '
          'random forest) and three graph-based variants. All are fitted on training patients only.'),
    ('h2', 'Q8. What is graph-based feature selection?'),
    ('p', 'Radiomic features are strongly correlated; many texture features measure almost the same property. Graph-FS '
          'builds a graph in which features are nodes and highly correlated features are connected, finds communities of '
          'redundant features and keeps one representative per community. Because the community structure hardly changes when '
          'the training patients change, the selection is stable. The spectral variant was the most stable strategy in this '
          'study (Kuncheva index 0.64–0.73 across folds, against 0.11–0.34 for LASSO). However, label-free Graph-FS ignores the '
          'outcome, and its stable features were not more predictive. The hybrid variant introduced in this study keeps the '
          'graph but chooses the most outcome-relevant feature in each community.'),
    ('fig', 'fig_graphfs_workflow.png', 17.0, 'Graph-based feature selection on the training data of the first fold (LR).'),
    ('h2', 'Q9. Why compare four topologies?'),
    ('p', 'The radiomic star of RadGraph selects supervoxels that resemble the GTV, which may lie anywhere in the 5 cm box, '
          'including skin or air. The spatial star selects the regions immediately around the tumour, where microscopic '
          'extension is expected. The kNN graphs test whether direct communication between regions helps. In this study the '
          'four topologies gave nearly identical mean LR AUCs (0.59–0.62); for OS, radiomic topologies tended to do better.'),
    ('h2', 'Q10. What does the network learn?'),
    ('p', 'It learns the projection matrices and attention vectors of every head and layer, the layer-normalisation scales and '
          'shifts, and the output weights that combine the GTV representation with the clinical variables (Section 3).'),
    ('h2', 'Q11. What happens if a patient has fewer than 20 supervoxels?'),
    ('p', 'The graph then contains all available supervoxels, and PyTorch Geometric handles graphs of different sizes in the '
          'same batch. In practice it never occurred: every patient in this study had exactly 100 supervoxels.'),
    ('h2', 'Q12. What is over-smoothing?'),
    ('p', 'After many layers all node representations become similar, because each is an average of an ever larger '
          'neighbourhood. Five layers on a 21-node graph is close to this limit. Layer normalisation and dropout mitigate it, and '
          'the near-uniform attention observed here is consistent with representations that are already smooth.'),
    ('h2', 'Q13. How is the model evaluated fairly?'),
    ('p', 'Following Mateus et al., models are developed on the four Canadian centres by 5-fold stratified cross-validation or '
          'by a centre-based split (HGJ and CHUS for training, HMR and CHUM for validation) and tested once on the external '
          'MAASTRO cohort. Feature selection, scaling, clinical encoding and training are refitted inside every fold, so that '
          'no information from validation or test patients leaks into the model. This replaces the single random 70/15/15 '
          'split of version 1, whose test set contained only 19 patients.'),
    ('h2', 'Q14. Why did validation performance not identify the best external model?'),
    ('p', 'Each cross-validation fold contains only two or three LR events, so validation AUC is extremely noisy. When more than '
          'one hundred configurations are compared, the best validation result largely reflects chance. Across configurations, '
          'the correlation between validation and external AUC ranged from −0.46 to 0.20. Reporting a pre-specified reference '
          'configuration and the distribution over the whole grid is therefore more informative than reporting the winner.'),
    ('fig', 'fig8_val_vs_test.png', 16.0, 'Validation AUC versus external AUC for every configuration.'),
    ('h2', 'Q15. How does this compare with the published CNN?'),
    ('p', 'Mateus et al. trained a convolutional neural network on the same collections with the same designs. For LR the '
          'radiomic GAT reached 0.71 with clinical data in the cohort split versus 0.59 for the CNN, and was similar in '
          'cross-validation. For DM and OS the CNN was clearly better; the development data of this study contain less than half '
          'of the events available to the CNN and only oropharyngeal tumours.'),
    ('h2', 'Q16. What are the main limitations?'),
    ('bullets', [
        'Small, event-poor development set (199 of 298 patients with images, all oropharyngeal).',
        'Only seven external DM events.',
        'About one fifth of supervoxels lie mostly in air, and the intensity normalisation compresses soft-tissue contrast.',
        'Hyper-parameters adopted from RadGraph rather than tuned; OS reuses the LR settings.',
        'Binary horizons exclude censored patients; survival models would use the data more efficiently.',
        'Attention is near-uniform and cannot yet support clinical interpretation.',
    ]),

    ('h1', H[6]),
    ('table', ['Term', 'Meaning'], [
        ['Graph', 'nodes connected by edges'],
        ['Node / edge', 'a region (GTV or supervoxel) / a connection along which information is exchanged'],
        ['Feature vector', 'the six radiomic values describing a node'],
        ['Message passing', 'one round in which every node updates itself from its neighbours'],
        ['Attention weight', 'learned weight in [0, 1] of a neighbour; weights of a node sum to 1'],
        ['Head', 'one independent attention computation; four run in parallel'],
        ['Read-out', 'conversion of node representations into one patient vector (here: the GTV node)'],
        ['Self-loop', 'edge from a node to itself, added automatically by the GAT layer'],
        ['Dropout', 'random removal of connections or values during training to prevent overfitting'],
        ['Layer normalisation', 'rescaling of node representations within each graph'],
        ['Epoch', 'one pass through all training patients'],
        ['AUC', 'probability that an event patient is ranked above a non-event patient'],
        ['Kuncheva index', 'agreement between two feature subsets, corrected for chance (1 identical, 0 chance)'],
        ['Graph-FS', 'feature selection on a graph of feature correlations'],
        ['Supervoxel', 'compact, homogeneous 3-D region of the peritumoral CT produced by SLIC-zero'],
    ], [4.0, 13.0], 'Glossary.'),
    ('h2', 'Key equations'),
    ('eq', 'Attention:   α_ij = softmax_j ( LeakyReLU( aᵀ [ W h_i ‖ W h_j ] ) )'),
    ('eq', 'Update:   h′_i = ‖_{m=1..4}  Σ_{j ∈ N(i) ∪ {i}}  α_ij^m W^m h_j'),
    ('eq', 'Prediction:   P(event) = σ( wᵀ [ h_GTV ‖ c ] + b )'),
]

if __name__ == '__main__':
    d = Doc('Graph Attention Networks in Radiomics',
            'Concepts, implementation and frequently asked questions for the radiomic graph attention network',
            'Graph attention networks explained',
            'Supersedes GAT_Questions_and_Answers (May 2026). Describes the current implementation (radgraph/models/gat.py).')
    d.contents(H)
    d.add(blocks)
    print(d.save('02_Graph_Attention_Networks_Explained_v2'))
