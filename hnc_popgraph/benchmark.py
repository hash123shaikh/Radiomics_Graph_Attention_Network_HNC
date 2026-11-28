"""External test AUCs of the published CNN (Mateus et al., Sci Rep 13, 18176, 2023): Table 3 (imaging)
and Table 4 (imaging and clinical data). CV: mean of the per-fold MAASTRO AUCs; cohort split: MAASTRO AUC."""

CNN_AUC = {('MATEUS_CV', 'imaging'): {'DM': 0.83, 'LR': 0.53, 'OS': 0.63},
           ('MATEUS_CV', 'imaging+clinical'): {'DM': 0.88, 'LR': 0.57, 'OS': 0.68},
           ('MATEUS_COHORT', 'imaging'): {'DM': 0.89, 'LR': 0.45, 'OS': 0.67},
           ('MATEUS_COHORT', 'imaging+clinical'): {'DM': 0.93, 'LR': 0.59, 'OS': 0.69}}
