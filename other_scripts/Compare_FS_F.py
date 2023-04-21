#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import pcntoolkit as ptk 
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from pcntoolkit.normative import predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, calibration_descriptives

from matplotlib import pyplot as plt
import seaborn as sns
import pingouin as pg

from scipy import stats
from scipy.stats import mannwhitneyu

import os
import numpy as np
from pcntoolkit.util.utils import calibration_descriptives
from pcntoolkit.dataio.fileio import load_nifti, save_nifti
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
#%% COMPARE DEVIATION COUNTS: 
##################
##LOAD DATA AND GET MEAN & STD FOR EACH CONDITION
##################
counts_df_faces = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run7_f/vox/NPM/counts_hist.csv')
counts_df_faces_shapes = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run7_fs/vox/NPM/counts_hist.csv')

reference_counts_F = counts_df_faces['reference_counts']
reference_counts_FS = counts_df_faces_shapes['reference_counts']
clinical_counts_F = counts_df_faces['clinical_counts']
clinical_counts_F = clinical_counts_F.dropna()
clinical_counts_FS = counts_df_faces_shapes['clinical_counts']
clinical_counts_FS = clinical_counts_FS.dropna()


print('REFERENCE: ')
print('Faces>Shapes: Mean: ', reference_counts_FS.mean(), 'STD: ', reference_counts_FS.std())
print('Faces>Baseline: Mean: ', reference_counts_F.mean(), 'STD: ', reference_counts_F.std(), '\n')
print('CLINICAL: ')
print('Faces>Shapes: Mean: ', clinical_counts_FS.mean(), 'STD: ', clinical_counts_FS.std())
print('Faces>Baseline: Mean: ', clinical_counts_F.mean(), 'STD: ', clinical_counts_F.std(), '\n')

##################
#COMPARE THE NUMBER OF DEVAITIONS BETWEEN CONTRASTS: 
##################
##CONTROLS:
stat, p_value = mannwhitneyu(reference_counts_FS, reference_counts_F)
print("Faces > Shapes Reference vs. Faces > Baseline Reference: \n Mann Whitney U Test: statistic= ", stat ,", p-value = ", p_value, '\n')

##PATIENTS:
stat, p_value = mannwhitneyu(clinical_counts_FS, clinical_counts_F)
print("Faces > Shapes Clinical vs. Faces > Baseline Clincial: \n Mann Whitney U Test: statistic= ", stat ,", p-value = ", p_value, '\n')

##################
#COMPARE THE NUMBER OF DEVAITIONS BETWEEN GROUPS - Faces>Shapes: 
##################
stat, p_value = mannwhitneyu(reference_counts_FS, clinical_counts_FS)
print("Faces > Shapes: Ref vs. Clincial: \n Mann Whitney U Test: statistic= ", stat ,", p-value = ", p_value, '\n')
    
##################
#COMPARE THE NUMBER OF DEVAITIONS BETWEEN GROUPS - Faces>Baseline: 
##################
stat, p_value = mannwhitneyu(reference_counts_F, clinical_counts_F)
print("Faces > Baseline: Ref vs. Clincial: \n Mann Whitney U Test: statistic= ", stat ,", p-value = ", p_value, '\n')


#%% COMPARE SCCA RESULTS:
#FACTORS:
scca_df_faces = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run7_f/vox/NPM/SCCA/Faces_scca_factors.csv')
scca_df_faces_shapes = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run7_fs/vox/NPM/SCCA/Faces_shapes_scca_factors.csv')
print('FACTORS:')
print('SCCA - Train: ')
print('Faces>Shapes: Mean: ', scca_df_faces_shapes['Train'].mean(), 'STD: ', scca_df_faces_shapes['Train'].std())
print('Faces>Baseline: Mean: ', scca_df_faces['Train'].mean(), 'STD: ', scca_df_faces['Train'].std(), '\n')

print('SCCA - Test: ')
print('Faces>Shapes: Mean: ', scca_df_faces_shapes['Test'].mean(), 'STD: ', scca_df_faces_shapes['Test'].std())
print('Faces>Baseline: Mean: ', scca_df_faces['Test'].mean(), 'STD: ', scca_df_faces['Test'].std(), '\n')

stat, p_value = mannwhitneyu(scca_df_faces_shapes['Test'], scca_df_faces['Test'])
print("Faces > Shapes SCCA vs. Faces > Baseline SCCA: \n Mann Whitney U Test: statistic= ", stat ,", p-value = ", p_value, '\n')


scca_df_faces = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run7_f/vox/NPM/SCCA/Faces_scca_diagnosis_flipped.csv')
scca_df_faces_shapes = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run7_fs/vox/NPM/SCCA/Faces_shapes_scca_diagnosis_flipped.csv')
print('DIAGNOSIS:')
print('SCCA - Train: ')
print('Faces>Shapes: Mean: ', scca_df_faces_shapes['Train'].mean(), 'STD: ', scca_df_faces_shapes['Train'].std())
print('Faces>Baseline: Mean: ', scca_df_faces['Train'].mean(), 'STD: ', scca_df_faces['Train'].std(), '\n')

print('SCCA - Test: ')
print('Faces>Shapes: Mean: ', scca_df_faces_shapes['Test'].mean(), 'STD: ', scca_df_faces_shapes['Test'].std())
print('Faces>Baseline: Mean: ', scca_df_faces['Test'].mean(), 'STD: ', scca_df_faces['Test'].std(), '\n')

stat, p_value = mannwhitneyu(scca_df_faces_shapes['Test'], scca_df_faces['Test'])
print("Faces > Shapes SCCA vs. Faces > Baseline SCCA: \n Mann Whitney U Test: statistic= ", stat ,", p-value = ", p_value, '\n')



