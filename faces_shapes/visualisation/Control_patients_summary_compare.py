#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:24:09 2023

@author: hansav
"""

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
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

##################
# global variables
##################
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run8_fs/'
data_dir = os.path.join(root_dir,'data/')
mask_nii = os.path.join('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')

# Load in the Z_est files
Z_est_control_test = ptkload(os.path.join(w_dir,'Z_estimate.pkl'), mask=mask_nii)
Z_est_clinical = ptkload(os.path.join(w_dir,'Z_predcl.pkl'), mask=mask_nii)

#Load in the diagnosis information
metadata_cl_diagnosis = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/data/MIND_Set_diagnoses.csv')


##################
#COMPUTE AND SAVE COUNTS
##################

mask_Diganosis = metadata_cl_diagnosis['diagnosis'].eq(1)
mask_MINDSet_control = metadata_cl_diagnosis['diagnosis'].eq(0)
Z_est_clinical_only = Z_est_clinical[mask_Diganosis]
Z_est_MINDSet_control = Z_est_clinical[mask_MINDSet_control]

reference_counts = []
clinical_counts = []
control_counts = []

for x in range(0,len(Z_est_control_test)):
    reference_counts.append(sum((Z_est_control_test[x] < -2.6) | (Z_est_control_test[x]  > 2.6)))

for y in range(0,len(Z_est_clinical_only)):
    clinical_counts.append(sum((Z_est_clinical_only[y] < -2.6) | (Z_est_clinical_only[y]  > 2.6)))

for z in range(0,len(Z_est_MINDSet_control)):
    control_counts.append(sum((Z_est_MINDSet_control[z] < -2.6) | (Z_est_MINDSet_control[z]  > 2.6)))

reference_counts_df = pd.DataFrame(reference_counts)
control_counts_df = pd.DataFrame(control_counts)
clinical_counts_df = pd.DataFrame(clinical_counts)

#Save as a dataframe -> csv to be able to easily re-run in the future if necessary.
counts_df = pd.concat([reference_counts_df, control_counts_df, clinical_counts_df], ignore_index=True, axis =1 )
counts_df = counts_df.rename(columns = {0:'reference_counts', 1:'control_counts', 2:'clinical_counts'})

counts_df.to_csv(os.path.join('/project_cephfs/3022017.02/projects/hansav/Run8_fs/vox/NPM/counts_hist.csv'),index=False)      

#This means once you have saved the file you can simply load in the csv in future - saves lots of time. 

##################
#LOAD IN COUNTS
##################
counts_df = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/vox/NPM/counts_hist.csv')

counts_df_ref = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/vox/NPM/counts_hist.csv')['reference_counts']
counts_df_cont = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/vox/NPM/counts_hist.csv')['control_counts']
counts_df_clin = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/vox/NPM/counts_hist.csv')['clinical_counts']


reference_counts_df = pd.DataFrame(counts_df_ref)
control_counts_df = pd.DataFrame(counts_df_cont)
clinical_counts_df = pd.DataFrame(counts_df_clin)

##################
#HISTOGRAM PLOTS
##################

bins = np.histogram_bin_edges(counts_df['reference_counts'], bins='auto')

sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()
sns.histplot(data=counts_df, x='reference_counts', color="#9D9D9D", label="Reference Cohort", kde=True,  stat="percent", alpha = 0.5, bins=bins)
sns.histplot(data=counts_df, x='clinical_counts', color="#AA3377", label="Patients", kde=True, stat="percent", alpha = 0.3, bins=bins)
#sns.histplot(data=counts_df, x='control_counts', color="#9D9D9D", label="MIND Set Controls", kde=True,  stat="percent")
sns.despine()
plt.xlim(0,20000)
plt.ylim(0,30)
plt.xlabel('Number of deviations ± 2.6')
plt.ylabel('Percentage of sample')
#plt.legend(frameon=False) 
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_fs/Figures/deviation_count_control_clinical.png', dpi=300)



##################
#BOX PLOTS
##################
boxplt_palette = ['#9D9D9D', '#AA3377']

plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()
sns.set_style("white")
sns.boxplot(data = counts_df[['reference_counts', 'clinical_counts']], showfliers = False, palette=boxplt_palette)
sns.stripplot(data = counts_df[['reference_counts', 'clinical_counts']], marker = "o", size = 2, alpha=0.3, color="black")
plt.xticks([0,1], ['Reference', 'MIND-Set\n Patients'])
sns.despine()
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_fs/Figures/deviation_count_control_clincial_boxplot_scat.png', dpi=300)


plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()
sns.set_style("white")
sns.boxplot(data = counts_df[['reference_counts', 'clinical_counts']], showfliers = False, palette=boxplt_palette)
plt.xticks([0,1], ['Reference', 'MIND-Set\n Patients'])
sns.despine(top=True, right=True, left=False, bottom=True)
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_fs/Figures/deviation_count_control_clincial_boxplot.png', dpi=300)
