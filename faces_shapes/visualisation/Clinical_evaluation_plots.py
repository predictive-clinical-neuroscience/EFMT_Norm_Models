#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:56:31 2023

@author: hansav
"""

import os
import numpy as np
from pcntoolkit.util.utils import calibration_descriptives
from pcntoolkit.dataio.fileio import load_nifti, save_nifti
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


root_dir = '/project_cephfs/3022017.06/projects/hansav/Run8_fs/'
data_dir = os.path.join(root_dir,'data/')
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
ex_nii = os.path.join(data_dir, 'faces_AOMIC_4D.nii.gz')

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')


EV = load_nifti("/project_cephfs/3022017.06/projects/hansav/Run8_fs/vox/EV_predcl.nii.gz")
Kurtosis = load_nifti("/project_cephfs/3022017.06/projects/hansav/Run8_fs/vox/kurtosis_predcl.nii.gz")
Skew = load_nifti("/project_cephfs/3022017.06/projects/hansav/Run8_fs/vox/skew_predcl.nii.gz")
SMSE = load_nifti('/project_cephfs/3022017.06/projects/hansav/Run8_fs/vox/SMSE_predcl.nii.gz')

#################################################################################
#FIG 1A: EV histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(EV)+0.005)
min_x = (min(EV)-0.005)

ax.hist(EV, bins = 100, ec = 'white', lw=0.2, fc = '#AA3377') 
plt.xlabel('Explained Variance')
plt.ylabel('Number of models (voxels)')
plt.axis([min_x,max_x, 0, 70000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
fig.savefig('/project_cephfs/3022017.06/projects/hansav/Run8_fs/Figures/EV_histogram_clinical.png', dpi=300)


# inset axes....
axins = ax.inset_axes([0.12, 0.1, 0.55, 0.47])
axins.hist(EV, bins = 100, ec = 'white', range = [min_x, -0.02],  lw = 0.2, fc = '#AA3377') 
#axins.yaxis.set_major_locator(ticker.MultipleLocator(1))
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
# sub region of the original image
#x1, x2, y1, y2 = 0.1, 0.5, 0, 2500
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#axins.set_xticklabels()
#axins.set_yticklabels()

#ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
fig.savefig('/project_cephfs/3022017.06/projects/hansav/Run8_fs/Figures/EV_histogram_clinical.png', dpi=300)


#################################################################################
#FIG 1B: Skew histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(Skew)+0.005)
min_x = (min(Skew)-0.005)

ax.hist(Skew, bins = 100, ec = 'white', lw=0.2, fc = '#AA3377') 
plt.xlabel('Skew')
plt.ylabel('Number of models (voxels)')
plt.axis([min_x,max_x, 0, 120000])
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
fig.savefig('/project_cephfs/3022017.06/projects/hansav/Run8_fs/Figures/Skew_histogram_clinical.png', dpi=300)

#################################################################################

#FIG 1C: Kurtosis histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(Kurtosis)+0.005)
min_x = (max(Kurtosis)+0.005)

ax.hist(Kurtosis, bins = 100, ec = 'white', lw=0.2, fc = '#AA3377') 
plt.xlabel('Kurtosis')
plt.ylabel('Number of models (voxels)')
plt.axis([min(Kurtosis),270, 0, 220000])
ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# inset axes....
axins = ax.inset_axes([0.22, 0.1, 0.75, 0.47])
axins.hist(Kurtosis, bins = 100, ec = 'white', range = [-5, 270],  lw = 0.2, fc = '#AA3377') 
axins.axis([10,225, 0, 200])
axins.xaxis.set_major_locator(ticker.MultipleLocator(20))
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
# sub region of the original image
#x1, x2, y1, y2 = 0.1, 0.5, 0, 2500
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#axins.set_xticklabels()
#axins.set_yticklabels()

#ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
fig.savefig('/project_cephfs/3022017.06/projects/hansav/Run8_fs/Figures/Kurtosis_histogram_clinical.png', dpi=300)

#################################################################################

#FIG 1C: SMSE histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(SMSE)+0.005)

ax.hist(SMSE, bins = 100, ec = 'white', lw=0.2, fc = '#AA3377', range=[0,2]) 
plt.xlabel('SMSE')
plt.ylabel('Number of models (voxels)')
plt.axis([min(SMSE),2, 0, 60000])
#ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# inset axes....
#axins = ax.inset_axes([0.22, 0.1, 0.75, 0.47])
#axins.hist(SMSE, bins = 100, ec = 'white', range = [0.9, 2],  lw = 0.2, fc = '#AA3377') 
#axins.axis([0.5,2.5, 0, 25000])
#axins.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
#axins.spines['top'].set_visible(False)
#axins.spines['right'].set_visible(False)
# sub region of the original image
#x1, x2, y1, y2 = 0.0, 0.5, 0, 2500
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#axins.set_xticklabels()
#axins.set_yticklabels()

#ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
fig.savefig('/project_cephfs/3022017.06/projects/hansav/Run8_fs/Figures/SMSE_histogram_clinical.png', dpi=300)
