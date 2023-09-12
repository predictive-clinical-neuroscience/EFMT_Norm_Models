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


root_dir = '/project_cephfs/3022017.02/projects/hansav/Run8_f/'
data_dir = os.path.join(root_dir,'data/')
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
ex_nii = os.path.join(data_dir, 'faces_shapes_AOMIC_4D.nii.gz')

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')


EV = load_nifti("/project_cephfs/3022017.02/projects/hansav/Run8_f/vox/EV_ref.nii.gz")
Kurtosis = load_nifti("/project_cephfs/3022017.02/projects/hansav/Run8_f/vox/kurtosis_ref.nii.gz")
Skew = load_nifti("/project_cephfs/3022017.02/projects/hansav/Run8_f/vox/skew_ref.nii.gz")
SMSE =load_nifti("/project_cephfs/3022017.02/projects/hansav/Run8_f/vox/SMSE_ref.nii.gz")


#################################################################################
#FIG 1A: EV histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(EV)+0.005)

ax.hist(EV, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('Explained Variance')
plt.ylabel('Number of models (voxels)')
plt.axis([min(EV),max_x, 0, 30000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# inset axes....
axins = ax.inset_axes([0.22, 0.3, 0.75, 0.47])
axins.hist(EV, bins = 100, ec = 'white', range = [0.1, max_x],  lw = 0.2, fc = '#9D9D9D') 
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
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_f/Figures/EV_histogram.png', dpi=300)


#################################################################################
#FIG 1B: Skew histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(Skew)+0.005)
min_x = (min(Skew)-0.005)

ax.hist(Skew, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('Skew')
plt.ylabel('Number of models (voxels)')
plt.axis([min_x,max_x, 0,30000])
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_f/Figures/Skew_histogram.png', dpi=300)

#################################################################################

#FIG 1C: Kurtosis histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(Kurtosis)+0.005)

ax.hist(Kurtosis, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('Kurtosis')
plt.ylabel('Number of models (voxels)')
plt.axis([min(Kurtosis),max(Kurtosis), 0, 175000])
ax.yaxis.set_major_locator(ticker.MultipleLocator(25000))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# inset axes....
axins = ax.inset_axes([0.22, 0.1, 0.75, 0.47])
axins.hist(Kurtosis, bins = 100, ec = 'white', range = [5, 100],  lw = 0.2, fc = '#9D9D9D') 
axins.axis([5,50, 0, 450])
axins.xaxis.set_major_locator(ticker.MultipleLocator(10))
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
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_f/Figures/Kurtosis_histogram.png', dpi=300)



#################################################################################

#FIG 1C: SMSE histogram
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(SMSE)+0.005)

ax.hist(SMSE, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('SMSE')
plt.ylabel('Number of models (voxels)')
plt.axis([min(SMSE),max(SMSE), 0, 20000])
#ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# inset axes....
#axins = ax.inset_axes([0.22, 0.1, 0.75, 0.47])
#axins.hist(SMSE, bins = 100, ec = 'white', range = [5, 100],  lw = 0.2, fc = '#9D9D9D') 
#axins.axis([5,50, 0, 450])
#axins.xaxis.set_major_locator(ticker.MultipleLocator(10))
#axins.spines['top'].set_visible(False)
#axins.spines['right'].set_visible(False)
# sub region of the original image
#x1, x2, y1, y2 = 0.1, 0.5, 0, 2500
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#axins.set_xticklabels()
#axins.set_yticklabels()

#ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run8_f/Figures/SMSE_histogram.png', dpi=300)
