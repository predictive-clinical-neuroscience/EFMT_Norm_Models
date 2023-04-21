import os
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pingouin as pg

from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, calibration_descriptives
from pcntoolkit.dataio.fileio import load as ptkload
import matplotlib.ticker as ticker

# globals

root_dir = '/project_cephfs/3022017.02/projects/hansav/Run7_f/'
data_dir = os.path.join(root_dir,'data/')
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
ex_nii = os.path.join(data_dir, 'faces_shapes_AOMIC_4D.nii.gz')

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')


# load covariates
print('loading covariate data ...')
df_all = pd.read_csv(os.path.join(data_dir,'control_metadata.csv'))
df_all = df_all.loc[(df_all['dataset'] == 'AOMIC') | \
                    (df_all['dataset'] == 'HCP_S1200') | \
                    (df_all['dataset'] == 'UKBiobank') | \
                    (df_all['dataset'] == 'HCP_Dev') | \
                    (df_all['dataset'] == 'DNS') | \
                    (df_all['dataset'] == 'MIND_Set')]
    
df_all_sex_split = df_all.groupby(df_all['sex'])
df_all_M = df_all_sex_split.get_group(0)
df_all_F = df_all_sex_split.get_group(1)
del(df_all_sex_split)
    
# load covariates
print('loading train split ...')
df_train = pd.read_csv(os.path.join(root_dir, 'metadata_tr.csv'))
df_train = df_train.loc[(df_train['dataset'] == 'AOMIC') | \
                    (df_train['dataset'] == 'HCP_S1200') | \
                    (df_train['dataset'] == 'UKBiobank') | \
                    (df_train['dataset'] == 'HCP_Dev') | \
                    (df_train['dataset'] == 'DNS') | \
                    (df_train['dataset'] == 'MIND_Set')]
    
df_train_sex_split = df_train.groupby(df_all['sex'])
df_train_M = df_train_sex_split.get_group(0)
df_train_F = df_train_sex_split.get_group(1)
del(df_train_sex_split)
        
      
# load covariates
print('loading test split ...')
df_test = pd.read_csv(os.path.join(root_dir, 'metadata_te.csv'))
df_test = df_test.loc[(df_test['dataset'] == 'AOMIC') | \
                    (df_test['dataset'] == 'HCP_S1200') | \
                    (df_test['dataset'] == 'UKBiobank') | \
                    (df_test['dataset'] == 'HCP_Dev') | \
                    (df_test['dataset'] == 'DNS') | \
                    (df_test['dataset'] == 'MIND_Set')]
df_test_sex_split = df_test.groupby(df_all['sex'])
df_test_M = df_test_sex_split.get_group(0)
df_test_F = df_test_sex_split.get_group(1)
del(df_test_sex_split)

#%% ALL SAMPLE AGESxSEX:    

color_coding = ['#4e79a7','#f28e2b','#76b7b2','#e15759','#59a14f','#b07aa1']
                #['#AOMIC_PIOP2','#HCP_S1200','UKBiobank','#HCP_Dev','#HCP_Dev','#MIND_Set']

bins = np.histogram_bin_edges(df_all_M['age'], bins='auto')

#ALL_PARTICIPANTS ON PYRAMID STYLE PLOT
plt.style.use('seaborn-white')
fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
#fig.suptitle('Age \n Full sample')
#Invert the x axis of
axs[0].invert_xaxis()
axs[0].set_xlim(1000, 0)
axs[0].invert_yaxis()
axs[0].set_ylim((85,5))
axs[0].set_yticklabels([])
axs[0].set(xlabel='Number of Males')
axs[0].spines['top'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].hist((df_all_M.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white', lw = 0.2, orientation = 'horizontal')#, ec='black')

axs[1].set_xlim(0,1000)
axs[1].hist((df_all_F.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white',  lw = 0.2, orientation = 'horizontal')#, ec='black')
axs[1].set(xlabel='Number of Females')
axs[1].set(ylabel='Age')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.ylim((85,5))
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run7_f/Figures/Age_sex_all_histogram.png', dpi=300)  

#%% TRAIN AGE SEX: 
    
color_coding = ['#4e79a7','#f28e2b','#76b7b2','#e15759','#59a14f','#b07aa1']
                #['#AOMIC_PIOP2','#HCP_S1200','UKBiobank','#HCP_Dev','#HCP_Dev','#MIND_Set']

bins = np.histogram_bin_edges(df_train_M['age'], bins='auto')


#ALL_PARTICIPANTS ON PYRAMID STYLE PLOT
plt.style.use('seaborn-white')
fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
#fig.suptitle('Age \n Full sample')
#Invert the x axis of
axs[0].invert_xaxis()
axs[0].set_xlim(400, 0)
axs[0].invert_yaxis()
axs[0].set_ylim((85,5))
axs[0].set_yticklabels([])
axs[0].set(xlabel='Number of Males')
axs[0].spines['top'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].hist((df_train_M.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white', lw = 0.2, orientation = 'horizontal')#, ec='black')

axs[1].set_xlim(0, 400)
axs[1].hist((df_train_F.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white',  lw = 0.2, orientation = 'horizontal')#, ec='black')
axs[1].set(xlabel='Number of Females')
axs[1].set(ylabel='Age')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.ylim((85,5))
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run7_f/Figures/Age_sex_train_histogram.png', dpi=300)  

#%% TEST AGE SEX: 
    
color_coding = ['#4e79a7','#f28e2b','#76b7b2','#e15759','#59a14f','#b07aa1']
                #['#AOMIC_PIOP2','#HCP_S1200','UKBiobank','#HCP_Dev','#HCP_Dev','#MIND_Set']


bins = np.histogram_bin_edges(df_train_M['age'], bins='auto')

#ALL_PARTICIPANTS ON PYRAMID STYLE PLOT
plt.style.use('seaborn-white')
fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
#fig.suptitle('Age \n Full sample')
#Invert the x axis of
axs[0].invert_xaxis()
axs[0].set_xlim(400, 0)
axs[0].invert_yaxis()
axs[0].set_ylim((85,5))
axs[0].set_yticklabels([])
axs[0].set(xlabel='Number of Males')
axs[0].spines['top'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].hist((df_test_M.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white', lw = 0.2, orientation = 'horizontal')#, ec='black')

axs[1].set_xlim(0, 400)
axs[1].hist((df_test_F.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white',  lw = 0.2, orientation = 'horizontal')#, ec='black')
axs[1].set(xlabel='Number of Females')
axs[1].set(ylabel='Age')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.ylim((85,5))
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run7_f/Figures/Age_sex_test_histogram.png', dpi=300)  



#%% CLINICAL SAMPLE: 
#load covariates
print('loading clnical sample ...')
df_clinical = pd.read_csv(os.path.join(root_dir, 'metadata_cl.csv'))
df_clinical = df_clinical.loc[(df_test['dataset'] == 'AOMIC') | \
                    (df_clinical['dataset'] == 'HCP_S1200') | \
                    (df_clinical['dataset'] == 'UKBiobank') | \
                    (df_clinical['dataset'] == 'HCP_Dev') | \
                    (df_clinical['dataset'] == 'DNS') | \
                    (df_clinical['dataset'] == 'MIND_Set')]
df_clinical_sex_split = df_clinical.groupby(df_all['sex'])
df_clinical_M = df_clinical_sex_split.get_group(0)
df_clinical_F = df_clinical_sex_split.get_group(1)
del(df_clinical_sex_split)

#%% CLINICAL AGE SEX: 
    
color_coding = ['#b07aa1']
                #['#AOMIC_PIOP2','#HCP_S1200','UKBiobank','#HCP_Dev','#HCP_Dev','#MIND_Set']

bins = np.histogram_bin_edges(df_clinical_M['age'], bins='auto')

#ALL_PARTICIPANTS ON PYRAMID STYLE PLOT
plt.style.use('seaborn-white')
fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
#fig.suptitle('Age \n Full sample')
#Invert the x axis of
axs[0].invert_xaxis()
axs[0].set_xlim(40, 0)
axs[0].invert_yaxis()
axs[0].set_ylim((85,5))
axs[0].set_yticklabels([])
axs[0].set(xlabel='Number of Males')
axs[0].spines['top'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].hist((df_clinical_M.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white', lw = 0.2, orientation = 'horizontal')#, ec='black')

axs[1].set_xlim(0,40)
axs[1].hist((df_clinical_F.pivot(columns='site', values='age')), bins = bins, stacked = True, color=color_coding, ec = 'white',  lw = 0.2, orientation = 'horizontal')#, ec='black')
axs[1].set(xlabel='Number of Females')
axs[1].set(ylabel='Age')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.ylim((85,5))
plt.show()
fig.savefig('/project_cephfs/3022017.02/projects/hansav/Run7_f/Figures/Age_sex_clinical_histogram.png', dpi=300)  

   