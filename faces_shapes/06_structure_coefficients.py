
import os
import pickle
import pandas as pd
import numpy as np
import nibabel as nib
import pcntoolkit as ptk 
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from pcntoolkit.dataio.fileio import save_nifti, load_nifti
from pcntoolkit.normative import predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, calibration_descriptives

from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns
import pingouin as pg

#%%
#Define:
def compute_pearsonr(A, B):
    """ Manually computes the Pearson correlation between two matrices.
        Basic usage::
            compute_pearsonr(A, B)
        :param A: an N * M data array
        :param cov: an N * M array
        :returns Rho: N dimensional vector of correlation coefficients
        :returns ys2: N dimensional vector of p-values
        Notes::
            This function is useful when M is large and only the diagonal entries
            of the resulting correlation matrix are of interest. This function
            does not compute the full correlation matrix as an intermediate step"""

    # N = A.shape[1]
    N = A.shape[0]

    # first mean centre
    Am = A - np.mean(A, axis=0)
    Bm = B - np.mean(B, axis=0)
    # then normalize
    An = Am / np.sqrt(np.sum(Am**2, axis=0))
    Bn = Bm / np.sqrt(np.sum(Bm**2, axis=0))
    del(Am, Bm)

    Rho = np.sum(An * Bn, axis=0)
    del(An, Bn)

    # Fisher r-to-z
    Zr = (np.arctanh(Rho) - np.arctanh(0)) * np.sqrt(N - 3)
    N = stats.norm()
    pRho = 2*N.cdf(-np.abs(Zr))
    # pRho = 1-N.cdf(Zr)
    
    return Rho, pRho

 #%%  
# globals
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run7_fs/'
data_dir = os.path.join(root_dir)
w_dir = os.path.join(root_dir,'vox/')
out_dir = os.path.join(root_dir,'vox/structure_coefficients/')
#%%
# load covariatea
print('loading covariate data ...')
df_dem = pd.read_csv(os.path.join(data_dir,'metadata_te.csv'))
df_dem = df_dem = df_dem.loc[(df_dem['dataset'] == 'AOMIC') | \
                        (df_dem['dataset'] == 'HCP_S1200') | \
                        (df_dem['dataset'] == 'HCP_Dev') | \
                        (df_dem['dataset'] == 'UKBiobank') | \
                        (df_dem['dataset'] == 'DNS') | \
                        (df_dem['dataset'] == 'MIND_Set')]
#load the pkl file
mask_nii = os.path.join('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
yhat_est = ptkload(os.path.join(w_dir,'yhat_estimate.pkl'), mask=mask_nii)
yhat_est_transf = np.transpose(yhat_est)

cols_cov = ['site', 'age','sex','TR','MB_F', 'volumes','task_length_s','target_blocks', 'instructions', 'target_stimuli', 'ICV']

#%%
#For each covariate of interest [n x 1]
for column in cols_cov:
    curr_cov = df_dem[column]
    curr_cov = curr_cov.astype(int)
    print(curr_cov)
    
    covariate_Rho = []
    covariate_pRho = []
    coviariate_Rho_Sqrd = []
    
    covariate_Rho_array = np.empty(0)
    covariate_pRho_array = np.empty(0)
    coviariate_Rho_Sqrd_array = np.empty(0)    
    
    #For each voxel of the brain [n x 1]
    for voxel in range(0, yhat_est.shape[1]):
        #Correlate the covariate with the predicted activity in that voxel [1 x 1 - with n data points]
        correlation = compute_pearsonr(curr_cov, yhat_est[:,voxel])
        #Accumulate the correlations to form one whole brain vector of 
        # vector of correlation coefficients
        covariate_Rho.append(correlation[0]) 
        coviariate_Rho_Sqrd.append(correlation[0]*correlation[0]) 
        #vector of p-values
        covariate_pRho.append(correlation[1]) 
        
    #Convert the lists to arrays
    covariate_Rho_array = np.array(covariate_Rho)
    coviariate_Rho_Sqrd_array = np.array(coviariate_Rho_Sqrd)
    covariate_pRho_array = np.array(covariate_pRho)
      
    #Build the filename for each output file    
    filename_Rho = (w_dir +'Rho_' +column +'.nii.gz')
    filename_Rho_Sqrd = (w_dir +'Rho_Sqrd_' +column +'.nii.gz')  
    filename_pRho = (w_dir +'pRho_' +column +'.nii.gz')
       
    #Save each output file as a nii
    save_nifti(covariate_Rho_array, filename_Rho, mask = mask_nii, examplenii = mask_nii, dtype='float32')
    save_nifti(coviariate_Rho_Sqrd_array, filename_Rho_Sqrd, mask = mask_nii, examplenii = mask_nii, dtype='float32')
    save_nifti(covariate_pRho_array, filename_pRho, mask = mask_nii, examplenii = mask_nii, dtype='float32')

#%% Threshold R2 <0.3   

for column in cols_cov:
    curr_cov = df_dem[column]
    print(column)
    
    in_filename = (out_dir +'Rho_Sqrd_' +column +'.nii.gz')  
    
    out_filename = (out_dir +'Rho_Sqrd_' +column +'_gt0pt3.nii.gz')

    #print(out_filename_pos, out_filename_neg)
    
    command = ('fslmaths ' +in_filename +' -thr 0.3 ' +out_filename)
    print(command)
    os.system(command)
    !command
    
    
#%% Mask Rho with regions only greater than R2 <0.3 
for column in cols_cov:
    curr_cov = df_dem[column]
    print(column)
 
    in_filename = (out_dir +'Rho_' +column +'.nii.gz')  
    out_filename = (out_dir +'Rho_' +column +'_thr_r2_03.nii.gz')
    mask_filename = (out_dir +'Rho_Sqrd_' +column +'_gt0pt3.nii.gz')
    
    command = ('fslmaths ' +in_filename +' -mas ' +mask_filename +' ' +out_filename)
    print(command)
    os.system(command)
    !command

    
    