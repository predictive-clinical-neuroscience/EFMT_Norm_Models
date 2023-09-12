 os
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
# globals
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run8_f/'
data_dir = os.path.join(root_dir)
z_dir = os.path.join(root_dir,'vox/')
w_dir = os.path.join(root_dir,'vox/NPM/')

##### load the metadata_te.csv
metadata = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_f/metadata_te.csv')
#print(metadata)

#%% GET THE INDEX OF EACH SITE
##FOR each site 
isAOMIC = metadata["dataset"].eq("AOMIC")
mask1 = isAOMIC & (~(isAOMIC.shift() & isAOMIC.shift(-1)) )
AOMIC_start_end = list(isAOMIC.index[mask1])
print("AOMIC indexes from: " +str(AOMIC_start_end))

isHCP_S1200 = metadata["dataset"].eq("HCP_S1200")
mask1 = isHCP_S1200 & (~(isHCP_S1200.shift() & isHCP_S1200.shift(-1)) )
HCP_S1200_start_end = list(isHCP_S1200.index[mask1])
print("HCP_S1200 indexes from: " +str(HCP_S1200_start_end))

isHCP_Dev = metadata["dataset"].eq("HCP_Dev")
mask1 = isHCP_Dev & (~(isHCP_Dev.shift() & isHCP_Dev.shift(-1)) )
HCP_Dev_start_end = list(isHCP_Dev.index[mask1])
print("HCP_Dev indexes from: " +str(HCP_Dev_start_end))

isUKB = metadata["dataset"].eq("UKBiobank")
mask1 = isUKB & (~(isUKB.shift() & isUKB.shift(-1)) )
UKB_start_end = list(isUKB.index[mask1])
print("DNS indexes from: " +str(UKB_start_end))

isDNS = metadata["dataset"].eq("DNS")
mask1 = isDNS & (~(isDNS.shift() & isDNS.shift(-1)) )
DNS_start_end = list(isDNS.index[mask1])
print("DNS indexes from: " +str(DNS_start_end))

isMIND_Set = metadata["dataset"].eq("MIND_Set")
mask1 = isMIND_Set & (~(isMIND_Set.shift() & isMIND_Set.shift(-1)) )
MIND_Set_start_end = list(isMIND_Set.index[mask1])
print("MIND_Set indexes from: " +str(MIND_Set_start_end))

#%% LOAD THE Z-EST FILE AND THE MASK (MNI 2MM)

mask_nii = os.path.join('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
Z_est = ptkload(os.path.join(z_dir,'Z_estimate.pkl'), mask=mask_nii)

#Example nii file:
examplenii_cope = '/project_cephfs/3022017.02/HCP_S1200/211114/MNINonLinear/Results/tfMRI_EMOTION/tfMRI_EMOTION_hp200_s4_level2vol.feat/cope1.feat/stats/cope1.nii.gz'
#%%THESE SPLIT EACH SITE'S Z-EST TO NII FILES

#AOMIC_Site - 1 nii
Z_est_AOMIC = Z_est[AOMIC_start_end[0] : (AOMIC_start_end[1] + 1),:] # +1 to include the last index
print(np.shape(Z_est_AOMIC))
print('There are ' +str(len(Z_est_AOMIC)) +' controls in this split')
part_trans = np.transpose(Z_est_AOMIC)
filename_part = (w_dir +'Z_est_AOMIC.nii.gz')
save_nifti(part_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')

#HCP_1200 -2 nii
Z_est_HCP_1200 = Z_est[HCP_S1200_start_end[0] : (HCP_S1200_start_end[1] + 1),:] # +1 to include the last index
Z_est_HCP_1200_split = np.array_split(Z_est_HCP_1200, 2) 
n = 1 

for part in Z_est_HCP_1200_split:
    print('There are ' +str(len(part)) +' controls in this split')
    print(np.shape(part))
    part_trans = np.transpose(part)
    filename_part = (w_dir +'Z_est_HCP_1200_' +str(n) +'.nii.gz')
    save_nifti(part_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')
    n = n +1

#HCP Dev - 1 nii
Z_est_HCP_Dev = Z_est[HCP_Dev_start_end[0] : (HCP_Dev_start_end[1] + 1),:] # +1 to include the last index
#Z_est_HCP_Dev_split = np.array_split(Z_est_HCP_Dev, 2) 
part_trans = np.transpose(Z_est_HCP_Dev)
filename_part = (w_dir +'Z_est_HCP_Dev.nii.gz')
save_nifti(part_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')

#UKBiobank - 5 nii
Z_est_UKB = Z_est[UKB_start_end[0] : (UKB_start_end[1] + 1),:] # +1 to include the last index
Z_est_UKB_split = np.array_split(Z_est_UKB, 5) 
n = 1 

for part in Z_est_UKB_split:
    print('There are ' +str(len(part)) +' controls in this split')
    print(np.shape(part))
    part_trans = np.transpose(part)
    filename_part = (w_dir +'Z_est_UKB_' +str(n) +'.nii.gz')
    save_nifti(part_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')
    n = n +1

#DNS - 2 nii
Z_est_DNS = Z_est[DNS_start_end[0] : (DNS_start_end[1] + 1), :] # +1 to include the last index
Z_est_DNS_split = np.array_split(Z_est_DNS, 2) 
n = 1 

for part in Z_est_DNS_split:
    print('There are ' +str(len(part)) +' controls in this split')
    print(np.shape(part))
    part_trans = np.transpose(part)
    filename_part = (w_dir +'Z_est_DNS_' +str(n) +'.nii.gz')
    save_nifti(part_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')
    n = n +1

#MIND_Set - 1 nii
Z_est_MIND_Set = Z_est[MIND_Set_start_end[0] : (MIND_Set_start_end[1] + 1), :] # +1 to include the last index
print('There are ' +str(len(Z_est_MIND_Set)) +' controls in this site')
part_trans = np.transpose(Z_est_MIND_Set)
filename_part = (w_dir +'Z_est_MIND_Set.nii.gz')
save_nifti(part_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')
