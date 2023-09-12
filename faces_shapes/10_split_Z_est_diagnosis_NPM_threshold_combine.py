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
# globals
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run8_fs/'
data_dir = os.path.join(root_dir)
z_dir = os.path.join(root_dir,'vox/')
w_dir = os.path.join(root_dir,'vox/NPM/Diagnoses/')

##### load the metadata.csv
metadata_cl_diagnosis = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/data/MIND_Set_diagnoses.csv')
#print(metadata_cl_diagnosis)

#%% LOAD THE Z-EST FILE AND THE MASK (MNI 2MM)

mask_nii = os.path.join('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
Z_est = ptkload(os.path.join(z_dir,'Z_predcl.pkl'), mask=mask_nii)

#Example nii file:
examplenii_cope = '/project_cephfs/3022017.02/HCP_S1200/211114/MNINonLinear/Results/tfMRI_EMOTION/tfMRI_EMOTION_hp200_s4_level2vol.feat/cope3.feat/stats/cope1.nii.gz'
#




#%% Get deviations for clinical sample - NO MIND-Set CONTROLS:
##### load the metadata.csv
metadata_all_clin_test = pd.read_csv('/project_cephfs/3022017.02/projects/hansav/Run8_fs/data/MIND_Set_metadata_control_split2_patients.csv')
#print(metadata_cl_diagnosis)

mask_Diganosis = metadata_all_clin_test['diagnosis'].eq(1)
print( sum(mask_Diganosis == True), col)
Z_est_Diganosis = Z_est[mask_Diganosis]
Z_est_Diganosis_trans = np.transpose(Z_est_Diganosis)
filename_part = (z_dir +'Z_est_MIND_Set_clinical.nii.gz')
save_nifti(Z_est_Diganosis_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')

z_dir = os.path.join(root_dir,'vox/NPM/')


##Threshold
in_filename = (z_dir +'Z_est_MIND_Set_clinical.nii.gz')
#print(filename)
out_filename_pos = (z_dir +'Z_est_MIND_Set_clinical' +'_pos2pt6.nii.gz')
out_filename_neg = (z_dir +'Z_est_MIND_Set_clinical' +'_neg2pt6.nii.gz')
#print(out_filename_pos, out_filename_neg)

command = ('fslmaths ' +in_filename +' -thr 2.6 ' +out_filename_pos)
print(command)
os.system(command)
!command

command = ('fslmaths ' +in_filename +' -mul -1 -thr 2.6 ' +out_filename_neg)
print(command)
os.system(command)
!command


##Binarise
in_filename_pos = (z_dir +'Z_est_MIND_Set_clinical' +'_pos2pt6.nii.gz')
in_filename_neg = (z_dir +'Z_est_MIND_Set_clinical' +'_neg2pt6.nii.gz')
#print(filename)
out_filename_pos = (z_dir +'Z_est_MIND_Set_clinical' +'_pos2pt6_bin.nii.gz')
out_filename_neg = (z_dir +'Z_est_MIND_Set_clinical' +'_neg2pt6_bin.nii.gz')
#print(out_filename_pos, out_filename_neg)

command = ('fslmaths ' +in_filename_pos +' -bin  ' +out_filename_pos)
print(command)
os.system(command)
!command

command = ('fslmaths ' +in_filename_neg +' -bin  ' +out_filename_neg)
print(command)
os.system(command)
!command

##Sum
in_filename_pos = (z_dir +'Z_est_MIND_Set_clinical' +'_pos2pt6_bin.nii.gz')
in_filename_neg = (z_dir +'Z_est_MIND_Set_clinical'+'_neg2pt6_bin.nii.gz')

thresh_bin_nii_pos = load_nifti(in_filename_pos, mask=mask_nii)
thresh_bin_nii_neg = load_nifti(in_filename_neg, mask=mask_nii)

out_filename_pos = (z_dir +'Z_est_MIND_Set_clinical' +'_count_pos.nii.gz')
out_filename_neg = (z_dir +'Z_est_MIND_Set_clinical' +'_count_neg.nii.gz')
#print(out_filename_pos, out_filename_neg)

#Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
sum_diagnosis_pos = np.sum(thresh_bin_nii_pos,axis=1)
#Save as nii
save_nifti(sum_diagnosis_pos, out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')

#Sum along the 2nd dimension(4D = volumes = participants) of the Negative thresholded, binarised file
sum_diagnosis_neg = np.sum(thresh_bin_nii_neg,axis=1)
#Save as nii
save_nifti(sum_diagnosis_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')





#%% SPLIT EACH DIGANOSIS Z-EST TO NII FILES

for col in metadata_cl_diagnosis.columns[3:]:
    mask_Diganosis = metadata_cl_diagnosis[col].eq(1)
    print( sum(mask_Diganosis == True), col)
    Z_est_Diganosis = Z_est[mask_Diganosis]
    Z_est_Diganosis_trans = np.transpose(Z_est_Diganosis)
    filename_part = (w_dir +'Z_est_' +col +'.nii.gz')
    save_nifti(Z_est_Diganosis_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')

#%% THRESHOLD THE NII FILES USING FSLMATHS
for col in metadata_cl_diagnosis.columns[3:]:
    in_filename = (w_dir +'Z_est_' +col +'.nii.gz')
    #print(filename)
    out_filename_pos = (w_dir +'Z_est_' +col +'_pos2pt6.nii.gz')
    out_filename_neg = (w_dir +'Z_est_' +col +'_neg2pt6.nii.gz')
    #print(out_filename_pos, out_filename_neg)
    
    command = ('fslmaths ' +in_filename +' -thr 2.6 ' +out_filename_pos)
    print(command)
    os.system(command)
    !command
    
    command = ('fslmaths ' +in_filename +' -mul -1 -thr 2.6 ' +out_filename_neg)
    print(command)
    os.system(command)
    !command
    
    
#%% BINARISE THE NII FILES USING FSLMATHS
for col in metadata_cl_diagnosis.columns[3:]:
    in_filename_pos = (w_dir +'Z_est_' +col +'_pos2pt6.nii.gz')
    in_filename_neg = (w_dir +'Z_est_' +col +'_neg2pt6.nii.gz')
    #print(filename)
    out_filename_pos = (w_dir +'Z_est_' +col +'_pos2pt6_bin.nii.gz')
    out_filename_neg = (w_dir +'Z_est_' +col +'_neg2pt6_bin.nii.gz')
    #print(out_filename_pos, out_filename_neg)
    
    command = ('fslmaths ' +in_filename_pos +' -bin  ' +out_filename_pos)
    print(command)
    os.system(command)
    !command
    
    command = ('fslmaths ' +in_filename_neg +' -bin  ' +out_filename_neg)
    print(command)
    os.system(command)
    !command

#%%SUM ACROSS DIGANOSES
for col in metadata_cl_diagnosis.columns[3:]:
    in_filename_pos = (w_dir +'Z_est_' +col +'_pos2pt6_bin.nii.gz')
    in_filename_neg = (w_dir +'Z_est_' +col +'_neg2pt6_bin.nii.gz')
    thresh_bin_nii_pos = load_nifti(in_filename_pos, mask=mask_nii)
    thresh_bin_nii_neg = load_nifti(in_filename_neg, mask=mask_nii)
    
    out_filename_pos = (w_dir +'Z_est_' +col +'_count_pos.nii.gz')
    out_filename_neg = (w_dir +'Z_est_' +col +'_count_neg.nii.gz')
    #print(out_filename_pos, out_filename_neg)
    
    #Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
    sum_diagnosis_pos = np.sum(thresh_bin_nii_pos,axis=1)
    #Save as nii
    save_nifti(sum_diagnosis_pos, out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')
    
    #Sum along the 2nd dimension(4D = volumes = participants) of the Negative thresholded, binarised file
    sum_diagnosis_neg = np.sum(thresh_bin_nii_neg,axis=1)
    #Save as nii
    save_nifti(sum_diagnosis_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')
    
    
#%%SPLIT EACH DIGANOSIS COUNT Z-EST TO NII FILES
#Remove lifetime or current status--> keep only current diagnosis, and remove other columns not relevant

#SUMMING OVER:
#PanicDisFinal
#AgoraPhFinal
#SocPhFinal
#SpecPhFinal
#OCDFinal
#PTSDFinal
#GADFinal
#MoodDisFinal
#AnxNOSFinal
#Conclusie_ADHD_Diagnose
#Conclusie_ASD_Diagnose

metadata_cl_diagnosis = metadata_cl_diagnosis[metadata_cl_diagnosis.columns.drop(['subj_id', 'dataset', 'diagnosis', 'UniDepPastFinal', 'BipLifetimeFinal', 'DysCurrFinal', 'CurrDepEpiFinal','CurrManEpiFinal'])]
for m in range(1,5):
    print(m)
    if m <4:     
        mask_number_diagnosis = metadata_cl_diagnosis.sum(axis=1) == m
    elif m ==4:
        mask_number_diagnosis = metadata_cl_diagnosis.sum(axis=1) > 3
    print(sum(mask_number_diagnosis == True), ' people have ', str(m), ' diagnoses')
    Z_est_number_diganosis = Z_est[mask_number_diagnosis]
    Z_est_number_diganosis_trans = np.transpose(Z_est_number_diganosis)
    filename_part = (w_dir +'Z_est_' +str(m) +'_diagnoses.nii.gz')
    save_nifti(Z_est_number_diganosis_trans, filename_part, examplenii=examplenii_cope, mask=mask_nii, dtype='float32')
    
#%% THRESHOLD THE NII FILES USING FSLMATHS
 for m in range(1,5):
    in_filename = (w_dir +'Z_est_' +str(m) +'_diagnoses.nii.gz')
    #print(filename)
    out_filename_pos = (w_dir +'Z_est_' +str(m) +'_pos2pt6.nii.gz')
    out_filename_neg = (w_dir +'Z_est_' +str(m) +'_neg2pt6.nii.gz')
    #print(out_filename_pos, out_filename_neg)
    
    command = ('fslmaths ' +in_filename +' -thr 2.6 ' +out_filename_pos)
    print(command)
    os.system(command)
    !command
    
    command = ('fslmaths ' +in_filename +' -mul -1 -thr 2.6 ' +out_filename_neg)
    print(command)
    os.system(command)
    !command
    
    
#%% BINARISE THE NII FILES USING FSLMATHS
for m in range(1,5):
    in_filename_pos = (w_dir +'Z_est_' +str(m) +'_pos2pt6.nii.gz')
    in_filename_neg = (w_dir +'Z_est_' +str(m) +'_neg2pt6.nii.gz')
    #print(filename)
    out_filename_pos = (w_dir +'Z_est_' +str(m) +'_pos2pt6_bin.nii.gz')
    out_filename_neg = (w_dir +'Z_est_' +str(m) +'_neg2pt6_bin.nii.gz')
    #print(out_filename_pos, out_filename_neg)
    
    command = ('fslmaths ' +in_filename_pos +' -bin  ' +out_filename_pos)
    print(command)
    os.system(command)
    !command
    
    command = ('fslmaths ' +in_filename_neg +' -bin  ' +out_filename_neg)
    print(command)
    os.system(command)
    !command
        
#%%SUM ACROSS DIGANOSES COUNTS
for m in range(1,5):
    in_filename_pos = (w_dir +'Z_est_' +str(m) +'_pos2pt6_bin.nii.gz')
    in_filename_neg = (w_dir +'Z_est_' +str(m) +'_neg2pt6_bin.nii.gz')
    thresh_bin_nii_pos = load_nifti(in_filename_pos, mask=mask_nii)
    thresh_bin_nii_neg = load_nifti(in_filename_neg, mask=mask_nii)
    
    out_filename_pos = (w_dir +'Z_est_' +str(m) +'_count_pos.nii.gz')
    out_filename_neg = (w_dir +'Z_est_' +str(m) +'_count_neg.nii.gz')
    #print(out_filename_pos, out_filename_neg)
    
    #Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
    sum_diagnosis_pos = np.sum(thresh_bin_nii_pos,axis=1)
    #Save as nii
    save_nifti(sum_diagnosis_pos, out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')
    
    #Sum along the 2nd dimension(4D = volumes = participants) of the Negative thresholded, binarised file
    sum_diagnosis_neg = np.sum(thresh_bin_nii_neg,axis=1)
    #Save as nii
    save_nifti(sum_diagnosis_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')
  