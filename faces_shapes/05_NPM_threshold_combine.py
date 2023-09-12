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

#%% SET GLOBALS
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run8_fs/'
data_dir = os.path.join(root_dir)
w_dir = os.path.join(root_dir,'vox/NPM/')
mask_nii = os.path.join('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

#%% THRESHOLD THE NII FILES USING FSLMATHS
#Threshold of 2.6
site = ['AOMIC', 'HCP_1200', 'HCP_Dev', 'UKB', 'DNS', 'MIND_Set']

for s in site:
    if (s == 'HCP_1200') or (s == 'DNS'):
        for img in range(1,3):
            in_filename = (w_dir +'Z_est_' +s +'_' +str(img) +'.nii.gz')
            #print(filename)
            out_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6.nii.gz')
            out_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6.nii.gz')
            #print(out_filename_pos, out_filename_neg)
            
            command = ('fslmaths ' +in_filename +' -thr 2.6 ' +out_filename_pos)
            print(command)
            os.system(command)
            !command
                        
            command = ('fslmaths ' +in_filename +' -mul -1 -thr 2.6 ' +out_filename_neg)
            print(command)
            os.system(command)
            !command
            
    elif s == 'UKB':
        n = range(1,5)
        for img in range(1,6):
            in_filename = (w_dir +'Z_est_' +s +'_' +str(img) +'.nii.gz')
            #print(filename)
            out_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6.nii.gz')
            out_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6.nii.gz')
            #print(out_filename_pos, out_filename_neg)
            
            command = ('fslmaths ' +in_filename +' -thr 2.6 ' +out_filename_pos)
            print(command)
            os.system(command)
            !command
            
            command = ('fslmaths ' +in_filename +' -mul -1 -thr 2.6 ' +out_filename_neg)
            print(command)
            os.system(command)
            !command
    else:
        in_filename = (w_dir +'Z_est_' +s +'.nii.gz')
        #print(filename)
        out_filename_pos = (w_dir +'Z_est_' +s +'_pos2pt6.nii.gz')
        out_filename_neg = (w_dir +'Z_est_' +s +'_neg2pt6.nii.gz')
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
site = ['AOMIC', 'HCP_1200', 'HCP_Dev', 'UKB', 'DNS', 'MIND_Set']

for s in site:
    if (s == 'HCP_1200') or (s == 'DNS'):
        for img in range(1,3):
            in_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6.nii.gz')
            in_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6.nii.gz')
            #print(filename)
            out_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6_bin.nii.gz')
            out_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6_bin.nii.gz')
            #print(out_filename_pos, out_filename_neg)
            
            command = ('fslmaths ' +in_filename_pos +' -bin  ' +out_filename_pos)
            print(command)
            os.system(command)
            !command
            
            command = ('fslmaths ' +in_filename_neg +' -bin  ' +out_filename_neg)
            print(command)
            os.system(command)
            !command
            
    elif s == 'UKB':
        for img in range(1,6):
            in_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6.nii.gz')
            in_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6.nii.gz')
            #print(filename)
            out_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6_bin.nii.gz')
            out_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6_bin.nii.gz')
            #print(out_filename_pos, out_filename_neg)
            
            command = ('fslmaths ' +in_filename_pos +' -bin  ' +out_filename_pos)
            print(command)
            os.system(command)
            !command
            
            command = ('fslmaths ' +in_filename_neg +' -bin  ' +out_filename_neg)
            print(command)
            os.system(command)
            !command
    else:
        in_filename_pos = (w_dir +'Z_est_' +s +'_pos2pt6.nii.gz')
        in_filename_neg = (w_dir +'Z_est_' +s +'_neg2pt6.nii.gz')
        #print(filename)
        out_filename_pos = (w_dir +'Z_est_' +s +'_pos2pt6_bin.nii.gz')
        out_filename_neg = (w_dir +'Z_est_' +s +'_neg2pt6_bin.nii.gz')
        #print(out_filename_pos, out_filename_neg)
    
        command = ('fslmaths ' +in_filename_pos +' -bin  ' +out_filename_pos)
        print(command)
        os.system(command)
        !command
        
        command = ('fslmaths ' +in_filename_neg +' -bin  ' +out_filename_neg)
        print(command)
        os.system(command)
        !command
        
#%% SUM THE NUMBER OF VOXELS

site = ['AOMIC', 'HCP_1200', 'HCP_Dev', 'UKB', 'DNS', 'MIND_Set']

for s in site:
    if (s == 'HCP_1200') or (s == 'DNS'):
        for img in range(1,3):
            print(s +' ' +str(img))
            #load thresholded, binarised files
            in_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6_bin.nii.gz')
            in_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6_bin.nii.gz')
            thresh_bin_nii_pos = load_nifti(in_filename_pos, mask=mask_nii)
            thresh_bin_nii_neg = load_nifti(in_filename_neg, mask=mask_nii)
            
            #print(filename)
            #Filenames for output count files
            out_filename_pos_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_pos.nii.gz')
            out_filename_neg_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_neg.nii.gz')
            #print(out_filename_pos, out_filename_neg)
           
            #Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
            sum_site_part_pos = np.sum(thresh_bin_nii_pos,axis=1)
            #Save as nii
            save_nifti(sum_site_part_pos,out_filename_pos_part, examplenii=mask_nii, mask=mask_nii, dtype='float32')
            
            #Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
            sum_site_part_neg = np.sum(thresh_bin_nii_neg,axis=1)
            #Save as nii
            save_nifti(sum_site_part_neg,out_filename_neg_part, examplenii=mask_nii, mask=mask_nii, dtype='float32')
            
    elif s == 'UKB':
         for img in range(1,6):
            print(s +' ' +str(img))
            in_filename_pos = (w_dir +'Z_est_' +s +'_' +str(img) +'_pos2pt6_bin.nii.gz')
            in_filename_neg = (w_dir +'Z_est_' +s +'_' +str(img) +'_neg2pt6_bin.nii.gz')
            thresh_bin_nii_pos = load_nifti(in_filename_pos, mask=mask_nii)
            thresh_bin_nii_neg = load_nifti(in_filename_neg, mask=mask_nii)
 
            out_filename_pos_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_pos.nii.gz')
            out_filename_neg_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_neg.nii.gz')
            #print(out_filename_pos, out_filename_neg)
           
            #Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
            sum_site_part_pos = np.sum(thresh_bin_nii_pos,axis=1)
            #Save as nii
            save_nifti(sum_site_part_pos,out_filename_pos_part, examplenii=mask_nii, mask=mask_nii, dtype='float32')
            
            #Sum along the 2nd dimension(4D = volumes = participants) of the Negative thresholded, binarised file
            sum_site_part_neg = np.sum(thresh_bin_nii_neg,axis=1)
            #Save as nii
            save_nifti(sum_site_part_neg,out_filename_neg_part, examplenii=mask_nii, mask=mask_nii, dtype='float32')
  
    else:
        print(s)
        in_filename_pos = (w_dir +'Z_est_' +s +'_pos2pt6_bin.nii.gz')
        in_filename_neg = (w_dir +'Z_est_' +s +'_neg2pt6_bin.nii.gz')
        thresh_bin_nii_pos = load_nifti(in_filename_pos, mask=mask_nii)
        thresh_bin_nii_neg = load_nifti(in_filename_neg, mask=mask_nii)

        out_filename_pos = (w_dir +'Z_est_' +s +'_count_pos.nii.gz')
        out_filename_neg = (w_dir +'Z_est_' +s +'_count_neg.nii.gz')
        #print(out_filename_pos, out_filename_neg)
       
        #Sum along the 2nd dimension(4D = volumes = participants) of the Positive thresholded, binarised file
        sum_site_pos = np.sum(thresh_bin_nii_pos,axis=1)
        #Save as nii
        save_nifti(sum_site_pos,out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')
        
        #Sum along the 2nd dimension(4D = volumes = participants) of the Negative thresholded, binarised file
        sum_site_neg = np.sum(thresh_bin_nii_neg,axis=1)
        #Save as nii
        save_nifti(sum_site_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')
     
#%% CREATE SUMMARY COUNT FILES
#Sites with more than 1 file

site = ['HCP_1200', 'UKB', "DNS"]
for s in site:
    if (s == 'HCP_1200') or (s == 'DNS'):
        combined_pos = []
        combined_neg = []

        for img in range(1,3):
            print(s +' ' +str(img))
            #load part count files
            in_filename_pos_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_pos.nii.gz')
            in_filename_neg_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_neg.nii.gz')
            count_pos = np.round(load_nifti(in_filename_pos_part, mask=mask_nii))
            count_neg = np.round(load_nifti(in_filename_neg_part, mask=mask_nii))
            
            combined_pos.append(count_pos)
            combined_neg.append(count_neg)
            
        combined_pos.append(np.sum(combined_pos[0:2],axis=0))
        combined_pos = combined_pos[2]
        combined_neg.append(np.sum(combined_neg[0:2],axis=0))
        combined_neg = combined_neg[2]

        out_filename_pos = (w_dir +'Z_est_' +s +'_count_pos.nii.gz')
        out_filename_neg = (w_dir +'Z_est_' +s +'_count_neg.nii.gz')

        #Save as nii
        save_nifti(combined_pos,out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')
        #Save as nii
        save_nifti(combined_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')

    elif s == 'UKB':
         UKB_combined_pos = []
         UKB_combined_neg = []
         
         for img in range(1,6):
             print(s +' ' +str(img))
             #load part count files
             in_filename_pos_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_pos.nii.gz')
             in_filename_neg_part = (w_dir +'Z_est_' +s +'_' +str(img) +'_count_neg.nii.gz')
             count_pos = np.round(load_nifti(in_filename_pos_part, mask=mask_nii))
             count_neg = np.round(load_nifti(in_filename_neg_part, mask=mask_nii))
             
             UKB_combined_pos.append(count_pos)
             UKB_combined_neg.append(count_neg)
             
         UKB_combined_pos.append(np.sum(UKB_combined_pos[0:6],axis=0))
         UKB_combined_pos = UKB_combined_pos[5]
         UKB_combined_neg.append(np.sum(UKB_combined_neg[0:6],axis=0))
         UKB_combined_neg = UKB_combined_neg[5]

         out_filename_pos = (w_dir +'Z_est_' +s +'_count_pos.nii.gz')
         out_filename_neg = (w_dir +'Z_est_' +s +'_count_neg.nii.gz')

         #Save as nii
         save_nifti(UKB_combined_pos,out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')
         #Save as nii
         save_nifti(UKB_combined_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')



#%% CREATE SUMMARY COUNT FILES
#Combine sites


site = ['AOMIC', 'HCP_1200', 'HCP_Dev', 'UKB', 'DNS', 'MIND_Set']

all_sites_pos = []
all_sites_neg = []

for s in site:
    #load site count files
    in_filename_pos_part = (w_dir +'Z_est_' +s +'_count_pos.nii.gz')
    in_filename_neg_part = (w_dir +'Z_est_' +s +'_count_neg.nii.gz')
    count_pos = np.round(load_nifti(in_filename_pos_part, mask=mask_nii))
    count_neg = np.round(load_nifti(in_filename_neg_part, mask=mask_nii))
    
    all_sites_pos.append(count_pos)
    all_sites_neg.append(count_neg)
    
all_sites_pos.append(np.sum(all_sites_pos[0:7],axis=0))
all_sites_pos = all_sites_pos[6]
all_sites_neg.append(np.sum(all_sites_neg[0:7],axis=0))
all_sites_neg = all_sites_neg[6]


out_filename_pos = (w_dir +'Z_est_all_count_pos.nii.gz')
out_filename_neg = (w_dir +'Z_est_all_count_neg.nii.gz')

#Save as nii
save_nifti(all_sites_pos,out_filename_pos, examplenii=mask_nii, mask=mask_nii, dtype='float32')
#Save as nii
save_nifti(all_sites_neg,out_filename_neg, examplenii=mask_nii, mask=mask_nii, dtype='float32')

    


