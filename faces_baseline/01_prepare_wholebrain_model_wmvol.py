import os
import pandas as pd
import numpy as np
import pcntoolkit as ptk 
from pcntoolkit.util.utils import create_design_matrix

# globals
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run7_f/'
data_dir = os.path.join(root_dir,'data/')
mask_nii = ('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

proc_dir = os.path.join(root_dir)

# load covariates
print('loading covariate data ...')
df_dem = pd.read_csv(os.path.join(data_dir,'control_metadata.csv'))
df_dem = df_dem.loc[(df_dem['dataset'] == 'AOMIC') | \
                    (df_dem['dataset'] == 'HCP_S1200') | \
                    (df_dem['dataset'] == 'HCP_Dev') | \
                    (df_dem['dataset'] == 'UKBiobank') | \
                    (df_dem['dataset'] == 'DNS') | \
                    (df_dem['dataset'] == 'MIND_Set')]
        
##############################
# split half training and test
##############################
tr = np.random.uniform(size=df_dem.shape[0]) > 0.5
te = ~tr

df_tr = df_dem.iloc[tr]
df_tr.to_csv(os.path.join(proc_dir,'metadata_tr.csv'))
df_te = df_dem.iloc[te]
df_te.to_csv(os.path.join(proc_dir,'metadata_te.csv'))

######################
# Configure covariates
######################
# design matrix parameters
xmin = 1 #REAL: 6 # boundaries for ages of participants +/- 5
xmax = 85 #REAL:81.0 
cols_cov = ['age','sex','TR','MB_F', 'volumes','task_length_s','target_blocks', 'instructions', 'target_stimuli', 'ICV']
site_ids =  sorted(set(df_tr['dataset'].to_list()))

print('configuring covariates ...')
X_tr = create_design_matrix(df_tr[cols_cov], site_ids = df_tr['dataset'],
                            basis = 'bspline', xmin = xmin, xmax = xmax)
print(X_tr)
X_te = create_design_matrix(df_te[cols_cov], site_ids = df_te['dataset'], all_sites=site_ids,
                            basis = 'bspline', xmin = xmin, xmax = xmax)

cov_file_tr = os.path.join(proc_dir, 'cov_bspline_tr.txt')
cov_file_te = os.path.join(proc_dir, 'cov_bspline_te.txt')
ptk.dataio.fileio.save(X_tr, cov_file_tr)
ptk.dataio.fileio.save(X_te, cov_file_te)

#########################
# configure response data
#########################
data_nii = []
data_nii.append(os.path.join(data_dir, 'faces_AOMIC_4D.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_HCP_S1200_4D_1.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_HCP_S1200_4D_2.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_HCP_Dev_4D.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_UKB_4D_1.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_UKB_4D_2.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_UKB_4D_3.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_UKB_4D_4.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_UKB_4D_5.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_DNS_4D_1.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_DNS_4D_2.nii.gz'))
data_nii.append(os.path.join(data_dir, 'faces_MIND_Set_4D_control_split1.nii.gz'))


# load the response data as nifti
print('loading wholebrain response data ...') 
for i, f in enumerate(data_nii):
    print('loading study', i, '[', f, '] ...')
    if i == 0:
        x = ptk.dataio.fileio.load(f, mask=mask_nii, vol=False).T
        print(x.shape)
        #x = ptk.dataio.fileio.load_nifti(f, mask=None, vol=False).T #without the  vol=False
    else: 
        x1 = ptk.dataio.fileio.load(f, mask=mask_nii, vol=False).T
        print(x1.shape)
        x = np.concatenate((x, ptk.dataio.fileio.load(f, mask=mask_nii, vol=False).T))
        print(x.shape)
        #x =  np.concatenate((x, ptk.dataio.fileio.load_nifti(f, mask=None, vol=False).T)) #without the  vol=False

# and write out as pkl
resp_file_tr = os.path.join(proc_dir,'resp_tr.pkl')
resp_file_te = os.path.join(proc_dir,'resp_te.pkl')
ptk.dataio.fileio.save(x[tr,:], resp_file_tr)
ptk.dataio.fileio.save(x[te,:], resp_file_te)
