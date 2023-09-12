import os
import numpy as np
from pcntoolkit.util.utils import calibration_descriptives
#from pcntoolkit.dataio.fileio import load_nifti, save_nifti
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave

root_dir = '/project_cephfs/3022017.02/projects/hansav/Run8_fs/'
data_dir = os.path.join(root_dir,'data/')
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
ex_nii = os.path.join(data_dir, 'faces_shapes_AOMIC_4D.nii.gz')

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')

output_suffix = '_predcl'

batch_size = 400 

# load data
y_te = ptkload(os.path.join(proc_dir,'resp_cl.pkl'))
EV = ptkload(os.path.join(w_dir,'EXPV_predcl.pkl')) 
Z = ptkload(os.path.join(w_dir,'Z_predcl.pkl'))
#Revisions - add SMSE
SMSE = ptkload(os.path.join(w_dir,'SMSE_predcl.pkl'))

Z[np.isnan(Z)] = 0
Z[np.isinf(Z)] = 0

[skew, sds, kurtosis, sdk, semean, sesd] = calibration_descriptives(Z)

badk = np.abs(kurtosis) > 10
kurtosis2 = kurtosis[~badk]

bads = np.abs(skew) > 10
skew2 = skew[~bads]

# fix some random bad voxels 
EV[EV < -1] = 0

ptksave(skew, os.path.join(w_dir,'skew' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(kurtosis, os.path.join(w_dir,'kurtosis' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
#Revisions - add SMSE
ptksave(SMSE.T, os.path.join(w_dir,'SMSE' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)

ptksave(EV.T, os.path.join(w_dir,'EV' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(Z, os.path.join(w_dir,'Z' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(y_te, os.path.join(w_dir,'y' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(np.arange(len(EV)), os.path.join(w_dir,'idx.nii.gz'), example=ex_nii, mask=mask_nii, dtype='uint32')

# load the index again in volumetric form
idxvol = ptkload(os.path.join(w_dir,'idx.nii.gz'), mask=mask_nii, vol=True)

# find the voxel coordinates for a given value
#vox_id = np.where(badk)[0][0]
#vox_coord = np.asarray(np.where(idxvol == vox_id)).T

# alternative method (opposite direction)
vox_coord = (11,13,60)
vox_id = int(idxvol[vox_coord])

# find batch id
#batch_num, mod_num = divmod(vox_id, batch_size)
#batch_num = batch_num + 1 # batch indexing starts at 1
