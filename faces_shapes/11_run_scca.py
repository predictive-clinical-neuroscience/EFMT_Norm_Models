import os
import sys
import pandas as pd
import numpy as np
import pcntoolkit as ptk 

from matplotlib import pyplot as plt
import seaborn as sns

#%% Set globals

scriptsdir = '/project_cephfs/3022017.02/projects/hansav/Run7_fs/scripts/'
datadir = '/project_cephfs/3022017.02/projects/hansav/Run7_fs/data/'
Zdir = '/project_cephfs/3022017.02/projects/hansav/Run7_fs/vox/'
sys.path.append(os.path.join(scriptsdir, 'saccade'))
from scca import SCCA

mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
ex_nii = os.path.join(Zdir, 'Z_predcl.nii.gz')

#%% load data
df = pd.read_csv(os.path.join(datadir,'MIND_Set_metadata_control_split2_patients.csv'))#,index_col=0)
df_diag = pd.read_csv(os.path.join(datadir,'MIND_Set_diagnoses.csv'))
df_efa = pd.read_csv(os.path.join(datadir,'MIND_Set_EFA.csv'))


X2 = ptk.dataio.fileio.load(os.path.join(Zdir, 'Z_predcl.nii.gz'), mask=mask_nii, vol=False).T

#%%Select which SCCA to run: 

##DIAGNOSES
X1 = df_diag[['PanicDisFinal','AgoraPhFinal', 'SocPhFinal', 'SpecPhFinal', 
              'OCDFinal','PTSDFinal', 'GADFinal', 'MoodDisFinal', 'AnxNOSFinal',
                'Conclusie_ADHD_Diagnose', 'Conclusie_ASD_Diagnose']].to_numpy()
X1[np.isnan(X1)] = 0

##FACTORS
# X1 = df_efa[['FAC1_1', 'FAC2_1', 'FAC3_1', 'FAC4_1']].to_numpy()
# X1[:,1] = -1 * X1[:,1] 
# idx = np.where(df['subj_id'].isin(df_efa['subj_id']))[0]
# X2 = X2[idx,:]


#%%Run SCCA
l1_1 = 0.9
l1_2 = 0.1 
niter = 1000; 
w1_sign = 0;
w2_sign = 0;


n_splits = 10
W1 = np.zeros((n_splits, X1.shape[1]))
W2 = np.zeros((n_splits, X2.shape[1]))
R = np.zeros(n_splits)
R_Train = np.zeros(n_splits)
for i in range(n_splits):
    tr = np.random.uniform(size=X1.shape[0]) < 0.7
    te = ~tr
    
    # standardize x1
    m1 = np.mean(X1[tr,:], axis = 0)
    s1 = np.std(X1[tr,:], axis = 0)
    X1tr = (X1[tr,:] - m1) / s1
    X1te = (X1[te,:] - m1) / s1
    
    # standardize x2
    m2 = np.mean(X2[tr,:], axis = 0)
    s2 = np.std(X2[tr,:], axis = 0)
    X2tr = (X2[tr,:] - m2) / s2
    X2te = (X2[te,:] - m2) / s2

    print('split', i, 'fitting scca...')
    C = SCCA()
    C.fit(X1tr,X2tr, l1_x=l1_1, l1_y=l1_2)
    x1_scores, x2_scores = C.transform(X1te, X2te)
    r = np.corrcoef(x1_scores, x2_scores)[0][1]
    
    print('r_train =', C.r, 'r_test =',r)
    W1[i,:] = C.wx
    W2[i,:] = C.wy
    R[i] = r
    R_Train[i] = C.r

ptk.dataio.fileio.save(W2, os.path.join(datadir,'W_faces_shapes_diagnosis.nii.gz'), example=ex_nii, mask=mask_nii)

# n_perm = 100
# n_splits = 5
# Rperm = np.zeros(n_perm)
# for p in range(n_perm):
#     print('perm', p)
#     R = np.zeros(n_splits)
#     for i in range(n_splits):
#         tr = np.random.uniform(size=X1.shape[0]) < 0.7
#         te = ~tr
    
#         # standardize x1
#         X1p = np.random.permutation(X1)
#         m1 = np.mean(X1p[tr,:], axis = 0)
#         s1 = np.std(X1p[tr,:], axis = 0)
#         X1tr = (X1p[tr,:] - m1) / s1
#         X1te = (X1p[te,:] - m1) / s1
    
#         # standardize x2
#         m2 = np.mean(X2[tr,:], axis = 0)
#         s2 = np.std(X2[tr,:], axis = 0)
#         X2tr = (X2[tr,:] - m2) / s2
#         X2te = (X2[te,:] - m2) / s2

#         print('split', i, 'fitting scca...')
#         C = SCCA()
#         C.fit(X1tr,X2tr, l1_x=l1_1, l1_y=l1_2)
#         x1_scores, x2_scores = C.transform(X1te, X2te)
#         r = np.corrcoef(x1_scores, x2_scores)[0][1]
        
#         R[i] = r
#     Rperm[p] = np.mean(R)
    
    
    
#%% Get mean train test values 
 
##################
##SCCA OUTPUT##
##################
faces_factors_train = R_Train
faces_factors_train_df = pd.DataFrame(faces_factors_train)
faces_factors_train_df.mean()

faces_factors_test = R
faces_factors_test_df = pd.DataFrame(faces_factors_test)
faces_factors_test_df.mean()


#Save as a dataframe -> csv to be able to easily re-run in the future if necessary.
faces_factors_scca = pd.concat([faces_factors_train_df, faces_factors_test_df], ignore_index=True, axis =1 )
faces_factors_scca = faces_factors_scca.rename(columns = {0:'Train', 1:'Test'})
#faces_factors_scca.to_csv(os.path.join('/project_cephfs/3022017.02/projects/hansav/Run7_fs/vox/NPM/SCCA/Faces_shapes_scca_factors.csv'),index=False)      
faces_factors_scca.to_csv(os.path.join('/project_cephfs/3022017.02/projects/hansav/Run7_fs/vox/NPM/SCCA/Faces_shapes_scca_diagnosis.csv'),index=False)      

 
#%% Plots:
##################
#BAR PLOTS - FACTORS
##################
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
faces_W = W1
#faces_W = np.load('/project_cephfs/3022017.02/projects/hansav/Run7_f/validation/eft/W1_faces_factors.npy')
#faces_shapes_W = np.load('/project_cephfs/3022017.02/projects/hansav/Run6_f/validation/eft/W1_faces_shapes_diagnosis.npy')
faces_W_df = pd.DataFrame(faces_W, columns = ['Negative\nValence', 'Cognitive\nFunction', 'Social\nProcesses', 'Arousal/\nInhibition'])
Faces_W_melt =pd.melt(faces_W_df)
Faces_W_melt.rename(columns = {'variable' : 'Factor'}, inplace = True)
sns.catplot(data = Faces_W_melt, x="Factor", y="value", palette = ['#e199c2','#f7e5ef','#eab8d4' , '#ca4e95'], kind="bar", ci ="sd")
                  #errorbar="sd", capsize=.2, join=False, color=".5")
sns.swarmplot(data = Faces_W_melt, x="Factor", y="value", size = 3,  dodge = False, color = "black")
#plt.xticks([0,1,2,3], ['Negative \nValence', 'Cognitive \nFunction', 'Social \nProcesses', 'Arousal/\nInhibition'])
plt.ylabel('Coefficients')
plt.xlabel('Factor')
#plt.xlim(reversed(plt.xlim()))
plt.ylim(-0.9,0.1)
sns.despine()
plt.savefig('/project_cephfs/3022017.02/projects/hansav/Run7_fs/Figures/scca_factors.png', dpi=300)


#%% Plots:
##################
#BAR PLOTS - DIAGNOSIS
##################
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
faces_W = np.load('/project_cephfs/3022017.02/projects/hansav/Run6_f/validation/eft/W1_faces_diagnosis.npy')
faces_W_df = pd.DataFrame(faces_W, columns = ['Panic Disorder','Agoraphobia', 'Social Phobia', 'Specific Phobia', 
                                                            'OCD','PTSD', 'GAD', 'Mood Disorder', 'Anxiety Disorder NOS',
                                                            'ADHD', 'ASD'])
Faces_W_melt =pd.melt(faces_W_df)
Faces_W_melt.rename(columns = {'variable' : 'Diagnosis'}, inplace = True)
sns.catplot(data = Faces_W_melt, x="Diagnosis", y="value", 
            palette = ['#f3d6e6','#ca4e95','#f7e5ef','#eec7dd','#f7e5ef','#eab8d4','#dc8ab9','#f3d6e6','#d36ca7','#eec7dd','#d36ca7'],
            kind="bar", ci ="sd")
                  #errorbar="sd", capsize=.2, join=False, color=".5")
sns.swarmplot(data = Faces_W_melt, x="Diagnosis", y="value", size = 3,  dodge = False, color = "black")
plt.xticks([0,1,2,3,4,5,6,7,8,9,10], ['Panic\nDisorder','Agora-\nphobia', 'Social\nPhobia', 'Specific\nPhobia', 
                                                            'OCD','PTSD', 'GAD', 'Mood\nDisorder', 'Anxiety\nDisorder\nNOS',
                                                            'ADHD', 'ASD'], fontsize =6)
plt.ylabel('Coefficients')
plt.xlabel('Diagnosis')
#plt.xlim(reversed(plt.xlim()))
plt.ylim(-0.8,0.6)
sns.despine()
plt.savefig('/project_cephfs/3022017.02/projects/hansav/Run7_fs/Figures/scca_diagnosis.png', dpi=300)


#%% Create mean W_faces_shapes_X.nii.gz file: 
    
in_filename = os.path.join(Zdir + 'NPM/SCCA/W_faces_shapes_factors.nii.gz')
out_filename = os.path.join (Zdir +'NPM/SCCA/W_faces_shapes_factors_mean.nii.gz')
 
command = ('fslmaths ' +in_filename +' -Tmean ' +out_filename)
print(command)
os.system(command)
!command   

    
in_filename = os.path.join(Zdir + 'NPM/SCCA/W_faces_shapes_diagnosis_flipped.nii.gz')
out_filename = os.path.join (Zdir +'NPM/SCCA/W_faces_shapes_diagnosis_flipped_mean.nii.gz')
 
command = ('fslmaths ' +in_filename +' -Tmean ' +out_filename)
print(command)
os.system(command)
!command   