# EFMT_Norm_Models
Code to run the Normative Models for the emotional face matching task and numerical source data, as detailed in the following manuscript: 

**Dissecting task-based fMRI activity using normative modelling: an application to the Emotional Face Matching Task**

Hannah S. Savage (1,2), Peter C. R. Mulders (1,3), Philip F. P. van Eijndhoven (1,3), Jasper van Oort (1,3), Indira Tendolkar (1,3), Janna N. Vrijsen (1,3,4), Christian F. Beckmann (1,2,5), Andre F. Marquand (1,2)

1 Donders Institute of Brain, Cognition and Behaviour, Radboud University, Nijmegen, The Netherlands
2 Department of Cognitive Neuroscience, Radboud University Medical Centre, Nijmegen, The Netherlands
3 Department of Psychiatry, Radboud University Medical Centre, Nijmegen, The Netherlands
4 Depression Expertise Centre, Pro Persona Mental Health Care, Nijmegen, The Netherlands 
5 Centre for Functional MRI of the Brain (FMRIB), Nuffield Department of Clinical Neurosciences, Wellcome Centre for Integrative Neuroimaging, University of Oxford, Oxford, UK


## CODE: 
For faces>shapes and faces>baseline the following code was run:
Normative Models: Reference cohort - build, test and evaluate: 
* 01_prepare_wholebrain_model_wmvol.py																							
* 02_run_wholebrain_model_wmvol.py																									
* 03_evaluate_wholebrain_model_wmvol.py																									
* 04_split_Z_est_site.py																									
* 05_NPM_threshold_combine.py																									

Normative Models: Reference cohort - structure coefficients (faces>shapes only):
* 06_structure_coefficients.py

Normative Models: Clinical cohort - test and evaluate: 
* 07_prepare_wholebrain_model_wmvol_clinical.py
* 08_predict_wholebrain_model_wmvol_clinical.py
* 09_evaluate_wholebrain_model_wmvol_clincial.py
* 10_split_Z_est_diagnosis_NPM_threshold_combine.py
* 11_run_scca.py
* 11_run_scca_flip_for_diagnosis.py


## NUMERICAL SOURCE DATA: 
Detailed information can be found in: Supplementary Data Files.docx

For each faces>shapes and faces>baseline the following outputs were generated and illustrated in figures in the manuscript:
Reference:
* EV_ref.nii.gz
* kurtosis_ref.nii.gz
* skew_ref.nii.gz
* SMSE_ref.nii.gz
* Z_est_all_count_neg.nii.gz
* Z_est_all_count_pos.nii.gz

Clinical cohort:
* EV_predcl.nii.gz
* kurtosis_predcl.nii.gz
* skew_predcl.nii.gz
* SMSE_predcl.nii.gz
* Z_est_clinical_count_neg.nii.gz
* Z_est_clinical_count_pos.nii.gz

Test-ReTest:
* ICC_3_1_ICCval.nii.gz
* ICC_3_1_pval.nii.gz
* Z_pred_TRT_ReTest_count_neg.nii.gz
* Z_pred_TRT_ReTest_count_pos.nii.gz
* Z_pred_TRT_Test_count_neg.nii.gz
* Z_pred_TRT_Test_count_pos.nii.gz

For faces>shapes the structure coefficients (Correlation coefficients (rho) thresholded by the respective coefficients of determination (rho2>0.3)) for each input variable are included.

The data for the Normative Probability Map Count comparisons (Figure 6) and SCCA results (Figure 7 and Supplementary Figure 8) are in: 
* NPMCounts_SCCA.xlsx

For the supplementary faces>shapes UKBiobank additional 5000: 
* EV_pred_UKB_extra.nii.gz
* kurtosis_pred_UKB_extra.nii.gz
* skew_pred_UKB_extra.nii.gz
* SMSE_pred_UKB_extra.nii.gz
* Z_est_UKB_extra5000_count_neg.nii.gz
* Z_est_UKB_extra5000_count_pos.nii.gz




