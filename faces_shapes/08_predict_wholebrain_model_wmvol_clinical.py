import os
from pcntoolkit.normative_parallel import execute_nm, collect_nm, delete_nm
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.normative import predict, estimate

# globals
root_dir = '/project_cephfs/3022017.02/projects/hansav/Run7_fs/'
proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')
os.makedirs(w_dir, exist_ok=True)

py_path = '/home/preclineu/hansav/.conda/envs/py38/bin/python'
log_path = '/project_cephfs/3022017.02/projects/hansav/Run7_fs/logs/'
job_name = 'Hariri_cl_predict'
batch_size = 400 
memory = '40gb'
duration = '05:00:00'
#warp ='WarpSinArcsinh'
#warp_reparam = 'True'
cluster = 'torque'

resp_file_tr = os.path.join(proc_dir,'resp_tr.pkl')
resp_file_te = os.path.join(proc_dir,'resp_te.pkl')
cov_file_tr = os.path.join(proc_dir, 'cov_bspline_tr.txt')
cov_file_te = os.path.join(proc_dir, 'cov_bspline_te.txt')

resp_file_cl = os.path.join(proc_dir,'resp_cl.pkl')
cov_file_cl = os.path.join(proc_dir, 'cov_bspline_cl.txt')

os.chdir(w_dir)

# Make prdictions with test data
execute_nm(processing_dir = w_dir,
            python_path = py_path, 
            job_name = job_name, 
            covfile_path = cov_file_cl, 
            respfile_path = resp_file_cl,
            batch_size = batch_size, 
            memory = memory, 
            duration = duration, 
            func = 'predict', 
            alg = 'blr', 
            binary=True, 
            outputsuffix='_predcl',
            inputsuffix='_estimate', 
            cluster_spec = cluster, 
            log_path=log_path)

#collect_nm(w_dir, job_name, collect=True, binary=True, func='predict', outputsuffix='_predcl')
#delete_nm(w_dir, binary=True)

