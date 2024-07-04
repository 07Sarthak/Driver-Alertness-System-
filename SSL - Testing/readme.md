# rPPG Estimation using Self - Supervised Learning

Brief description of your project.

## Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.6+ recommended)
- Required Python packages (listed in `requirements.txt`)

## Setup

1. **Download the zip file**
   Should be of the name SSL - Testing
2. **Check if the zip file consist of a job file**
   If not then use the code given below [also ensure that the model weights and file-list are present in the zip file]
'''
job.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name self_sup_2
#SBATCH --nodes=1
#SBATCH --time=0-00:30:00
#SBATCH --error=error_test_2
#SBATCH --output=output_ssl
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_NTASKS="$SLURM_NTASKS
ulimit -s unlimited
ulimit -c unlimited


source /home/apps/DL/DL-CondaPy3.7/bin/activate
source activate /scratch/siddharths.scee.iitmandi/envs/sslconda
python main_rppgtest.py --model_path weights/rppg_model_19.pth --file_list testlist.txt > output_test_24_04_2024.txt
'''
3.**Run the code**
open your terminal in the folder and run:
'''
sbatch job.sh
'''

This should run the job file


