#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100#
#SBATCH --job-name=t5_test
#SBATCH --mem-per-cpu=30000
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.vogel@alumnos.upm.es
#SBATCH --output=out-%j.log
##------------------------ End job description ------------------------

module purge && module load CUDA


source /home/s730/s730251/projects/T5env/bin/activate

# rutas absolutas! 
srun python3 train_t5_XLSum.py 

deactivate
