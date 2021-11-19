#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100#
#SBATCH --job-name=nlp_test
#SBATCH --mem-per-cpu=10000
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pcalleja@fi.upm.es
#SBATCH --output=out-%j.log
##------------------------ End job description ------------------------

module purge && module load CUDA


source /home/s730/s730251/projects/envexp/bin/activate

# rutas absolutas! 
srun python3 run_ner.py --app-param --data_dir=/home/s730/s730251/projects/BERT-NER/data/  --bert_model=bert-base-cased --task_name=ner --output_dir=/home/s730/s730251/projects/BERT-NER/out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1

deactivate
