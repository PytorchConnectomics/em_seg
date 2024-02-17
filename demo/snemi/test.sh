#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -p cox
#SBATCH -n 4 # number of cores
#SBATCH --gres=gpu:4 # memory pool for all cores
#SBATCH --mem 50000 # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)

#SBATCH -o db/slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e db/slurm/slurm.%N.%j.err # STDERR

module load cuda/10.2.89-fasrc01 cudnn/7.6.5.32_cuda10.2-fasrc01 boost

source /n/pfister_lab2/Lab/donglai/lib/miniconda2/bin/activate pytorchzudi

python /n/pfister_lab2/Lab/donglai/lib/pipeline/zudi/pytorch_connectomics/scripts/main.py --config-file snemi_db_4min_orig_x2_z29.yaml --inference --checkpoint pc/output/checkpoint_50000.pth.tar

