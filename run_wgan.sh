#!/bin/bash

# Generic job script for all experiments.

#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -t24:00:00

#PRINCE PRINCE_GPU_COMPUTE_MODE=default

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0

# Default model.
PYTHONPATH=. python unsupervised_wgan.py --src_lang zh --tgt_lang en --src_emb /scratch/nn1119/infgan/wiki.zh.vec --tgt_emb  /scratch/nn1119/infgan/wiki.en.vec --refinement True --n_epochs 10