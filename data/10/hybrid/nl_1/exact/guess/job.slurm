#!/bin/bash
#SBATCH -J 10/hybrid/nl_1/exact/guess
#SBATCH -p epyc-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=60G
#SBATCH -t 01:30:00
#SBATCH -o out.txt

module load python/3.13.11
source /storage/home/zgj518/projects/PowerGridQuantumOpt/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
time python /storage/home/zgj518/projects/PowerGridQuantumOpt/run_parallel.py -df /storage/home/zgj518/projects/PowerGridQuantumOpt/data/10 -s hybrid -ae -mct 0.5 -t 1 -sa exact -ia guess
jobstats $SLURM_JOB_ID
