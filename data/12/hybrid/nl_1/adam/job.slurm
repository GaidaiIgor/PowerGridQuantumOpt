#!/bin/bash
#SBATCH -J 12/hybrid/nl_1/adam
#SBATCH -p epyc-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=60G
#SBATCH -t 02:30:00
#SBATCH -o out.txt

module load python/3.13.11
source /storage/home/zgj518/projects/PowerGridQuantumOpt/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
time python /storage/home/zgj518/projects/PowerGridQuantumOpt/run_parallel.py -df /storage/home/zgj518/projects/PowerGridQuantumOpt/data/12 -s hybrid -ae -mct 1 -t 2 -sa finite -om adam -oo '{"maxiter": 4000}'
jobstats $SLURM_JOB_ID
