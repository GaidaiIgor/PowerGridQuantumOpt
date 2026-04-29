#!/bin/bash
#SBATCH -J 5/hybrid/nl_1
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
time python /storage/home/zgj518/projects/PowerGridQuantumOpt/run_parallel.py -df /storage/home/zgj518/projects/PowerGridQuantumOpt/data/5 -s hybrid -mct 0.5 -t 2 -ae -nl 1
jobstats $SLURM_JOB_ID
