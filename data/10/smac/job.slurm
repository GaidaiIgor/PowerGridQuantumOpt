#!/bin/bash
#SBATCH -J 10
#SBATCH -p epyc-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=100G
#SBATCH -t 00:30:00
#SBATCH -o out.txt
#SBATCH -e err.txt

module load python/3.13.11
source /storage/home/zgj518/projects/PowerGridQuantumOpt/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /storage/home/zgj518/projects/PowerGridQuantumOpt/run_parallel.py .. smac
