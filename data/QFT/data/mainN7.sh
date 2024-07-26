#!/bin/bash
#SBATCH --job-name FTN7
#SBATCH --ntasks=1  # Adjust if using multiprocessing
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00  # Adjust desired duration

source ~/.bashrc

# Load Anaconda environment
module load python/miniconda3

python3 generate_magic_power.py 7 1 1 100 > logN7state1method1realizations100
python3 generate_magic_power.py 7 2 1 100 > logN7state2method1realizations100
python3 generate_magic_power.py 7 3 1 100 > logN7state3method1realizations100


python3 generate_magic_power.py 7 1 2 100 > logN7state1method2realizations100 
python3 generate_magic_power.py 7 2 2 100 > logN7state2method2realizations100
python3 generate_magic_power.py 7 3 2 100 > logN7state3method2realizations100


