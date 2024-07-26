#!/bin/bash
#SBATCH --job-name FTN4
#SBATCH --ntasks=1  # Adjust if using multiprocessing
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00  # Adjust desired duration

source ~/.bashrc

# Load Anaconda environment
module load python/miniconda3

python3 generate_magic_power.py 4 1 1 100 > logN4state1method1realizations100
python3 generate_magic_power.py 4 2 1 100 > logN4state2method1realizations100
python3 generate_magic_power.py 4 3 1 100 > logN4state3method1realizations100


python3 generate_magic_power.py 4 1 2 100 > logN4state1method2realizations100 
python3 generate_magic_power.py 4 2 2 100 > logN4state2method2realizations100
python3 generate_magic_power.py 4 3 2 100 > logN4state3method2realizations100


