#!/bin/bash
#SBATCH --job-name FTN9
#SBATCH --ntasks=1  # Adjust if using multiprocessing
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00  # Adjust desired duration

source ~/.bashrc

# Load Anaconda environment
module load python/miniconda3

python3 generate_magic_power.py 9 1 1 100 > logN9state1method1realizations100
python3 generate_magic_power.py 9 2 1 100 > logN9state2method1realizations100
python3 generate_magic_power.py 9 3 1 100 > logN9state3method1realizations100


python3 generate_magic_power.py 9 1 2 100 > logN9state1method2realizations100 
python3 generate_magic_power.py 9 2 2 100 > logN9state2method2realizations100
python3 generate_magic_power.py 9 3 2 100 > logN9state3method2realizations100


