#!/bin/bash
#SBATCH --job-name=sarima_p1_q1_P1_Q1         # Job name
#SBATCH --partition=batch             # Partition (queue) name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=13gb                     # Job memory request
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x.%j.out            # Standard output log
#SBATCH --error=%x.%j.err             # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fs47816@uga.edu  # Where to send mail	

cd /scratch/fs47816/workdir/sample_scripts/time_series_dl/stats_models

ml Anaconda3/2023.09-0

python print_script.py

python sarima.py -p 1 -q 1 -P 0 -Q 0

