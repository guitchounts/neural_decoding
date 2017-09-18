#!/bin/bash
#SBATCH -p general 
#SBATCH -N 1 
#SBATCH -n 6 
#SBATCH --mem 240000
#SBATCH -t 1-00:00 
#SBATCH -o /n/home11/guitchounts/ephys/GRat32/636407614258558769/exp1.out
#SBATCH -e /n/home11/guitchounts/ephys/GRat32/636407614258558769/exp1.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guitchounts@fas.harvard.edu
python ~/code/neural_decoding/svr_gridcv.py 'svr'