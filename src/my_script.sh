#!/bin/bash -l

module load python/3.6.2 
# OR: module load python/2.7.13
module load cuda/8.0 
module load cudnn/6.0 
module load tensorflow/r1.3

python main_threads.py

