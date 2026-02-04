#!/bin/sh
### General options
### –- specify queue: gpuv100, gputitanxpascal, gpuk40, gpum2050 --
#BSUB -q gpuh100
### -- set the job Name --
#BSUB -J coop
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### -- request 24GB of memory
#BSUB -R "rusage[mem=5GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc_submit/logs/coop/%J.out
#BSUB -e hpc_submit/logs/coop/%J.err
# -- end of LSF options --

# Exits if any errors occur at any point (non-zero exit code)
set -e

# Load virtual Python environment
source /zhome/de/d/169059/vlms-initial-testing/hpc_submit/load_modules.sh
##################################################################
# Execute your own code by replacing the sanity check code below #
##################################################################
# Print available GPU devices with Tensorflow
python scripts/train_models/run_video_coop.py