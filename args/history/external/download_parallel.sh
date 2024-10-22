#!/bin/bash
#SBATCH --job-name=download_parallel
#SBATCH --mail-user=nevyn.neal@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=guralnick
#SBATCH --qos=guralnick

# Where to put the outputs: %j expands into the job number (a unique identifier for this job)
#SBATCH --output download_parallel%j.out
#SBATCH --error download_parallel%j.err

# Number of nodes to use
#SBATCH --nodes=1

#number of cores to parallelize with:
#SBATCH --cpus-per-task=1

# Memory per cpu core. Default is megabytes, but units can be specified with M
# or G for megabytes or Gigabytes.
#SBATCH --mem-per-cpu=128G

# Job run time in [DAYS]
# HOURS:MINUTES:SECONDS
# [DAYS] are optional, use when it is convenient
#SBATCH --time=0:20:00

## activate conda
source /home/${USER}/.bashrc
source conda activate database


# Save some useful information to the "output" file
date;hostname;pwd

python -u /home/neal.nevyn/blue_guralnick/share/gbif_data/scripts/py/download_parallel.py
