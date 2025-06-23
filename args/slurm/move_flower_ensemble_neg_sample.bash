#!/bin/bash

#SBATCH --job-name=move_images_template

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=4:00:00

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge
mkdir -p /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2

cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1091312287_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056950830_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056659915_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2

tar czf /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2.tgz /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2
