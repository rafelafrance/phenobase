#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

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
mkdir -p /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract

cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1056559911_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1055894971_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056632563_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055907377_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055635568_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056048765_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1055719411_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056084521_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1055778484_3.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056006432_3.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1055659049_3.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055929661_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056525558_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056247886_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056257006_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055736052_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056141874_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056446844_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1056063317_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055761752_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055927721_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1055832109_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056176407_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056222018_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056050596_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1055704174_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056072457_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056216407_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1055648500_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1055619467_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056075044_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055932257_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056264959_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056413183_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1055786787_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1055745007_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055897702_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056200148_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055761095_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056280384_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056602002_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056211830_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056577821_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056441957_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055839404_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055765505_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055883388_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055912420_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055937770_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055981672_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056443694_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056236824_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055640337_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1055653793_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056526999_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056378868_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055635248_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056469344_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056319453_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055789969_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056475342_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056533054_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056500267_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056373485_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056593062_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056528726_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056254284_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056630948_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056596005_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055809899_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056331008_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056386512_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055875535_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056558700_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055827556_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056586443_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056012188_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056402568_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1055775720_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056305949_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056204036_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1055690143_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056217929_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055678507_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056140848_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055826647_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1055678412_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056360060_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056105491_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056250558_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056271843_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056125895_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056599886_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056336440_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055924544_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056221555_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056503232_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056494369_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1055712193_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055881993_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract

tar czf /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract.tgz /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
