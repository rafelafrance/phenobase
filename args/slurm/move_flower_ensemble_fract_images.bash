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

cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1056481656_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056083568_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056012269_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056225709_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056428687_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056348997_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0102a/1055981555_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055804330_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056397849_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055930042_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1055806306_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056174364_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1056099502_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056157534_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056090704_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1055749314_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056324457_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055795043_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1055942891_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055927488_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056206340_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056207083_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1055680496_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056582825_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1056367027_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056042045_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055770356_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055659576_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056576435_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055794418_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055747717_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1055825076_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056622028_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056046974_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056410159_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055615934_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056531907_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056155315_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1056352663_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056094269_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1055721175_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055934287_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056389014_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056458125_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056352339_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056566602_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055744918_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056483370_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056101828_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056492834_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056030354_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056410753_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1055772142_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055985425_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055679803_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056152255_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056421666_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056132027_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055619660_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056095538_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1055846861_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055646840_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056349112_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056502661_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1056158650_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1056358838_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056309788_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056417279_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056439081_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1056550203_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1056265381_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055900202_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055935795_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0003/1055836588_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056212339_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056504245_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056400565_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055674772_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056460028_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1056293754_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056230875_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055840272_3.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056245626_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1055606308_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056530822_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056363940_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1055648050_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0086a/1056253460_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1055771378_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056266623_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0017a/1056285203_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056406749_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0004/1056019690_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0106a/1056355598_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056442546_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0095a/1055857247_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1055687615_2.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0047a/1056209310_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0034a/1055710820_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0111a/1056344045_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract
