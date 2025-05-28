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
mkdir -p /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud

cp /blue/guralnick/share/phenobase_specimen_data/images/images_0137b/4912057687_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0214e/2234416117_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0129j/3766337533_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0202j/4022431660_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0191g/3391077308_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0299f/437052968_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0165b/4898133043_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0144i/4876026005_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0295d/3344478092_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0156j/3758949965_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0360c/1891107282_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0126h/438177354_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0158b/4850316290_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0153g/3709856982_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0154f/2573391190_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0121f/2859419164_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0060d/3775547745_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0168g/3969577356_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0275e/4855961963_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0297g/4912110135_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0274h/2234415539_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0220c/2640539445_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0164f/3312847386_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0264c/2859481814_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0077d/3710027877_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0216e/2515397807_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0346d/3331188069_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0083g/1426301024_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0248e/438384283_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0260h/1990012864_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0153g/3886961127_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0149h/4134268559_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0018b/2012886359_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0060d/3969544710_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0194e/2012882341_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0244d/3969736701_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0347b/3710019333_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0276b/2421768285_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0276b/2012879350_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0198g/3709958407_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0149h/4101976892_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0200d/1990012894_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0065i/2012884195_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0264c/3331194962_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0275e/4850314118_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0099d/3469892423_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0116g/1457764970_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0258g/1457797882_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0171f/3091298051_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0070d/4022431650_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0336e/3893486445_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0125b/1228512475_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0088e/4102306665_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0112b/1849134700_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0159f/2565955575_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0266b/2012875962_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0014d/3978877334_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0074a/1457783540_3.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0213c/1322439445_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0328e/3766337455_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0234f/1038927772_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0124g/4022426658_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1318920727_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0121f/2859450178_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0174a/4078024860_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0253e/4134262344_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0061a/1426300871_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0342e/3886936870_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0077d/3336606359_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0196a/4134263229_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0315e/4072579360_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0105i/2012881085_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0168g/3709807035_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0324c/1978063476_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0337a/2012885786_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0243a/1990238188_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0206j/3766337556_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0207g/3829935007_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0343j/1318307217_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0104h/1978010363_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0099d/3344479100_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0276b/2234320349_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0202j/4072331078_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0246i/2421732546_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0185h/4101994095_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0146e/2242404180_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0140c/1318056184_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0334f/4536542780_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0010d/4072325682_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0244d/3766338336_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0349e/4075457129_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0140c/1457823213_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0067b/1990012838_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0018b/2234440587_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0110i/2012882999_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0146e/2234415023_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0254b/2234417405_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0064e/4101976872_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0199c/1586166300_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
cp /blue/guralnick/share/phenobase_specimen_data/images/images_0042d/3925832296_1.jpg /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud

tar czf /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud.tgz /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud
