file_path = '../gbif_data/multimedia.txt'  # Replace with your actual file path
gbif_d = data.table::fread(file = file_path)

head(gbif_d)
gbif_d[302940,]

count(gbif_d, type)
filter(gbif_d, type == "")[1:10,]
#                   type        n
#                   <char>    <int>
# 1:                      2680820
# 2: InteractiveResource  2611011
# 3:          StillImage 35314854

gbif_d = filter(gbif_d, type == "StillImage")
names(gbif_d)

gbif_d = select(gbif_d, gbifID, format, identifier, description) %>% distinct()

write_dataset(gbif_d, path = "metadata/gbif_images_url", format = "parquet")
