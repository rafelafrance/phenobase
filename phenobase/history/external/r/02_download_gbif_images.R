gbif_d = arrow::open_dataset(source = "metadata/gbif_images_url")
gbif_d = collect(gbif_d)
count(gbif_d, format)

image <- image_read(gbif_d$identifier[1])

# Image details
info <- image_info(image)
