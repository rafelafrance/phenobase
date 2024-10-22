import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image

# Load the pickle file
pickle_file_path = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/multimedia.pkl"
df = pd.read_pickle(pickle_file_path)  # noqa: S301

# Filter the dataframe for rows where the 'type' is 'StillImage'
filtered_df = df[df["type"] == "StillImage"]

# Remove rows with NaN or invalid URLs in the 'identifier' column
filtered_df = filtered_df.dropna(subset=["identifier"])

# Limit the dataframe to 10000 rows
filtered_df = filtered_df.head(10000)
print(f"Total filtered rows: {len(filtered_df)}")

# Check for duplicate gbif IDs (assuming the 'gbifID' column exists)
if "gbifID" in filtered_df.columns:
    filtered_df["duplicate_count"] = filtered_df.groupby("gbifID").cumcount() + 1
else:
    print("'gbifID' column not found in the dataframe.")

# Directory to save the images
image_directory = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images"
resized_image_directory = (
    "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images_resized"
)
os.makedirs(image_directory, exist_ok=True)  # noqa: PTH103
os.makedirs(resized_image_directory, exist_ok=True)  # noqa: PTH103


# Function to download an image using wget
def download_image(row):
    gbif_id = row["gbifID"]
    url = row["identifier"]
    duplicate_count = row["duplicate_count"]

    if not isinstance(url, str):
        return f"Skipping invalid URL for gbifID {gbif_id}"

    image_filename = (
        f"{gbif_id}_{duplicate_count}.jpg" if duplicate_count > 1 else f"{gbif_id}.jpg"
    )
    image_path = os.path.join(image_directory, image_filename)  # noqa: PTH118

    # Command to download using wget with a 10-second timeout and 1 retry
    wget_command = [
        "wget",
        "-O",
        image_path,
        url,
        "--timeout=10",
        "--tries=1",
        "--waitretry=0",
    ]

    try:
        subprocess.run(wget_command, check=True)
        return f"Downloaded {image_path}"  # noqa: TRY300
    except subprocess.CalledProcessError as e:
        return f"Failed to download {image_path}: {e}"


# Function to resize the image to 528x528
def resize_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Resize to 528x528
            resized_img = img.resize((528, 528), Image.Resampling.LANCZOS)
            resized_image_path = os.path.join(
                resized_image_directory, os.path.basename(image_path)
            )
            resized_img.save(resized_image_path)
            return f"Resized {image_path} and saved to {resized_image_path}"
    except Exception as e:
        return f"Failed to resize {image_path}: {e}"


# Function to delete images from the image directory
def delete_images():
    image_files = [
        os.path.join(image_directory, f)
        for f in os.listdir(image_directory)
        if f.endswith(".jpg")
    ]
    for image_path in image_files:
        try:
            os.remove(image_path)
            print(f"Deleted {image_path}")
        except OSError as e:
            print(f"Error deleting {image_path}: {e}")


# Track total resized images
total_resized = 0

# Step to process images in one batch of 10,000
batch_size = 10000
batch_df = filtered_df.iloc[0:batch_size]

# Step 1: Download images using ThreadPoolExecutor
start_time = time.time()
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(download_image, row) for _, row in batch_df.iterrows()]
    for future in as_completed(futures):
        print(future.result())

end_time = time.time()
download_time = end_time - start_time
print(f"Download of {batch_size} images completed in {download_time:.2f} seconds.")

# Step 2: Resize images using ThreadPoolExecutor
image_files = [
    os.path.join(image_directory, f)
    for f in os.listdir(image_directory)
    if f.endswith(".jpg")
]

start_time = time.time()
with ThreadPoolExecutor(max_workers=20) as executor:
    resize_futures = [
        executor.submit(resize_image, image_path) for image_path in image_files
    ]
    for future in as_completed(resize_futures):
        result = future.result()
        print(result)
        if "Resized" in result:
            total_resized += 1

end_time = time.time()
resize_time = end_time - start_time
print(f"Resizing of {total_resized} images completed in {resize_time:.2f} seconds.")

# Step 3: Delete the original images after resizing
start_time = time.time()
delete_images()
end_time = time.time()
delete_time = end_time - start_time
print(f"Deletion of original images completed in {delete_time:.2f} seconds.")

# Summary
print(f"Total images resized: {total_resized}")
print(f"Total time taken for download: {download_time:.2f} seconds.")
print(f"Total time taken for resizing: {resize_time:.2f} seconds.")
print(f"Total time taken for deletion: {delete_time:.2f} seconds.")
