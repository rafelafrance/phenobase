# cols: gbif_id, url, path_on_blue, download_status
# (can be filled out by running previous script)
import os
import subprocess

import pandas as pd

# Load the pickle file
pickle_file_path = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/multimedia.pkl"
df = pd.read_pickle(pickle_file_path)  # noqa: S301

# Filter the dataframe for rows where the 'type' is 'StillImage'
filtered_df = df[df["type"] == "StillImage"]

# Select the first 10,000 rows
filtered_df = filtered_df.head(10000)

# Directory to save the images
image_directory = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images"
os.makedirs(image_directory, exist_ok=True)  # noqa: PTH103

# Download images using wget with timeout and limited retries
for _, row in filtered_df.iterrows():
    gbif_id = row["gbifID"]
    url = row["identifier"]
    image_path = os.path.join(image_directory, f"{gbif_id}.jpg")  # noqa: PTH118

    # Command to download using wget with a 10-second timeout and 1 retry
    wget_command = [
        "wget",
        "-O",
        image_path,
        url,
        "--timeout=10",  # Set the connection timeout to 10 seconds
        "--tries=1",  # Limit to 1 attempt (no retries)
        "--waitretry=0",  # Don't wait before retrying (won't retry anyway)
    ]

    try:
        # Run the wget command using subprocess
        subprocess.run(wget_command, check=True)
        print(f"Downloaded {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {image_path}: {e}")

print("Download completed.")
