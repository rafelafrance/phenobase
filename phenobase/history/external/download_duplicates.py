import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Load the pickle file
pickle_file_path = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/multimedia.pkl"
df = pd.read_pickle(pickle_file_path)

# Filter the dataframe for rows where the 'type' is 'StillImage'
filtered_df = df[df["type"] == "StillImage"]

# Remove rows with NaN or invalid URLs in the 'identifier' column
filtered_df = filtered_df.dropna(subset=["identifier"])

# Limit the dataframe to 10,000 rows for testing
filtered_df = filtered_df.head(10000)
print(f"Total filtered rows for testing: {len(filtered_df)}")

# Check for duplicate gbif IDs (assuming the 'gbifID' column exists)
if "gbifID" in filtered_df.columns:
    # Track duplicates
    filtered_df["duplicate_count"] = filtered_df.groupby("gbifID").cumcount() + 1
    duplicates = filtered_df[filtered_df.duplicated(subset=["gbifID"], keep=False)]
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate gbifIDs.")
        print(duplicates[["gbifID", "identifier"]].head())
    else:
        print("No duplicate gbifIDs found.")
else:
    print("'gbifID' column not found in the dataframe.")

# Directory to save the images
image_directory = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images"
os.makedirs(image_directory, exist_ok=True)


# Function to download an image using wget
def download_image(row):
    gbif_id = row["gbifID"]
    url = row["identifier"]
    duplicate_count = row["duplicate_count"]

    if not isinstance(url, str):
        return f"Skipping invalid URL for gbifID {gbif_id}"

    # If the duplicate count is greater than 1, add a suffix to the filename
    if duplicate_count > 1:
        image_filename = f"{gbif_id}_{duplicate_count}.jpg"
    else:
        image_filename = f"{gbif_id}.jpg"

    image_path = os.path.join(image_directory, image_filename)

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
        return f"Downloaded {image_path}"
    except subprocess.CalledProcessError as e:
        return f"Failed to download {image_path}: {e}"


# Use ThreadPoolExecutor to download files in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [
        executor.submit(download_image, row) for _, row in filtered_df.iterrows()
    ]

    for future in as_completed(futures):
        print(future.result())

print("Download completed.")
