import pandas as pd

# Load the pickle file
pickle_file_path = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/multimedia.pkl"
df = pd.read_pickle(pickle_file_path)

# Filter the dataframe for rows where the 'type' is 'StillImage'
filtered_df = df[df["type"] == "StillImage"]

# Remove rows with NaN or invalid URLs in the 'identifier' column
filtered_df = filtered_df.dropna(subset=["identifier"])
print(len(filtered_df))

""" # Check for duplicate gbif IDs (assuming the 'gbif_id' column exists)
if 'gbifID' in filtered_df.columns:
    duplicates = filtered_df[filtered_df.duplicated(subset=['gbifID'], keep=False)]
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate gbifIDs.")
        print(duplicates[['gbifID', 'identifier']].head())
    else:
        print("No duplicate gbifIDs found.")
else:
    print("'gbifID' column not found in the dataframe.") """

# Directory to save the images
""" image_directory = '/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images'
os.makedirs(image_directory, exist_ok=True)

# Function to download an image using wget
def download_image(row):
    gbif_id = row['gbifID']
    url = row['identifier']

    if not isinstance(url, str):
        return f"Skipping invalid URL for gbifID {gbif_id}"

    image_path = os.path.join(image_directory, f"{gbif_id}.jpg")

    # Command to download using wget with a 10-second timeout and 1 retry
    wget_command = [
        "wget", "-O", image_path, url,
        "--timeout=10",     # Set the connection timeout to 10 seconds
        "--tries=1",        # Limit to 1 attempt (no retries)
        "--waitretry=0"     # Don't wait before retrying (won't retry anyway)
    ]

    try:
        # Run the wget command using subprocess
        subprocess.run(wget_command, check=True)
        return f"Downloaded {image_path}"
    except subprocess.CalledProcessError as e:
        return f"Failed to download {image_path}: {e}"

# Use ThreadPoolExecutor to download files in parallel
with ThreadPoolExecutor(max_workers=40) as executor:
    futures = [executor.submit(download_image, row) for _, row in filtered_df.iterrows()]

    for future in as_completed(futures):
        print(future.result())

print("Download completed.")
 """
