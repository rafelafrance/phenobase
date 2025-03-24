# cols: gbif_id, url, path_on_blue, download_status (can be filled out by running previous script)
import os

import pandas as pd

# Path to the text file
file_path = "/home/neal.nevyn/blue_guralnick/share/gbif_data/multimedia.txt"  # Replace with your actual file path

# Load the large .txt file as if it's a .tsv
# Using low_memory=False to avoid DtypeWarning for large mixed-type columns
# Specifying tab ('\t') as the delimiter
multimedia_df = pd.read_csv(file_path, delimiter="\t", low_memory=False)
print(multimedia_df.head())

# save as pkl file

pickle_file_path = "home/neal.nevyn/blue_guralnick/share/gbif_data/data/multimedia.pkl"  # Choose a path to save the pickle file
# Create the directory if it does not exist
os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)

multimedia_df.to_pickle(pickle_file_path)

print(f"DataFrame saved as a pickle file at: {pickle_file_path}")
