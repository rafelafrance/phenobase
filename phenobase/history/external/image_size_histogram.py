import os

import matplotlib.pyplot as plt

# Directory path
directory = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images_resized"

# List to store file sizes in megabytes (MB)
file_sizes_mb = []
zero_byte_files_count = 0  # Counter for zero-byte files

# Walk through the directory and subdirectories
for root, dirs, files in os.walk(directory):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            # Get the file size in bytes
            file_size_bytes = os.path.getsize(file_path)

            # Only process files with a size greater than 0 bytes
            if file_size_bytes > 0:
                file_size_mb = file_size_bytes / (1024 * 1024)  # Convert to MB
                file_sizes_mb.append(file_size_mb)
            else:
                zero_byte_files_count += 1  # Count zero-byte files
        except FileNotFoundError:
            # In case a file is deleted during execution or inaccessible
            continue

# Print the number of valid (non-zero) files
print(f"Number of non-zero files: {len(file_sizes_mb)}")
print(f"Number of zero-byte files: {zero_byte_files_count}")

# Plotting the histogram of file sizes in megabytes
plt.figure(figsize=(10, 6))
plt.hist(file_sizes_mb, bins=150, edgecolor="black", color="skyblue")
plt.title("Histogram of File Sizes in Images Directory (MB)")
plt.xlabel("File Size (MB)")
plt.ylabel("Frequency")

# Save the plot to a file
plt.savefig(
    "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/file_sizes_histogram_mb.png"
)

# Optional: Close the plot after saving
plt.close()
