import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import PIL
from PIL import Image
from tqdm import tqdm

# Initialize a global lock
lock = Lock()


def process_image(file_path, error_log_path):
    results = []
    fname = Path(file_path).name

    try:
        print("img opened")
        img = open(file_path, "rb").read()
        if len(img) == 0:
            results.append([fname, "len(img) == 0"])
            return results
        i = Image.open(file_path)
    except PIL.UnidentifiedImageError:
        results.append([fname, "PIL.UnidentifiedImageError"])
        return results
    except OSError:
        results.append([fname, "OSError, truncated"])
        return results
    except ValueError as e:
        if "Unsupported SGI image mode" in str(e):
            results.append([fname, "Unsupported SGI image mode"])
        return results
    except Exception as e:
        results.append([fname, str(e)])
        return results

    try:
        print("bytestream opened")
        i = i.tobytes("xbm", "RGB")
    except OSError:
        results.append([fname, "OSError (bytestream), truncated"])
    except Exception as e:
        results.append([fname, str(e)])

    return results


def record_corrupt_images(base_dir, num_workers):
    error_log_path = "corrupt_images.csv"

    # Initialize the CSV file
    with open(error_log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Error Reason"])

    # Gather all file paths directly in the base directory
    file_paths = [entry.path for entry in os.scandir(base_dir) if entry.is_file()]

    # Process files in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_image, file_path, error_log_path): file_path
            for file_path in file_paths
        }

        # Use tqdm to display the progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Files"
        ):
            results = (
                future.result()
            )  # This will also raise any exceptions caught during the execution

            # Write results to the CSV file in a thread-safe manner
            if results:
                with lock:
                    with open(error_log_path, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(results)


if __name__ == "__main__":
    base_dir = "/home/neal.nevyn/blue_guralnick/share/gbif_data/data/images"
    record_corrupt_images(base_dir, num_workers=20)
