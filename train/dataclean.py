import os
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp
from functools import partial
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import csv

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def process_row(row, image_dir, invalid_ids):
    if row['ID'] in invalid_ids:
        return None
    image_path = os.path.join(image_dir, f"{row['ID']}.png")
    if is_valid_image(image_path):
        return row
    return None

def filter_parquet_by_valid_images(input_parquet, output_parquet, image_dir, invalid_ids_csv):
    # Ensure the image directory path ends with a slash
    image_dir = os.path.join(image_dir, '')

    # Read or create the invalid IDs set
    if os.path.exists(invalid_ids_csv):
        with open(invalid_ids_csv, 'r') as f:
            invalid_ids = set(f.read().splitlines())
        print(f"Loaded {len(invalid_ids)} invalid IDs from existing CSV")
    else:
        invalid_ids = set()

    # Read the Parquet file
    df = pd.read_parquet(input_parquet)

    # Convert DataFrame to list of dictionaries
    rows = df.to_dict('records')

    # Create a partial function with fixed arguments
    process_row_partial = partial(process_row, image_dir=image_dir, invalid_ids=invalid_ids)

    # Use all available CPU cores
    num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_row_partial, rows),
            total=len(rows),
            desc="Processing rows"
        ))

    # Filter out None results and update invalid_ids
    valid_rows = []
    for row, result in zip(rows, results):
        if result is None:
            invalid_ids.add(row['ID'])
        else:
            valid_rows.append(result)

    # Save the updated invalid IDs to CSV
    with open(invalid_ids_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID'])
        writer.writerows([[id] for id in invalid_ids])
    print(f"Saved {len(invalid_ids)} invalid IDs to {invalid_ids_csv}")

    # Create a new DataFrame from valid rows
    valid_df = pd.DataFrame(valid_rows)

    # Save the filtered DataFrame as a Parquet file
    table = pa.Table.from_pandas(valid_df)
    pq.write_table(table, output_parquet)

    print(f"Filtered Parquet file saved to {output_parquet}")

# Usage
input_parquet = 'F:/CODE/AI/e621-tagger/dataset.parquet'
output_parquet = 'F:/CODE/AI/e621-tagger/dataset2.parquet'
image_dir = 'F:/Temp_SSD_Data/ME621/images_300/'
invalid_ids_csv = 'F:/CODE/AI/e621-tagger/invalid_ids.csv'

if __name__ == '__main__':
    filter_parquet_by_valid_images(input_parquet, output_parquet, image_dir, invalid_ids_csv)