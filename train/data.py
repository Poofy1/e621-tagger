import pandas as pd
import os
from ast import literal_eval
from tqdm import tqdm
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp


current_dir = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(current_dir)

def count_tags(chunk):
    local_counter = Counter()
    for _, row in chunk.iterrows():
        for tag_column in ['General Tags', 'Copyright Tags', 'Character Tags', 'Species Tags']:
            tags = literal_eval(row[tag_column])
            local_counter.update(tags)
    return local_counter

def convert_tags_to_indices(chunk, tag_to_index):
    new_data = []
    for _, row in chunk.iterrows():
        id = row['ID']
        all_tags = []
        for tag_column in ['General Tags', 'Copyright Tags', 'Character Tags', 'Species Tags']:
            all_tags.extend(literal_eval(row[tag_column]))
        tag_indices = [tag_to_index[tag] for tag in all_tags if tag in tag_to_index]
        valid = row['Valid']
        new_data.append({'ID': id, 'tag_indices': tag_indices, 'Valid': valid})
    return new_data

def convert_chunk(args):
    chunk, tag_to_index = args
    return convert_tags_to_indices(chunk, tag_to_index)

def process_csv(input_file, output_file, tag_map_file, min_occurrences=100, chunk_size=10000):
    print("Loading CSV data...")
    df = pd.read_csv(input_file, usecols=['ID', 'General Tags', 'Copyright Tags', 'Character Tags', 'Species Tags', 'Valid'])

    print("Collecting and counting tags...")
    num_cores = max(1, int(mp.cpu_count() / 2))
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(count_tags, chunks), total=len(chunks), desc="Counting tags"))
    
    tag_counter = sum(results, Counter())

    print(f"Total unique tags: {len(tag_counter)}")

    # Filter tags that appear at least min_occurrences times
    frequent_tags = {tag for tag, count in tag_counter.items() if count >= min_occurrences}
    print(f"Tags appearing at least {min_occurrences} times: {len(frequent_tags)}")

    # Create tag to index mapping for frequent tags
    tag_to_index = {tag: idx for idx, tag in enumerate(sorted(frequent_tags))}

    # Save tag to index mapping as CSV
    tag_map_df = pd.DataFrame([{'tag': tag, 'index': idx} for tag, idx in tag_to_index.items()])
    tag_map_df.to_csv(tag_map_file, index=False)

    print("Creating new dataframe with tag indices...")
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(convert_chunk, [(chunk, tag_to_index) for chunk in chunks]), 
                            total=len(chunks), desc="Converting tags to indices"))
    
    new_data = [item for sublist in results for item in sublist]

    new_df = pd.DataFrame(new_data)

    print("Saving new Parquet file...")
    table = pa.Table.from_pandas(new_df)
    pq.write_table(table, output_file)

    print("Done!")

# Usage
input_file = 'D:/DATA/E621/source_images.csv'
output_file = f'{env}/data/dataset2.parquet'
tag_map_file = f'{env}/data/tag_map.csv'

if __name__ == '__main__':
    process_csv(input_file, output_file, tag_map_file, min_occurrences=100, chunk_size=10000)