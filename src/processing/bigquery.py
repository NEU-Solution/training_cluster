from google.cloud import bigquery
import math
import os
import pandas as pd
import json
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from processing.utils import process_messages


credentials_path = os.path.join(current_dir, "../../neusolution.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

class BigQueryChunkedDownloader:
    def __init__(self, project_id, dataset_id, table_id, chunk_size=5000, hard_limit=1000000):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.chunk_size = chunk_size
        self.hard_limit = hard_limit



    def _get_total_row_count(self):
        query = f"""
            SELECT COUNT(*) AS total
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        """
        return self.client.query(query).result().to_dataframe().iloc[0]["total"]

    def fetch_chunks(self, order_by_column="id", custom_query=None):
        all_rows = self._get_total_row_count()


        total_rows = min(all_rows, self.hard_limit)
        rows_per_chunk = max(1, self.chunk_size)
        num_chunks = math.ceil(total_rows / rows_per_chunk)

        print(f"Total selected rows: {total_rows} in {all_rows} rows")
        print(f"Fetching in {num_chunks} chunks of ~{rows_per_chunk} rows each...")

        for chunk in range(num_chunks):
            start = chunk * rows_per_chunk + 1
            end = start + rows_per_chunk - 1
            query = f"""
    WITH outer_rows AS (
      SELECT
        ROW_NUMBER() OVER (ORDER BY {order_by_column}) AS rn,
        id, version, messages
      FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
    )
    SELECT *
    FROM outer_rows
    WHERE rn BETWEEN {start} AND {end}
            """

            query = custom_query if custom_query else query

            df = self.client.query(query).result()
            result = [dict(row) for row in df]
            yield result




def bigquery_fetch_data(version, template = None):

    size_per_file = 50000

    project_id = "neusolution"
    dataset_id = "message"
    
    if template == 'qwen3':
        table_ids = ['thinking_huggingface', 'sft_huggingface']
    elif template == 'r1':
        table_ids = ['thinking_huggingface']
    else:
        table_ids = ['sft_huggingface']


    # Need version to get exact table

    # Test
    chunk_size = 5
    hard_limit = 20

    data_files = []


    for table_id in table_ids:
        print(table_id)
        downloader = BigQueryChunkedDownloader(project_id, dataset_id, table_id, chunk_size, hard_limit)

        counter = 1
        len_processed = 0

        for chunk in downloader.fetch_chunks(order_by_column="date_added"):
            
            # Process messages
            chunk = process_messages(chunk, template)

            print(len(chunk))
            
            save_path = os.path.join(current_dir, f"../../LLaMA-Factory/data/{table_id}_{counter}.jsonl")

            # New patch
            if len_processed == 0:
                with open(save_path, 'w') as f:
                    for row in chunk:
                        f.write(json.dumps(row) + '\n')
            
            elif len_processed + len(chunk) > size_per_file:
                counter += 1
                save_path = os.path.join(current_dir, f"../../LLaMA-Factory/data/{table_id}_{counter}.jsonl")
                with open(save_path, 'w') as f:
                    for row in chunk:
                        f.write(json.dumps(row) + '\n')
                len_processed = 0
            else:
                with open(save_path, 'a') as f:
                    for row in chunk:
                        f.write(json.dumps(row) + '\n')
            len_processed += len(chunk)

            data_files.append(save_path)

    return data_files
