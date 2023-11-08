import os
import pyarrow.feather as feather
from src.data import database

def write_feather_data(table, df, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    feather_file_path = os.path.join(data_dir, f'{table}.feather')
    feather.write_feather(df, feather_file_path)

    print(f"Table '{table}' saved as '{feather_file_path}'")


def pull_raw_data(list_of_tables):
    try:
        for table in list_of_tables:
            query = f"SELECT * FROM {table}"
            df = database.query_database(query)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, '../..', 'datasets', 'raw')

            write_feather_data(table, df, data_dir)

    except Exception as e:
        print("Error:", e)


def pull_raw_video_data(list_of_tables):
    try:
        for table in list_of_tables:
            query = f"SELECT * FROM {table} LIMIT 10000"
            df = database.query_database(query)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, '../..', 'datasets', 'raw_new')

            write_feather_data(table, df, data_dir)

    except Exception as e:
        print("Error:", e)
