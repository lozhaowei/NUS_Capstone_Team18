import os
import pyarrow.feather as feather
from src.data import database

def write_feather_data(table, df, data_dir):
    """ 
    Helper function to create feather files and saving it to the desired directories
    """
    os.makedirs(data_dir, exist_ok=True)

    feather_file_path = os.path.join(data_dir, f'{table}.feather')
    feather.write_feather(df, feather_file_path)

    print(f"Table '{table}' saved as '{feather_file_path}'")

def pull_raw_data(list_of_tables):
    """ 
    Function which pulls raw data from the MySQL Database 
    :param list_of_tables: refers to the list of tables from which we will query all the rows
    """
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
    """ 
    Function which pulls raw video relarted data from the MySQL Database 
    :param list_of_tables: refers to the list of tables from which we will query all the rows
    """
    try:
        for table in list_of_tables:
            query = f"SELECT * FROM {table}"
            df = database.query_database(query)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, '../..', 'datasets', 'raw_new')

            write_feather_data(table, df, data_dir)

    except Exception as e:
        print("Error:", e)


def pull_latest_data_and_combine(list_of_tables, existing_data_dir, latest_data_dir):
    """ 
    Pulls the last 24 hours data for from the specified tables and combines this new data with the existing saved tables 
    :param list_of_tables: refers to the list of tables from which we will query the last 24 hours
    :param existing_data_dir: refers to the path of the existing tables which we want to combine our new data with
    :param latest_data_dir: refers to the path of where we would like to save our combined tables
    """
    try:
        for table in list_of_tables:
            # Define the appropriate datetime column for each table
            datetime_column = {
                'user_interest': 'updated_at',
                'season': 'created_at',
                'video': 'created_at',
                'user': 'updated_at',
                'vote': 'created_at'
            }.get(table)

            if not datetime_column:
                print(f"Error: Datetime column not defined for table {table}")
                continue

            # Fetch the latest date from the existing data
            existing_data_path = os.path.join(existing_data_dir, f'{table}.feather')
            existing_df = pd.read_feather(existing_data_path)

            latest_date = existing_df[datetime_column].max()

            # Fetch only the latest data from the database
            query = f"SELECT * FROM {table} WHERE {datetime_column} = '{latest_date}'"
            latest_data_df = database.query_database(query)

            # Save the latest data to the "datasets/latest" directory
            latest_data_dir_path = os.path.join(latest_data_dir, f'{table}.feather')
            write_feather_data(table, latest_data_df, latest_data_dir)

            # Combine the existing data with the latest data
            combined_df = pd.concat([existing_df, latest_data_df], ignore_index=True)

            # Save the combined data back to the "datasets/raw_new" directory
            combined_data_dir_path = os.path.join(existing_data_dir, f'{table}.feather')
            write_feather_data(table, combined_df, existing_data_dir)

    except Exception as e:
        print("Error:", e)