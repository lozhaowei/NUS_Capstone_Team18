import pymysql
import pandas as pd
from decouple import config
import os 

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}

def query_database(query):
    try:
        conn = pymysql.connect(**CONN_PARAMS)

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    except Exception as e:
        print("Error:", e)

def insert_data(table_name, data):
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Drop the table if it exists
        drop_table_query = f'DROP TABLE IF EXISTS {table_name}'
        cursor.execute(drop_table_query)

        # Create a table with backticks for column names
        create_table_query = f'''
        CREATE TABLE {table_name} (
            `id` INT PRIMARY KEY AUTO_INCREMENT,
            `dt` DATE,
            `roc_auc_score` FLOAT,
            `accuracy` FLOAT,
            `precision` FLOAT,
            `recall` FLOAT,
            `f1_score` FLOAT,
            `hit_ratio_k` FLOAT,
            `ndcg_k` FLOAT,
            `model` VARCHAR(255)
        )
        '''

        cursor.execute(create_table_query)

        data_values = ', '.join(['%s'] * len(data.columns))
        insert_query = f'INSERT INTO {table_name} (`dt`, `roc_auc_score`, `accuracy`, `precision`, `recall`, `f1_score`, `hit_ratio_k`, `ndcg_k`, `model`) VALUES ({data_values})'

        # Convert 'dt' column to string before insertion
        data['dt'] = data['dt'].astype(str)
        # Convert NaN values to None for proper insertion
        data = data.where(pd.notna(data), None)

        cursor.executemany(insert_query, data.values.tolist())

        conn.commit()
        conn.close()

        print(f"Data updated in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)

def combine_tables_video():
    table1 = pd.read_csv('datasets/final/random_forest_video.csv')
    table2 = pd.read_csv('datasets/final/knn_video.csv')
    table3 = pd.read_csv('datasets/final/neural_networks_video.csv')

    # Combine tables
    combined_table = pd.concat([table1, table2, table3], ignore_index=True)

    output_folder = 'datasets/final'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "nus_video_eval.csv")
    combined_table.to_csv(output_path, index=False)
    
    print(f"Combined data saved to '{output_path}' successfully.")
    return output_path

def combine_tables_convo():
    table1 = pd.read_csv('datasets/final/random_forest_convo.csv')
    table2 = pd.read_csv('datasets/final/knn_convo.csv')

    # Combine tables
    combined_table = pd.concat([table1, table2], ignore_index=True)

    output_folder = 'datasets/final'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "nus_convo_eval.csv")
    combined_table.to_csv(output_path, index=False)
    
    print(f"Combined data saved to '{output_path}' successfully.")
    return output_path


