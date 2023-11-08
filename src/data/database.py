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

        # Check if the table exists
        table_exists_query = f"SHOW TABLES LIKE '{table_name}'"
        cursor.execute(table_exists_query)
        table_exists = cursor.fetchone()

        if not table_exists:
            # If the table doesn't exist, create a new table with specified columns
            create_table_query = f'''
            CREATE TABLE {table_name} (
                `dt` DATE,
                `roc_auc_score` FLOAT,
                `accuracy` FLOAT,
                `precision` FLOAT,
                `recall` FLOAT,
                `f1_score` FLOAT,
                `hit_ratio_k` FLOAT,
                `ndcg_k` FLOAT,
                `model` VARCHAR(255),
                PRIMARY KEY (`dt`, `model`)
            )
            '''
            cursor.execute(create_table_query)
            print(f"Table '{table_name}' created successfully.")

        # Convert 'dt' column to string before insertion
        data['dt'] = data['dt'].astype(str)
        # Convert NaN values to None for proper insertion
        data = data.where(pd.notna(data), None)

        # Insert data only if the combination of 'dt' and 'model' is unique
        for _, row in data.iterrows():
            insert_query = f'''
            INSERT INTO {table_name} (`dt`, `roc_auc_score`, `accuracy`, `precision`, `recall`, `f1_score`, `hit_ratio_k`, `ndcg_k`, `model`) 
            SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s 
            WHERE NOT EXISTS (
                SELECT 1 FROM {table_name} WHERE `dt` = %s AND `model` = %s
            )
            ON DUPLICATE KEY UPDATE
            `roc_auc_score` = VALUES(`roc_auc_score`),
            `accuracy` = VALUES(`accuracy`),
            `precision` = VALUES(`precision`),
            `recall` = VALUES(`recall`),
            `f1_score` = VALUES(`f1_score`),
            `hit_ratio_k` = VALUES(`hit_ratio_k`),
            `ndcg_k` = VALUES(`ndcg_k`)
            '''
            cursor.execute(insert_query, (row['dt'], row['roc_auc_score'], row['accuracy'], row['precision'],
                                         row['recall'], row['f1_score'], row['hit_ratio_k'], row['ndcg_k'],
                                         row['model'], row['dt'], row['model']))

        conn.commit()
        conn.close()

        print(f"Data updated in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)

def combine_tables_video():
    table1 = pd.read_csv('datasets/final_new/random_forest_video.csv')
    table2 = pd.read_csv('datasets/final_new/knn_video.csv')
    table3 = pd.read_csv('datasets/final_new/svd_video.csv')
    # table4 = pd.read_csv('datasets/final_new/ncf_video.csv')

    # Combine tables
    combined_table = pd.concat([table1, table2, table3], ignore_index=True)

    output_folder = 'datasets/final_new'
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


