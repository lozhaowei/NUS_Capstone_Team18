import pymysql
import pandas as pd
from decouple import config

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

    finally:
        if conn.is_connected():
            conn.close()
            print('MySQL connection is closed')

def insert_data(table_name, data):
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Create a table if it doesn't exist with backticks for column names
        create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            dt DATETIME,
            `roc_auc_score` FLOAT,
            accuracy FLOAT,
            `precision` FLOAT,
            recall FLOAT,
            `f1_score` FLOAT,
            `hit_ratio_k` FLOAT,
            `ndcg_k` FLOAT
        )
        '''
        cursor.execute(create_table_query)

        data_values = ', '.join(['%s'] * len(data.columns))
        insert_query = f'INSERT INTO {table_name} VALUES ({data_values})'
        cursor.executemany(insert_query, data.values.tolist())

        conn.commit()
        conn.close()

        print(f"Data updated in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)
