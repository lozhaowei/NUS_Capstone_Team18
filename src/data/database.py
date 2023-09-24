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

def insert_data(table_name, data):
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create a new table with the same schema
        data.to_sql(name=table_name, con=conn, index=False)

        conn.commit()
        conn.close()

    except Exception as e:
        print("Error:", e)

def table_exists(cursor, table_name):
    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    result = cursor.fetchone()
    return result is not None
