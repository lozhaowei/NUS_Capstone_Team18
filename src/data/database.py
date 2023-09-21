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