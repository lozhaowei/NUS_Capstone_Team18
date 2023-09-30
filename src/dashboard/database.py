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

# data is in the form of a dictionary: column names are keys
def insert_model_feedback(data):
    if data is None:
        print("Error getting feedback data")
        return 1

    try:
        # conn = pymysql.connect(**CONN_PARAMS)
        # cursor = conn.cursor()

        insert_query = f"INSERT INTO nus_model_feedback (rating, feedback) " \
                       f"VALUES ({data['rating']}, {data['feedback']})"
        print(insert_query)
        # cursor.execute(insert_query, data)
        #
        # conn.commit()
        # conn.close()

        print(f"Data inserted into MySQL table nus_model_feedback successfully.")
        return 0

    except Exception as e:
        print("Error:", e)
        return 1

    # finally:
    #     if conn.is_connected():
    #         cursor.close()
    #         conn.close()
    #         print('MySQL connection is closed')