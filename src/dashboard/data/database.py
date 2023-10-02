import pymysql
import logging as log
from decouple import config

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}

def insert_model_feedback(data):
    """
    only inserts one row at a time into nus_model_feedback table
    :param data: dictionary - column names are keys: rating, feedback, model, recommended_item
    :return: 0 if success, 1 if failed
    """
    if data is None:
        print("Error getting feedback data")
        return 1

    if len(data['feedback']) > 500:
        print("Feedback length cannot exceed 500 char")
        return 1

    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        insert_query = f"INSERT INTO nus_model_feedback (rating, feedback, model, recommended_item) " \
                       f"VALUES ({data['rating']}, '{data['feedback']}', '{data['model']}', " \
                       f"'{data['recommended_item']}')"

        print(insert_query)
        cursor.execute(insert_query, data)

        conn.commit()
        conn.close()

        print(f"Data inserted into MySQL table nus_model_feedback successfully.")
        return 0

    except Exception as e:
        log.error(e)
        return 1
