import pymysql
import pandas as pd
from decouple import config

from src.data import database

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}

def get_dashboard_data(entity):
    """
    queries database to obtain metrics data of specific model, renames the columns for frontend use,
    then writes it as a feather file
    :param entity: recommended item
    """
    try:
        query = f"SELECT * FROM nus_{entity}_eval"
        df = database.query_database(query)
        df["dt"] = pd.to_datetime(df["dt"])

        # Rename columns for frontend use
        df.rename(columns={'roc_auc_score': 'ROC AUC Score', 'accuracy': 'Accuracy', 'precision': 'Precision',
                           'recall': 'Recall', 'f1_score': 'F1 Score', 'hit_ratio_k': 'HitRatio@K',
                           'ndcg_k': 'NDCG@K'}, inplace=True)

        return df

    except Exception as e:
        print("Error:", e)

def insert_model_feedback(data):
    """
    only inserts one row at a time into nus_model_feedback table
    :param data: dictionary - column names are keys: rating, feedback, model, recommended_item
    :return: 0 if success, 1 if failed
    """
    if data is None:
        log.error("Error getting feedback data")
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
        print("Error:", e)
        return 1
