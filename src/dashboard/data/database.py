import pymysql
import pandas as pd
import streamlit as st
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
    queries database to obtain metrics data of specific model, renames the columns for frontend use
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

def get_upvote_percentage_for_day(table_name, interacted_entity, interacted_pct):
    """
    Queries the upvoted entity summary table (nus_rs_video_upvote, nus_rs_conversation_like)
    Chooses the most recent updated entry for each recommendation date

    :param table_name: upvoted entity summary table
    :param interacted_entity: column representing the interacted entity in the table (upvoted_videos / liked_conversations)
    :param interacted_pct: column representing the liked entity in the table (upvote_percentage / like_percentage)
    :return: upvoted entity summary table
    """
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        query = f"""
        WITH ranked_like_ratio AS (
            SELECT *,
                   ROW_NUMBER()
                           OVER (PARTITION BY recommendation_date ORDER BY dt DESC) AS rn
            FROM {table_name}
        )
        SELECT {interacted_pct}, {interacted_entity}, number_recommended, recommendation_date, dt  
        FROM ranked_like_ratio WHERE rn = 1;
        """

        cursor.execute(query)
        result = cursor.fetchall()
        df = pd.DataFrame(result, columns=[i[0] for i in cursor.description])
        df["recommendation_date"] = pd.to_datetime(df["recommendation_date"]).dt.date

        conn.close()

        return df

    except Exception as e:
        print("Error, ", e)

def insert_model_feedback(data):
    """
    only inserts one row at a time into nus_model_feedback table
    :param data: dictionary - column names are keys: rating, feedback, model, recommended_item, user_id, role
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

        insert_query = f"INSERT INTO nus_model_feedback " \
                       f"(rating, feedback, model, recommended_item, user_id, role) " \
                       f"VALUES (%(rating)s, %(feedback)s, %(model)s, %(recommended_item)s, %(user_id)s, %(role)s)"
        print(insert_query)
        cursor.execute(insert_query, data)

        conn.commit()
        conn.close()

        print(f"Data inserted into MySQL table nus_model_feedback successfully.")
        return 0

    except Exception as e:
        print("Error:", e)
        return 1

def get_model_ratings(recommended_item):
    """
    calculates the average rating for each model based on the recommended item
    :param recommended_item: recommended item: video, conversation, must follow the format: lowercase, singular
    :return: returns a dictionary: keys - model name, rating - average rating
    if error fetching data, returns an empty dictionary
    """
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        query = '''
                SELECT model, FORMAT(AVG(rating), 2) FROM nus_model_feedback 
                WHERE recommended_item = %s
                GROUP BY model;
                '''

        cursor.execute(query, recommended_item)
        result = cursor.fetchall()

        conn.close()

        # return result example: (('knn', '4.25'), ('random_forest', '4.50'))
        return {model: rating for (model, rating) in result}

    except Exception as e:
        print("Error:", e)
