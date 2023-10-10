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

@st.cache_data
def get_data_for_real_time_section_videos(recommendation_table_name):
    """
    Queries database for latest 3 dates in the list of recommendations produced by the model, for use in the "Latest Model Metrics" section.
    Args:
    :param recommendation_table_name: Name of table in AWS database to query from. Table should contain the recommendations produced by the relevant model,
    together with the date it was produced at.
    """
    try:
        recommendation_query = f"SELECT * FROM {recommendation_table_name} ORDER BY created_at DESC LIMIT 5000"
        df = database.query_database(recommendation_query)
        df["created_at"] = pd.to_datetime(df["created_at"])

        video_query = "SELECT * FROM video"
        video_df = database.query_database(video_query)
        video_df["created_at"] = pd.to_datetime(video_df["created_at"]).dt.date

        season_query = "SELECT * FROM season"
        season_df = database.query_database(season_query)

        vote_query = "SELECT * FROM vote"
        vote_df = database.query_database(vote_query)
        vote_df["timestamp"] = pd.to_datetime(vote_df["timestamp"]).dt.date

        user_interest_query = "SELECT * FROM user_interest"
        user_interest_df = database.query_database(user_interest_query)
        user_interest_df["updated_at"] = pd.to_datetime(user_interest_df["updated_at"]).dt.date

        return df, video_df, season_df, vote_df, user_interest_df

    except Exception as e:
        print("Error, ", e)

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

        query = f"SELECT model, FORMAT(AVG(rating), 2) FROM nus_model_feedback " \
                f"WHERE recommended_item = %s" \
                f"GROUP BY model;"

        cursor.execute(query, recommended_item)
        result = cursor.fetchall()

        conn.close()

        # return result example: (('knn', '4.25'), ('random_forest', '4.50'))
        return {model: rating for (model, rating) in result}

    except Exception as e:
        print("Error:", e)
