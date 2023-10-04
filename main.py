import pandas as pd
from src.data.database import query_database, insert_data, CONN_PARAMS, combine_tables
from src.data.make_datasets import pull_raw_data
from src.video_recommend.knn import run_knn_recommender
from src.video_recommend.random_forest import run_model
import schedule 
import time 


def main():
    print("process starting")
    pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
                    'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
                    'user', 'user_interest', 'video', 'vote'])

    time.sleep(5)

    nus_knn_eval = run_knn_recommender('2023-07-01', 10, 32)
    print(nus_knn_eval)

    time.sleep(5)

    nus_random_forest_eval = run_model()
    print(nus_random_forest_eval)

    time.sleep(5)

    combine_tables()
    combined_data = pd.read_csv("datasets/final/nus_video_eval.csv")
    insert_data("nus_video_eval", combined_data)

    # get dashboard metrics (commented out because i transferred this directly to the dashboard)
    # get_dashboard_data()

if __name__ == "__main__":
    schedule.every().day.at("00:47").do(main)

while True:
    schedule.run_pending()
    time.sleep(10)