import pandas as pd
from src.data.database import query_database, insert_data, CONN_PARAMS, combine_tables
from src.data.make_datasets import pull_raw_data, get_dashboard_data
from src.video_recommend.knn import run_knn_recommender
from src.video_recommend.random_forest import run_Model


def main():
    # pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
    #                 'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
    #                 'user', 'user_interest', 'video', 'vote'])

    # nus_knn_eval = run_knn_recommender('2023-08-01', 10, 32)
    # print(nus_knn_eval)
    # nus_random_forest_eval = run_Model()
    # print(nus_random_forest_eval)

    eval_table = combine_tables()
    combined_data = pd.read_csv(eval_table)
    # insert_data("nus_video_eval", combined_data)


    # get dashboard metrics
    # get_dashboard_data()

if __name__ == "__main__":
    main()