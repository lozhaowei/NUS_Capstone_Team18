import pandas as pd
from src.data.database import query_database, insert_data, CONN_PARAMS
from src.data.make_datasets import pull_raw_data
from src.video_recommend.knn import run_knn_recommender


def main():
    # pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
    #                 'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
    #                 'user', 'user_interest', 'video', 'vote'])

    nus_knn_eval = run_knn_recommender('2023-08-01', 10, 32)
    print(nus_knn_eval)

if __name__ == "__main__":
    main()