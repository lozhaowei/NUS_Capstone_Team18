import pandas as pd
from src.data.database import query_database, insert_data, CONN_PARAMS, combine_tables_video, combine_tables_convo
from src.data.make_datasets import pull_raw_data
from src.video_recommend.knn import run_knn_recommender,get_num_cycles
from src.video_recommend.random_forest import run_model
from src.conversation_recommend.cosine_similarity import run_collaborative_recommender
from src.conversation_recommend.random_forest_convo import run_model_convo
import schedule
import time 
conversation_like = pd.read_feather("datasets/raw/conversation_like.feather")


def main():
    # pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
    #                 'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
    #                 'user', 'user_interest', 'video', 'vote'])

    # knn_eval_video = run_knn_recommender('2023-07-01', 10, get_num_cycles('2023-07-01'))
    # print(knn_eval_video)
    # random_forest_eval_video = run_model()
    # print(random_forest_eval_video)

    # time.sleep(5)

    # combine_tables_video()
    # combined_data = pd.read_csv("datasets/final/nus_video_eval.csv")
    # insert_data("nus_video_eval", combined_data)

    knn_eval_convo = run_collaborative_recommender('2023-09-02', 3, 4, conversation_like)
    print(knn_eval_convo)
    random_forest_eval_convo = run_model_convo()
    print(random_forest_eval_convo)

    combine_tables_convo()
    combined_data_2 = pd.read_csv("datasets/final/nus_convo_eval.csv")
    insert_data("nus_convo_eval", combined_data_2)

    # get dashboard metrics (commented out because i transferred this directly to the dashboard)
    # get_dashboard_data()

if __name__ == "__main__":
    schedule.every().day.at("02:34").do(main)

while True:
    schedule.run_pending()
    time.sleep(10)