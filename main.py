import pandas as pd
from src.data.database import query_database, insert_data, CONN_PARAMS, combine_tables_video, combine_tables_convo
from src.data.make_datasets import pull_raw_data
from src.video_recommend.knn import run_knn_recommender,get_num_cycles
from src.video_recommend.svd import run_svd_recommender
from src.video_recommend.random_forest import run_random_forest
from src.video_recommend.neural_networks import run_ncf
from src.conversation_recommend.cosine_similarity import run_collaborative_recommender
from src.conversation_recommend.random_forest_convo import run_model_convo
import schedule
import time 


def main():
    # Step 1: pull data from database
    pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
                    'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
                    'user', 'user_interest', 'video', 'vote'])

    # Step 2: Run the 3 models for Video Recommendations
    knn_eval_video = run_knn_recommender('2023-07-01', 10, get_num_cycles('2023-07-01'))
    print(knn_eval_video)


    random_forest_eval_video = run_random_forest('2023-07-01', 10, get_num_cycles('2023-07-01'))
    print(random_forest_eval_video)


    run_svd_recommender('2023-07-01', 10, get_num_cycles('2023-07-01'))

    run_ncf('2023-08-01')
    
    # Step 3: Combine the 4 evaluation tables into 1 mega table
    combine_tables_video()
    combined_data = pd.read_csv("datasets/final/nus_video_eval.csv")
    insert_data("nus_video_eval", combined_data)

    # Step 4: Run the 3 models for Conversations Recommendations
    conversation_like = pd.read_feather("datasets/raw/conversation_like.feather")
    conversation_categories = pd.read_feather("datasets/final/conversation_with_categories.feather")
    knn_eval_convo = run_collaborative_recommender('2023-09-02', 10, 4, conversation_like, conversation_categories)
    print(knn_eval_convo)
    random_forest_eval_convo = run_model_convo()
    print(random_forest_eval_convo)

    # Step 5: Combine the 3 evaluation tables into 1 mega table
    combine_tables_convo()
    combined_data_2 = pd.read_csv("datasets/final/nus_convo_eval.csv")
    insert_data("nus_convo_eval", combined_data_2)

    # get dashboard metrics (commented out because i transferred this directly to the dashboard)
    # get_dashboard_data()

if __name__ == "__main__":
    main()
    schedule.every().day.at("21:46").do(main)

while True:
    schedule.run_pending()
    time.sleep(3)