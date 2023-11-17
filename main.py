import pandas as pd
from src.data.database import query_database, insert_data, CONN_PARAMS, combine_tables_video, combine_tables_convo, clean_csv
from src.data.make_datasets import pull_raw_data, pull_raw_video_data, pull_latest_data_and_combine
from src.video_recommend.knn import run_knn_recommender,get_num_cycles
from src.video_recommend.svd import run_svd_recommender
from src.video_recommend.random_forest import run_random_forest
from src.video_recommend.neural_networks import run_ncf
from src.conversation_recommend.cosine_similarity import run_collaborative_recommender
from src.conversation_recommend.alternating_least_squares import run_als_recommender
from src.dashboard.data.spark_pipeline import SparkPipeline
import schedule
import threading
import time 
import os
from datetime import datetime
from datetime import timedelta

def main():    

    ######### RUN STEP 1 AND 2 ONLY ONCE. THEN COMMENT IT OUT AND DO NOT RUN AGAIN ON A DAILY BASIS #########
       
    # Step 1: pull data from database - keep it commented out
    # pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
    #                 'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
    #                 'user', 'user_interest', 'video', 'vote'])

    # Step 2: pull video datasets - this is only needed for the first training iteration
    # pull_raw_video_data(['post_feed', 'season', 'user', 'user_interest', 'video', 'vote'])

    # ######### EVERYTHING BELOW THIS WILL BE RUN EVERYDAY EVERYDAY THROUGH THE SCHEDULER #########

    today_date = datetime.now()
    start_date = today_date - timedelta(days=89)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # Step 3: Extracting the latest Video Data 
    list_of_v_tables = ['user_interest', 'season', 'video', 'user', 'vote']
    existing_v_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'raw_new')
    latest_v_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'Last-24hour-data')
    pull_latest_data_and_combine(list_of_v_tables, existing_v_data_dir, latest_v_data_dir)

    # Step 4: Run KNN (Video) Model
    knn_eval_video = run_knn_recommender(start_date_str, 3, get_num_cycles(start_date_str))
    print(knn_eval_video)

    # Step 5: Run Random Forest (Video) Model
    random_forest_eval_video = run_random_forest(start_date_str, 10, get_num_cycles(start_date_str))
    print(random_forest_eval_video)

    # Step 6: Run SVD (Video) Model
    run_svd_recommender(start_date_str, 10, get_num_cycles(start_date_str))

    # Step 7: Run NCF (Video) Model
    run_ncf(start_date_str)
    
    # Step 8: Combine the 4 evaluation tables into 1 mega table
    combine_tables_video()

    # Step 9: Send the combined table into the DB
    clean_csv("datasets/final_new/nus_video_eval_2.csv", "datasets/final_new/nus_video_eval_2.csv")
    combined_data = pd.read_csv("datasets/final_new/nus_video_eval_2.csv")

    insert_data("nus_video_eval_2", combined_data)

    # Step 10: Extracting the latest Conversation Data 
    list_of_c_tables = ['conversation_like']
    existing_c_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'raw_new')
    latest_c_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'Last-24hour-data')
    pull_latest_data_and_combine(list_of_c_tables, existing_c_data_dir, latest_c_data_dir)

    # Step 11: Run the 3 models for Conversations Recommendations
    # conversation_like = pd.read_feather("datasets/raw/conversation_like.feather")
    # conversation_categories = pd.read_feather("datasets/final/conversation_with_categories.feather")
    knn_eval_convo = run_collaborative_recommender(start_date_str, 10, get_num_cycles(start_date_str), conversation_like)
    als_eval_convo = run_als_recommender(start_date_str, 10, get_num_cycles(start_date_str), conversation_like)
    # print(knn_eval_convo)
    # random_forest_eval_convo = run_model_convo()
    # print(random_forest_eval_convo)

    # Step 12: Combine the 3 evaluation tables into 1 mega table
    combine_tables_convo()
    clean_csv("datasets/final_new/nus_convo_eval_2.csv", "datasets/final_new/nus_convo_eval_2.csv")
    combined_data_2 = pd.read_csv("datasets/final_new/nus_convo_eval_2.csv")
    insert_data("nus_convo_eval_2", combined_data_2)

def dashboard_video_spark_job():
    print("Start Spark video like job %s" % threading.current_thread())
    spark_pipeline = SparkPipeline()
    spark_pipeline.initialize_spark_session()
    spark_pipeline.run_video_upvote_percentage_pipeline()
    spark_pipeline.close_spark_session()

def dashboard_conversation_spark_job():
    print("Start Spark conversation like job %s" % threading.current_thread())
    spark_pipeline = SparkPipeline()
    spark_pipeline.initialize_spark_session()
    spark_pipeline.run_conversation_like_percentage_pipeline()
    spark_pipeline.close_spark_session()

def run_threaded(job):
    job_thread = threading.Thread(target=job)
    job_thread.start()

if __name__ == "__main__":
    schedule.every().day.at("00:00").do(main)
    schedule.every().day.at("00:00").do(run_threaded, dashboard_video_spark_job)
    schedule.every().day.at("00:00").do(run_threaded, dashboard_conversation_spark_job)
    schedule.every().day.at("06:00").do(run_threaded, dashboard_video_spark_job)
    schedule.every().day.at("06:00").do(run_threaded, dashboard_conversation_spark_job)
    schedule.every().day.at("12:00").do(run_threaded, dashboard_video_spark_job)
    schedule.every().day.at("12:00").do(run_threaded, dashboard_conversation_spark_job)
    schedule.every().day.at("18:00").do(run_threaded, dashboard_video_spark_job)
    schedule.every().day.at("18:00").do(run_threaded, dashboard_conversation_spark_job)

while True:
    schedule.run_pending()
    time.sleep(3)
