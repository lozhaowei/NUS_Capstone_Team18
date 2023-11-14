import pandas as pd
import warnings
from typing import Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ndcg_score
import numpy as np
import random
from datetime import datetime, timedelta
# from src.data.database import CONN_PARAMS, insert_data
from decouple import config

warnings.filterwarnings('ignore')

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}
def get_end_date() -> str:
    # Calculate end date as 2 weeks before today
    today = datetime.now()
    end_date = (today - timedelta(weeks=2)).strftime('%Y-%m-%d')
    return end_date

def create_df(user_interest_df: pd.DataFrame, user_df: pd.DataFrame, season_df: pd.DataFrame, 
                              video_df: pd.DataFrame, vote_df: pd.DataFrame, date: datetime):
    
    #Create user interest matrix
    user_interest_df = user_df["id"].to_frame().merge(user_interest_df[user_interest_df["updated_at"] <= date], left_on="id", right_on="user_id", how="left", suffixes=["_user", "_interest"])
    user_interest_df["count"] = 1
    user_interest_df = user_interest_df.pivot(index="id", columns="name", values="count")
    user_interest_df = user_interest_df.loc[:, user_interest_df.columns.notna()].fillna(0)

    #Filter out 1 month of upvote data created before training date and create user upvote category matrix
    upvotes_df = vote_df[(vote_df["created_at"] > date - timedelta(weeks=4)) & (vote_df["created_at"] <= date)].groupby(["voter_id", "video_id"])["id"].nunique().reset_index(name="upvotes")
    upvotes_df = upvotes_df.merge(video_df[["id", "season_id", "created_at"]], left_on="video_id", right_on="id", how="left", suffixes=["", "_video"])
    upvotes_category_df = upvotes_df.merge(season_df[["id", "category"]], left_on="season_id", right_on="id", how="left", suffixes=["_video", "_season"])
    upvotes_category_df = upvotes_category_df.pivot_table(columns="category", index="voter_id", values="upvotes", aggfunc=sum, fill_value=0)
    upvotes_category_df["CRYPTO"] = 0
    upvotes_category_df["FINANCE"] = 0

    #add user interest and user upvote category matrix in a 4:6 ratio and remove users without any values
    user_interest_df = user_interest_df.map(lambda x: x*0.4).add(upvotes_category_df.map(lambda x: x*0.6), fill_value=0)
    user_interest_df.reset_index(names="user_id", inplace=True)
    user_interest_df = user_interest_df[user_interest_df.loc[:, user_interest_df.columns != "user_id"].sum(1) > 0]

    #Join each user to all of their upvoted videos
    interaction_df = user_interest_df.merge(upvotes_df[["voter_id", "video_id", "upvotes"]], left_on="user_id", right_on="voter_id", how="left", suffixes=["_user", "_upvotes"])
    interaction_df.drop(columns="voter_id", inplace=True)
    interaction_df.dropna(subset="video_id", axis=0, inplace=True)
    
    #Get category of all videos
    video_category_df = video_df[["id", "created_at", "season_id"]].merge(season_df[["id", "category"]], left_on="season_id", right_on="id", how="left", suffixes=["_video", "_season"])

    #Get category of all videos upvoted
    interaction_df = interaction_df.merge(video_category_df, left_on="video_id", right_on="id_video", how="left")
    interaction_df.drop(columns=["id_video", "season_id", "id_season"], inplace=True)
    interaction_df.set_index(["user_id", "video_id"], inplace=True)
    
    #Create test dataset, which is all possible combination of filtered users and videos created after training split date
#     test_df = user_interest_df.merge(video_category_df[(video_category_df["created_at"] > date) & (video_category_df["created_at"] <= get_end_date())], how="cross")
    test_df = user_interest_df.merge(video_category_df, how="cross")
    test_df.drop(columns=["season_id", "id_season"], inplace=True)
    test_df.rename(columns={"id_video": "video_id"}, inplace=True)
    test_df.set_index(["user_id", "video_id"], inplace=True)

    #encode video categories
    enc = LabelEncoder()
    enc.fit(video_category_df["category"].to_frame())
    interaction_df["category"] = enc.transform(interaction_df["category"].to_frame())
    test_df["category"] = enc.transform(test_df["category"].to_frame())
    interaction_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    return interaction_df, test_df

def find_top_k_videos(user_id, k, prediction_df):
    return prediction_df[prediction_df.index.get_level_values("user_id") == user_id].nlargest(k, "prediction")

def hit_ratio_at_k(y_true, y_pred, K):
    top_k_indices = np.argsort(-np.array(y_pred))[:K]
    return int(any(y_true[i] == 1 for i in top_k_indices))  # 1 if at least one relevant item is in top-K, 0 otherwise

def ndcg_at_k(y_true, y_pred, K):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    return ndcg_score([y_true], [y_pred], k=K)

def get_summary_statistics(vote_df, video_df, train_df, test_df, date, K):
    train_df.drop(columns="created_at", inplace=True)
    test_df.drop(columns="created_at", inplace=True)
    vote_test = vote_df[vote_df["created_at"] > date]
    vote_test['created_at'] = vote_test['created_at'].dt.date
    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])
    model = RandomForestRegressor(n_estimators=50, max_features=8, max_samples=0.7)
    model.fit(train_df.loc[:, train_df.columns != "upvotes"], train_df["upvotes"])
    test_df["prediction"] = model.predict(test_df)

    #find voted videos for each day after the training split date
    for day in sorted(vote_test['created_at'].unique()):
        print(day)
        voted_videos_for_day = vote_test[vote_test['created_at'] == day]
        summary_statistics = pd.DataFrame(columns=['user_id', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])
        voted_videos_for_day = voted_videos_for_day[voted_videos_for_day["voter_id"].isin(train_df.index.get_level_values("user_id"))]
        voted_videos_for_day = voted_videos_for_day[voted_videos_for_day["video_id"].isin(test_df.index.get_level_values("video_id"))]

        for user_id in voted_videos_for_day['voter_id'].unique():
            #Get voted videos for each user
            user_voted_videos = voted_videos_for_day[voted_videos_for_day['voter_id'] == user_id]
            
            
            #create dataframe to calculate confusion matrix
            y_true_and_pred = pd.DataFrame(index=test_df.index.get_level_values("video_id").unique())
            y_true_and_pred['true'] = np.where(y_true_and_pred.index.isin(user_voted_videos['video_id']), 1, 0)

            recommendations = find_top_k_videos(user_id, 20, test_df)
            y_true_and_pred['pred'] = np.where(y_true_and_pred.index.isin(recommendations.index.get_level_values("video_id")), 1, 0)

            try:
                roc_auc = roc_auc_score(y_true_and_pred['true'], y_true_and_pred['pred'])
                accuracy = accuracy_score(y_true_and_pred['true'], y_true_and_pred['pred'])
                precision = precision_score(y_true_and_pred['true'], y_true_and_pred['pred'])
                recall = recall_score(y_true_and_pred['true'], y_true_and_pred['pred'])
                f1 = f1_score(y_true_and_pred['true'], y_true_and_pred['pred'])
                hit_ratio = hit_ratio_at_k(y_true_and_pred['true'], y_true_and_pred['pred'], K)
                ndcg = ndcg_at_k(y_true_and_pred['true'], y_true_and_pred['pred'], K)
            except ValueError:
                print(f'ROC AUC for {user_id} not valid')

            summary_statistics.loc[len(summary_statistics)] = [user_id, roc_auc, accuracy, precision, recall, f1, hit_ratio, ndcg]
        
        model_statistics.loc[len(model_statistics)] = np.append(np.array(day), summary_statistics.iloc[:,1:].mean().values)

    return model_statistics

def run_random_forest(date, K):
    user_interest_df = pd.read_feather('datasets/raw_new/user_interest.feather')
    user_df = pd.read_feather('datasets/raw_new/user.feather')
    season_df = pd.read_feather('datasets/raw_new/season.feather')
    video_df = pd.read_feather('datasets/raw_new/video.feather')
    vote_df = pd.read_feather('datasets/raw_new/vote.feather')

    date = pd.to_datetime(date)

    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])
    train_df, test_df = create_df(user_interest_df, user_df, season_df, video_df, vote_df, date)
    model_statistics_for_training_cycle = get_summary_statistics(vote_df, video_df, train_df, test_df, date, K)
    model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])

    model_statistics['model'] = 'random_forest'
    model_statistics.to_csv('datasets/final_new/random_forest_video.csv', index=False)
