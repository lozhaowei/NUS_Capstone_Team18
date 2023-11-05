import pandas as pd
import warnings
from typing import Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
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

def get_num_cycles(start_date: str) -> int:
    # Get today's date
    today_date = datetime.now().strftime('%Y-%m-%d')
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(today_date, '%Y-%m-%d')
    date_difference = (end_datetime - start_datetime).days

    return date_difference

def train_test_split_for_data(data: pd.DataFrame, date_col: str, start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = data[data[date_col] <= start_date,]
    test_data = data[(data[date_col] > start_date) & (data[date_col] > get_end_date())]
    return train_data, test_data

def create_interaction_df(user_interest_df: pd.DataFrame, user_df: pd.DataFrame, season_df: pd.DataFrame, 
                              video_df: pd.DataFrame, vote_df: pd.DataFrame, date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_interest_df = user_df["id"].to_frame().merge(user_interest_df, left_on="id", right_on="user_id", how="left", suffixes=["_user", "_interaction"])
    user_interest_df["count"] = 1
    user_interest_df = user_interest_df.pivot(index="id", columns="name", values="count")
    user_interest_df = user_interest_df.loc[:, user_interest_df.columns.notna()].fillna(0)

    upvotes_df = vote_df[vote_df["created_at"] > datetime.strptime(date, '%Y-%m-%d') - timedelta(weeks=1)].groupby(["voter_id", "video_id"])["id"].nunique().reset_index(name="upvotes")
    upvotes_df = upvotes_df.merge(video_df[["id", "season_id", "created_at"]], left_on="video_id", right_on="id", how="left", suffixes=["", "_video"])

    upvotes_category_df = upvotes_df.merge(season_df[["id", "category"]], left_on="season_id", right_on="id", how="left", suffixes=["_video", "_season"])
    upvotes_category_df = upvotes_category_df.pivot_table(columns="category", index="voter_id", values="upvotes", aggfunc=sum, fill_value=0)
    upvotes_category_df["CRYPTO"] = 0
    upvotes_category_df["FINANCE"] = 0

    user_interest_df = user_interest_df.add(upvotes_category_df, fill_value=0)
    user_interest_df.reset_index(names="user_id", inplace=True)

    interaction_df = user_interest_df.merge(upvotes_df[["voter_id", "video_id", "upvotes"]], left_on="user_id", right_on="voter_id", how="left", suffixes=["_user", "_upvotes"])
    interaction_df.drop(columns="voter_id", inplace=True)
    no_likes = interaction_df[interaction_df["video_id"].isna()]
    interaction_df.dropna(subset="video_id", axis=0, inplace=True)

    no_likes = no_likes.drop(columns=["video_id"]).merge(pd.DataFrame(interaction_df["video_id"].unique(), columns=["video_id"]), how="cross", suffixes=["_no_like", ""])
    no_likes = no_likes.merge(video_df[["id", "created_at", "season_id"]], left_on="video_id", right_on="id", how="left")
    no_likes = no_likes.merge(season_df[["id", "category"]], left_on="season_id", right_on="id", how="left", suffixes=["_video", "_season"])
    no_likes.drop(columns=["id_video", "season_id", "id_season"], inplace=True)
    no_likes.set_index(["user_id", "video_id"], inplace=True)
    no_likes = no_likes[(no_likes["created_at"] > datetime.strptime(date, '%Y-%m-%d') - timedelta(weeks=1)) & (no_likes["created_at"] <= date)]

    interaction_df = interaction_df.merge(video_df[["id", "created_at", "season_id"]], left_on="video_id", right_on="id", how="left")
    interaction_df = interaction_df.merge(season_df[["id", "category"]], left_on="season_id", right_on="id", how="left", suffixes=["_video", "_season"])
    interaction_df.drop(columns=["id_video", "season_id", "id_season"], inplace=True)
    interaction_df.set_index(["user_id", "video_id"], inplace=True)

    interaction_df = pd.concat([interaction_df, no_likes])

    enc = OrdinalEncoder(encoded_missing_value=-1)
    enc.fit(interaction_df["category"].to_frame())
    interaction_df["category"] = enc.transform(interaction_df["category"].to_frame())
    interaction_df.fillna(0, inplace=True)
    
    return interaction_df

def find_top_k_videos(user_id, k, prediction_df):
    return prediction_df.nlargest(k, "prediction")

def hit_ratio_at_k(y_true, y_pred, K):
    top_k_indices = np.argsort(-np.array(y_pred))[:K]
    return int(any(y_true[i] == 1 for i in top_k_indices))  # 1 if at least one relevant item is in top-K, 0 otherwise

def ndcg_at_k(y_true, y_pred, K):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    return ndcg_score([y_true], [y_pred], k=K)

def get_summary_statistics(vote_df, interaction_df, date, K):
    train_df, test_df = train_test_split_for_data(interaction_df, 'created_at', date)
    train_df.drop(columns="created_at", inplace=True)
    test_df.drop(columns="created_at", inplace=True)
    _, vote_test = train_test_split_for_data(vote_df, 'created_at', date)
    vote_test['created_at'] = vote_test['created_at'].dt.date
    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])
    model = RandomForestClassifier()
    model.fit(train_df.loc[:, train_df.columns != "upvotes"], train_df["upvotes"])
    test_df["prediction"] = model.predict(test_df.loc[:, test_df.columns != "upvotes"])

    for day in sorted(vote_test['created_at'].unique()):
        print(day)
        voted_videos_for_day = vote_test[vote_test['created_at'] == day]
        summary_statistics = pd.DataFrame(columns=['user_id', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])

        for user_id in voted_videos_for_day['voter_id'].unique():
            if user_id not in train_df.index:
                continue
            
            # create dataframe to calculate confusion matrix
            user_voted_videos = voted_videos_for_day[voted_videos_for_day['voter_id'] == user_id]
            y_true_and_pred = pd.DataFrame(index=test_df.index.get_level_values("video_id"))
            y_true_and_pred['true'] = np.where(y_true_and_pred.index.isin(user_voted_videos['video_id']), 1, 0)

            recommendations = find_top_k_videos(user_id, 20, test_df)
            y_true_and_pred['pred'] = np.where(y_true_and_pred.index.isin(recommendations.index), 1, 0)

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

def run_random_forest(date, K, num_cycles):
    user_interest_df = pd.read_feather('datasets/raw/user_interest.feather')
    user_df = pd.read_feather('datasets/raw/user.feather')
    season_df = pd.read_feather('datasets/raw/season.feather')
    video_df = pd.read_feather('datasets/raw/video.feather')
    vote_df = pd.read_feather('datasets/raw/vote.feather')

    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])
    # for cycle in range(num_cycles):
    interaction_df = create_interaction_df(user_interest_df, user_df, season_df, video_df, vote_df, date)
    model_statistics_for_training_cycle = get_summary_statistics(vote_df, interaction_df, date, K)
    model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])
    # date = get_end_date()

    model_statistics['model'] = 'random_forest'
    model_statistics.to_csv('datasets/final/random_forest_video.csv', index=False)
# def generate_random_data(start_date, end_date):
#     dates = pd.date_range(start_date, end_date)
#     data = []

#     for date in dates:
#         roc_auc = random.uniform(0.49, 0.55)
#         accuracy = random.uniform(0.987, 0.989)
#         precision = random.uniform(0.0, 0.05)
#         recall = random.uniform(0.0, 0.1)
#         f1_score = random.uniform(0.0, 0.03)
#         hit_ratio_k = random.uniform(0.0, 0.1)
#         ndcg_k = random.uniform(0.0, 0.03)

#         data.append([date.strftime('%Y-%m-%d'), roc_auc, accuracy, precision, recall, f1_score, hit_ratio_k, ndcg_k])

#     return data

# def run_model():
    # start_date = datetime(2023, 7, 1)
    # end_date = datetime.now()
    # num_cycles = (end_date - start_date).days + 1

    # data = []
    # for _ in range(num_cycles):
    #     data += generate_random_data(start_date, start_date)
    #     start_date += timedelta(days=1)

    # columns = ['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k']
    # model_statistics = pd.DataFrame(data, columns=columns)

    # model_statistics['model'] = 'random_forest'
    # model_statistics.to_csv('datasets/final/random_forest_video.csv', index=False)

