## Importing Libraries 
import warnings
import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
from scipy import spatial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ndcg_score, roc_auc_score
from src.data import database

warnings.filterwarnings('ignore')

def get_end_date(start_date: str) -> str:
    start_datetime_object = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime_object = start_datetime_object + timedelta(days=3)
    end_date = end_datetime_object.strftime('%Y-%m-%d')
    return end_date

def train_test_split_for_data(data: pd.DataFrame, date_col: str, start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = data[data[date_col] <= start_date]
    test_data = data[(data[date_col] > start_date) & (data[date_col] <= get_end_date(start_date))]
    return train_data, test_data

def create_embedding_matrices(user_interest_df: pd.DataFrame, user_df: pd.DataFrame, season_df: pd.DataFrame, 
                              video_df: pd.DataFrame, vote_df: pd.DataFrame, date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # create the initial user interest matrix
    user_interest_train, _ = train_test_split_for_data(user_interest_df, 'updated_at', date)
    user_interest_train['count'] = 1
    user_interest_matrix = user_interest_train.pivot(columns='name', index='user_id', values='count')
    user_interest_matrix.fillna(0, inplace=True)

    # create video embedding
    video_train, _ = train_test_split_for_data(video_df, 'created_at', date)
    video_category = pd.merge(video_train[['id', 'season_id']], season_df[['id', 'category']], 
                              how='inner', left_on='season_id', right_on='id', suffixes=('_video_df', '_season_df'))
    video_category['count'] = 1
    video_category_matrix = video_category.pivot(columns='category', index='id_video_df', values='count')
    video_category_matrix.fillna(0, inplace=True)
    video_category_matrix = video_category_matrix.reindex(columns=user_interest_matrix.columns, fill_value=0)

    # update the initial user interest matrix with vote counts
    vote_train, _ = train_test_split_for_data(vote_df, 'created_at', date)
    vote_categories = pd.merge(vote_train[['video_id', 'voter_id', 'status']], video_category[['id_video_df', 'category']], 
                           how='left', left_on='video_id', right_on='id_video_df')
    vote_categories['value'] = np.where(vote_categories['status'] == 'UPVOTE', 1, -1)
    vote_categories = vote_categories.groupby(['voter_id', 'category'], as_index=False).agg({'value': 'sum'})
    vote_categories_matrix = vote_categories.pivot(columns='category', index='voter_id', values='value')
    vote_categories_matrix.fillna(0, inplace=True)
    vote_categories_matrix = vote_categories_matrix.reindex(columns=user_interest_matrix.columns, fill_value=0)
    user_interest_matrix = user_interest_matrix.reindex(index=vote_categories_matrix.index, fill_value=0)
    user_interest_matrix = user_interest_matrix + vote_categories_matrix

    return user_interest_matrix, video_category_matrix

def similarity(user_id, video_id, user_interest_matrix, video_category_matrix):
    user = user_interest_matrix.loc[user_id]
    video = video_category_matrix.loc[video_id]
    category_distance = spatial.distance.cosine(user, video)

    return category_distance

def find_top_k_videos(user_id, k, user_interest_matrix, video_category_matrix):
    recommendations = pd.DataFrame(index=video_category_matrix.index)
    recommendations['similarity'] = recommendations.index.map(lambda x: similarity(user_id, x, user_interest_matrix, video_category_matrix))
    return recommendations.nsmallest(k, 'similarity')

def hit_ratio_at_k(y_true, y_pred, K):
    top_k_indices = np.argsort(-np.array(y_pred))[:K]
    return int(any(y_true[i] == 1 for i in top_k_indices))  # 1 if at least one relevant item is in top-K, 0 otherwise

def ndcg_at_k(y_true, y_pred, K):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    return ndcg_score([y_true], [y_pred], k=K)

def get_summary_statistics(vote_df, user_interest_matrix, video_category_matrix, date, K):
    _, vote_test = train_test_split_for_data(vote_df, 'created_at', date)
    vote_test['created_at'] = vote_test['created_at'].dt.date
    model_statistics = pd.DataFrame(columns=['datetime', 'roc auc score', 'accuracy', 'precision', 'recall', 'f1 score', 'hitratio@k', 'ndcg@k'])
    
    for day in sorted(vote_test['created_at'].unique()):
        print(day)
        voted_videos_for_day = vote_test[vote_test['created_at'] == day]
        summary_statistics = pd.DataFrame(columns=['user_id', 'roc auc score', 'accuracy', 'precision', 'recall', 'f1 score', 'hitratio@k', 'ndcg@k'])

        for user_id in voted_videos_for_day['voter_id'].unique():
            if user_id not in user_interest_matrix.index:
                continue
            
            # create dataframe to calculate confusion matrix
            user_voted_videos = voted_videos_for_day[voted_videos_for_day['voter_id'] == user_id]
            y_true_and_pred = pd.DataFrame(index=video_category_matrix.index)
            y_true_and_pred['true'] = np.where(y_true_and_pred.index.isin(user_voted_videos['video_id']), 1, 0)

            recommendations = find_top_k_videos(user_id, 20, user_interest_matrix, video_category_matrix)
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

def run_knn_recommender(date, K, num_cycles):
    user_interest_df = pd.read_feather('datasets/raw/user_interest.feather')
    user_df = pd.read_feather('datasets/raw/user.feather')
    season_df = pd.read_feather('datasets/raw/season.feather')
    video_df = pd.read_feather('datasets/raw/video.feather')
    vote_df = pd.read_feather('datasets/raw/vote.feather')
    model_statistics = pd.DataFrame(columns=['datetime', 'roc auc score', 'accuracy', 'precision', 'recall', 'f1 score', 'hitratio@k', 'ndcg@k'])

    for cycle in range(num_cycles):
        user_interest_matrix, video_category_matrix = create_embedding_matrices(user_interest_df, user_df, season_df, video_df, vote_df, date)
        model_statistics_for_training_cycle = get_summary_statistics(vote_df, user_interest_matrix, video_category_matrix, date, K)
        model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])
        date = get_end_date(date)

    model_statistics.to_csv('datasets/final/nus_knn_eval.csv', index=False)

    try:
        table_name = 'nus_knn_eval'
        csv_data = pd.read_csv('datasets/final/nus_knn_eval.csv')

        if database.table_exists(table_name):
            database.update_data(table_name, csv_data)
            print(f"Data updated in MySQL table '{table_name}' successfully.")
        else:
            database.create_table(table_name, csv_data)
            print(f"New MySQL table '{table_name}' created and data inserted successfully.")

    except Exception as e:
        print("Error:", e)
