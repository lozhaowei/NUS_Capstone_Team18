## Importing Libraries 
import warnings
import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
from scipy import spatial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ndcg_score, roc_auc_score
from src.data.database import CONN_PARAMS, insert_data
import pymysql
import pandas as pd
from decouple import config
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}

def get_end_date() -> str:
    """
    Calculate end date as 2 weeks before today.

    Returns:
    :return: End date in the format '%Y-%m-%d'.
    """
    today = datetime.now()
    end_date = (today - timedelta(weeks=1)).strftime('%Y-%m-%d')
    return end_date


def get_num_cycles(start_date: str) -> int:
    """
    Get the number of cycles based on the start date.

    Parameters:
    :param start_date: Starting date for training cycles.

    Returns:
    :return: Number of cycles.
    """
    today_date = datetime.now().strftime('%Y-%m-%d')
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(today_date, '%Y-%m-%d')
    date_difference = (end_datetime - start_datetime).days

    return date_difference

def train_test_split_for_data(data: pd.DataFrame, date_col: str, start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on the provided start date.

    Parameters:
    :param data: DataFrame to be split.
    :param date_col: Name of the column containing date information.
    :param start_date: Starting date for training.

    Returns:
    :return: Tuple containing training and testing sets.
    """
    train_data = data[data[date_col] <= start_date]
    test_data = data[(data[date_col] > start_date) & (data[date_col] <= get_end_date())]
    return train_data, test_data

def create_interaction_matrices(user_interest_df: pd.DataFrame, user_df: pd.DataFrame, season_df: pd.DataFrame, 
                              video_df: pd.DataFrame, vote_df: pd.DataFrame, post_feed_df: pd.DataFrame, date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create interaction matrices based on user interest, video embeddings, and user votes.

    Parameters:
    :param user_interest_df: DataFrame containing user interest data.
    :param user_df: DataFrame containing user data.
    :param season_df: DataFrame containing season data.
    :param video_df: DataFrame containing video data.
    :param vote_df: DataFrame containing vote data.
    :param post_feed_df: DataFrame containing post feed data.
    :param date: The date until which the data needs to be considered.

    Returns:
    :return: Tuple containing weighted interaction matrix and video category matrix.
    """
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
    vote_categories = pd.merge(vote_df[['video_id', 'voter_id', 'status']], video_category[['id_video_df', 'category']], 
                           how='left', left_on='video_id', right_on='id_video_df')
    vote_categories['value'] = np.where(vote_categories['status'] == 'UPVOTE', 1, -1)
    vote_categories = vote_categories.groupby(['voter_id', 'category'], as_index=False).agg({'value': 'sum'})
    vote_categories_matrix = vote_categories.pivot(columns='category', index='voter_id', values='value')
    vote_categories_matrix.fillna(0, inplace=True)
    vote_categories_matrix = vote_categories_matrix.reindex(columns=user_interest_matrix.columns, fill_value=0)
    user_interest_matrix = user_interest_matrix.reindex(index=vote_categories_matrix.index, fill_value=0)
    user_interest_matrix = user_interest_matrix + vote_categories_matrix

    # create weighted interaction matrix using watch_time / video_duration
    video_interaction = post_feed_df[post_feed_df['media_type'] == 'VIDEO']
    video_interaction['interaction'] = video_interaction['watch_time'] / video_interaction['video_duration']
    video_interaction = video_interaction.groupby(['user_id', 'post_id'], as_index=False)['interaction'].mean()
    video_interaction['value'] = video_interaction['interaction']  # Use 'interaction' as the interaction value
    video_interaction_matrix = video_interaction.pivot(index='user_id', columns='post_id', values='value').fillna(0)
    video_interaction_matrix = video_interaction_matrix.reindex(columns=user_interest_matrix.columns, fill_value=0)

    # Apply weights to the interaction matrix
    weighted_interaction_matrix = 0.7 * user_interest_matrix + 0.3 * video_interaction_matrix  # Example weights: 0.7 for votes, 0.3 for watch_time interaction

    return weighted_interaction_matrix, video_category_matrix

def svd_similarity(user_id, video_id, user_features, video_features):
    """
    Calculate SVD similarity between a user and a video.

    Parameters:
    :param user_id: User ID.
    :param video_id: Video ID.
    :param user_features: User features matrix.
    :param video_features: Video features matrix.

    Returns:
    :return: Similarity score.
    """
    user_features_vector = user_features.loc[user_id].values
    video_features_vector = video_features.loc[video_id].values
    similarity_score = np.dot(user_features_vector, video_features_vector)
    return similarity_score

def svd_similarity_2(user_id, video_id, user_features, video_features):
    """
    Calculate an alternative SVD similarity between a user and a video.

    Parameters:
    :param user_id: User ID.
    :param video_id: Video ID.
    :param user_features: User features matrix.
    :param video_features: Video features matrix.

    Returns:
    :return: Similarity score.
    """
    user_features_vector = user_features.loc[user_id].values
    video_features_vector = video_features.loc[video_id].values
    distance = np.linalg.norm(user_features_vector - video_features_vector)

    similarity_score = 1 / (1 + distance)
    return similarity_score

def find_top_videos(user_id, k, weighted_interaction_matrix, video_category_matrix):
    """
    Find the top K videos for a user based on SVD similarity.

    Parameters:
    :param user_id: User ID.
    :param k: Number of top videos to retrieve.
    :param weighted_interaction_matrix: Weighted interaction matrix.
    :param video_category_matrix: Video category matrix.

    Returns:
    :return: Indices of the top K videos.
    """
    similarity_matrix = svd_similarity(weighted_interaction_matrix, video_category_matrix)
    user_index = weighted_interaction_matrix.index.get_loc(user_id)
    similarities = similarity_matrix[user_index]
    top_k_indices = np.argsort(-similarities)[:k]
    return top_k_indices

def find_top_k_videos(user_id, k, weighted_interaction_matrix, video_category_matrix):
    """
    Find the top K videos for a user based on SVD similarity.

    Parameters:
    :param user_id: User ID.
    :param k: Number of top videos to retrieve.
    :param weighted_interaction_matrix: Weighted interaction matrix.
    :param video_category_matrix: Video category matrix.

    Returns:
    :return: DataFrame with top K videos and their similarity scores.
    """
    recommendations = pd.DataFrame(index=video_category_matrix.index)
    recommendations['similarity'] = recommendations.index.map(lambda x: svd_similarity(user_id, x, weighted_interaction_matrix, video_category_matrix))
    return recommendations.nsmallest(k, 'similarity')

def hit_ratio_at_k(y_true, y_pred, K):
    """
    Find the top K videos for a user based on SVD similarity.

    Parameters:
    :param user_id: User ID.
    :param k: Number of top videos to retrieve.
    :param weighted_interaction_matrix: Weighted interaction matrix.
    :param video_category_matrix: Video category matrix.

    Returns:
    :return: DataFrame with top K videos and their similarity scores.
    """
    top_k_indices = np.argsort(-np.array(y_pred))[:K]
    return int(any(y_true[i] == 1 for i in top_k_indices))  # 1 if at least one relevant item is in top-K, 0 otherwise

def ndcg_at_k(y_true, y_pred, K):
    """
    Calculate NDCG at K.

    Parameters:
    :param y_true: True values.
    :param y_pred: Predicted values.
    :param K: Number of top items to consider.

    Returns:
    :return: NDCG score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    return ndcg_score([y_true], [y_pred], k=K)

def get_summary_statistics(vote_df, weighted_interaction_matrix, video_category_matrix, date, K):
    """
    Get summary statistics for the model.

    Parameters:
    :param vote_df: DataFrame containing vote data.
    :param weighted_interaction_matrix: Weighted interaction matrix.
    :param video_category_matrix: Video category matrix.
    :param date: The date until which the data needs to be considered.
    :param K: Number of top items to consider.

    Returns:
    :return: DataFrame containing summary statistics.
    """
    _, vote_test = train_test_split_for_data(vote_df, 'created_at', date)
    vote_test['created_at'] = vote_test['created_at'].dt.date
    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])
    
    for day in sorted(vote_test['created_at'].unique()):
        print(day)
        voted_videos_for_day = vote_test[vote_test['created_at'] == day]
        summary_statistics = pd.DataFrame(columns=['user_id', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])

        for user_id in voted_videos_for_day['voter_id'].unique():
            if user_id not in weighted_interaction_matrix.index:
                continue
            
            # create dataframe to calculate confusion matrix
            user_voted_videos = voted_videos_for_day[voted_videos_for_day['voter_id'] == user_id]
            y_true_and_pred = pd.DataFrame(index=video_category_matrix.index)
            y_true_and_pred['true'] = np.where(y_true_and_pred.index.isin(user_voted_videos['video_id']), 1, 0)

            recommendations = find_top_k_videos(user_id, 20, weighted_interaction_matrix, video_category_matrix)
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

def run_svd_recommender(date, K, num_cycles):
    """
    Run SVD recommender model and evaluate its performance.

    Parameters:
    :param date: The date until which the data needs to be considered.
    :param K: Number of top items to consider.
    :param num_cycles: Number of training cycles.
    """
    user_interest_df = pd.read_feather('datasets/raw/user_interest.feather')
    user_df = pd.read_feather('datasets/raw/user.feather')
    season_df = pd.read_feather('datasets/raw/season.feather')
    video_df = pd.read_feather('datasets/raw/video.feather')
    vote_df = pd.read_feather('datasets/raw/vote.feather')
    post_feed_df = pd.read_feather('datasets/raw/post_feed.feather')
    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])

    for cycle in range(num_cycles):
        # Ensure the correct order of parameters in the function call
        weighted_interaction_matrix, video_category_matrix = create_interaction_matrices(user_interest_df, user_df, season_df, video_df, vote_df, post_feed_df, date)
        
        # Create the weighted interaction matrix
        weighted_interaction_matrix, _ = create_interaction_matrices(user_interest_df, user_df, season_df, video_df, vote_df, post_feed_df, date)
        
        # Apply Truncated SVD using the weighted interaction matrix
        svd = TruncatedSVD(n_components=50)  # Adjust the number of components as needed        
        model_statistics_for_training_cycle = get_summary_statistics(vote_df, weighted_interaction_matrix, video_category_matrix, date, K)
        model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])
        date = get_end_date()

    model_statistics['model'] = 'svd'
    model_statistics.to_csv('datasets/final_new/svd_video.csv', index=False)






