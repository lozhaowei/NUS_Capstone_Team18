## Importing Libraries 
import warnings
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta
from scipy import spatial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ndcg_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


def get_end_date(start_date: str) -> str:
    start_datetime_object = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime_object = start_datetime_object + timedelta(days=3)
    end_date = end_datetime_object.strftime('%Y-%m-%d')
    return end_date

def get_predicted_scores(user_id, reconstructed_matrix, train_likes_table):
    try:
        user_index = list(train_likes_table.columns).index(user_id)
    except ValueError:
        return np.zeros(train_likes_table.shape[0])

    return np.dot(reconstructed_matrix[user_index], train_likes_table.T.values)

#return metrics in tabular form
def statistics(df, k, date):
    #Implement Collaborative Filtering
    df_conversation_like = df

    df_likes = df_conversation_like[['conversation_id','like_giver_id','timestamp']]
    df_likes = df_likes.drop_duplicates()

    temp_df = df_likes[['conversation_id','like_giver_id']]
    likes_table = temp_df.pivot_table(index='conversation_id', columns='like_giver_id', aggfunc=len, fill_value=0)

    # Computing user_conversation similarity
    # computational heavy version:
    #user_conversation_similarity = cosine_similarity(likes_matrix.T)

    #split by date, date to be changed for the function
    split_date = date
    train_data = df_likes[df_likes['timestamp'] < split_date]
    test_data = df_likes[df_likes['timestamp'] >= split_date]

    temp_df = train_data[['conversation_id','like_giver_id']]
    train_likes_table = temp_df.pivot_table(index='conversation_id', columns='like_giver_id', aggfunc=len, fill_value=0)
    train_likes_matrix = train_likes_table.values

    temp_df = test_data[['conversation_id','like_giver_id']]
    test_likes_table = temp_df.pivot_table(index='conversation_id', columns='like_giver_id', aggfunc=len, fill_value=0)
    test_likes_matrix = test_likes_table.values

    #optimising cosine similarity calculations:
    # Define batch size
    batch_size = 500

    # Calculate number of batches
    num_batches = int(train_likes_matrix.shape[1] / batch_size)

    # Handle the case where the number of users is not evenly divisible by the batch size
    if train_likes_matrix.shape[1] % batch_size != 0:
        num_batches += 1

    # Initialize an empty array for results
    user_conversation_similarity = np.zeros((train_likes_matrix.shape[1], train_likes_matrix.shape[1]))

    # Calculate cosine similarity in batches
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, train_likes_matrix.shape[1])
        user_conversation_similarity[start:end, :] = cosine_similarity(train_likes_matrix.T[start:end, :], train_likes_matrix.T)

    # Apply NMF to fill in missing values (dont use svd because its computationally heavy)
    # Define the number of features you want to extract (you can tune this parameter)
    n_features = 10
    nmf = NMF(n_components=n_features, init='random', random_state=42)
    W = nmf.fit_transform(user_conversation_similarity)
    H = nmf.components_

    # Reconstruct the Matrix
    reconstructed_matrix = np.dot(W, H)

    # Lists to hold actual and predicted scores
    y_true = []
    y_scores = []

    # Loop through each user-item pair in the test set
    for user in test_data['like_giver_id'].unique():
        for conversation in test_likes_table.index:
            actual_interaction = min(test_likes_table.loc[conversation, user], 1)

            # Check if conversation exists in training data
            if conversation in train_likes_table.index:
                predicted_score = get_predicted_scores(user, reconstructed_matrix, train_likes_table)[list(train_likes_table.index).index(conversation)]
            else:
                predicted_score = 0

            y_true.append(actual_interaction)
            y_scores.append(predicted_score)

    # Binarize the scores for accuracy, precision, recall and F1-score calculations
    y_pred = [1 if score > 0.1 else 0 for score in y_scores]

    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    def get_top_k_recommendations(user, k):
        scores = get_predicted_scores(user, reconstructed_matrix, train_likes_table)
        return np.argsort(scores)[-k:]

    hit_count = 0
    total = 0
    ndcgs = []

    for user in test_data['like_giver_id'].unique():
        top_k_rec = get_top_k_recommendations(user, k)
        actual_liked = test_likes_table[test_likes_table[user] > 0].index.tolist()

        # For Hit Ratio
        hits = len(set(top_k_rec) & set(actual_liked))
        if hits > 0:
            hit_count += 1

        # For NDCG
        actual_scores = [1 if conversation in actual_liked else 0 for conversation in top_k_rec]
        dcg = sum([actual_scores[i] / np.log2(i+2) for i in range(len(actual_scores))])
        idcg = sum([1 / np.log2(i+2) for i in range(len(actual_scores))])
        ndcgs.append(dcg/idcg if idcg > 0 else 0)

        total += 1

    hit_ratio_at_k = hit_count / total
    ndcg_at_k = np.mean(ndcgs)

    new_date = date - timedelta(days=1)

    data = {
        'datetime': [new_date],
        'roc auc score': [roc_auc],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1 score': [f1],
        f'hitratio@k': [hit_ratio_at_k],
        f'ndcg@k': [ndcg_at_k]
    }

    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(data)

    return metrics_df

def run_collaborative_recommender(date, k, num_cycles, df):
    conversation_like = df
    model_statistics = pd.DataFrame(columns=['datetime', 'roc auc score', 'accuracy', 'precision', 'recall', 'f1 score', 'hitratio@k', 'ndcg@k'])

    for cycle in range(num_cycles):
        model_statistics_for_training_cycle = statistics(df, k, date)
        model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])
        date = date + timedelta(days=1)

    return model_statistics