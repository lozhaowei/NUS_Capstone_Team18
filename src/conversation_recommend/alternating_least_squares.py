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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
import implicit

def statistics(df_conversation_like, k, date):
    df_likes = df_conversation_like[['conversation_id','like_giver_id','timestamp']]
    df_likes = df_likes.drop_duplicates()

    temp_df = df_likes[['conversation_id','like_giver_id']]

    #split by date, date to be changed for the function
    split_date = date
    train_data = df_likes[df_likes['timestamp'].dt.date < pd.to_datetime(split_date).date()]
    test_data = df_likes[df_likes['timestamp'].dt.date == pd.to_datetime(split_date).date()]
    
    # for train data
    temp_df = train_data[['conversation_id','like_giver_id']]
    conversation_ids = pd.factorize(temp_df['conversation_id'])[0]
    like_giver_ids = pd.factorize(temp_df['like_giver_id'])[0]
    # Create mappings for conversation_id and like_giver_id
    conversation_id_mapping = pd.factorize(temp_df['conversation_id'])
    like_giver_id_mapping = pd.factorize(temp_df['like_giver_id'])

    conversation_ids = conversation_id_mapping[0]
    like_giver_ids = like_giver_id_mapping[0]

    # Store the unique values for later lookup
    conversation_id_to_index = {id: index for index, id in enumerate(conversation_id_mapping[1])}
    like_giver_id_to_index = {id: index for index, id in enumerate(like_giver_id_mapping[1])}

    # Original mapping from conversation_id to index
    conversation_id_to_index = {id: index for index, id in enumerate(conversation_id_mapping[1])}

    # Reverse mapping from index to conversation_id
    index_to_conversation_id = {index: id for id, index in conversation_id_to_index.items()}

    data = np.ones(len(temp_df))
    sparse_matrix = coo_matrix((data, (conversation_ids, like_giver_ids)), 
                            shape=(len(np.unique(conversation_ids)), len(np.unique(like_giver_ids))))
    train_likes_matrix = sparse_matrix.tocsr()

    #for test data
    temp_df = test_data[['conversation_id','like_giver_id']]
    conversation_ids = pd.factorize(temp_df['conversation_id'])[0]
    like_giver_ids = pd.factorize(temp_df['like_giver_id'])[0]
    data = np.ones(len(temp_df))
    sparse_matrix = coo_matrix((data, (conversation_ids, like_giver_ids)), 
                            shape=(len(np.unique(conversation_ids)), len(np.unique(like_giver_ids))))
    test_likes_matrix = sparse_matrix.tocsr()

    #fitting ALS model
    model = implicit.als.AlternatingLeastSquares(num_threads=1)
    model.fit(train_likes_matrix)
    
    #generating predictions
    def hit_ratio(predicted, actual):
        return int(actual in predicted)

    def ndcg(predicted, actual):
        if actual in predicted:
            index = predicted.index(actual)
            return np.reciprocal(np.log2(index + 2))
        return 0

    hit_ratios = []
    ndcg_scores = []

    K = 20  # Top-K items

    y_pred = []
    num = 0
    for i in range(len(test_data['like_giver_id'])):
        user_id = like_giver_id_to_index.get(test_data['like_giver_id'].iloc[i])
        if user_id == None:
            pred = 0
            num += 1
        else:
            recommended = model.recommend(user_id, train_likes_matrix[user_id], N=k)
            recommended_item_ids = []
            for convo in recommended[0]:
                recommended_item_ids.append(index_to_conversation_id.get(convo))
            hit_ratios.append(hit_ratio(recommended_item_ids, test_data['conversation_id'].iloc[i]))
            ndcg_scores.append(ndcg(recommended_item_ids, test_data['conversation_id'].iloc[i]))
            pred = 0
            for convo in recommended[0]:
                if index_to_conversation_id.get(convo) == test_data['conversation_id'].iloc[i]:
                    pred = 1
        y_pred.append(pred)

    # Average metrics across all users
    average_hit_ratio = np.mean(hit_ratios)
    average_ndcg = np.mean(ndcg_scores)

    #generating metrics
    #ROC AUC set to 0 since it is not applicable in ALS, it only has binary output.
    test_data['liked'] = 1
    y_test = test_data['liked']

    roc_auc = 0
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    data = {
        'dt': [split_date],
        'roc_auc_score': [roc_auc],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        f'hit_ratio_k': [average_hit_ratio],
        f'ndcg_k': [average_ndcg]
    }

    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(data)

    return metrics_df

def run_als_recommender(date, k, num_cycles, convo_likes_df):
    date = datetime.strptime(date, '%Y-%m-%d')
    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])

    for cycle in range(num_cycles):
        model_statistics_for_training_cycle = statistics(convo_likes_df, k, date)
        model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])
        date = date + timedelta(days=1)

    model_statistics['model'] = 'als'
    model_statistics.to_csv('datasets/final/als_convo.csv', index=False)