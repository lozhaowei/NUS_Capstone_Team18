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
from scipy.sparse import csr_matrix

#return metrics in tabular form
def statistics(df_conversation_like, date):
    #Implement Collaborative Filtering
    df_likes = df_conversation_like[['conversation_id','like_giver_id','timestamp']]
    df_likes = df_likes.drop_duplicates()

    temp_df = df_likes[['conversation_id','like_giver_id']]
    #likes_table = temp_df.pivot_table(index='conversation_id', columns='like_giver_id', aggfunc=len, fill_value=0)

    # Computing user_conversation similarity
    # computational heavy version:
    #user_conversation_similarity = cosine_similarity(likes_matrix.T)

    #split by date, date to be changed for the function
    split_date = date
    train_data = df_likes[df_likes['timestamp'].dt.date < pd.to_datetime(split_date).date()]
    test_data = df_likes[df_likes['timestamp'].dt.date == pd.to_datetime(split_date).date()]

    train_data = train_data.drop_duplicates(subset=['like_giver_id', 'conversation_id'])
    test_data = test_data.drop_duplicates(subset=['like_giver_id', 'conversation_id'])

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
    # Create mappings for conversation_id and like_giver_id
    conversation_id_mapping_test = pd.factorize(temp_df['conversation_id'])
    like_giver_id_mapping_test = pd.factorize(temp_df['like_giver_id'])

    conversation_ids = conversation_id_mapping_test[0]
    like_giver_ids = like_giver_id_mapping_test[0]

    # Store the unique values for later lookup
    conversation_id_to_index_test = {id: index for index, id in enumerate(conversation_id_mapping_test[1])}
    like_giver_id_to_index_test = {id: index for index, id in enumerate(like_giver_id_mapping_test[1])}

    # Reverse mapping from index to conversation_id
    index_to_conversation_id_test = {index: id for id, index in conversation_id_to_index_test.items()}
    index_to_like_giver_id_test = {index: id for id, index in like_giver_id_to_index_test.items()}


    data = np.ones(len(temp_df))
    sparse_matrix = coo_matrix((data, (conversation_ids, like_giver_ids)), 
                            shape=(len(np.unique(conversation_ids)), len(np.unique(like_giver_ids))))
    test_likes_matrix = sparse_matrix.tocsr()

    user_similarity_matrix = cosine_similarity(train_likes_matrix, dense_output=False)

    y_true = []
    y_scores = []

    for i in range(test_likes_matrix.shape[0]):
        conversation_id = index_to_conversation_id_test.get(i)
        for ii in range(test_likes_matrix.shape[1]):
            like_giver_id = index_to_like_giver_id_test.get(ii)
            if like_giver_id_to_index.get(like_giver_id) != None:
                if conversation_id_to_index.get(conversation_id) != None:
                    score = user_similarity_matrix[conversation_id_to_index.get(conversation_id),like_giver_id_to_index.get(like_giver_id)]
                    if score > 1:
                        y_scores.append(1)
                    else:
                        y_scores.append(score)
                    y_true.append(test_likes_matrix[i,ii])

    # Binarize the scores for accuracy, precision, recall and F1-score calculations
    y_pred = [1 if score > 0.1 else 0 for score in y_scores]

    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    data = {
        'datetime': [split_date],
        'roc auc score': [roc_auc],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1 score': [f1],
        'hit_ratio_k': 0,
        'ndcg_k': 0 
    }

    metrics_df = pd.DataFrame(data)

    return metrics_df

def run_collaborative_recommender(date, k, num_cycles, convo_likes_df):
    date = datetime.strptime(date, '%Y-%m-%d')
    model_statistics = pd.DataFrame(columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])

    for cycle in range(num_cycles):
        model_statistics_for_training_cycle = statistics(convo_likes_df, k, date)
        model_statistics = pd.concat([model_statistics, model_statistics_for_training_cycle])
        date = date + timedelta(days=1)

    model_statistics['model'] = 'knn'
    model_statistics.to_csv('datasets/final/knn_convo.csv', index=False)