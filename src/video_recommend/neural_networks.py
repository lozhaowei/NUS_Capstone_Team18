import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import ndcg_score

def get_num_cycles(start_date: str) -> int:
    # Get today's date
    today_date = datetime.now().strftime('%Y-%m-%d')
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(today_date, '%Y-%m-%d')
    date_difference = (end_datetime - start_datetime).days
    return date_difference

def train_test_split_for_data(data: pd.DataFrame, date_col: str, start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = data[data[date_col] <= start_date]
    test_data = data[(data[date_col] > start_date) & (data[date_col] <= get_end_date(start_date))]
    return train_data, test_data

def create_embedding_matrices(user_interest_df: pd.DataFrame, user_df: pd.DataFrame, season_df: pd.DataFrame,
                              video_df: pd.DataFrame, vote_df: pd.DataFrame, date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_interest_train, _ = train_test_split_for_data(user_interest_df, 'updated_at', date)
    user_interest_train['count'] = 1
    user_interest_matrix = user_interest_train.pivot(columns='name', index='user_id', values='count')
    user_interest_matrix.fillna(0, inplace=True)

    video_train, _ = train_test_split_for_data(video_df, 'created_at', date)
    video_category = pd.merge(video_train[['id', 'season_id']], season_df[['id', 'category']],
                              how='inner', left_on='season_id', right_on='id', suffixes=('_video_df', '_season_df'))
    video_category['count'] = 1
    video_category_matrix = video_category.pivot(columns='category', index='id_video_df', values='count')
    video_category_matrix.fillna(0, inplace=True)
    video_category_matrix = video_category_matrix.reindex(columns=user_interest_matrix.columns, fill_value=0)

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

def hit_ratio_k(y_true, y_score, k=10):
    top_k_indices = np.argsort(y_score)[-k:]
    return int(np.any(y_true[top_k_indices]))

def ndcg_k(y_true, y_score, k=10):
    y_true = y_true.reshape(1, -1)
    y_score = y_score.reshape(1, -1)
    return ndcg_score(y_true, y_score, k=k)

def build_ncf_model(hp):
    user_input = Input(shape=(1, ))
    user_embedding = Embedding(input_dim=len(all_user_ids), output_dim=hp.Int('user_embedding_dim', min_value=8, max_value=32))(user_input)
    user_embedding = Flatten()(user_embedding)

    video_input = Input(shape=(1,))
    video_embedding = Embedding(input_dim=len(all_video_ids), output_dim=hp.Int('video_embedding_dim', min_value=8, max_value=32))(video_input)
    video_embedding = Flatten()(video_embedding)

    concat = Concatenate()([user_embedding, video_embedding])

    # Add Batch Normalization after concatenation
    concat = BatchNormalization()(concat)

    # Add more hidden layers with dropout and batch normalization
    fc1 = Dense(hp.Int('fc1_units', min_value=64, max_value=256, step=32), activation='relu')(concat)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(hp.Float('fc1_dropout', min_value=0.0, max_value=0.5))(fc1)

    fc2 = Dense(hp.Int('fc2_units', min_value=32, max_value=128, step=16), activation='relu')(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Dropout(hp.Float('fc2_dropout', min_value=0.0, max_value=0.5))(fc2)

    # Output layer
    output = Dense(1, activation='sigmoid')(fc2)

    model = Model(inputs=[user_input, video_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model

def run_model(start_date):

    num_cycles = get_num_cycles(start_date)

    user_interest_df = pd.read_feather('datasets/raw/user_interest.feather')
    user_df = pd.read_feather('datasets/raw/user.feather')
    season_df = pd.read_feather('datasets/raw/season.feather')
    video_df = pd.read_feather('datasets/raw/video.feather')
    vote_df = pd.read_feather('datasets/raw/vote.feather')
    results = []

    for cycle in range(num_cycles):
        date = (datetime.now() - timedelta(days=cycle)).strftime('%Y-%m-%d')

        user_interest_matrix, video_category_matrix = create_embedding_matrices(user_interest_df, user_df, season_df, video_df, vote_df, date)

        # Define constants and hyperparameters
        BATCH_SIZE = 64
        EPOCHS = 15

        # Create label encoders for user and video IDs
        user_encoder = LabelEncoder()
        video_encoder = LabelEncoder()

        all_user_ids = np.concatenate([train_data[:, 0], val_data[:, 0], test_data[:, 0]])
        all_video_ids = np.concatenate([train_data[:, 1], val_data[:, 1], test_data[:, 1]])

        user_encoder.fit(all_user_ids)
        video_encoder.fit(all_video_ids)

        # Transform user and video IDs
        train_users = user_encoder.transform(train_data[:, 0])
        train_videos = video_encoder.transform(train_data[:, 1])
        val_users = user_encoder.transform(val_data[:, 0])
        val_videos = video_encoder.transform(val_data[:, 1])
        test_users = user_encoder.transform(test_data[:, 0])
        test_videos = video_encoder.transform(test_data[:, 1])

        # Build a Neural Collaborative Filtering Model
        # model = build_ncf_model(user_interest_matrix.shape[0], video_category_matrix.shape[0])

        tuner.search([train_users, train_videos], train_data[:, 2],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=([val_users, val_videos], val_data[:, 2]))

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = build_ncf_model(best_hps)

        # Train the model
        history = model.fit(
            [train_users, train_videos], train_data[:, 2],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([val_users, val_videos], val_data[:, 2]),
            verbose=1
        )

        # Evaluate the model
        test_preds = model.predict([test_users, test_videos])
        test_preds = test_preds.flatten()
        accuracy = accuracy_score(test_data[:, 2], (test_preds > 0.5).astype(int))
        precision = precision_score(test_data[:, 2], (test_preds > 0.3).astype(int))
        recall = recall_score(test_data[:, 2], (test_preds > 0.3).astype(int))
        f1 = f1_score(test_data[:, 2], (test_preds > 0.3).astype(int))
        roc_auc = roc_auc_score(test_data[:, 2], test_preds)
        hit_ratio = hit_ratio_k(test_data[:, 2], test_preds, k=10)
        ndcg = ndcg_k(test_data[:, 2], test_preds, k=10)

        results.append({
            'dt': date,
            'roc_auc_score': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hit_ratio_k': hit_ratio,
            'ndcg_k': ndcg
        })

    results_df = pd.DataFrame(results)
    results_df['model'] = 'ncf'
    results_df = results_df.sort_values(by='dt', ascending=True)
    results_df.to_csv('datasets/final/neural_networks_video.csv', index=False)