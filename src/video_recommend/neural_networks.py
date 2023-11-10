import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, concatenate, Dense, Input, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
from typing import Tuple

def run_ncf(start_date: str) -> pd.DataFrame:
    # Load your data
    user_interest_df = pd.read_feather('datasets/raw/user_interest.feather')
    user_df = pd.read_feather('datasets/raw/user.feather')
    season_df = pd.read_feather('datasets/raw/season.feather')
    video_df = pd.read_feather('datasets/raw/video.feather')
    vote_df = pd.read_feather('datasets/raw/vote.feather')

    # Preprocessing steps (merging, label encoding, etc.)
    # Merge user_interest_df, user_df, season_df, video_df, and vote_df
    merged_df = pd.merge(user_interest_df, user_df, left_on='user_id', right_on='id', how='inner')
    merged_df = pd.merge(merged_df, vote_df, left_on='user_id', right_on='voter_id', how='inner')
    columns_to_remove = ['recce_status', 'created_at_x', 'id_x', 'type', 'age_consent', 'created_at_y', 'id_y', 'voter_id', 'timestamp', 'video_creator_id']
    merged_df = merged_df.drop(columns=columns_to_remove)
    merged_df = pd.merge(merged_df, season_df, left_on='season_id', right_on='id', how='inner')
    merged_df = merged_df.drop(columns=['created_at_x', 'id', 'creator_id', 'season_number', 'contest_id', 'participant_count', 'description', 'recce_status', 'created_at_y'])

    # Train-test split using the provided function
    def get_end_date() -> str:
        today = datetime.now()
        end_date = (today - timedelta(weeks=2)).strftime('%Y-%m-%d')
        return end_date

    def train_test_split_for_data(data: pd.DataFrame, date_col: str, start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = data[data[date_col] <= start_date]
        test_data = data[(data[date_col] > start_date) & (data[date_col] <= get_end_date())]
        return train_data, test_data

    label_encoders = {}
    for column in ['user_id', 'name', 'video_id', 'season_id', 'category', 'status']:
        le = LabelEncoder()
        merged_df[column] = le.fit_transform(merged_df[column])
        label_encoders[column] = le

    train_data, test_data = train_test_split_for_data(merged_df, 'updated_at', start_date)

    user_input = Input(shape=(1,))
    video_input = Input(shape=(1,))

    user_embedding = Embedding(len(label_encoders['user_id'].classes_), 50)(user_input)
    video_embedding = Embedding(len(label_encoders['video_id'].classes_), 50)(video_input)

    user_flat = Flatten()(user_embedding)
    video_flat = Flatten()(video_embedding)

    concat = concatenate([user_flat, video_flat])

    # Adding Dropout layer for regularization
    concat_dropout = Dropout(0.2)(concat)

    dense1 = Dense(100, activation='relu')(concat_dropout)

    # Output layer with softmax for multi-class classification
    output = Dense(len(label_encoders['category'].classes_), activation='softmax')(dense1)

    model = Model(inputs=[user_input, video_input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    train_users = train_data['user_id']
    train_videos = train_data['video_id']
    train_labels = train_data['category']

    test_users = test_data['user_id']
    test_videos = test_data['video_id']
    test_labels = test_data['category']

    # Train the model with Dropout and Early Stopping
    model.fit([train_users, train_videos], train_labels, epochs=50, batch_size=64, validation_split=0.2)

    evaluation_results = []

    for day in sorted(test_data['updated_at'].dt.date.unique()):
        test_data_for_day = test_data[test_data['updated_at'].dt.date == day]
        test_users_day = test_data_for_day['user_id']
        test_videos_day = test_data_for_day['video_id']
        test_labels_day = test_data_for_day['category']

        predictions_day = model.predict([test_users_day, test_videos_day])
        predicted_categories_day = np.argmax(predictions_day, axis=1)

        try:
            actual_categories_day = label_encoders['category'].inverse_transform(test_labels_day)
            predicted_categories_day = label_encoders['category'].inverse_transform(predicted_categories_day)

            accuracy_day = accuracy_score(actual_categories_day, predicted_categories_day)
            precision_day = precision_score(actual_categories_day, predicted_categories_day, average='weighted')
            recall_day = recall_score(actual_categories_day, predicted_categories_day, average='weighted')
            f1_day = f1_score(actual_categories_day, predicted_categories_day, average='weighted')

            # Appending evaluation metrics to the results
            evaluation_results.append({
                'dt': day,
                'roc_auc_score': 0,  # placeholder
                'accuracy': accuracy_day,
                'precision': precision_day,
                'recall': recall_day,
                'f1_score': f1_day,
                'hit_ratio_k': 0,  # placeholder
                'ndcg_k': 0  # placeholder
            })

        except KeyError as e:
            print(f"Skipping metric calculation for {day} due to unseen labels: {str(e)}")

    # Convert the list of dictionaries to a pandas DataFrame
    evaluation_df = pd.DataFrame(evaluation_results, columns=['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k'])

    evaluation_df['model'] = 'ncf'

    # Save the evaluation results to a CSV file
    evaluation_df.to_csv('datasets/final_new/ncf_video.csv', index=False)

    return evaluation_df