import os
import pyarrow.feather as feather
import pandas as pd

from src.data import database


def write_feather_data(table, df, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    feather_file_path = os.path.join(data_dir, f'{table}.feather')
    feather.write_feather(df, feather_file_path)

    print(f"Table '{table}' saved as '{feather_file_path}'")


def pull_raw_data(list_of_tables):
    try:
        for table in list_of_tables:
            query = f"SELECT * FROM {table}"
            df = database.query_database(query)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, '../..', 'datasets', 'raw')

            write_feather_data(table, df, data_dir)

    except Exception as e:
        print("Error:", e)

def get_dashboard_data(entity):
    try:
        query = f"SELECT * FROM nus_{entity}_eval"
        df = database.query_database(query)
        df["dt"] = pd.to_datetime(df["dt"])

        df.rename(columns={'roc_auc_score': 'ROC AUC Score', 'accuracy': 'Accuracy', 'precision': 'Precision',
                           'recall': 'Recall', 'f1_score': 'F1 Score', 'hit_ratio_k': 'HitRatio@K',
                           'ndcg_k': 'NDCG@K'}, inplace=True)
        
        return df

    except Exception as e:
        print("Error:", e)