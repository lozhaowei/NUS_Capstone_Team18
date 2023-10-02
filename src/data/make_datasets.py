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

def get_dashboard_data(model):
    """
    queries database to obtain metrics data of specific model, renames the columns for frontend use,
    then writes it as a feather file
    :param model: model name
    """
    try:
        table_name = f'nus_{model}_eval'
        query = f"SELECT * FROM {table_name}"
        # df = database.query_database(query)
        df = pd.read_csv(f'datasets/final/{table_name}.csv')

        # Rename columns for frontend use
        df.rename(columns={'roc_auc_score': 'ROC AUC Score', 'accuracy': 'Accuracy', 'precision': 'Precision',
                           'recall': 'Recall', 'f1_score': 'F1 Score', 'hit_ratio_k': 'HitRatio@K',
                           'ndcg_k': 'NDCG@K'}, inplace=True)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, '../..', 'datasets', 'final')

        write_feather_data(table_name, df, data_dir)

    except Exception as e:
        print("Error:", e)
