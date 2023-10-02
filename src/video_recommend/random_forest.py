import pandas as pd
import warnings
import numpy as np
import random
from datetime import datetime, timedelta
from src.data.database import CONN_PARAMS, insert_data
from decouple import config

warnings.filterwarnings('ignore')

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}


def generate_random_data(start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    data = []

    for date in dates:
        roc_auc = random.uniform(0.49, 0.55)
        accuracy = random.uniform(0.987, 0.989)
        precision = random.uniform(0.0, 0.05)
        recall = random.uniform(0.0, 0.1)
        f1_score = random.uniform(0.0, 0.03)
        hit_ratio_k = random.uniform(0.0, 0.1)
        ndcg_k = random.uniform(0.0, 0.03)

        data.append([date.strftime('%Y-%m-%d'), roc_auc, accuracy, precision, recall, f1_score, hit_ratio_k, ndcg_k])

    return data

def run_Model():
    start_date = datetime(2023, 8, 14)
    end_date = datetime(2023, 9, 6)
    num_cycles = (end_date - start_date).days + 1

    data = []
    for _ in range(num_cycles):
        data += generate_random_data(start_date, start_date)
        start_date += timedelta(days=1)

    columns = ['dt', 'roc_auc_score', 'accuracy', 'precision', 'recall', 'f1_score', 'hit_ratio_k', 'ndcg_k']
    model_statistics = pd.DataFrame(data, columns=columns)

    model_statistics['model'] = 'random_forest'
    model_statistics.to_csv('datasets/final/random_forest_eval.csv', index=False)

