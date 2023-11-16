import pymysql
import pandas as pd
from decouple import config
import os 
import csv
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

def get_secret():

    secret_name = "Capstone-Team18"
    region_name = "ap-southeast-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    host = get_secret_value_response['host']
    user = get_secret_value_response['username']
    password = get_secret_value_response['password']
    port = get_secret_value_response['port']
    name = get_secret_value_response['name']
    
    return [host, user, password, port, name]

CONN_PARAMS = {
    'host': get_secret()[0],
    'user': get_secret()[1],
    'password': get_secret()[2],
    'port': int(get_secret()[3]),
    'database': get_secret()[4],
}

# CONN_PARAMS = {
#     'host': config('DB_HOST'),
#     'user': config('DB_USER'),
#     'password': config('DB_PASSWORD'),
#     'port': int(config('DB_PORT')),
#     'database': config('DB_NAME'),
# }

def query_database(query):
    """
    Establishes a connection with MySQL Database to read queries
    :param query: will be placed as the param of the function
    """
    try:
        conn = pymysql.connect(**CONN_PARAMS)

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    except Exception as e:
        print("Error:", e)

def insert_data(table_name, data):
    """
    Inserts data into the desired table of the Database. 
    If the table does not exist, it creates a new one.
    If the table exists, it simply updates the values of the additional rows
    :param table_name: identifier of the table in the DB
    :param data: In the form of a dataframe which is expected to be sent to the DB
    """
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Check if the table exists
        table_exists_query = f"SHOW TABLES LIKE '{table_name}'"
        cursor.execute(table_exists_query)
        table_exists = cursor.fetchone()

        if not table_exists:
            # If the table doesn't exist, create a new table with specified columns
            create_table_query = f'''
            CREATE TABLE {table_name} (
                `dt` DATE,
                `roc_auc_score` FLOAT,
                `accuracy` FLOAT,
                `precision` FLOAT,
                `recall` FLOAT,
                `f1_score` FLOAT,
                `hit_ratio_k` FLOAT,
                `ndcg_k` FLOAT,
                `model` VARCHAR(255),
                PRIMARY KEY (`dt`, `model`)
            )
            '''
            cursor.execute(create_table_query)
            print(f"Table '{table_name}' created successfully.")

        # Convert 'dt' column to string before insertion
        data['dt'] = data['dt'].astype(str)
        # Convert NaN values to None for proper insertion
        data = data.where(pd.notna(data), None)

        # Insert data only if the combination of 'dt' and 'model' is unique
        for _, row in data.iterrows():
            insert_query = f'''
            INSERT INTO {table_name} (`dt`, `roc_auc_score`, `accuracy`, `precision`, `recall`, `f1_score`, 
            `hit_ratio_k`, `ndcg_k`, `model`) 
            SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s 
            WHERE NOT EXISTS (
                SELECT 1 FROM {table_name} WHERE `dt` = %s AND `model` = %s
            )
            ON DUPLICATE KEY UPDATE
            `roc_auc_score` = VALUES(`roc_auc_score`),
            `accuracy` = VALUES(`accuracy`),
            `precision` = VALUES(`precision`),
            `recall` = VALUES(`recall`),
            `f1_score` = VALUES(`f1_score`),
            `hit_ratio_k` = VALUES(`hit_ratio_k`),
            `ndcg_k` = VALUES(`ndcg_k`)
            '''
            cursor.execute(insert_query, (row['dt'], row['roc_auc_score'], row['accuracy'], row['precision'],
                                         row['recall'], row['f1_score'], row['hit_ratio_k'], row['ndcg_k'],
                                         row['model'], row['dt'], row['model']))

        conn.commit()
        conn.close()

        print(f"Data updated in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)

def combine_tables_video():
    """ 
    Combined the different evaluation tables into a mega table which could directly be sent to the DB
    """
    table1 = pd.read_csv('datasets/final_new/random_forest_video.csv')
    table2 = pd.read_csv('datasets/final_new/knn_video.csv')
    table3 = pd.read_csv('datasets/final_new/svd_video.csv')
    table4 = pd.read_csv('datasets/final_new/ncf_video.csv')

    # Combine tables
    combined_table = pd.concat([table1, table2, table3, table4], ignore_index=True)

    output_folder = 'datasets/final_new'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "nus_video_eval_2.csv")
    combined_table.to_csv(output_path, index=False)
    
    print(f"Combined data saved to '{output_path}' successfully.")
    return output_path

def combine_tables_convo():
    """ 
    Combined the different evaluation tables into a mega table which could directly be sent to the DB
    """
    table1 = pd.read_csv('datasets/final_new/random_forest_convo.csv')
    table2 = pd.read_csv('datasets/final_new/als_conversation.csv')

    # Combine tables
    combined_table = pd.concat([table1, table2], ignore_index=True)

    output_folder = 'datasets/final_new'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "nus_convo_eval_2.csv")
    combined_table.to_csv(output_path, index=False)
    
    print(f"Combined data saved to '{output_path}' successfully.")
    return output_path

def is_valid_datetime(date_str):
    """ 
    Helper function to convert the date-string value into a date-time object
    :param date_str: it is expressed as YYYY/MM/DD
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def clean_csv(input_csv_path, output_csv_path):
    """ 
    Helper function to clean the CSV before being pushed into the DB
    :param input_csv_path: the path for extracting the pre-processed CSV
    :param output_csv_path: the path for extracting the processed CSV
    """
    with open(input_csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = list(csv_reader)

    cleaned_rows = [row for row in rows if is_valid_datetime(row.get('dt'))]

    with open(output_csv_path, 'w', newline='') as cleaned_csv:
        fieldnames = csv_reader.fieldnames
        csv_writer = csv.DictWriter(cleaned_csv, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(cleaned_rows)
