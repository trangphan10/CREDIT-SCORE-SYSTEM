from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
import pandas as pd 


def extract_csv_data(file_csv, **kwargs):
    ti = kwargs['ti']
    if os.path.exists(file_csv):
        file = pd.read_csv(file_csv)
        ti.xcom_push(key='read_data', value=file)
    else:
        raise FileNotFoundError(f"The file {file_csv} does not exist")

def transform(x, y, is_dummy=True):
    data = pd.concat([x, y], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    data.drop('DEBTINC', axis=1, inplace=True)
    if is_dummy:
        data = pd.get_dummies(data, columns=['REASON', 'JOB'])
    y = data['BAD']
    x = data.drop(['BAD'], axis=1)
    return x, y

def onehotcoding_label(x):
    labelEncoder = LabelEncoder()
    for column in x.columns:
        if x[column].dtype == 'object':
            x[column] = labelEncoder.fit_transform(x[column])
    pickle.dump(labelEncoder, open('/opt/airflow/dags/LabelEncoder.pickle', 'wb'))

def standardScaler_data(X, y):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled_features = scaler.fit_transform(X)
    scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    pickle.dump(scaler, open('/opt/airflow/dags/StandardScaler.pickle', 'wb'))
    return scaled_features_df, y

def transform_csv_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='read_data', task_ids='extract_data')
    if data.duplicated().any():
        data = data.drop_duplicates()
    y = data['BAD']
    X = data.drop(['BAD'], axis=1)
    X, y = transform(X, y)
    onehotcoding_label(X)
    X, y = standardScaler_data(X, y)
    data = pd.concat([X, y], axis=1)
    ti.xcom_push(key='transform_data', value=data)
    data.to_csv('/opt/airflow/dags/clean_train.csv',index=False)
# def get_connection():
#     return create_engine(
#         f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
#     )

# def load_data_to_DB(**kwargs):
#     ti = kwargs['ti']
#     data = ti.xcom_pull(key='transform_data', task_ids='transform_data')
#     if isinstance(data, pd.DataFrame):
#         engine = get_connection()
#         data.to_sql('credit', con=engine, if_exists='append', index=False)
#     else:
#         print("The data pulled from XCom is not a DataFrame")

def train_model(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='transform_data', task_ids='transform_data')
    
    y = data['BAD']
    X = data.drop(['BAD'], axis=1)
    
    model = LogisticRegression(C=0.01, penalty='l2', solver='liblinear')
    model.fit(X, y)
    pickle.dump(model, open('/opt/airflow/dags/LogisticRegression.pickle', 'wb'))

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='Credit_score_pipeline_v04',
    default_args=default_args,
    start_date=datetime(2024, 8, 2, 10),
    schedule_interval='@daily', 
    catchup=False
) as dag:
    
    extract_data = PythonOperator(
        task_id='extract_data',
        python_callable=extract_csv_data,
        op_kwargs={'file_csv': '/opt/airflow/dags/train.csv'},
        do_xcom_push=True
    )
    
    transform_data = PythonOperator(
        task_id='transform_data',
        python_callable=transform_csv_data,
        do_xcom_push=True
    )
    
    # load_data = PostgresOperator(
    #     task_id='load_data',
    #     postgres_conn_id="postgresql",  
    #     sql="""
    #    COPY credit FROM '/opt/airflow/dags/clean_train.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

    #     """,
    # )
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    
    extract_data >> transform_data >> train_model_task
