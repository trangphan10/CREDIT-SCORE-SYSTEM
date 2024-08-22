from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
# from sklearn.linear_model import LogisticRegression
import pandas as pd 


def extract_csv_data(file_csv, **kwargs):
    ti = kwargs['ti']
    if os.path.exists(file_csv):
        file = pd.read_csv(file_csv)
        ti.xcom_push(key='read_data', value=file)
    else:
        raise FileNotFoundError(f"The file {file_csv} does not exist")
    
def strategy_clean(x,y,is_drop=True):
    # Loại bỏ các hàng có 4 bản ghi nếu is_drop == True
    if is_drop:
        data = pd.concat([x,y],axis=1)
        data.dropna(axis = 0, thresh=3,inplace=True)
        y = data['BAD']
        x = data.drop(['BAD'],axis=1)
    for column in x.columns[1:]:
        if x[column].dtype == 'object':
            # Fill giá trị nhiều nhất vào các cột Reason và job
            x[column].fillna(x[column].mode().iloc[0], inplace=True)
        else:
            # Thay thế các vị trí NA bằng giá trị xuất hiện nhiều nhất xét từng trường hợp y = 0 và y=1
            x.loc[(x[column].isna()) & (y == 0), column] = x[y == 0][column].value_counts().idxmax()
            x.loc[(x[column].isna()) & (y == 1), column] = x[y == 1][column].value_counts().idxmax()
    return x,y

def remove_outliers_iqr(x, y, columns):
    df = pd.concat([x, y], axis=1)
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_final = df[~outlier_mask]

    x = df_final.iloc[:, :len(x.columns)]
    y = df_final.iloc[:, len(x.columns):]

    return x, y

def labelencoding(x):
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
    # Không bỏ các hàng có từ 4 vị trí null trở lên và Triển khai theo chiến lược 2
    X,y = strategy_clean(X,y,False)
    # x_test_3,y_test_3 = strategy_clean(x_test_3,y_test_3,False)
    # Loại bỏ ngoại lệ các cột có giá trị skew trung bình là 3
    remove_outliers_iqr(X,y,['VALUE','DEBTINC'])
    # remove_outliers_iqr(x_test_3,y_test_3,['VALUE','DEBTINC'])
    # LabelEncode
    labelencoding(X)
    # labelencoding(x_test_3,3,False)
# Chuẩn hóa dạng Standard Scaler
    X, y = standardScaler_data(X, y)
    # x_test_3, y_test_3 = standardScaler_data(x_test_3, y_test_3, 3, False)
    data = pd.concat([X,y],axis = 1)
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
    
    model = RandomForestClassifier(max_depth= 8, min_samples_split= 5, n_estimators= 200)
    model.fit(X, y)
    pickle.dump(model, open('/opt/airflow/dags/RandomForest.pickle', 'wb'))

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='Credit_score_pipeline_v01',
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
