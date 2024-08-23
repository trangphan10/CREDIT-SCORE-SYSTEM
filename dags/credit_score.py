from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.providers.postgres.operators.postgres import PostgresOperator
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.metrics import make_scorer
# from imblearn.metrics import specificity_score
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd 
# from sqlalchemy import 
# # from sklearn.linear_model import LogisticRegression
# import mlflow
# from mlflow.tracing import MlflowClient
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("train_model")


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
def transform(data): 
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
    data = pd.concat([X,y],axis = 1)
    return data

def transform_csv_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='read_data', task_ids='extract_data')
    data = transform(data)
    ti.xcom_push(key='transform_data', value=data)
    data.to_csv('/opt/airflow/dags/clean_data.csv',index=False)
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
def split_train_test(**kwargs): 
    ti = kwargs['ti']
    data = ti.xcom_pull(key='transform_data', task_ids='transform_data')
    X = data.drop(['BAD'],axis=1)  # x là các biến quan sát, lược bỏ cột BAD
    y = data['BAD'] # các giá trị cột BAD là biến mục tiêu
    x_train, x_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42, test_size=0.2)
    train_data = pd.concat([x_train,y_train],axis = 1)
    test_data = pd.concat([x_test,y_test],axis=1)
    train_data.to_csv('/opt/airflow/dags/train_data.csv',index=False)
    test_data.to_csv('/opt/airflow/dags/test_data.csv',index = False)
    
def train_model():
    
    train_data = pd.read_csv('/opt/airflow/dags/train_data.csv')
    y_train = train_data['BAD']
    X_train = train_data.drop(['BAD'],axis = 1)
    
    model = RandomForestClassifier(max_depth= 8, min_samples_split= 5, n_estimators= 200)
    
    # with mlflow.start_run():
    #     mlflow.sklearn.log_model(model,
    #                          artifact_path="rf",
    #                          registered_model_name="rf")
    #     # mlflow.log_artifact(local_path="/opt/airflow/dags/credit_score.py",
    #     #                 artifact_path="train_model code")
    #     mlflow.end_run()
    
    model.fit(X_train, y_train)
    pickle.dump(model, open('/opt/airflow/dags/RandomForest.pickle', 'wb'))
def train_and_tune_model(**kwargs):  
    ti = kwargs['ti']
    data = ti.xcom_pull(key='transform_data', task_ids='transform_data')
    
    y = data['BAD']
    X = data.drop(['BAD'],axis = 1)
    
    train_data = pd.read_csv('/opt/airflow/dags/train_data.csv')
    y_train = train_data['BAD']
    X_train = train_data.drop(['BAD'],axis = 1)
    rf = RandomForestClassifier(random_state=99)

    parameters = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 3, 4, 5],
}

    scoring = {
    "auc": "roc_auc",
    # "specificity": make_scorer(specificity_score),
    "recall": "recall",
    "accuracy": "accuracy",
}

    gs_rf = GridSearchCV(rf, parameters, cv=5, scoring=scoring, refit="auc", n_jobs=-1)


    gs_rf.fit(X_train, y_train)

    rf_scores = cross_validate(gs_rf.best_estimator_, X, y, cv=5, n_jobs=-1, verbose=1,
                               return_train_score=True, scoring=scoring)
    print('Score: ',rf_scores)
    pickle.dump(gs_rf, open('/opt/airflow/dags/RandomForest.pickle', 'wb'))
    return rf_scores

    
def test_model(): 
    model = RandomForestClassifier()
    with open('/opt/airflow/dags/RandomForest.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
    test_data = pd.read_csv('/opt/airflow/dags/test_data.csv')
    y_test = test_data['BAD']
    x_test = test_data.drop(['BAD'],axis=1)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:,1]
    with open('/opt/airflow/dags/result.txt','w') as f: 
        f.write(classification_report(y_test,y_pred))
        f.write('Roc_auc_score of model is ' + str(roc_auc_score(y_test,y_score)))
    assert roc_auc_score(y_test,y_score) > 0.8, 'Model is not good.'
    
    
default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='Credit_score_pipeline_v03',
    default_args=default_args,
    start_date=datetime(2024, 8, 2, 10),
    schedule_interval='@daily', 
    catchup=False
) as dag:
    
    extract_data = PythonOperator(
        task_id='extract_data',
        python_callable=extract_csv_data,
        op_kwargs={'file_csv': '/opt/airflow/dags/hmeq.csv'},
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
    split_train_test_task = PythonOperator(
        task_id='train_test_split',
        python_callable=split_train_test
    )
    
    # train_model_task = PythonOperator(
    #     task_id='train_model',
    #     python_callable=train_model
    # )
    
    train_and_tune_model_task = PythonOperator(
        task_id='train_and_tune_model',
        python_callable=train_model
    )
    test_model_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model
    )
    
    
    # extract_data >> transform_data >> split_train_test_task >> train_model_task >> test_model_task
    extract_data >> transform_data >> split_train_test_task >> train_and_tune_model_task >> test_model_task
