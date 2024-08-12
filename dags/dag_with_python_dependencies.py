from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator 

default_args={ 
              'owner':'airflow', 
              'retry':5, 
              'retry_delay':timedelta(minutes=5)}

def get_sklearn(): 
    import sklearn 
    print(f'scikit_learn with version: {sklearn.__version__}')
with DAG(
    default_args=default_args,
    dag_id ='dag_with_python_dependencies_v01',
    start_date=datetime(2024,8,1,10),
    schedule_interval='@daily'
) as dag: 
    
    get_sklearn=PythonOperator(
        task_id ='get_sklearn', 
        python_callable=get_sklearn
    )
    get_sklearn