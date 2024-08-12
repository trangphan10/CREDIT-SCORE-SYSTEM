from datetime import datetime, timedelta
from airflow import DAG 
from airflow.operators.python import PythonOperator
import pandas as pd
import json 
from pytz import timezone
import os 
import openai

local_tz = timezone('Asia/Ho_Chi_Minh')

def extract_data(file_excel, **kwargs): 
    ti = kwargs['ti']
    if os.path.exists(file_excel):
        file = pd.read_excel(file_excel)
        ti.xcom_push(value=file.to_dict(), key='read_data')
    else:
        raise FileNotFoundError(f"The file {file_excel} does not exist")

def transform_data(**kwargs): 
    ti = kwargs['ti']
    file_dict = ti.xcom_pull(key='read_data', task_ids='extract_data')
    if file_dict is None:
        raise ValueError("No data found from extract_data task")
        
    file = pd.DataFrame.from_dict(file_dict)
    system_title = 'This is a chatbot for supporting students in tackling math problems.'
    all_conversation = []
    for idx, row in file.iterrows(): 
        all_conversation.append({'messages': [
            {'role': 'system', 'content': system_title},
            {'role': 'user', 'content': row['Problem']}, 
            {'role': 'assistant', 'content': row['Solution']}
        ]})
    
    output_path = '/opt/airflow/dags/instances.jsonl'
    with open(output_path, 'w') as f: 
        for conversation in all_conversation: 
            json.dump(conversation, f)
            f.write('\n')

def finetuning(**kwargs): 
    # api_key = os.getenv('OPENAI_API_KEY')
    # if not api_key:
    #     raise ValueError("API key is not set")
    
    client = openai.OpenAI(api_key='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')
    with open('/opt/airflow/dags/instances.jsonl', 'rb') as file: 
        response = client.files.create(file=file, purpose='fine-tune')  
    
    file_id = response.id
    response = client.fine_tuning.jobs.create( 
        training_file=file_id,
        model='gpt-4o-mini-2024-07-18'
    )    
    job_id = response.id 
    print('List:', client.fine_tuning.jobs.list(limit=1))
    print('Retrieve:', client.fine_tuning.jobs.retrieve(job_id))

default_args = {
    'owner': 'admin',
    'retries': 5, 
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='finetune_gpt_v1', 
    default_args=default_args, 
    description='This is the workflow for finetuning GPT', 
    start_date=datetime(2024, 8, 1, 10),
    schedule_interval='@daily', 
    catchup=False
) as dag: 
    task1 = PythonOperator(
        task_id='extract_data', 
        python_callable=extract_data,
        do_xcom_push=True, 
        op_kwargs={'file_excel': '/opt/airflow/dags/Sample_data.xlsx'}
    ) 
    task2 = PythonOperator( 
        task_id='transform_data', 
        python_callable=transform_data
    )
    # task3 = PythonOperator( 
    #     task_id='finetuning_by_GPT', 
    #     python_callable=finetuning
    # )
    
    task1 >> task2
