# import os
# import sys
# sys.path.append('/home/ajinkya/airflow_with_code_env/steps/train_pred')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import logging
import warnings
import random
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from datetime import datetime
import pytz
from my_company.common_packages.common_module import model_prediction_write_result

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


dag = DAG(
    'prediction_pipeline',
    default_args={'start_date': days_ago(1)},
    schedule_interval='*/2 * * * *',
    catchup=False
)

run_prediction_task = PythonOperator(
    task_id='run_prediction',
    python_callable=model_prediction_write_result,
    # op_kwargs={"registered_model_name": "ElasticnetWineModel"},
    dag=dag
)

# Set the dependencies between the tasks
run_prediction_task

# For debugging purpose only
# if __name__ == "__main__":
#     dag.test()