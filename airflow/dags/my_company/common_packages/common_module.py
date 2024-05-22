import pandas as pd
import numpy as np
import logging
import warnings
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from datetime import datetime
import pytz
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def training_data():
    # Read the wine-quality csv file from the URL
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    print('Training data prepared successfully')
    return train_x, train_y, test_x, test_y


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2    


def model_training():
    experiment_name = "Wine Quality"
    registered_model = "ElasticnetWineModel"
    alpha = round(random.random(), 1)
    l1_ratio = round(random.random(), 1)
    train_x, train_y, test_x, test_y = training_data()

    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    list_runs = mlflow.search_runs(experiment.experiment_id, run_view_type=ViewType.ACTIVE_ONLY, order_by=['metric.r2 DESC'])
    highest_accuracy_run = mlflow.search_runs(experiment.experiment_id,run_view_type=ViewType.ACTIVE_ONLY,max_results=1,order_by=["metrics.r2 DESC"]) 

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Since this is regression problem, lets use R2 as the evaluation metric to decide good quality model
        accuracy = r2
        # Register the model
        if len(list_runs) == 0:
            print("First time running, use default threshold!")
            threshold = 0.05
        else:                       
            threshold = highest_accuracy_run['metrics.r2'].iloc[0]
        
        # If higher than threshold -> deploy model
        if accuracy > threshold:
            print("New model has higher accuracy than that of the current model, model will be deployed shortly!")
            mlflow.sklearn.log_model(lr, "model", registered_model_name=registered_model)
            print("Model has been deployed successfully")
        else:
            print("Model is NOT deployed because accuracy is not higher than current model or is too low!")
            print("Current model accuracy: ", threshold)    


def model_prediction_write_result():
    registered_model_name = "ElasticnetWineModel"
    # Prepare random data for prediction. Use distribution from the original data
    # Read the wine-quality csv file from the URL
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    column_names = list(data.columns)
    column_names = column_names[:-1] # Drop the last name, as it's the target
    predict_df = pd.DataFrame(columns=column_names)
    for column in column_names:
        temp = random.uniform(data[column].min(), data[column].max()) 
        predict_df[column] = [temp]

    print(predict_df)

    # mlflow.set_experiment(experiment_name)
    # client = MlflowClient()
    # experiment = client.get_experiment_by_name(experiment_name)

    # High accuracy model for prediction
    # highest_accuracy_run = mlflow.search_runs(experiment.experiment_id,run_view_type=ViewType.ACTIVE_ONLY,max_results=1,order_by=["metrics.r2 DESC"]) 
    # run_id = highest_accuracy_run['run_id'].iloc[0]
    # model_load = mlflow.sklearn.load_model("runs:/" + run_id + "/model")
    # print('Model is loaded successfully')

    models = mlflow.search_registered_models(filter_string=f"name = '{registered_model_name}'")
    if models:
        latest_model_version = models[0].latest_versions[0].version
        model_load = mlflow.sklearn.load_model(
            model_uri=f"models:/{registered_model_name}/{latest_model_version}"
        )
        print(f"Latest model version in the model registry used for prediction: {latest_model_version}")
    else:
        print(f"No model in the model registry under the name: {registered_model_name}.")  

    predicted_quality = model_load.predict(predict_df)
    print(predicted_quality)

    # Get current timestamp
    utc_now = datetime.utcnow()
    utc_now = pytz.utc.localize(utc_now)
    est_now = utc_now.astimezone(pytz.timezone('US/Eastern'))
    current_time_est = est_now.strftime('%Y-%m-%d %H:%M:%S')

    # Write prediction results in the csv file
    column_names = list(predict_df.columns)
    predict_df['Time'] = current_time_est
    predict_df['Predicted_Quality'] = predicted_quality[0]

    column_names = ['Time'] + column_names + ['Predicted_Quality']
    predict_df = predict_df[column_names]

    filename = 'result.csv'
    # Check if the CSV file exists
    if os.path.exists(filename):
        # If file exists, append DataFrame to it
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, predict_df], ignore_index=True)
        updated_df.to_csv(filename, index=False)
    else:
        # If file doesn't exist, create it and write DataFrame to it
        predict_df.to_csv(filename, index=False)
    
    return None    

# warnings.filterwarnings("ignore")
# np.random.seed(22)
# experiment_name = "Wine Quality"
# registered_model = "ElasticnetWineModel"
# alpha = float(sys.argv[1]) if len(sys.argv) > 1 else round(random.random(), 1)
# l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else round(random.random(), 1) 

# train_x, train_y, test_x, test_y = training_data()
# model_training()

# model_prediction_write_result()
# print('YES+')

