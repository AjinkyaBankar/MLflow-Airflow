# MLflow Airflow Demo
In this example, I train ElasticNet machine learning model for wine quality prediction using MLflow. 

Important note: To avoid airflow not recognizing user defined modules, maintain the `airflow/dag` directory structure for custom modules as follows-
```
| .airflowignore  -- only needed in ``dags`` folder, see below
| -- my_company
              | __init__.py
              | common_packages
              |               |  __init__.py
              |               | common_module.py
              |
              | my_custom_dags
                              | __init__.py
                              | my_dag0.py
                              | my_dag1.py
                              | my_dag2.py
```

We can see MLflow UI with the help of the following command:
```
$ mlflow ui
```
Then, I use the Apache Airflow for periodic scheduling of the training and prediction pipelines. Before starting Airflow scheduler, set the `AIRFLOW_HOME` directory inside `.bashrc` file or use the following command inside the created `airflow` directory as:
```
$ export AIRFLOW_HOME="$(pwd)"
```
Then start the scheduler in terminals as follows:
```
$ airflow scheduler
```  
Finally, open one more terminal window and start the web server as:
```
$ airflow webserver
```
Then, we can log into the airflow UI to see the running dags. If dags aren't active, we can manually activate or modify the `dags_are_paused_at_creation = False` inside `airflow.cfg` file.
