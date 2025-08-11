import dagshub
import mlflow


mlflow.set_tracking_uri('https://dagshub.com/sudarshansahane1044/mlopsminiproject.mlflow')
dagshub.init(repo_owner='sudarshansahane1044', repo_name='mlopsminiproject', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)