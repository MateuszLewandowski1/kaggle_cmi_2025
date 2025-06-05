import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from loguru import logger


def run(X_train, y_train, config):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        logger.info("Starting fitting the model")
        model.fit(X_train, y_train)

        mlflow.log_param("model_type", config['model_type'])
        mlflow.sklearn.log_model(model, "model")

    return model