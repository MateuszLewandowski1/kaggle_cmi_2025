import mlflow
from sklearn.metrics import f1_score

def run(model, X_test, y_test, config):
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')

    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run(nested=True):
        mlflow.log_metric("f1_score", f1)
    print(f"F1 Score: {f1:.4f}")