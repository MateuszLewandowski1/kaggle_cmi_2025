import os
import yaml
from pipeline import preprocess, features, model, evaluate, optimize

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Preprocessing
data = preprocess.run(config)

# Feature Engineering
X_train, X_test, y_train, y_test = features.run(data, config)

# Modeling
trained_model = model.run(X_train, y_train, config)

# Evaluation
evaluate.run(trained_model, X_test, y_test, config)

# Optimization (decide next step)
optimize.run(config)