from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run(df, config):
    # Basic features: mean, std over rolling windows (as example)
    window_size = 20
    df['acc_mean'] = df['acc_x'].rolling(window_size).mean().fillna(0)
    df['acc_std'] = df['acc_x'].rolling(window_size).std().fillna(0)

    features = ['acc_mean', 'acc_std']

    X = df[features]
    y = df['sequence_type']

    return train_test_split(X, y, test_size=0.2, random_state=42)
