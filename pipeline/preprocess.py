import os
import pandas as pd

def run(config):
    raw_path = os.path.join(config['data_dir'], 'cmi-detect-behavior-with-sensor-data', 'train.csv')
    df = pd.read_csv(raw_path)
    df.dropna(inplace=True)
    # Example cleanup; customize as needed
    return df