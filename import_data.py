import pandas as pd

source_file = "data/spotify_data.csv"

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
