import os
import yaml
import pandas as pd


def load_config(path="config/settings.yaml"):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def detect_id_columns(df):
    """
    Detect columns that are likely to be IDs.
    Rules:
    - high uniqueness ratio (> 95%)
    - mostly numeric or string without semantic meaning
    """
    id_columns = []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > 0.95:
            id_columns.append(col)

    return id_columns


def safe_read_csv(path):
    """Read CSV with automatic encoding detection."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def log(message):
    """Simple logger."""
    print(f"[Preprocessing] {message}")
