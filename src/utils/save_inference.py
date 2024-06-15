import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def save_inference(results_table, dest_dir):
    """
    Parameters
    ----------
    results_table: dict
        The table of results where keys are columns and values are a list of rows.

    dest_dir: str
        The inference directory where the results will be saved.
    """
    
    df = pd.DataFrame(results_table)
    save_path = os.path.join(dest_dir, "tables")

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    df.to_parquet(os.path.join(save_path, "inference_results.parquet", index=False))