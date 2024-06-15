import os
import pandas as pd
from typing import Dict, List

def save_inference(
        results_table: Dict[str, List], 
        dest_dir: str
        ) -> None:
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

    print("prediction", df["prediction"].shape)
    df["prediction"] = df["prediction"].map(lambda x: x.flatten())

    df.to_parquet(os.path.join(save_path, "inference_results.parquet"), index=False)