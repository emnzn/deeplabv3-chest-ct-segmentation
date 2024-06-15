import os
import yaml
from typing import Dict, Any

def get_args(arg_dir: str) -> Dict[str, Any]:
    """
    Parameters
    ----------
    arg_dir: str
        The path to the YAML file.

    Returns
    -------
    args: Dict[str, Any]:
        A dictionary containing the YAML file contents.
    """
    
    with open(arg_dir, "r") as f:
        args = yaml.safe_load(f)

    return args

def save_args(args, dest_dir):
    path = os.path.join(dest_dir, "run_config.yaml")
    with open(path, "w") as f:
        yaml.dump(args, f)