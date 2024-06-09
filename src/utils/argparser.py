import os
import yaml

def get_args(arg_dir):
    with open(arg_dir, "r") as f:
        args = yaml.safe_load(f)

    return args

def save_args(args, dest_dir):
    path = os.path.join(dest_dir, "run_config.yaml")
    with open(path, "w") as f:
        yaml.dump(args, f)