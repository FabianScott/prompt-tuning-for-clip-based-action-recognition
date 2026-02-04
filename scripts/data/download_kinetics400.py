import os
import sys

if os.getcwd().endswith("notebooks"):
    os.chdir("..")
print(f"Current working directory: {os.getcwd()}")
sys.path.append(os.getcwd())

import fiftyone as fo
import fiftyone.zoo as foz
from src.configs.paths import DATA_RAW_PATH


fo.config.dataset_zoo_dir = DATA_RAW_PATH
dataset = foz.load_zoo_dataset("kinetics-400", split="train", retry_errors=True, num_workers=8, cleanup=False)
