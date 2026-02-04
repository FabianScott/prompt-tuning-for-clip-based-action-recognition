import sys
from pathlib import Path

# Find project root
project_root = Path(__file__).resolve().parent
while not (project_root / "README.md").exists() and project_root.parent != project_root:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

import fiftyone as fo
import fiftyone.zoo as foz
from src.configs.paths import DATA_RAW_PATH


fo.config.dataset_zoo_dir = DATA_RAW_PATH
dataset = foz.load_zoo_dataset("kinetics-400", split="train", retry_errors=True, num_workers=8, cleanup=False)
