import os

from ..configs.read_token import read_token
from ..configs.paths import HUGGINGFACE_CACHE_DIR

def huggingface_login(token_file: str='tokens/huggingface_token.txt'): # type: ignore
    """Login to Hugging Face using a token stored in a file."""
    from huggingface_hub import login
    token = read_token(token_file=token_file, token_name="Hugging Face")

    login(token=token)
    print("Logged in to Hugging Face successfully.")

def set_cache_path(PATH: str | None = None):
    """Set the cache path for Hugging Face Transformers."""
    if PATH is None:
        PATH = HUGGINGFACE_CACHE_DIR

    os.environ['TRANSFORMERS_CACHE'] = PATH
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH
    os.environ['TORCH_HOME'] = PATH

    print(f"Cache path set to: {PATH}")
