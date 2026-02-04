import os


def read_token(token_file: str, token_name=None):
    token_name = "Unamed" if token_name is None else token_name
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"{token_name} token file not found. Please create '{token_file}' with your token.")
    with open(token_file, 'r') as f:
        token = f.read().strip()
    return token