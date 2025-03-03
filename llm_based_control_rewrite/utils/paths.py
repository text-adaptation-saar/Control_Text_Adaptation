from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
data_auxiliary_dir = Path(REPO_DIR) / "data_auxiliary"

def get_data_auxiliary_dir(lang):
    return data_auxiliary_dir / lang


