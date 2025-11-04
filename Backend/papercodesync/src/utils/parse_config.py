from pathlib import Path
import yaml

def load_config(file_path):
    p = Path(file_path)
    if not p.is_absolute():
        # resolve relative to project root: src/.. = ROOT
        p = (Path(__file__).resolve().parent.parent / p).resolve()
    with open(p, "r") as f:
        return yaml.safe_load(f)
