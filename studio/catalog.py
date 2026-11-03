import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def load_catalog():
    data = json.loads((ROOT / "config" / "models.json").read_text(encoding="utf-8"))
    models = data.get("models", [])
    ids = [m["id"] for m in models]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate model IDs")
    return models
def get_model(model_id):
    for model in load_catalog():
        if model["id"] == model_id: return model
    raise KeyError(model_id)
