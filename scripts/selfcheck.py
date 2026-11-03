"""Catalogue and import smoke test; does not download model weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio.catalog import load_catalog
models=load_catalog()
assert models
assert len({m["id"] for m in models})==len(models)
for m in models:
    assert m.get("name") and m.get("repo_id")
    assert m.get("modality") in {"image","video"}
    if m.get("state")=="implemented": assert m.get("adapter")
print(f"SNAPGASP Studio catalogue OK: {len(models)} models.")
