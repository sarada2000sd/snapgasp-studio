import os, re, shutil
from pathlib import Path
def data_root():
    p=Path(os.environ.get("SNAPGASP_DATA_DIR","./snapgasp-data")).expanduser(); p.mkdir(parents=True,exist_ok=True); return p
def model_dir(model_id): return data_root()/"models"/re.sub(r"[^a-zA-Z0-9_.-]+","_",model_id)
def output_dir(kind):
    p=data_root()/"outputs"/kind; p.mkdir(parents=True,exist_ok=True); return p
def is_installed(model_id):
    p=model_dir(model_id); return p.exists() and any(p.iterdir())
def install_model(spec, force=False):
    from huggingface_hub import snapshot_download
    p=model_dir(spec["id"]); p.parent.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo_id=spec["repo_id"],repo_type="model",local_dir=str(p),force_download=force)
    return str(p)
def uninstall_model(model_id):
    p=model_dir(model_id)
    if p.exists(): shutil.rmtree(p)
    return f"Removed {p}" if p.exists() else f"Model cache removed: {p}"
