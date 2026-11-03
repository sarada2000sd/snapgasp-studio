import os, re, shutil
from pathlib import Path

def data_root():
    p=Path(os.environ.get("SNAPGASP_DATA_DIR","./snapgasp-data")).expanduser()
    p.mkdir(parents=True,exist_ok=True)
    return p

def model_root():
    # Optional separate cache for large model weights (e.g. Colab's temporary /content disk).
    p=Path(os.environ.get("SNAPGASP_MODEL_DIR",str(data_root()/"models"))).expanduser()
    p.mkdir(parents=True,exist_ok=True)
    return p

def model_dir(model_id):
    return model_root()/re.sub(r"[^a-zA-Z0-9_.-]+","_",model_id)

def output_dir(kind):
    p=data_root()/"outputs"/kind
    p.mkdir(parents=True,exist_ok=True)
    return p

def is_installed(model_id):
    p=model_dir(model_id)
    return p.exists() and any(p.iterdir())

def install_model(spec, force=False):
    from huggingface_hub import snapshot_download
    p=model_dir(spec["id"])
    p.parent.mkdir(parents=True,exist_ok=True)
    required_gb=float(spec.get("estimated_min_disk_gb",0))
    if required_gb:
        used_gb=sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/(1024**3) if p.exists() else 0
        free_gb=shutil.disk_usage(p.parent).free/(1024**3)
        if free_gb+used_gb < required_gb:
            cache_location=str(p.parent)
            raise RuntimeError(
                f"Not enough free space in the model cache ({cache_location}). "
                f"This model needs about {required_gb:.0f} GB; this disk has "
                f"{free_gb:.1f} GB free and {used_gb:.1f} GB already downloaded. "
                "Choose a cache location with more free space or remove unused files."
            )
    snapshot_download(repo_id=spec["repo_id"],repo_type="model",local_dir=str(p),force_download=force)
    return str(p)

def uninstall_model(model_id):
    p=model_dir(model_id)
    if p.exists():
        shutil.rmtree(p)
        return f"Removed model cache: {p}"
    return f"No model cache found: {p}"
