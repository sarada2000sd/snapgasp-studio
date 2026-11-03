import os, platform, sys
def inspect_runtime():
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cuda_available": False, "gpu_name": "No CUDA GPU detected",
            "gpu_total_gb": 0.0, "gpu_free_gb": 0.0, "system_ram_gb": 0.0,
            "system_ram_available_gb": 0.0,
            "gpu_compute_capability": None,
            "data_dir": os.environ.get("SNAPGASP_DATA_DIR", "./snapgasp-data")}
    try:
        import psutil
        mem=psutil.virtual_memory()
        info["system_ram_gb"] = round(mem.total / 2**30, 1)
        info["system_ram_available_gb"] = round(mem.available / 2**30, 1)
    except Exception: pass
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            info.update(gpu_name=p.name, gpu_total_gb=round(total/2**30,1),
                        gpu_free_gb=round(free/2**30,1),
                        gpu_compute_capability=f"{p.major}.{p.minor}",
                        cuda_version=torch.version.cuda)
        else: info["warning"] = "No CUDA GPU attached."
    except Exception as e: info["warning"] = str(e)
    return info
def can_attempt_model(spec, runtime):
    if spec.get("state") not in ("implemented", "experimental") or not spec.get("adapter"):
        return False, "Adapter not implemented."
    if not runtime.get("cuda_available"): return False, "Attach a CUDA GPU first."
    need_gpu, total_gpu = float(spec.get("estimated_min_vram_gb",0)), float(runtime.get("gpu_total_gb",0))
    need_ram, total_ram = float(spec.get("estimated_min_ram_gb",0)), float(runtime.get("system_ram_gb",0))
    reasons=[]
    if total_gpu < need_gpu: reasons.append(f"estimated minimum VRAM {need_gpu:g} GB; detected {total_gpu:g} GB")
    if total_ram < need_ram: reasons.append(f"estimated minimum system RAM {need_ram:g} GB; detected {total_ram:g} GB")
    if reasons: return False, "Hardware check: " + "; ".join(reasons) + "."
    warnings=[]
    recommended=spec.get("recommended_compute_capability")
    actual=runtime.get("gpu_compute_capability")
    if recommended and actual:
        try:
            if float(actual) < float(recommended):
                warnings.append(
                    f"GPU capability {actual} is below the model's recommended {recommended}. "
                    "The model authors' optimized setup is aimed at Ampere 8.0+; this older-GPU run is experimental."
                )
        except (TypeError, ValueError): pass
    base="Hardware estimate passed; actual peak memory and runtime compatibility are not guaranteed."
    if warnings: base += " WARNING: " + " ".join(warnings)
    return True, base
