import os, platform, sys
def inspect_runtime():
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cuda_available": False, "gpu_name": "No CUDA GPU detected",
            "gpu_total_gb": 0.0, "gpu_free_gb": 0.0, "system_ram_gb": 0.0,
            "system_ram_available_gb": 0.0,
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
                        gpu_free_gb=round(free/2**30,1), cuda_version=torch.version.cuda)
        else: info["warning"] = "No CUDA GPU attached."
    except Exception as e: info["warning"] = str(e)
    return info
def can_attempt_model(spec, runtime):
    if spec.get("state") != "implemented" or not spec.get("adapter"):
        return False, "Adapter not implemented."
    if not runtime.get("cuda_available"): return False, "Attach a CUDA GPU first."
    need_gpu, total_gpu = float(spec.get("estimated_min_vram_gb",0)), float(runtime.get("gpu_total_gb",0))
    need_ram, total_ram = float(spec.get("estimated_min_ram_gb",0)), float(runtime.get("system_ram_gb",0))
    reasons=[]
    if total_gpu < need_gpu: reasons.append(f"estimated minimum VRAM {need_gpu:g} GB; detected {total_gpu:g} GB")
    if total_ram < need_ram: reasons.append(f"estimated minimum system RAM {need_ram:g} GB; detected {total_ram:g} GB")
    if reasons: return False, "Hardware check: " + "; ".join(reasons) + "."
    return True, "Hardware check passed as a best-effort estimate; actual peak memory can still be higher."
