from .adapters.diffusers_image import DiffusersImageAdapter
from .adapters.cogvideox import CogVideoXAdapter
from .adapters.cogvideox15 import CogVideoX15Adapter
from .manager import is_installed,model_dir
from .runtime import inspect_runtime,can_attempt_model

class StudioController:
    def __init__(self): self.active_model_id=None; self.active_adapter=None
    def load(self,spec):
        if self.active_model_id==spec["id"] and self.active_adapter: return "Already loaded."
        self.unload()
        runtime=inspect_runtime()
        allowed,msg=can_attempt_model(spec,runtime)
        if not allowed: raise RuntimeError(msg)
        if not is_installed(spec["id"]): raise RuntimeError("Install model first in Model Manager.")
        cls={"diffusers_image":DiffusersImageAdapter,"cogvideox":CogVideoXAdapter,"cogvideox15":CogVideoX15Adapter}.get(spec.get("adapter"))
        if cls is None: raise RuntimeError("No adapter for this model.")
        adapter=cls(spec,model_dir(spec["id"]))
        adapter.load(runtime)
        self.active_model_id,self.active_adapter=spec["id"],adapter
        return f"Loaded {spec['name']}. {msg}"
    def unload(self):
        if self.active_adapter:
            try: self.active_adapter.unload()
            finally: self.active_adapter=None; self.active_model_id=None
            return "Model unloaded."
        return "No model loaded."
    def require_active(self,model_id):
        if not self.active_adapter or self.active_model_id!=model_id: raise RuntimeError("Load selected model first.")
        return self.active_adapter
