import inspect, torch
from diffusers import DiffusionPipeline
from .base import ModelAdapter

class DiffusersImageAdapter(ModelAdapter):
    def load(self, runtime):
        dtype = torch.float16
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): dtype=torch.bfloat16
        except Exception: pass
        self.pipeline=DiffusionPipeline.from_pretrained(str(self.local_path),torch_dtype=dtype)
        if torch.cuda.is_available() and hasattr(self.pipeline,"enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload()
        elif torch.cuda.is_available() and hasattr(self.pipeline,"enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
        else: self.pipeline.to("cpu")
    def capabilities(self):
        if self.pipeline is None: return {}
        try: params=inspect.signature(self.pipeline.__call__).parameters
        except (TypeError,ValueError): params={}
        return {k:k in params for k in ("prompt","negative_prompt","width","height","num_inference_steps","guidance_scale","generator","max_sequence_length","true_cfg_scale")}
    def generate(self,prompt,negative_prompt="",width=1024,height=1024,steps=4,guidance=0,seed=0):
        if self.pipeline is None: raise RuntimeError("Load image model first.")
        args={"prompt":prompt,"negative_prompt":negative_prompt or None,"width":int(width),"height":int(height),
              "num_inference_steps":int(steps),"guidance_scale":float(guidance),
              "generator":torch.Generator(device="cpu").manual_seed(int(seed))}
        args={k:v for k,v in args.items() if self.capabilities().get(k)}
        result=self.pipeline(**args)
        if not getattr(result,"images",None): raise RuntimeError("Pipeline returned no images.")
        return result.images[0]
