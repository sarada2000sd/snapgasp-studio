import inspect, time, torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from .base import ModelAdapter
from ..manager import output_dir

class CogVideoXAdapter(ModelAdapter):
    def load(self,runtime):
        dtype=torch.float16
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): dtype=torch.bfloat16
        except Exception: pass
        self.pipeline=CogVideoXPipeline.from_pretrained(str(self.local_path),torch_dtype=dtype)
        if torch.cuda.is_available() and hasattr(self.pipeline,"enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload()
        elif torch.cuda.is_available() and hasattr(self.pipeline,"enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
        else: self.pipeline.to("cpu")
    def capabilities(self):
        if self.pipeline is None: return {}
        try: params=inspect.signature(self.pipeline.__call__).parameters
        except (TypeError,ValueError): params={}
        return {k:k in params for k in ("prompt","negative_prompt","height","width","num_frames","num_inference_steps","guidance_scale","generator")}
    def generate(self,prompt,negative_prompt="",width=720,height=480,frames=49,fps=8,steps=30,guidance=6,seed=0):
        if self.pipeline is None: raise RuntimeError("Load CogVideoX first.")
        values={"prompt":prompt,"negative_prompt":negative_prompt or None,"width":int(width),"height":int(height),
                "num_frames":int(frames),"num_inference_steps":int(steps),"guidance_scale":float(guidance),
                "generator":torch.Generator(device="cpu").manual_seed(int(seed))}
        args={k:v for k,v in values.items() if self.capabilities().get(k)}
        result=self.pipeline(**args)
        frames_out=getattr(result,"frames",None)
        if not frames_out: raise RuntimeError("CogVideoX returned no frames.")
        path=output_dir("videos")/f"cogvideox_{int(time.time())}_seed{seed}.mp4"
        export_to_video(frames_out[0],str(path),fps=int(fps))
        return str(path)
