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
        return {k:k in params for k in ("prompt","negative_prompt","width","height","num_inference_steps","guidance_scale","generator","max_sequence_length","true_cfg_scale","callback_on_step_end","callback_on_step_end_tensor_inputs")}
    def generate(self,prompt,negative_prompt="",width=1024,height=1024,steps=4,guidance=0,seed=0,progress_callback=None):
        if self.pipeline is None: raise RuntimeError("Load image model first.")
        total_steps=int(steps)
        if progress_callback:
            progress_callback.update(2,"Preparing","Preparing image pipeline…")
        args={"prompt":prompt,"negative_prompt":negative_prompt or None,"width":int(width),"height":int(height),
              "num_inference_steps":total_steps,"guidance_scale":float(guidance),
              "generator":torch.Generator(device="cpu").manual_seed(int(seed))}
        if progress_callback and self.capabilities().get("callback_on_step_end"):
            def callback(_pipe, step, _timestep, callback_kwargs):
                completed=min(total_steps,int(step)+1)
                progress_callback.update(5+round(90*completed/total_steps),"Generating",f"Image diffusion step {completed}/{total_steps}")
                return callback_kwargs
            args["callback_on_step_end"]=callback
            if self.capabilities().get("callback_on_step_end_tensor_inputs"):
                args["callback_on_step_end_tensor_inputs"]=[]
        args={k:v for k,v in args.items() if self.capabilities().get(k) or k=="callback_on_step_end_tensor_inputs"}
        result=self.pipeline(**args)
        if not getattr(result,"images",None): raise RuntimeError("Pipeline returned no images.")
        if progress_callback: progress_callback.update(98,"Saving","Saving generated image…")
        return result.images[0]
