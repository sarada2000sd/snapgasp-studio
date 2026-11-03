from pathlib import Path
import os, time, traceback
import ipywidgets as w
from IPython.display import display, HTML, Video, Image
from .catalog import load_catalog
from .controller import StudioController
from .manager import install_model, uninstall_model, is_installed, output_dir
from .runtime import inspect_runtime

CSS = """
<style>
.sg {background:#090b0e;color:#f5f5f5;border:1px solid #333;border-radius:14px;padding:18px;font-family:Arial}
.sg h2{color:#ffd400;margin:0}.sg small{color:#adb5c0}
</style>
"""

def build_dashboard():
    models=load_catalog()
    by_id={m["id"]:m for m in models}
    controller=StudioController()
    display(HTML(CSS+'<div class="sg"><small>PRIVATE NOTEBOOK WORKSPACE · FREE-TIER FIRST</small><h2>SNAPGASP Studio</h2><p>One model at a time · Google Drive cache · live GPU check</p></div>'))
    tabs=w.Tab()
    im=[m for m in models if m["modality"]=="image"]
    vm=[m for m in models if m["modality"]=="video"]
    im_model=w.Dropdown(options=[(m["name"],m["id"]) for m in im],description="Model")
    im_prompt=w.Textarea(description="Prompt",placeholder="Describe the image…",layout=w.Layout(width="100%",height="90px"))
    im_neg=w.Textarea(description="Negative",layout=w.Layout(width="100%",height="55px"))
    im_w=w.Dropdown(options=[512,768,1024],value=768,description="Width")
    im_h=w.Dropdown(options=[512,768,1024],value=768,description="Height")
    im_steps=w.IntSlider(value=4,min=1,max=40,description="Steps",continuous_update=False)
    im_guidance=w.FloatSlider(value=0,min=0,max=8,step=.5,description="Guidance",continuous_update=False)
    im_seed=w.IntText(value=12345,description="Seed")
    im_load=w.Button(description="Load model",button_style="warning")
    im_go=w.Button(description="Generate image",button_style="success")
    im_status=w.Output()
    im_out=w.Output()
    def image_load(_):
        with im_status:
            im_status.clear_output()
            try: print(controller.load(by_id[im_model.value]))
            except Exception as e: print("ERROR:",e)
    def image_generate(_):
        with im_status:
            im_status.clear_output()
            try:
                adapter=controller.require_active(im_model.value)
                result=adapter.generate(prompt=im_prompt.value,negative_prompt=im_neg.value,width=im_w.value,height=im_h.value,steps=im_steps.value,guidance=im_guidance.value,seed=im_seed.value)
                path=output_dir("images")/f"image_{int(time.time())}_seed{im_seed.value}.png"
                result.save(path)
                with im_out:
                    im_out.clear_output()
                    display(Image(filename=str(path)))
                    print("Saved:",path)
            except Exception as e: print("ERROR:",e)
    im_load.on_click(image_load); im_go.on_click(image_generate)
    image_tab=w.VBox([im_model,im_prompt,im_neg,w.HBox([im_w,im_h]),im_steps,im_guidance,im_seed,w.HBox([im_load,im_go]),im_status,im_out])

    vid_model=w.Dropdown(options=[(m["name"]+" · "+m["state"],m["id"]) for m in vm],description="Model")
    vid_prompt=w.Textarea(description="Prompt",placeholder="Describe the scene and motion…",layout=w.Layout(width="100%",height="90px"))
    vid_neg=w.Textarea(description="Negative",layout=w.Layout(width="100%",height="55px"))
    vid_w=w.Dropdown(options=[480,640,720],value=480,description="Width")
    vid_h=w.Dropdown(options=[320,384,480],value=320,description="Height")
    vid_frames=w.Dropdown(options=[17,25,49],value=17,description="Frames")
    vid_fps=w.Dropdown(options=[8,12],value=8,description="FPS")
    vid_steps=w.IntSlider(value=20,min=10,max=40,description="Steps",continuous_update=False)
    vid_seed=w.IntText(value=12345,description="Seed")
    vid_load=w.Button(description="Load model",button_style="warning")
    vid_go=w.Button(description="Generate video",button_style="success")
    vid_status=w.Output(); vid_out=w.Output()
    def video_load(_):
        with vid_status:
            vid_status.clear_output()
            try: print(controller.load(by_id[vid_model.value]))
            except Exception as e: print("ERROR:",e)
    def video_generate(_):
        with vid_status:
            vid_status.clear_output()
            try:
                adapter=controller.require_active(vid_model.value)
                path=adapter.generate(prompt=vid_prompt.value,negative_prompt=vid_neg.value,width=vid_w.value,height=vid_h.value,frames=vid_frames.value,fps=vid_fps.value,steps=vid_steps.value,seed=vid_seed.value)
                with vid_out:
                    vid_out.clear_output()
                    display(Video(path,embed=True))
                    print("Saved:",path)
            except Exception as e: print("ERROR:",e)
    vid_load.on_click(video_load); vid_go.on_click(video_generate)
    video_tab=w.VBox([vid_model,vid_prompt,vid_neg,w.HBox([vid_w,vid_h,vid_frames,vid_fps]),vid_steps,vid_seed,w.HBox([vid_load,vid_go]),vid_status,vid_out])

    mgr_model=w.Dropdown(options=[(m["name"]+" · "+m["state"],m["id"]) for m in models],description="Model")
    install=w.Button(description="Install / resume",button_style="success")
    update=w.Button(description="Force update")
    remove=w.Button(description="Uninstall",button_style="danger")
    unload=w.Button(description="Unload active model")
    mgr_status=w.Output()
    def manager_action(action):
        with mgr_status:
            mgr_status.clear_output()
            spec=by_id[mgr_model.value]
            try:
                if action=="install": print(install_model(spec))
                elif action=="update": print(install_model(spec,force=True))
                elif action=="remove":
                    if controller.active_model_id==spec["id"]: controller.unload()
                    print(uninstall_model(spec["id"]))
                else: print(controller.unload())
            except Exception as e: print("ERROR:",e)
    install.on_click(lambda _:manager_action("install"))
    update.on_click(lambda _:manager_action("update"))
    remove.on_click(lambda _:manager_action("remove"))
    unload.on_click(lambda _:manager_action("unload"))
    manager_tab=w.VBox([mgr_model,w.HBox([install,update,remove,unload]),mgr_status])

    refresh=w.Button(description="Refresh GPU status",icon="refresh")
    sys_out=w.Output()
    history=w.Button(description="Refresh output history",icon="history")
    hist_out=w.Output()
    def show_runtime(_=None):
        with sys_out:
            sys_out.clear_output()
            info=inspect_runtime()
            for k,v in info.items(): print(f"{k}: {v}")
    def show_history(_=None):
        with hist_out:
            hist_out.clear_output()
            root=Path(os.environ.get("SNAPGASP_DATA_DIR","./snapgasp-data"))/"outputs"
            files=sorted([p for p in root.rglob("*") if p.is_file()],key=lambda p:p.stat().st_mtime,reverse=True)[:20] if root.exists() else []
            if not files: print("No generated files yet.")
            for p in files: print(p)
    refresh.on_click(show_runtime); history.on_click(show_history)
    system_tab=w.VBox([refresh,sys_out,w.HTML("<h4>Recent outputs</h4>"),history,hist_out])
    tabs.children=[image_tab,video_tab,manager_tab,system_tab]
    for i,t in enumerate(["Image Studio","Video Studio","Model Manager","System & History"]): tabs.set_title(i,t)
    show_runtime(); show_history()
    return tabs
