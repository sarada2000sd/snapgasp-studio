from pathlib import Path
import os, time, html, uuid
from IPython.display import display, HTML, Video, Image
from google.colab import output

from .catalog import load_catalog
from .controller import StudioController
from .manager import install_model, uninstall_model, is_installed, output_dir
from .runtime import inspect_runtime, can_attempt_model


def build_dashboard():
    models = load_catalog()
    by_id = {m["id"]: m for m in models}
    image_models = [m for m in models if m.get("modality") == "image"]
    video_models = [m for m in models if m.get("modality") == "video"]
    controller = StudioController()
    prefix = "sg_" + uuid.uuid4().hex[:8]

    def opt(model):
        state = "" if model.get("state") == "implemented" else " · planned"
        return f'<option value="{html.escape(model["id"])}">{html.escape(model["name"] + state)}</option>'

    def status(target, message, ok=False):
        from google.colab import output as colab_output
        color = "#16a34a" if ok else "#f59e0b"
        safe = html.escape(str(message))
        with colab_output.redirect_to_element("#" + target):
            display(HTML(f'<div style="padding:10px;border-radius:8px;background:#15191f;color:{color};white-space:pre-wrap">{safe}</div>'))

    def check_spec(model_id):
        if model_id not in by_id:
            raise ValueError("Unknown model selection.")
        spec = by_id[model_id]
        if spec.get("state") != "implemented" or not spec.get("adapter"):
            raise RuntimeError("This model is marked Planned. Its adapter is not implemented yet.")
        allowed, reason = can_attempt_model(spec, inspect_runtime())
        if not allowed:
            raise RuntimeError(reason)
        return spec, reason

    def install_cb(model_id):
        target = prefix + "_manager_status"
        try:
            spec, reason = check_spec(model_id)
            status(target, reason + "\\nDownloading model weights to Google Drive…", True)
            path = install_model(spec)
            status(target, f"Model installed successfully.\\nSaved in: {path}", True)
        except Exception as exc:
            status(target, f"INSTALL ERROR: {type(exc).__name__}: {exc}")

    def update_cb(model_id):
        target = prefix + "_manager_status"
        try:
            spec, reason = check_spec(model_id)
            status(target, reason + "\\nUpdating model files…", True)
            path = install_model(spec, force=True)
            status(target, f"Model updated.\\nSaved in: {path}", True)
        except Exception as exc:
            status(target, f"UPDATE ERROR: {type(exc).__name__}: {exc}")

    def uninstall_cb(model_id):
        target = prefix + "_manager_status"
        try:
            if controller.active_model_id == model_id:
                controller.unload()
            status(target, uninstall_model(model_id), True)
        except Exception as exc:
            status(target, f"UNINSTALL ERROR: {type(exc).__name__}: {exc}")

    def unload_cb():
        status(prefix + "_manager_status", controller.unload(), True)

    def load_cb(model_id, modality):
        target = prefix + "_" + modality + "_status"
        try:
            spec, reason = check_spec(model_id)
            result = controller.load(spec)
            status(target, result + "\\n" + reason, True)
        except Exception as exc:
            status(target, f"LOAD ERROR: {type(exc).__name__}: {exc}")

    def image_generate_cb(model_id, prompt, negative, width, height, steps, guidance, seed):
        target = prefix + "_image_result"
        try:
            if not str(prompt).strip():
                raise ValueError("Please enter an image prompt.")
            spec, _ = check_spec(model_id)
            adapter = controller.require_active(model_id)
            image = adapter.generate(prompt=str(prompt), negative_prompt=str(negative),
                                     width=int(width), height=int(height), steps=int(steps),
                                     guidance=float(guidance), seed=int(seed))
            path = output_dir("images") / f"image_{int(time.time())}_seed{int(seed)}.png"
            image.save(path)
            with output.redirect_to_element("#" + target):
                display(HTML(f'<div style="color:#16a34a;padding:8px">Image generated · saved to Drive: {html.escape(str(path))}</div>'))
                display(Image(filename=str(path)))
        except Exception as exc:
            status(target, f"IMAGE ERROR: {type(exc).__name__}: {exc}")

    def video_generate_cb(model_id, prompt, negative, width, height, frames, fps, steps, seed):
        target = prefix + "_video_result"
        try:
            if not str(prompt).strip():
                raise ValueError("Please enter a video prompt.")
            spec, _ = check_spec(model_id)
            adapter = controller.require_active(model_id)
            path = adapter.generate(prompt=str(prompt), negative_prompt=str(negative),
                                    width=int(width), height=int(height), frames=int(frames),
                                    fps=int(fps), steps=int(steps), seed=int(seed))
            with output.redirect_to_element("#" + target):
                display(HTML(f'<div style="color:#16a34a;padding:8px">Video generated · saved to Drive: {html.escape(str(path))}</div>'))
                display(Video(path, embed=True))
        except Exception as exc:
            status(target, f"VIDEO ERROR: {type(exc).__name__}: {exc}")

    def gpu_cb():
        info = inspect_runtime()
        lines = "\\n".join(f"{k}: {v}" for k, v in info.items())
        status(prefix + "_system_status", lines, info.get("cuda_available", False))

    def history_cb():
        root = Path(os.environ.get("SNAPGASP_DATA_DIR", "./snapgasp-data")) / "outputs"
        files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:30] if root.exists() else []
        message = "\\n".join(str(p) for p in files) if files else "No generated files yet."
        status(prefix + "_history_status", message, True)

    callbacks = {
        "install": install_cb, "update": update_cb, "uninstall": uninstall_cb, "unload": unload_cb,
        "image_load": lambda model_id: load_cb(model_id, "image"),
        "video_load": lambda model_id: load_cb(model_id, "video"),
        "image_generate": image_generate_cb, "video_generate": video_generate_cb,
        "gpu": gpu_cb, "history": history_cb
    }
    for name, function in callbacks.items():
        output.register_callback(prefix + "_" + name, function)

    image_options = "".join(opt(m) for m in image_models)
    video_options = "".join(opt(m) for m in video_models)
    manager_options = "".join(opt(m) for m in models)

    page = f"""
    <div id="{prefix}" class="sg-shell">
      <style>
        #{prefix}{{font-family:Arial,sans-serif;background:#090b0e;color:#f4f4f5;border:1px solid #333;border-radius:16px;padding:18px;max-width:1100px}}
        #{prefix} .brand{{border-bottom:1px solid #333;padding:4px 4px 16px;margin-bottom:14px}}
        #{prefix} .brand small{{color:#a1a1aa;letter-spacing:1px}}
        #{prefix} h2{{color:#ffd400;font-size:25px;margin:5px 0}}
        #{prefix} .tabs{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 18px}}
        #{prefix} button{{cursor:pointer;border:1px solid #555;border-radius:8px;padding:10px 14px;background:#20242b;color:#eee;font-weight:600}}
        #{prefix} button.active{{background:#ffd400;color:#111;border-color:#ffd400}}
        #{prefix} button.primary{{background:#ffd400;color:#111;border-color:#ffd400}}
        #{prefix} button.danger{{border-color:#9f3333;color:#ffb4b4}}
        #{prefix} .panel{{display:none;background:#11151a;border:1px solid #333;border-radius:12px;padding:16px}}
        #{prefix} .panel.active{{display:block}}
        #{prefix} label{{display:block;color:#d4d4d8;font-size:13px;margin:10px 0 5px}}
        #{prefix} input,#{prefix} select,#{prefix} textarea{{box-sizing:border-box;width:100%;padding:10px;border-radius:7px;border:1px solid #454b55;background:#080a0d;color:#fff;font:14px Arial}}
        #{prefix} textarea{{min-height:72px;resize:vertical}}
        #{prefix} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
        #{prefix} .actions{{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}}
        #{prefix} .status{{margin-top:12px;min-height:4px}}
        #{prefix} .hint{{color:#a1a1aa;font-size:12px;line-height:1.5;margin-top:8px}}
        #{prefix} .result{{margin-top:12px;overflow:auto}}
        #{prefix} .result img,#{prefix} .result video{{max-width:100%;height:auto;border-radius:10px}}
        @media(max-width:650px){{#{prefix} .grid{{grid-template-columns:1fr}}}}
      </style>
      <div class="brand"><small>PRIVATE NOTEBOOK WORKSPACE · FREE-TIER FIRST</small><h2>SNAPGASP Studio</h2><div>Image & video generation · Google Drive cache · one active model</div></div>
      <div class="tabs">
        <button class="active" onclick="sgTab('image')">Image Studio</button>
        <button onclick="sgTab('video')">Video Studio</button>
        <button onclick="sgTab('manager')">Model Manager</button>
        <button onclick="sgTab('system')">System & History</button>
      </div>
      <section class="panel active" id="{prefix}_image">
        <h3>Image Studio</h3>
        <label>Model</label><select id="{prefix}_im_model">{image_options}</select>
        <label>Prompt</label><textarea id="{prefix}_im_prompt" placeholder="Describe the image you want to create…"></textarea>
        <label>Negative prompt</label><textarea id="{prefix}_im_negative" placeholder="Elements to avoid (optional)"></textarea>
        <div class="grid">
          <div><label>Width</label><select id="{prefix}_im_width"><option>512</option><option selected>768</option><option>1024</option></select></div>
          <div><label>Height</label><select id="{prefix}_im_height"><option>512</option><option selected>768</option><option>1024</option></select></div>
          <div><label>Steps</label><input id="{prefix}_im_steps" type="number" min="1" max="50" value="4"></div>
          <div><label>Guidance</label><input id="{prefix}_im_guidance" type="number" min="0" max="12" step="0.5" value="0"></div>
          <div><label>Seed</label><input id="{prefix}_im_seed" type="number" value="12345"></div>
        </div>
        <div class="actions">
          <button onclick="sgCall('image_load',[sgVal('im_model')],'image_status')">Load selected model</button>
          <button class="primary" onclick="sgImage()">Generate image</button>
        </div>
        <div class="hint">Install the selected model first in Model Manager. The Studio checks GPU/RAM estimates before installation.</div>
        <div class="status" id="{prefix}_image_status"></div><div class="result" id="{prefix}_image_result"></div>
      </section>
      <section class="panel" id="{prefix}_video">
        <h3>Video Studio</h3>
        <label>Model</label><select id="{prefix}_vi_model">{video_options}</select>
        <label>Prompt</label><textarea id="{prefix}_vi_prompt" placeholder="Describe the scene, subject and motion…"></textarea>
        <label>Negative prompt</label><textarea id="{prefix}_vi_negative" placeholder="Elements to avoid (optional)"></textarea>
        <div class="grid">
          <div><label>Width</label><select id="{prefix}_vi_width"><option selected>720</option></select></div>
          <div><label>Height</label><select id="{prefix}_vi_height"><option selected>480</option></select></div>
          <div><label>Frames</label><select id="{prefix}_vi_frames"><option selected>49</option></select></div>
          <div><label>FPS</label><select id="{prefix}_vi_fps"><option selected>8</option></select></div>
          <div><label>Steps</label><input id="{prefix}_vi_steps" type="number" min="30" max="50" step="10" value="30"></div>
          <div><label>Seed</label><input id="{prefix}_vi_seed" type="number" value="12345"></div>
        </div>
        <div class="actions">
          <button onclick="sgCall('video_load',[sgVal('vi_model')],'video_status')">Load selected model</button>
          <button class="primary" onclick="sgVideo()">Generate video</button>
        </div>
        <div class="hint">CogVideoX-2B settings are fixed to 720×480, 49 frames, 8 FPS in this starter adapter. Video generation may run out of T4 memory.</div>
        <div class="status" id="{prefix}_video_status"></div><div class="result" id="{prefix}_video_result"></div>
      </section>
      <section class="panel" id="{prefix}_manager">
        <h3>Model Manager</h3>
        <label>Select model</label><select id="{prefix}_manager_model">{manager_options}</select>
        <div class="actions">
          <button class="primary" onclick="sgCall('install',[sgVal('manager_model')],'manager_status')">Install / resume</button>
          <button onclick="sgCall('update',[sgVal('manager_model')],'manager_status')">Force update</button>
          <button class="danger" onclick="sgCall('uninstall',[sgVal('manager_model')],'manager_status')">Uninstall</button>
          <button onclick="sgCall('unload',[],'manager_status')">Unload active model</button>
        </div>
        <div class="hint">Model weights are saved in Google Drive, not GitHub. Planned models cannot be installed until their adapters are implemented.</div>
        <div class="status" id="{prefix}_manager_status"></div>
      </section>
      <section class="panel" id="{prefix}_system">
        <h3>System & History</h3>
        <div class="actions"><button class="primary" onclick="sgCall('gpu',[],'system_status')">Refresh GPU status</button><button onclick="sgCall('history',[],'history_status')">Refresh output history</button></div>
        <pre class="status" id="{prefix}_system_status" style="white-space:pre-wrap;color:#eee"></pre>
        <h4>Recent outputs</h4><pre class="status" id="{prefix}_history_status" style="white-space:pre-wrap;color:#eee"></pre>
      </section>
    </div>
    <script>
      const sgRoot = document.getElementById("{prefix}");
      function sgTab(name) {{
        sgRoot.querySelectorAll(".panel").forEach(x=>x.classList.remove("active"));
        sgRoot.querySelectorAll(".tabs button").forEach(x=>x.classList.remove("active"));
        document.getElementById("{prefix}_"+name).classList.add("active");
        const b=[...sgRoot.querySelectorAll(".tabs button")].find(x=>x.textContent.toLowerCase().startsWith(name==="manager"?"model":name==="system"?"system":name));
        if(b)b.classList.add("active");
      }}
      function sgVal(key) {{ return document.getElementById("{prefix}_"+key).value; }}
      function sgCall(action,args,target) {{
        document.getElementById("{prefix}_"+target).textContent="Working…";
        google.colab.kernel.invokeFunction("{prefix}_"+action,args,{{}});
      }}
      function sgImage() {{
        sgCall("image_generate",[sgVal("im_model"),sgVal("im_prompt"),sgVal("im_negative"),sgVal("im_width"),sgVal("im_height"),sgVal("im_steps"),sgVal("im_guidance"),sgVal("im_seed")],"image_result");
      }}
      function sgVideo() {{
        sgCall("video_generate",[sgVal("vi_model"),sgVal("vi_prompt"),sgVal("vi_negative"),sgVal("vi_width"),sgVal("vi_height"),sgVal("vi_frames"),sgVal("vi_fps"),sgVal("vi_steps"),sgVal("vi_seed")],"video_result");
      }}
    </script>
    """
    display(HTML(page))
    gpu_cb()
    history_cb()
    return {"controller": controller, "prefix": prefix}
