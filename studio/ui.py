from pathlib import Path
import os, time, html, uuid
from IPython.display import display, HTML, Video, Image
from google.colab import output

from .catalog import load_catalog
from .controller import StudioController
from .manager import install_model, uninstall_model, is_installed, output_dir
from .runtime import inspect_runtime, can_attempt_model
from .progress import ColabProgress


def build_dashboard():
    models = load_catalog()
    by_id = {m["id"]: m for m in models}
    image_models = [m for m in models if m.get("modality") == "image"]
    video_models = [m for m in models if m.get("modality") == "video"]
    controller = StudioController()
    prefix = "sg_" + uuid.uuid4().hex[:8]

    def opt(model):
        state = "" if model.get("state") == "implemented" else " · EXPERIMENTAL" if model.get("state") == "experimental" else " · planned"
        selected = " selected" if model["id"] == "cogvideox15-5b" else ""
        return f'<option value="{html.escape(model["id"])}"{selected}>{html.escape(model["name"] + state)}</option>'

    def check_spec(model_id):
        if model_id not in by_id:
            raise ValueError("Unknown model selection.")
        spec = by_id[model_id]
        if spec.get("state") not in ("implemented", "experimental") or not spec.get("adapter"):
            raise RuntimeError("This model is marked Planned. Its adapter is not implemented yet.")
        allowed, reason = can_attempt_model(spec, inspect_runtime())
        if not allowed:
            raise RuntimeError(reason)
        return spec, reason

    def run_status(target, start_detail, action, finished_label="Completed"):
        panel = ColabProgress(target)
        panel.update(0, "Working", start_detail)
        try:
            result = action()
            panel.update(100, finished_label, result or "Done.")
        except Exception as exc:
            panel.update(0, "Failed", f"{type(exc).__name__}: {exc}")

    def install_cb(model_id):
        def action():
            spec, reason = check_spec(model_id)
            path = install_model(spec)
            return f"{reason}\\nModel installed in Google Drive: {path}"
        run_status(prefix + "_manager_status", "Checking Drive space and downloading model files…", action)

    def update_cb(model_id):
        def action():
            spec, reason = check_spec(model_id)
            path = install_model(spec, force=True)
            return f"{reason}\\nModel updated in Google Drive: {path}"
        run_status(prefix + "_manager_status", "Checking Drive space and updating model files…", action)

    def uninstall_cb(model_id):
        def action():
            if controller.active_model_id == model_id:
                controller.unload()
            return uninstall_model(model_id)
        run_status(prefix + "_manager_status", "Removing the selected model cache…", action)

    def unload_cb():
        run_status(prefix + "_manager_status", "Unloading active model and clearing GPU cache…", controller.unload)

    def load_cb(model_id, modality):
        def action():
            spec, reason = check_spec(model_id)
            result = controller.load(spec)
            return f"{result}\\n{reason}"
        run_status(prefix + "_" + modality + "_progress", "Loading the selected model. This can take several minutes…", action)

    def image_generate_cb(model_id, prompt, negative, width, height, steps, guidance, seed):
        target = prefix + "_image_progress"
        panel = ColabProgress(target)
        panel.update(0, "Working", "Preparing image generation…")
        try:
            if not str(prompt).strip():
                raise ValueError("Please enter an image prompt.")
            check_spec(model_id)
            adapter = controller.require_active(model_id)
            image = adapter.generate(prompt=str(prompt), negative_prompt=str(negative),
                                     width=int(width), height=int(height), steps=int(steps),
                                     guidance=float(guidance), seed=int(seed),
                                     progress_callback=panel)
            path = output_dir("images") / f"image_{int(time.time())}_seed{int(seed)}.png"
            image.save(path)
            panel.update(100, "Completed", f"Image saved to Google Drive: {path}")
            with output.redirect_to_element("#" + prefix + "_image_result"):
                display(Image(filename=str(path)))
        except Exception as exc:
            panel.update(0, "Failed", f"{type(exc).__name__}: {exc}")

    def video_generate_cb(model_id, prompt, negative, resolution, aspect_ratio, duration,
                          steps, guidance, seed):
        panel = ColabProgress(prefix + "_video_progress")
        panel.update(0, "Working", "Checking settings and preparing video generation…")
        try:
            if not str(prompt).strip():
                raise ValueError("Please enter a video prompt in English.")
            spec, reason = check_spec(model_id)
            adapter = controller.require_active(model_id)
            panel.update(2, "Preparing", f"{reason}\\nUsing {spec['name']}…")
            if model_id == "cogvideox15-5b":
                result = adapter.generate(
                    prompt=str(prompt), negative_prompt=str(negative),
                    duration_seconds=int(duration), resolution=str(resolution),
                    aspect_ratio=str(aspect_ratio), steps=int(steps),
                    guidance=float(guidance), seed=int(seed),
                    progress_callback=panel,
                )
            elif model_id == "cogvideox-2b":
                result = adapter.generate(
                    prompt=str(prompt), negative_prompt=str(negative),
                    width=720, height=480, frames=49, fps=8,
                    steps=int(steps), guidance=float(guidance), seed=int(seed),
                    resolution=str(resolution), aspect_ratio=str(aspect_ratio),
                    progress_callback=panel,
                )
            else:
                raise RuntimeError("No video generation adapter is available for this model.")
            path = result["path"]
            panel.update(100, "Completed",
                         f"Saved: {path}\\nOutput: {result['width']}×{result['height']} · "
                         f"{result['duration_seconds']} sec · {result['fps']} FPS\\n"
                         f"Model render: {result['native_resolution']} · {result['quantization']}\\n"
                         f"{result['note']}")
            with output.redirect_to_element("#" + prefix + "_video_result"):
                display(Video(path, embed=True))
        except Exception as exc:
            panel.update(0, "Failed", f"{type(exc).__name__}: {exc}")

    def gpu_cb():
        def action():
            info = inspect_runtime()
            return "\\n".join(f"{key}: {value}" for key, value in info.items())
        run_status(prefix + "_system_status", "Refreshing GPU and RAM information…", action)

    def history_cb():
        def action():
            root = Path(os.environ.get("SNAPGASP_DATA_DIR", "./snapgasp-data")) / "outputs"
            files = sorted((p for p in root.rglob("*") if p.is_file()),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:30] if root.exists() else []
            return "\\n".join(str(p) for p in files) if files else "No generated files yet."
        run_status(prefix + "_history_status", "Reading generated output history…", action)

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
        #{prefix} .hint{{color:#a1a1aa;font-size:12px;line-height:1.55;margin-top:8px}}
        #{prefix} .status{{margin-top:10px;min-height:4px}}
        #{prefix} .result{{margin-top:12px;overflow:auto}}
        #{prefix} .result img,#{prefix} .result video{{max-width:100%;height:auto;border-radius:10px}}
        #{prefix} .section-title{{font-size:17px;font-weight:700;margin:4px 0 12px}}
        @media(max-width:650px){{#{prefix} .grid{{grid-template-columns:1fr}}}}
      </style>

      <div class="brand">
        <small>PRIVATE NOTEBOOK WORKSPACE · GOOGLE DRIVE STORAGE</small>
        <h2>SNAPGASP Studio</h2>
        <div>Image & video generation · one active model · free Colab GPU</div>
      </div>

      <div class="tabs">
        <button class="active" onclick="sgTab('image')">Image Studio</button>
        <button onclick="sgTab('video')">Video Studio</button>
        <button onclick="sgTab('manager')">Model Manager</button>
        <button onclick="sgTab('system')">System & History</button>
      </div>

      <section class="panel active" id="{prefix}_image">
        <div class="section-title">Image Studio</div>
        <label>Model</label><select id="{prefix}_im_model">{image_options}</select>
        <label>Prompt</label><textarea id="{prefix}_im_prompt" placeholder="Describe the image you want to create…"></textarea>
        <label>Negative prompt</label><textarea id="{prefix}_im_negative" placeholder="Optional: unwanted objects, text, distortions…"></textarea>
        <div class="grid">
          <div><label>Width</label><select id="{prefix}_im_width"><option>512</option><option selected>768</option><option>1024</option></select></div>
          <div><label>Height</label><select id="{prefix}_im_height"><option>512</option><option selected>768</option><option>1024</option></select></div>
          <div><label>Steps</label><input id="{prefix}_im_steps" type="number" min="1" max="50" value="25"></div>
          <div><label>Guidance</label><input id="{prefix}_im_guidance" type="number" min="0" max="12" step="0.5" value="7"></div>
          <div><label>Seed</label><input id="{prefix}_im_seed" type="number" value="12345"></div>
        </div>
        <div class="actions">
          <button onclick="sgCall('image_load',[sgVal('im_model')],'image_progress')">Load selected model</button>
          <button class="primary" onclick="sgImage()">Generate image</button>
        </div>
        <div class="status" id="{prefix}_image_status"></div>
        <div class="status" id="{prefix}_image_progress"></div>
        <div class="result" id="{prefix}_image_result"></div>
      </section>

      <section class="panel" id="{prefix}_video">
        <div class="section-title">Video Studio</div>
        <label>Video model</label><select id="{prefix}_vi_model" onchange="sgUpdateVideoControls()">{video_options}</select>
        <label>Prompt <span style="color:#a1a1aa">(English works best)</span></label>
        <textarea id="{prefix}_vi_prompt" placeholder="Describe the subject, action, camera and setting…"></textarea>
        <label>Negative prompt</label>
        <textarea id="{prefix}_vi_negative">visible watermark, logo, text, subtitles, signature, blurry details, flicker, unstable motion, distorted anatomy</textarea>

        <div class="section-title" style="margin-top:18px">Output settings</div>
        <div class="grid">
          <div>
            <label>Resolution</label>
            <select id="{prefix}_vi_resolution" onchange="sgUpdateVideoNote()">
              <option value="360p">360p · smaller file</option>
              <option value="720p" selected>720p · HD</option>
              <option value="1080p">1080p · upscale</option>
            </select>
          </div>
          <div>
            <label>Aspect ratio</label>
            <select id="{prefix}_vi_aspect" onchange="sgUpdateVideoNote()">
              <option value="16:9" selected>16:9 · Landscape</option>
              <option value="9:16">9:16 · Vertical</option>
            </select>
          </div>
          <div>
            <label>Duration</label>
            <select id="{prefix}_vi_duration" onchange="sgUpdateVideoNote()">
              <option value="5" selected>5 seconds</option>
              <option value="10">10 seconds</option>
            </select>
          </div>
          <div>
            <label>Generation quality</label>
            <select id="{prefix}_vi_steps">
              <option value="20">Draft · 20 steps</option>
              <option value="30" selected>Standard · 30 steps</option>
              <option value="50">High · 50 steps</option>
            </select>
          </div>
          <div>
            <label>Guidance</label>
            <select id="{prefix}_vi_guidance">
              <option value="4">4 · Lower prompt influence</option>
              <option value="6" selected>6 · Balanced</option>
              <option value="8">8 · Stronger prompt influence</option>
            </select>
          </div>
          <div>
            <label>Seed</label>
            <input id="{prefix}_vi_seed" type="number" value="12345">
          </div>
        </div>

        <div class="hint" id="{prefix}_vi_note">
          CogVideoX1.5-5B creates a native 1360×768 landscape video at 16 FPS. 1080p is upscaled; 9:16 is centre-cropped from landscape.
        </div>

        <div class="actions">
          <button onclick="sgCall('video_load',[sgVal('vi_model')],'video_progress')">Load selected model</button>
          <button class="primary" onclick="sgVideo()">Generate video</button>
        </div>

        <div class="status" id="{prefix}_video_progress"></div>
        <div class="result" id="{prefix}_video_result"></div>
      </section>

      <section class="panel" id="{prefix}_manager">
        <div class="section-title">Model Manager</div>
        <label>Select model</label><select id="{prefix}_manager_model">{manager_options}</select>
        <div class="actions">
          <button class="primary" onclick="sgCall('install',[sgVal('manager_model')],'manager_status')">Install / resume</button>
          <button onclick="sgCall('update',[sgVal('manager_model')],'manager_status')">Force update</button>
          <button class="danger" onclick="sgCall('uninstall',[sgVal('manager_model')],'manager_status')">Uninstall</button>
          <button onclick="sgCall('unload',[],'manager_status')">Unload active model</button>
        </div>
        <div class="hint">
          CogVideoX1.5-5B is experimental on a T4. It uses INT8 transformer weights with FP16 activations and needs approximately 32 GB of free Google Drive space for installation. Its 9:16 and 1080p outputs are post-processed crops/upscales, not native model output.
        </div>
        <div class="status" id="{prefix}_manager_status"></div>
      </section>

      <section class="panel" id="{prefix}_system">
        <div class="section-title">System & History</div>
        <div class="actions">
          <button class="primary" onclick="sgCall('gpu',[],'system_status')">Refresh GPU status</button>
          <button onclick="sgCall('history',[],'history_status')">Refresh output history</button>
        </div>
        <pre class="status" id="{prefix}_system_status" style="white-space:pre-wrap;color:#eee"></pre>
        <h4>Recent outputs</h4>
        <pre class="status" id="{prefix}_history_status" style="white-space:pre-wrap;color:#eee"></pre>
      </section>
    </div>

    <script>
      const sgRoot=document.getElementById("{prefix}");
      function sgTab(name){{
        sgRoot.querySelectorAll(".panel").forEach(x=>x.classList.remove("active"));
        sgRoot.querySelectorAll(".tabs button").forEach(x=>x.classList.remove("active"));
        document.getElementById("{prefix}_"+name).classList.add("active");
        const b=[...sgRoot.querySelectorAll(".tabs button")].find(x=>x.textContent.toLowerCase().startsWith(name==="manager"?"model":name==="system"?"system":name));
        if(b)b.classList.add("active");
      }}
      function sgVal(key){{return document.getElementById("{prefix}_"+key).value;}}
      function sgCall(action,args,target){{
        document.getElementById("{prefix}_"+target).textContent="Working…";
        google.colab.kernel.invokeFunction("{prefix}_"+action,args,{{}});
      }}
      function sgImage(){{
        sgCall("image_generate",[sgVal("im_model"),sgVal("im_prompt"),sgVal("im_negative"),sgVal("im_width"),sgVal("im_height"),sgVal("im_steps"),sgVal("im_guidance"),sgVal("im_seed")],"image_progress");
      }}
      function sgVideo(){{
        sgCall("video_generate",[sgVal("vi_model"),sgVal("vi_prompt"),sgVal("vi_negative"),sgVal("vi_resolution"),sgVal("vi_aspect"),sgVal("vi_duration"),sgVal("vi_steps"),sgVal("vi_guidance"),sgVal("vi_seed")],"video_progress");
      }}
      function sgUpdateVideoControls(){{
        const model=sgVal("vi_model");
        const d=document.getElementById("{prefix}_vi_duration");
        const note=document.getElementById("{prefix}_vi_note");
        if(model==="cogvideox15-5b"){{
          d.innerHTML='<option value="5">5 seconds</option><option value="10">10 seconds</option>';
          d.value="5";
          note.textContent="CogVideoX1.5-5B: native 1360×768 landscape, 16 FPS, 5 or 10 seconds. 1080p is upscaled; 9:16 is centre-cropped.";
        }}else{{
          d.innerHTML='<option value="6">6 seconds (fixed by model)</option>';
          d.value="6";
          note.textContent="CogVideoX-2B: native 720×480, 49 frames, 8 FPS, 6 seconds. Resolution/aspect choices resize or crop the exported video.";
        }}
        sgUpdateVideoNote();
      }}
      function sgUpdateVideoNote(){{
        const model=sgVal("vi_model");
        const res=sgVal("vi_resolution");
        const ratio=sgVal("vi_aspect");
        const note=document.getElementById("{prefix}_vi_note");
        let messages=[];
        if(model==="cogvideox15-5b") messages.push("Model renders at 1360×768.");
        else messages.push("Model renders at 720×480.");
        if(res==="1080p") messages.push("1080p is an upscale; it adds no new AI detail.");
        else if(res==="720p") messages.push("720p is resized from the model's native render.");
        else messages.push("360p is downscaled to reduce file size.");
        if(ratio==="9:16") messages.push("Vertical output is a centre crop; keep the main subject near the centre.");
        else messages.push("Landscape output uses 16:9.");
        note.textContent=messages.join(" ");
      }}
      sgUpdateVideoControls();
    </script>
    """
    display(HTML(page))
    gpu_cb()
    history_cb()
    return {"controller": controller, "prefix": prefix}
