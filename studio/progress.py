import html
import uuid
from IPython.display import HTML, display, update_display
from google.colab import output

class ColabProgress:
    """Progress card updated in-place through Colab's output redirect."""
    def __init__(self, target_id):
        self.target_id = target_id
        self.display_id = "sg-progress-" + uuid.uuid4().hex
        self.started = False
        self.percent = 0

    def update(self, percent, state, detail=""):
        percent = max(0, min(100, int(percent)))
        self.percent = percent
        safe_state = html.escape(str(state))
        safe_detail = html.escape(str(detail))
        color = "#16a34a" if state == "Completed" else "#dc2626" if state == "Failed" else "#facc15"
        markup = f"""
        <div style="background:#11151a;border:1px solid #343a40;border-radius:10px;padding:12px;margin:8px 0;color:#f4f4f5;font-family:Arial,sans-serif">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
            <b style="color:{color}">{safe_state}</b>
            <b>{percent}%</b>
          </div>
          <div style="height:10px;background:#303640;border-radius:8px;overflow:hidden;margin:10px 0">
            <div style="height:10px;width:{percent}%;background:{color};border-radius:8px;transition:width .25s"></div>
          </div>
          <div style="font-size:12px;color:#c5cbd3;white-space:pre-wrap">{safe_detail}</div>
        </div>
        """
        with output.redirect_to_element("#" + self.target_id):
            if not self.started:
                display(HTML(markup), display_id=self.display_id)
                self.started = True
            else:
                update_display(HTML(markup), display_id=self.display_id)

    def step_callback(self, total_steps, prefix="Generating"):
        total_steps = max(1, int(total_steps))
        def callback(_pipe, step, _timestep, callback_kwargs):
            completed = min(total_steps, int(step) + 1)
            percent = 5 + round(88 * completed / total_steps)
            self.update(percent, "Generating", f"{prefix}: step {completed}/{total_steps}")
            return callback_kwargs
        return callback
