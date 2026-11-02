"""Record a walkthrough of the dashboard to showcase/out/walkthrough.mp4.

Needs the API (port 8000) and the dashboard (port 3000) running.
"""

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "out"
APP = "http://localhost:3000"
SIZE = {"width": 1440, "height": 900}

# React ignores a plain value assignment on inputs; go through the native setter.
SET_RANGE = """([id, v]) => {
  const el = document.getElementById(id);
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
  el.dispatchEvent(new Event('input', { bubbles: true }));
}"""


def main() -> None:
    picks = json.loads((OUT / "picks.json").read_text(encoding="utf-8"))["picks"]
    gk = picks[0]
    raw = OUT / "raw"
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport=SIZE, record_video_dir=raw, record_video_size=SIZE,
                                  color_scheme="dark")  # fmt: skip
        page = ctx.new_page()
        page.set_default_timeout(120_000)

        page.goto(APP)  # lineup: the gaps glow
        page.wait_for_selector("text=GK")
        page.wait_for_timeout(4500)

        page.goto(f"{APP}/recruit?slot=GK")
        page.wait_for_selector("ol li")
        page.wait_for_timeout(2500)
        age = int(page.get_attribute("#age", "value"))
        for v in range(age, 24, -1):  # drag the age slider down to 25
            page.evaluate(SET_RANGE, ["age", v])
            page.wait_for_timeout(450)
        page.wait_for_timeout(2500)
        for slot in ["ST", "LB", "AM"]:
            page.get_by_role("button", name=slot, exact=True).click()
            page.wait_for_timeout(3000)

        page.goto(f"{APP}/compare?ids={gk['candidate']['player_id']}&ids={gk['starter_profile']['player_id']}")
        page.wait_for_selector("svg")
        page.wait_for_timeout(3500)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(3000)

        video = page.video.path()
        ctx.close()
        browser.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(OUT / "walkthrough.mp4")], check=True)  # fmt: skip
    Path(video).unlink()


if __name__ == "__main__":
    main()
