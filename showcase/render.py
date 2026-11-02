"""Render the showcase cards to PNGs and a reel (MP4) from showcase/out/picks.json.

    python -m showcase.export      # data
    python -m showcase.render      # out/card-*.png, out/reel.mp4
"""

import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE / "out"
FPS = 30
HOLD_MS = 1600  # time a finished card stays on screen in the reel


def build() -> Path:
    html = (HERE / "cards.html").read_text(encoding="utf-8")
    data = (OUT / "picks.json").read_text(encoding="utf-8")
    built = OUT / "cards.built.html"
    built.write_text(html.replace("/*DATA*/null", data), encoding="utf-8")
    return built


def seek(page, ms: float) -> None:
    page.evaluate("t => document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; })", ms)


def write(path: Path, data: bytes) -> None:
    """Write, retrying while an image viewer briefly holds the old file open."""
    for _ in range(20):
        try:
            path.write_bytes(data)
            return
        except OSError:
            time.sleep(0.5)
    path.write_bytes(data)


def main() -> None:
    url = build().as_uri()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    reel = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", str(OUT / "reel.mp4")],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1080, "height": 1350})
        page.goto(f"{url}?card=0")
        count = page.evaluate("window.CARD_COUNT")
        for n in range(count):
            page.goto(f"{url}?card={n}")
            page.evaluate("document.fonts.ready")
            end = page.evaluate(
                "Math.max(...document.getAnimations().map(a => a.effect.getComputedTiming().endTime))"
            )
            for frame in range(int((end + HOLD_MS) / 1000 * FPS)):
                seek(page, frame * 1000 / FPS)
                reel.stdin.write(page.screenshot(type="png"))
            seek(page, end)
            write(OUT / f"card-{n + 1:02d}.png", page.screenshot(type="png"))
            print(f"card {n + 1}/{count}")
        browser.close()
    reel.stdin.close()
    reel.wait()


if __name__ == "__main__":
    main()
