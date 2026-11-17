"""Render labelled synthetic photos of a puzzle.

The pictures are drawn by the real 3D model in Chrome (web/datagen.html), driven here with
Playwright, under random light, camera, sticker shades, background and (sometimes) a hand; the
labels come from the engine state, so they are exact.

    Maltese Gear Cube      one face seen roughly face-on per picture; a scan is six pictures
    Tangram Cube Extreme   one face seen roughly face-on per picture, in any shape; a scan is six

    python tools/gen_dataset.py --puzzle maltese --out D:/cube_data/synth/v1 --count 40000
    python tools/gen_dataset.py --puzzle tangram --out D:/cube_data/tangram/v1/train --count 60000
    python tools/gen_dataset.py --puzzle tangram --out ... --count 200 --headed     # watch it work

Writes  <out>/images/000123.jpg  and one line per picture in  <out>/meta.jsonl.
It can be stopped and started again: pictures already there are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

CHROME_ARGS = [
    "--ignore-gpu-blocklist",
    "--enable-gpu",
    "--use-angle=d3d11",
    "--enable-unsafe-swiftshader",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
]


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def start_dev_server(port: int) -> tuple[subprocess.Popen | None, str]:
    """The page is served by Vite. Use a server that is already up, or start one."""
    url = f"http://127.0.0.1:{port}/datagen.html"
    if _reachable(url):
        return None, url
    cmd = f"npx vite --port {port} --strictPort --host 127.0.0.1"
    proc = subprocess.Popen(cmd, cwd=WEB, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(120):
        if _reachable(url):
            return proc, url
        if proc.poll() is not None:
            raise RuntimeError(f"`{cmd}` exited with code {proc.returncode}; run `npm install` in web/ first")
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("the Vite dev server did not come up")


PUZZLES = {"maltese": "maltese-gear-cube", "tangram": "tangram-cube-extreme"}
SHOTS_PER_SCAN = {"maltese": 6, "tangram": 6}


async def generate(args: argparse.Namespace, url: str) -> None:
    from playwright.async_api import async_playwright

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"
    done: set[int] = set()
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    if m.get("puzzle", "maltese-gear-cube") != PUZZLES[args.puzzle]:
                        sys.exit(f"{meta_path} holds pictures of another puzzle ({m.get('puzzle', 'maltese-gear-cube')}); use another --out")
                    done.add(m["seed"])
    shots = SHOTS_PER_SCAN[args.puzzle]
    if args.scans:
        # A scan is several pictures of one puzzle; its pictures are numbered scan * 8 + shot.
        seeds = [s for s in range(args.start_seed, args.start_seed + args.count) if s * 8 + shots - 1 not in done]
        print(f"{len(done) // shots} scans already in {out}; rendering {len(seeds)} more ({shots} pictures each)", flush=True)
    else:
        seeds = [s for s in range(args.start_seed, args.start_seed + args.count) if s not in done]
        print(f"{len(done)} pictures already in {out}; rendering {len(seeds)} more", flush=True)
    if not seeds:
        return

    total = len(seeds) * (shots if args.scans else 1)
    call = "runScans" if args.scans else "run"
    meta_file = meta_path.open("a", encoding="utf-8")
    t0 = time.time()
    saved = 0

    def save(meta_json: str, jpeg_b64: str) -> None:
        nonlocal saved
        meta = json.loads(meta_json)
        name = f"{meta['seed']:07d}.jpg"
        (images / name).write_bytes(base64.b64decode(jpeg_b64))
        meta["file"] = f"images/{name}"
        meta["size"] = args.size
        meta.setdefault("puzzle", PUZZLES[args.puzzle])
        meta_file.write(json.dumps(meta, separators=(",", ":")) + "\n")
        saved += 1
        if saved % args.report == 0:
            meta_file.flush()
            rate = saved / (time.time() - t0)
            print(f"  {saved}/{total}  {rate:.1f} pictures/s  about {(total - saved) / rate / 60:.1f} min left", flush=True)

    async with async_playwright() as p:
        chrome_args = CHROME_ARGS + (["--force_high_performance_gpu"] if args.discrete_gpu else [])
        browser = await p.chromium.launch(channel=args.channel, headless=not args.headed, args=chrome_args)
        context = await browser.new_context(viewport={"width": 1000, "height": 1000})
        await context.expose_function("__save", save)

        async def tab(index: int) -> None:
            page = await context.new_page()
            page.on("pageerror", lambda err: print(f"  [tab {index}] page error: {err}", file=sys.stderr, flush=True))
            extra = (f"&layouts={args.layouts}" if args.layouts else "") + ("&shades=wide" if args.wide_shades else "")
            await page.goto(f"{url}?puzzle={PUZZLES[args.puzzle]}&size={args.size}{extra}")
            info = await page.evaluate("() => window.datagen.ready")
            if info.get("puzzle", "maltese-gear-cube") != PUZZLES[args.puzzle] or info.get("shotsPerScan", 6) != shots:
                raise RuntimeError(f"the page renders {info.get('puzzle')}, not {PUZZLES[args.puzzle]}: rebuild or restart the dev server")
            if index == 0:
                print(f"  renderer: {info['renderer']}", flush=True)
                if "swiftshader" in info["renderer"].lower():
                    print("  warning: Chrome is rendering in software; this will be slow", flush=True)
            mine = seeds[index::args.tabs]
            for i in range(0, len(mine), args.chunk):
                await page.evaluate(f"([seeds, q]) => window.datagen.{call}(seeds, q)", [mine[i:i + args.chunk], args.quality])
            await page.close()

        try:
            await asyncio.gather(*(tab(i) for i in range(args.tabs)))
        finally:
            meta_file.close()
            await browser.close()
    print(f"done: {saved} pictures in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="maltese")
    ap.add_argument("--out", required=True, help="dataset folder (keep it out of synced folders: it gets large)")
    ap.add_argument("--count", type=int, default=1000, help="pictures, or scans with --scans")
    ap.add_argument("--scans", action="store_true",
                    help="render whole scans: all the pictures of one scrambled puzzle in capture order (for end-to-end scoring)")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=640, help="picture width and height in pixels")
    ap.add_argument("--quality", type=float, default=0.92, help="JPEG quality, 0-1")
    ap.add_argument("--tabs", type=int, default=3, help="Chrome tabs rendering in parallel")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--report", type=int, default=500)
    ap.add_argument("--port", type=int, default=5183)
    ap.add_argument("--channel", default="chrome", help="Playwright browser channel (chrome, msedge, chromium)")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--layouts", type=float, default=0.0,
                    help="share of puzzles stickered another way (a random colour layout; labels are the colours shown), e.g. 0.5")
    ap.add_argument("--wide-shades", action="store_true",
                    help="vary sticker shades more from puzzle to puzzle (reds towards orange, oranges towards red)")
    ap.add_argument("--discrete-gpu", action="store_true",
                    help="ask Chrome for the discrete GPU on a laptop with two (leave off while a model is training on it)")
    args = ap.parse_args()

    server, url = start_dev_server(args.port)
    try:
        asyncio.run(generate(args, url))
    finally:
        if server is not None:
            # `npx` runs through a shell on Windows: end the whole tree.
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
            else:
                server.terminate()


if __name__ == "__main__":
    main()
