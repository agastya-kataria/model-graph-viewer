#!/usr/bin/env python3
"""Capture Netron's topology view without loading weights or contacting telemetry services."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/tinyllama/openvino_model.xml")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/netron-tinyllama.png")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    import netron
    from playwright.sync_api import sync_playwright

    address = ("127.0.0.1", args.port)
    origin = f"http://{address[0]}:{address[1]}"
    # Netron can inspect IR topology from XML alone. Its browser does not need
    # the 2.2 GB weight payload for this comparison.
    with TemporaryDirectory(prefix="netron-topology-") as directory:
        xml = Path(directory) / args.model.name
        shutil.copyfile(args.model, xml)
        netron.start(str(xml), address=address, browse=False)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.route(
                    "**/*",
                    lambda route: (
                        route.continue_()
                        if route.request.url.startswith(origin + "/")
                        else route.abort()
                    ),
                )
                page.goto(origin, wait_until="domcontentloaded")
                page.get_by_text("Accept", exact=True).click()
                page.wait_for_selector("svg g.node", timeout=60000)
                page.wait_for_selector("body.default", timeout=60000)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.output))
                print(f"Captured Netron topology: {args.output}")
                browser.close()
        finally:
            netron.stop(address)


if __name__ == "__main__":
    main()
