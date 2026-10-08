#!/usr/bin/env python3

"""Generate the Jetsam application icons from the Jetsam brand mark.

The vector sources live next to the output, in
``jetsam_gui/assets/app-icons/source/``:

* ``jetsam-mark.svg``: the Jetsam mark on its deep-water tile;
* ``jetsam-mark-small.svg``: the small-size optical cut of the same mark
  (heavier strokes, no jib), used at 24 px and below where the full drawing
  stops reading as a sail.

Rasterisation uses ``rsvg-convert`` (librsvg). The Windows ``.ico`` embeds
the PNG renditions byte for byte.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "jetsam_gui" / "assets" / "app-icons"
SOURCE = OUTPUT / "source"
MARK = SOURCE / "jetsam-mark.svg"
MARK_SMALL = SOURCE / "jetsam-mark-small.svg"
SMALL_CUT_MAX_SIZE = 24
PNG_SIZES = (16, 32, 48, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 32, 48, 64, 128, 256)


def rasterize(size: int) -> bytes:
    source = MARK_SMALL if size <= SMALL_CUT_MAX_SIZE else MARK
    png = subprocess.run(
        [
            "rsvg-convert",
            "--width",
            str(size),
            "--height",
            str(size),
            "--format",
            "png",
            str(source),
        ],
        check=True,
        capture_output=True,
    ).stdout
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"rsvg-convert did not return a PNG for {size} px")
    width, height = struct.unpack(">II", png[16:24])
    if (width, height) != (size, size):
        raise RuntimeError(
            f"rsvg-convert returned {width}x{height} for a {size} px icon"
        )
    return png


def encode_ico(images: list[tuple[int, bytes]]) -> bytes:
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = bytearray()
    payload = bytearray()
    for size, png in images:
        encoded_size = 0 if size == 256 else size
        entries.extend(
            struct.pack(
                "<BBBBHHII",
                encoded_size,
                encoded_size,
                0,
                0,
                1,
                32,
                len(png),
                offset,
            )
        )
        payload.extend(png)
        offset += len(png)
    return header + bytes(entries) + bytes(payload)


def main() -> int:
    if shutil.which("rsvg-convert") is None:
        print("rsvg-convert (librsvg) is required", file=sys.stderr)
        return 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    encoded: dict[int, bytes] = {}

    for size in PNG_SIZES:
        png = rasterize(size)
        encoded[size] = png
        (OUTPUT / f"Jetsam-{size}.png").write_bytes(png)

    ico = encode_ico([(size, encoded[size]) for size in ICO_SIZES])
    (OUTPUT / "Jetsam.ico").write_bytes(ico)
    print(f"generated application icons in {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
