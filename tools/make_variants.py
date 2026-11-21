"""
Derives extra test images from assets/test_image.png so the golden comparison exercises
different letterbox paths (wide, tall, small, plate touching the border).

Usage: .venv/bin/python tools/make_variants.py
"""

from pathlib import Path

import cv2

assets = Path(__file__).resolve().parent.parent / "assets"
src = cv2.imread(str(assets / "test_image.png"))
h, w = src.shape[:2]
print("source", w, "x", h)

variants = {
    # very wide strip around the plate -> padding on top/bottom
    "variant_wide": src[max(h // 2 - 120, 0) : h // 2 + 120, :],
    # tall crop -> padding on the sides
    "variant_tall": src[:, w // 4 : w // 4 + w // 3],
    # downscaled by ~2.3 (odd size -> fractional padding)
    "variant_small": cv2.resize(src, (round(w / 2.3), round(h / 2.3)), interpolation=cv2.INTER_AREA),
    # crop that cuts the plate on the right edge -> exercises coordinate clamping
    "variant_edge": src[:, : 387 + 10],
}

for name, img in variants.items():
    out = assets / f"{name}.png"
    cv2.imwrite(str(out), img)
    print(name, img.shape[1], "x", img.shape[0])
