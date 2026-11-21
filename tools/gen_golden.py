"""
Generates reference outputs with the original Python `fast-alpr`, to compare the Node port against.

Usage (from the project root):
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python "fast-alpr[onnx]"
    .venv/bin/python tools/gen_golden.py assets/test_image.png [more images...]

Writes test/golden/<image-name>.json for each image.
"""

import json
import sys
from pathlib import Path

from fast_alpr import ALPR


def to_jsonable(result) -> dict:
    det = result.detection
    bbox = det.bounding_box
    ocr = result.ocr
    return {
        "detection": {
            "label": det.label,
            "confidence": float(det.confidence),
            "boundingBox": {"x1": int(bbox.x1), "y1": int(bbox.y1), "x2": int(bbox.x2), "y2": int(bbox.y2)},
        },
        "ocr": None
        if ocr is None
        else {
            "text": ocr.text,
            "confidence": [float(c) for c in ocr.confidence]
            if isinstance(ocr.confidence, list)
            else float(ocr.confidence),
            "region": ocr.region,
            "regionConfidence": None if ocr.region_confidence is None else float(ocr.region_confidence),
        },
    }


def main() -> None:
    images = sys.argv[1:]
    if not images:
        sys.exit(__doc__)

    out_dir = Path(__file__).resolve().parent.parent / "test" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
        ocr_device="cpu",
        detector_providers=["CPUExecutionProvider"],
        ocr_providers=["CPUExecutionProvider"],
    )

    for image in images:
        results = [to_jsonable(r) for r in alpr.predict(image)]
        target = out_dir / f"{Path(image).stem}.json"
        target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{image}: {len(results)} plate(s) -> {target}")


if __name__ == "__main__":
    main()
