# fast-alpr-node

**Language:** [English](README.md) · [Português](README.pt.md) · [Español](README.es.md)

Port of [fast-alpr](https://github.com/ankandrew/fast-alpr) (Automatic License Plate Recognition) to Node.js / TypeScript.

It runs the same ONNX models as the Python library through `onnxruntime-node`; only the pre/post-processing is
reimplemented. Models are downloaded on first use into `~/.cache/open-image-models` and `~/.cache/fast-plate-ocr`
(the same cache the Python libraries use).

## Install

```sh
npm install fast-alpr-node
```

Requires **Node.js ≥ 20**. Native dependencies (`sharp`, `onnxruntime-node`) ship platform-specific binaries.
Node only — not a browser package. ONNX weights are fetched on first use into the cache above.

## Usage

```ts
import { ALPR } from "fast-alpr-node";

const alpr = await ALPR.create({
  detectorModel: "yolo-v9-t-384-license-plate-end2end", // default
  ocrModel: "cct-xs-v2-global-model", // default
});

const results = await alpr.predict("assets/test_image.png");
for (const { detection, ocr } of results) {
  console.log(detection.boundingBox, detection.confidence, ocr?.text, ocr?.region);
}

// Also accepted: Buffer / Uint8Array, base64 / data URL, byte stream, or a BGR `Image`
// await alpr.predict(buffer);
// await alpr.predict(base64String);
// await alpr.predict(fs.createReadStream("foto.jpg"));
```

`predict` returns one `ALPRResult` per plate: `detection` (box, label, confidence) and `ocr`
(`text`, per-character `confidence`, optional `region` / `regionConfidence`, or `null`).

Accepted inputs: file path, encoded image bytes (`Buffer` / `Uint8Array` / `ArrayBuffer`),
base64 string or `data:image/...;base64,...` URL, async byte stream (e.g. Node `Readable`),
or an already-decoded BGR `Image`.

CLI example: `npm run predict -- path/to/image.png`

## How it works

The pipeline has two stages: a YOLOv9 plate detector, then a CCT OCR on each crop.
See the full walkthrough (flowcharts, letterbox geometry, OCR slot decoding, module map,
and the test-image result):

- [English](docs/how-it-works.en.md)
- [Português](docs/how-it-works.pt.md)
- [Español](docs/how-it-works.es.md)

## Differences from the Python version

- **OpenCV backend**: `@techstark/opencv-js` (WASM, same C++ code) instead of `opencv-python`. Only `src/cv.ts`
  touches OpenCV, so the backend can be swapped for a native binding. File decoding uses `sharp`.
- **Async API**: models are created with `await ALPR.create()` and `predict` returns a Promise.
- **Not ported yet**: `draw_predictions`, batch inference.
- **Providers**: default is `["cpu"]`. Pass `detectorProviders` / `ocrProviders` (e.g. `["coreml", "cpu"]`) to change it.
- Pixel values can differ by 1 gray level from the Python version after resizing (different OpenCV builds), so
  confidences match to ~1e-3 on confident characters rather than bit-for-bit. Text, region and boxes match.

## Development

```sh
npm install
npm run typecheck
npm test            # unit tests + comparison against the Python library
npm run build
```

### Regenerating the Python reference outputs

`test/golden/*.json` come from the original library:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "fast-alpr[onnx]"
.venv/bin/python tools/make_variants.py                 # derives extra images from assets/test_image.png
.venv/bin/python tools/gen_golden.py assets/*.png
```

## License

MIT, like the original. The model weights are downloaded from the upstream projects and have their own terms.
