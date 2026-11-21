/**
 * Usage: npm run predict -- path/to/image.png
 * Prints the ALPR results as JSON.
 */
import { ALPR } from "../src/index.js";

const imagePath = process.argv[2];
if (!imagePath) {
  console.error("Usage: npm run predict -- <image>");
  process.exit(1);
}

const alpr = await ALPR.create({
  detectorModel: "yolo-v9-t-384-license-plate-end2end",
  ocrModel: "cct-xs-v2-global-model",
});

const results = await alpr.predict(imagePath);
console.log(JSON.stringify(results, null, 2));
process.exit(0); // the OpenCV WASM runtime keeps the event loop alive otherwise
