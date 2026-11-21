/**
 * YOLOv9 preprocessing. Port of `open_image_models/detection/core/yolo_v9/preprocess.py`.
 */
import { copyMakeBorder, resize } from "../cv.js";
import type { Image } from "../types.js";
import { roundHalfEven } from "../util.js";

export interface LetterboxResult {
  image: Image;
  /** Scale ratio (same for both axes). */
  ratio: [number, number];
  /** Padding added on each side, (dw, dh). Kept as floats, exactly like the Python version. */
  padding: [number, number];
}

/**
 * Resizes and pads to `newShape` keeping aspect ratio.
 * `newShape` is `[height, width]`.
 */
export function letterbox(
  im: Image,
  newShape: [number, number],
  color: [number, number, number] = [114, 114, 114],
  scaleup = true,
): LetterboxResult {
  const [newH, newW] = newShape;

  let r = Math.min(newH / im.height, newW / im.width);
  if (!scaleup) r = Math.min(r, 1.0);

  const unpadW = roundHalfEven(im.width * r);
  const unpadH = roundHalfEven(im.height * r);
  const dw = (newW - unpadW) / 2;
  const dh = (newH - unpadH) / 2;

  let out = im;
  if (im.width !== unpadW || im.height !== unpadH) {
    out = resize(im, unpadW, unpadH, "linear");
  }

  const top = roundHalfEven(dh - 0.1);
  const bottom = roundHalfEven(dh + 0.1);
  const left = roundHalfEven(dw - 0.1);
  const right = roundHalfEven(dw + 0.1);
  out = copyMakeBorder(out, top, bottom, left, right, color);

  return { image: out, ratio: [r, r], padding: [dw, dh] };
}

export interface PreprocessResult {
  /** Float32 tensor data, shape [1, 3, H, W], RGB, 0..1. */
  tensor: Float32Array;
  shape: [number, number, number, number];
  ratio: [number, number];
  padding: [number, number];
}

/** Letterbox, BGR -> RGB, HWC -> CHW, scale to 0..1 as float32. Input image must be BGR. */
export function preprocess(img: Image, imgSize: number | [number, number]): PreprocessResult {
  const size: [number, number] = typeof imgSize === "number" ? [imgSize, imgSize] : imgSize;
  const { image, ratio, padding } = letterbox(img, size);

  const { width: w, height: h, data } = image;
  const plane = w * h;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const p = i * 3;
    tensor[i] = data[p + 2]! / 255; // R
    tensor[plane + i] = data[p + 1]! / 255; // G
    tensor[2 * plane + i] = data[p]! / 255; // B
  }

  return { tensor, shape: [1, 3, h, w], ratio, padding };
}
