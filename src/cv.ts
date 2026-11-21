/**
 * Thin image layer on top of OpenCV.
 *
 * Everything else in the library works with plain `Image` objects (Uint8Array + shape);
 * only this file knows about OpenCV. That keeps the backend swappable (WASM build today,
 * a native binding later) without touching the detector / OCR logic.
 *
 * File / byte decoding uses `sharp`, because the WASM build of OpenCV has no file/codec
 * support in Node.
 */
import { Buffer } from "node:buffer";
import { createRequire } from "node:module";
import sharp from "sharp";
import type { Image, ImageInterpolation, PredictInput } from "./types.js";

// The emscripten module object is a thenable. If it goes through an ESM `import` (or any
// `await`), loaders such as vite-node wait on it forever. Loading it with `require` hands
// back the plain object; we then wait for `onRuntimeInitialized` ourselves.
type Cv = any; // The bundled typings are incomplete; we only use a small, well-known subset.
const cv: Cv = createRequire(import.meta.url)("@techstark/opencv-js");

let readyPromise: Promise<void> | undefined;

/** Initializes OpenCV. Safe to call many times. */
export function initCv(): Promise<void> {
  readyPromise ??= new Promise<void>((resolve) => {
    if (cv.Mat) {
      resolve();
      return;
    }
    cv.onRuntimeInitialized = () => resolve();
  });
  return readyPromise;
}

function assertReady(): void {
  if (!cv.Mat) {
    throw new Error("OpenCV is not initialized. Await initCv() (or use ALPR.create()) first.");
  }
}

const cvType = (channels: number): number => (channels === 1 ? cv.CV_8UC1 : cv.CV_8UC3);

function toMat(img: Image): any {
  return cv.matFromArray(img.height, img.width, cvType(img.channels), img.data);
}

function fromMat(mat: any, channels: 1 | 3): Image {
  const image: Image = {
    data: new Uint8Array(mat.data), // copy out of the WASM heap
    width: mat.cols,
    height: mat.rows,
    channels,
  };
  mat.delete();
  return image;
}

const DATA_URL_RE = /^data:image\/[a-z0-9.+-]+;base64,/i;
const IMAGE_EXT_RE = /\.(png|jpe?g|webp|bmp|gif|tiff?|avif)$/i;
const BASE64_BODY_RE = /^[A-Za-z0-9+/=\s]+$/;
// Characters common in file paths but outside the standard base64 alphabet. We must not treat
// `/` alone as a path marker: PNG/JPEG base64 payloads contain many slashes.
const PATH_HINT_RE = /[._:\\~-]|^(\.\/|\.\.\/)/;

/** True when `value` is an already-decoded BGR `Image`. */
export function isImage(value: unknown): value is Image {
  if (!value || typeof value !== "object") return false;
  const img = value as Image;
  return (
    img.data instanceof Uint8Array &&
    typeof img.width === "number" &&
    typeof img.height === "number" &&
    (img.channels === 1 || img.channels === 3)
  );
}

function stripDataUrl(value: string): string {
  const match = DATA_URL_RE.exec(value);
  return match ? value.slice(match[0].length) : value;
}

function looksLikeBase64(value: string): boolean {
  if (DATA_URL_RE.test(value)) return true;
  // Paths usually have an extension, a drive letter, `./`, or characters like `.` `_` `-`.
  if (IMAGE_EXT_RE.test(value) || PATH_HINT_RE.test(value) || /^[A-Za-z]:[\\/]/.test(value)) {
    return false;
  }
  const body = value.replace(/\s+/g, "");
  // A real image encoded as base64 is never tiny; reject short strings that might be filenames.
  return body.length >= 64 && body.length % 4 === 0 && BASE64_BODY_RE.test(body);
}

async function bufferFromStream(stream: AsyncIterable<Uint8Array | Buffer>): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) {
    throw new Error("Failed to load image from stream: stream produced no bytes");
  }
  return Buffer.concat(chunks);
}

function toNodeBuffer(bytes: Buffer | Uint8Array | ArrayBuffer | ArrayBufferView): Buffer {
  if (Buffer.isBuffer(bytes)) return bytes;
  if (bytes instanceof ArrayBuffer) return Buffer.from(bytes);
  if (ArrayBuffer.isView(bytes)) {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  return Buffer.from(bytes);
}

/** Decodes an encoded image (path, bytes, base64, …) into a BGR `Image`. */
async function decodeEncodedImage(input: string | Buffer, label: string): Promise<Image> {
  try {
    const { data, info } = await sharp(input)
      .rotate() // honour EXIF orientation, as cv2.imread does
      .removeAlpha()
      .toColourspace("srgb")
      .raw({ depth: "uchar" })
      .toBuffer({ resolveWithObject: true });
    if (info.channels !== 3) {
      throw new Error(`unexpected channels=${info.channels}`);
    }
    // sharp gives RGB, OpenCV convention is BGR.
    const out = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i += 3) {
      out[i] = data[i + 2]!;
      out[i + 1] = data[i + 1]!;
      out[i + 2] = data[i]!;
    }
    return { data: out, width: info.width, height: info.height, channels: 3 };
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    throw new Error(`Failed to load image from ${label}: ${detail}`);
  }
}

/** Reads an image file as BGR (like `cv2.imread` with default flags). */
export async function readImage(path: string): Promise<Image> {
  return decodeEncodedImage(path, `path: ${path}`);
}

/** Decodes encoded image bytes (JPEG, PNG, WebP, …) into a BGR `Image`. */
export async function readImageFromBuffer(bytes: Buffer | Uint8Array | ArrayBuffer | ArrayBufferView): Promise<Image> {
  return decodeEncodedImage(toNodeBuffer(bytes), "buffer");
}

/** Decodes a base64 string or `data:image/...;base64,...` URL into a BGR `Image`. */
export async function readImageFromBase64(value: string): Promise<Image> {
  const body = stripDataUrl(value).replace(/\s+/g, "");
  if (!body) {
    throw new Error("Failed to load image from base64: empty string");
  }
  return decodeEncodedImage(Buffer.from(body, "base64"), "base64");
}

/**
 * Turns any `PredictInput` into a BGR `Image`.
 *
 * Strings that look like file paths are loaded from disk; other strings are treated as base64
 * (including data URLs). Encoded byte buffers and streams are decoded with `sharp`.
 */
export async function resolveImage(input: PredictInput): Promise<Image> {
  if (isImage(input)) return input;

  if (typeof input === "string") {
    return looksLikeBase64(input) ? readImageFromBase64(input) : readImage(input);
  }

  if (Buffer.isBuffer(input) || input instanceof Uint8Array || input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
    return readImageFromBuffer(input);
  }

  if (input && typeof input === "object" && Symbol.asyncIterator in input) {
    return readImageFromBuffer(await bufferFromStream(input as AsyncIterable<Uint8Array | Buffer>));
  }

  throw new Error("Unsupported image input. Expected a path, Image, buffer, base64 string, or byte stream.");
}

const interpolationFlag = (method: ImageInterpolation): number => {
  switch (method) {
    case "nearest":
      return cv.INTER_NEAREST;
    case "linear":
      return cv.INTER_LINEAR;
    case "cubic":
      return cv.INTER_CUBIC;
    case "area":
      return cv.INTER_AREA;
    case "lanczos4":
      return cv.INTER_LANCZOS4;
  }
};

export function resize(img: Image, width: number, height: number, interpolation: ImageInterpolation): Image {
  assertReady();
  const src = toMat(img);
  const dst = new cv.Mat();
  try {
    cv.resize(src, dst, new cv.Size(width, height), 0, 0, interpolationFlag(interpolation));
    return fromMat(dst, img.channels);
  } catch (e) {
    dst.delete();
    throw e;
  } finally {
    src.delete();
  }
}

/** Constant-colour border, equivalent to `cv2.copyMakeBorder(..., BORDER_CONSTANT, value)`. */
export function copyMakeBorder(
  img: Image,
  top: number,
  bottom: number,
  left: number,
  right: number,
  color: [number, number, number],
): Image {
  assertReady();
  const src = toMat(img);
  const dst = new cv.Mat();
  try {
    cv.copyMakeBorder(
      src,
      dst,
      top,
      bottom,
      left,
      right,
      cv.BORDER_CONSTANT,
      new cv.Scalar(color[0], color[1], color[2], 0),
    );
    return fromMat(dst, img.channels);
  } catch (e) {
    dst.delete();
    throw e;
  } finally {
    src.delete();
  }
}

export function bgrToGray(img: Image): Image {
  assertReady();
  if (img.channels !== 3) throw new Error("bgrToGray expects a 3-channel image");
  const src = toMat(img);
  const dst = new cv.Mat();
  try {
    cv.cvtColor(src, dst, cv.COLOR_BGR2GRAY);
    return fromMat(dst, 1);
  } catch (e) {
    dst.delete();
    throw e;
  } finally {
    src.delete();
  }
}

export function bgrToRgb(img: Image): Image {
  if (img.channels !== 3) throw new Error("bgrToRgb expects a 3-channel image");
  const out = new Uint8Array(img.data.length);
  for (let i = 0; i < out.length; i += 3) {
    out[i] = img.data[i + 2]!;
    out[i + 1] = img.data[i + 1]!;
    out[i + 2] = img.data[i]!;
  }
  return { ...img, data: out };
}

/** `img[y1:y2, x1:x2]`. Coordinates must already be clamped to the image. */
export function crop(img: Image, x1: number, y1: number, x2: number, y2: number): Image {
  const w = x2 - x1;
  const h = y2 - y1;
  const c = img.channels;
  const out = new Uint8Array(Math.max(w, 0) * Math.max(h, 0) * c);
  for (let row = 0; row < h; row++) {
    const srcStart = ((y1 + row) * img.width + x1) * c;
    out.set(img.data.subarray(srcStart, srcStart + w * c), row * w * c);
  }
  return { data: out, width: Math.max(w, 0), height: Math.max(h, 0), channels: c };
}
