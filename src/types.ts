/**
 * Public types. They mirror the dataclasses in the Python `fast_alpr` and
 * `open_image_models` packages.
 */

/** Raw 8-bit image in HWC layout. Colour order depends on how it was produced (BGR by default). */
export interface Image {
  /** Pixel data, length = width * height * channels. */
  data: Uint8Array;
  width: number;
  height: number;
  channels: 1 | 3;
}

/**
 * Inputs accepted by `ALPR.predict`.
 *
 * - `Image`: already-decoded BGR pixels
 * - `string`: file path, raw base64, or a `data:image/...;base64,...` URL
 * - `Buffer` / `Uint8Array` / `ArrayBuffer` / `ArrayBufferView`: encoded image bytes (JPEG, PNG, WebP, …)
 * - async iterable / Node `Readable`: stream of those encoded bytes
 */
export type PredictInput =
  | Image
  | string
  | Buffer
  | Uint8Array
  | ArrayBuffer
  | ArrayBufferView
  | AsyncIterable<Uint8Array | Buffer>;

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionResult {
  label: string;
  confidence: number;
  boundingBox: BoundingBox;
}

export interface OcrResult {
  /** Recognized plate text. */
  text: string;
  /** One confidence per character slot, or a single value. */
  confidence: number | number[];
  /** Optional region / country prediction. */
  region?: string | null;
  regionConfidence?: number | null;
}

export interface ALPRResult {
  detection: DetectionResult;
  /** `null` when the OCR does not return a result. */
  ocr: OcrResult | null;
}

export type ImageInterpolation = "nearest" | "linear" | "cubic" | "area" | "lanczos4";
export type ImageColorMode = "grayscale" | "rgb";

/** A detector receives a BGR image and returns plate detections. */
export interface BaseDetector {
  predict(frame: Image): Promise<DetectionResult[]>;
}

/** An OCR receives a cropped BGR plate image and returns the recognized text. */
export interface BaseOCR {
  predict(croppedPlate: Image): Promise<OcrResult | null>;
}

/** Execution providers accepted by onnxruntime-node (e.g. "cpu", "coreml", "cuda"). */
export type ExecutionProvider = string | { name: string; [option: string]: unknown };
