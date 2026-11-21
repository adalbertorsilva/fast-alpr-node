/**
 * Model hub: URLs, download and cache.
 *
 * Cache locations match the Python libraries (`~/.cache/open-image-models/<name>` and
 * `~/.cache/fast-plate-ocr/<name>`), so models already downloaded by Python are reused.
 */
import { createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

// ---------------------------------------------------------------- detectors

const DETECTOR_BASE_URL = "https://github.com/ankandrew/open-image-models/releases/download/assets";

export const PLATE_DETECTOR_MODELS = {
  "yolo-v9-s-608-license-plate-end2end": "yolo-v9-s-608-license-plates-end2end.onnx",
  "yolo-v9-t-640-license-plate-end2end": "yolo-v9-t-640-license-plates-end2end.onnx",
  "yolo-v9-t-512-license-plate-end2end": "yolo-v9-t-512-license-plates-end2end.onnx",
  "yolo-v9-t-416-license-plate-end2end": "yolo-v9-t-416-license-plates-end2end.onnx",
  "yolo-v9-t-384-license-plate-end2end": "yolo-v9-t-384-license-plates-end2end.onnx",
  "yolo-v9-t-256-license-plate-end2end": "yolo-v9-t-256-license-plates-end2end.onnx",
} as const;

export type PlateDetectorModel = keyof typeof PLATE_DETECTOR_MODELS;

export const DETECTOR_CLASS_LABELS = ["License Plate"] as const;

// --------------------------------------------------------------------- OCR

const OCR_BASE_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates";

/** model name -> [onnx file, plate config file] */
export const OCR_MODELS = {
  "cct-s-v2-global-model": ["cct_s_v2_global.onnx", "cct_s_v2_global_plate_config.yaml"],
  "cct-xs-v2-global-model": ["cct_xs_v2_global.onnx", "cct_xs_v2_global_plate_config.yaml"],
  "cct-s-v1-global-model": ["cct_s_v1_global.onnx", "cct_s_v1_global_plate_config.yaml"],
  "cct-xs-v1-global-model": ["cct_xs_v1_global.onnx", "cct_xs_v1_global_plate_config.yaml"],
  "cct-s-relu-v1-global-model": ["cct_s_relu_v1_global.onnx", "cct_s_relu_v1_global_plate_config.yaml"],
  "cct-xs-relu-v1-global-model": ["cct_xs_relu_v1_global.onnx", "cct_xs_relu_v1_global_plate_config.yaml"],
  "argentinian-plates-cnn-model": ["arg_cnn_ocr.onnx", "arg_cnn_ocr_config.yaml"],
  "argentinian-plates-cnn-synth-model": ["arg_cnn_ocr_synth.onnx", "arg_cnn_ocr_config.yaml"],
  "european-plates-mobile-vit-v2-model": ["european_mobile_vit_v2_ocr.onnx", "european_mobile_vit_v2_ocr_config.yaml"],
  "global-plates-mobile-vit-v2-model": ["global_mobile_vit_v2_ocr.onnx", "global_mobile_vit_v2_ocr_config.yaml"],
} as const;

export type OcrModel = keyof typeof OCR_MODELS;

// ---------------------------------------------------------------- download

const exists = (path: string): Promise<boolean> =>
  stat(path).then(
    (s) => s.isFile(),
    () => false,
  );

/** Downloads `url` to `dest` through a temp file, so an interrupted download never leaves a corrupt model. */
async function download(url: string, dest: string): Promise<void> {
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download file from ${url}. Status code: ${response.status}`);
  }
  const tmp = `${dest}.${process.pid}.part`;
  try {
    await pipeline(Readable.fromWeb(response.body as never), createWriteStream(tmp));
    await rename(tmp, dest);
  } catch (e) {
    await rm(tmp, { force: true });
    throw e;
  }
}

async function ensureFile(url: string, dir: string, force: boolean): Promise<string> {
  const dest = join(dir, basename(url));
  if (force || !(await exists(dest))) {
    await download(url, dest);
  }
  return dest;
}

export interface DownloadOptions {
  /** Directory to store the model in. Defaults to the shared cache directory. */
  saveDirectory?: string;
  forceDownload?: boolean;
}

export async function downloadDetectorModel(
  modelName: PlateDetectorModel,
  { saveDirectory, forceDownload = false }: DownloadOptions = {},
): Promise<string> {
  const file = PLATE_DETECTOR_MODELS[modelName];
  if (!file) {
    throw new Error(`Unknown model ${modelName}. Use one of [${Object.keys(PLATE_DETECTOR_MODELS).join(", ")}]`);
  }
  const dir = saveDirectory ?? join(homedir(), ".cache", "open-image-models", modelName);
  await mkdir(dir, { recursive: true });
  return ensureFile(`${DETECTOR_BASE_URL}/${file}`, dir, forceDownload);
}

export async function downloadOcrModel(
  modelName: OcrModel,
  { saveDirectory, forceDownload = false }: DownloadOptions = {},
): Promise<{ modelPath: string; configPath: string }> {
  const entry = OCR_MODELS[modelName];
  if (!entry) {
    throw new Error(`Unknown model ${modelName}. Use one of [${Object.keys(OCR_MODELS).join(", ")}]`);
  }
  const dir = saveDirectory ?? join(homedir(), ".cache", "fast-plate-ocr", modelName);
  await mkdir(dir, { recursive: true });
  const [modelFile, configFile] = entry;
  const modelPath = await ensureFile(`${OCR_BASE_URL}/${modelFile}`, dir, forceDownload);
  const configPath = await ensureFile(`${OCR_BASE_URL}/${configFile}`, dir, forceDownload);
  return { modelPath, configPath };
}
