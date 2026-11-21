/**
 * OCR input/output processing. Port of `fast_plate_ocr/core/process.py`.
 */
import { copyMakeBorder, resize } from "../cv.js";
import type { Image } from "../types.js";
import { roundHalfEven } from "../util.js";
import type { PlateConfig } from "./config.js";

/**
 * Resizes to the model input size, optionally keeping aspect ratio with letterbox padding.
 * The image must already be in the colour mode the model expects.
 */
export function resizeImage(img: Image, cfg: PlateConfig): Image {
  const { imgHeight, imgWidth, interpolation } = cfg;

  if (!cfg.keepAspectRatio) {
    return resize(img, imgWidth, imgHeight, interpolation);
  }

  const r = Math.min(imgHeight / img.height, imgWidth / img.width);
  const unpadW = roundHalfEven(img.width * r);
  const unpadH = roundHalfEven(img.height * r);

  let out = img;
  if (img.width !== unpadW || img.height !== unpadH) {
    out = resize(img, unpadW, unpadH, interpolation);
  }

  const dw = (imgWidth - unpadW) / 2;
  const dh = (imgHeight - unpadH) / 2;
  const top = roundHalfEven(dh - 0.1);
  const bottom = roundHalfEven(dh + 0.1);
  const left = roundHalfEven(dw - 0.1);
  const right = roundHalfEven(dw + 0.1);

  return copyMakeBorder(out, top, bottom, left, right, cfg.paddingColor);
}

export interface PlatePrediction {
  plate: string;
  /** Max probability of each character slot (not trimmed when the pad is stripped). */
  charProbs: number[] | null;
  region: string | null;
  regionProb: number | null;
}

/** Removes trailing occurrences of `padChar` (Python's `str.rstrip(pad_char)`). */
function rstripChar(text: string[], padChar: string): string {
  let end = text.length;
  while (end > 0 && text[end - 1] === padChar) end--;
  return text.slice(0, end).join("");
}

export interface PostprocessOptions {
  maxPlateSlots: number;
  alphabet: string;
  padChar?: string | null;
  removePadChar?: boolean;
  returnConfidence?: boolean;
  /** Flattened plate head output, `batch * slots * vocab` values. */
  plateOutput: ArrayLike<number>;
  /** Flattened region head output, `batch * numRegions` values (already softmaxed). */
  regionOutput?: ArrayLike<number> | null;
  regionLabels?: readonly string[] | null;
}

/** Decodes raw model outputs into per-image predictions (one entry per batch element). */
export function postprocessOutput(opts: PostprocessOptions): PlatePrediction[] {
  const { maxPlateSlots, padChar = null, removePadChar = true, returnConfidence = false } = opts;
  const alphabet = Array.from(opts.alphabet);
  const vocab = alphabet.length;
  const stride = maxPlateSlots * vocab;

  if (opts.plateOutput.length % stride !== 0) {
    throw new Error(
      `Plate output length ${opts.plateOutput.length} is not a multiple of slots*vocab (${maxPlateSlots}*${vocab})`,
    );
  }
  const batch = opts.plateOutput.length / stride;

  const wantRegion = opts.regionOutput != null && opts.regionLabels != null;
  const numRegions = wantRegion ? opts.regionLabels!.length : 0;

  const results: PlatePrediction[] = [];
  for (let b = 0; b < batch; b++) {
    const chars: string[] = [];
    const probs: number[] = [];

    for (let slot = 0; slot < maxPlateSlots; slot++) {
      const base = b * stride + slot * vocab;
      let best = 0;
      let bestVal = opts.plateOutput[base]!;
      for (let k = 1; k < vocab; k++) {
        const v = opts.plateOutput[base + k]!;
        if (v > bestVal) {
          bestVal = v;
          best = k;
        }
      }
      chars.push(alphabet[best]!);
      probs.push(bestVal);
    }

    const plate = removePadChar && padChar != null ? rstripChar(chars, padChar) : chars.join("");

    let region: string | null = null;
    let regionProb: number | null = null;
    if (wantRegion) {
      const base = b * numRegions;
      let best = 0;
      let bestVal = opts.regionOutput![base]!;
      for (let k = 1; k < numRegions; k++) {
        const v = opts.regionOutput![base + k]!;
        if (v > bestVal) {
          bestVal = v;
          best = k;
        }
      }
      region = opts.regionLabels![best] ?? null;
      if (returnConfidence) regionProb = bestVal;
    }

    results.push({
      plate,
      charProbs: returnConfidence ? probs : null,
      region,
      regionProb,
    });
  }

  return results;
}
