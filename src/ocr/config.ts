/**
 * Plate OCR model config. Port of `fast_plate_ocr/inference/config.py`.
 */
import { readFile } from "node:fs/promises";
import { parse } from "yaml";
import type { ImageColorMode, ImageInterpolation } from "../types.js";

export interface PlateConfig {
  /** Number of character positions predicted by the model. */
  maxPlateSlots: number;
  /** Characters the model can output; index = class id. */
  alphabet: string;
  /** Padding character for plates shorter than `maxPlateSlots`. */
  padChar: string;
  imgHeight: number;
  imgWidth: number;
  keepAspectRatio: boolean;
  interpolation: ImageInterpolation;
  imageColorMode: ImageColorMode;
  /** Letterbox colour (only when `keepAspectRatio`). */
  paddingColor: [number, number, number];
  /** Optional list of regions/countries the model can recognize. */
  plateRegions: string[] | null;
}

export const numChannels = (cfg: PlateConfig): 1 | 3 => (cfg.imageColorMode === "rgb" ? 3 : 1);
export const hasRegionRecognition = (cfg: PlateConfig): boolean => (cfg.plateRegions?.length ?? 0) > 0;

const INTERPOLATIONS: readonly ImageInterpolation[] = ["nearest", "linear", "cubic", "area", "lanczos4"];

function required<T>(data: Record<string, unknown>, key: string): T {
  const value = data[key];
  if (value === undefined || value === null) {
    throw new Error(`Missing required key '${key}' in plate config`);
  }
  return value as T;
}

/** Accepts a scalar or a list; YAML configs may use either. */
function parsePaddingColor(value: unknown): [number, number, number] {
  if (Array.isArray(value)) {
    const [a = 114, b = a, c = a] = value.map(Number);
    return [a, b, c];
  }
  if (typeof value === "number") return [value, value, value];
  return [114, 114, 114];
}

export function parsePlateConfig(yamlText: string): PlateConfig {
  const data = parse(yamlText) as Record<string, unknown>;
  if (!data || typeof data !== "object") {
    throw new Error("Invalid plate config: expected a YAML mapping");
  }

  const interpolation = (data.interpolation as ImageInterpolation | undefined) ?? "linear";
  if (!INTERPOLATIONS.includes(interpolation)) {
    throw new Error(`Unsupported interpolation '${interpolation}'`);
  }
  const imageColorMode = (data.image_color_mode as ImageColorMode | undefined) ?? "grayscale";
  if (imageColorMode !== "grayscale" && imageColorMode !== "rgb") {
    throw new Error(`Unsupported image_color_mode '${imageColorMode}'`);
  }

  return {
    maxPlateSlots: required<number>(data, "max_plate_slots"),
    alphabet: required<string>(data, "alphabet"),
    padChar: required<string>(data, "pad_char"),
    imgHeight: required<number>(data, "img_height"),
    imgWidth: required<number>(data, "img_width"),
    keepAspectRatio: (data.keep_aspect_ratio as boolean | undefined) ?? false,
    interpolation,
    imageColorMode,
    paddingColor: parsePaddingColor(data.padding_color),
    plateRegions: (data.plate_regions as string[] | null | undefined) ?? null,
  };
}

export async function loadPlateConfig(path: string): Promise<PlateConfig> {
  return parsePlateConfig(await readFile(path, "utf-8"));
}
