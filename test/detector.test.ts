import { beforeAll, describe, expect, it } from "vitest";
import { initCv } from "../src/cv.js";
import { convertToDetectionResult } from "../src/detector/postprocess.js";
import { letterbox, preprocess } from "../src/detector/preprocess.js";
import type { Image } from "../src/types.js";

beforeAll(async () => {
  await initCv();
});

function solid(width: number, height: number, bgr: [number, number, number]): Image {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) data.set(bgr, i * 3);
  return { data, width, height, channels: 3 };
}

describe("letterbox", () => {
  it("pads a wide image to a square and reports ratio and float padding", () => {
    // 200x100 -> 384x384: r = 1.92, unpadded = 384x192, dh = 96
    const { image, ratio, padding } = letterbox(solid(200, 100, [10, 20, 30]), [384, 384]);
    expect(image.width).toBe(384);
    expect(image.height).toBe(384);
    expect(ratio).toEqual([1.92, 1.92]);
    expect(padding).toEqual([0, 96]);
    // padding rows use 114, content rows keep the colour
    expect(Array.from(image.data.slice(0, 3))).toEqual([114, 114, 114]);
    const mid = (192 * 384 + 192) * 3;
    expect(Array.from(image.data.slice(mid, mid + 3))).toEqual([10, 20, 30]);
  });

  it("splits odd padding like the Python version (round(dh -/+ 0.1))", () => {
    // 100x99 -> 100x100: r=1.0 (width-limited), unpadded=100x99, dh=0.5 -> top=0, bottom=1
    const { image, padding } = letterbox(solid(100, 99, [1, 2, 3]), [100, 100]);
    expect(padding).toEqual([0, 0.5]);
    expect(image.height).toBe(100);
    const lastRow = 99 * 100 * 3;
    expect(Array.from(image.data.slice(lastRow, lastRow + 3))).toEqual([114, 114, 114]);
    expect(Array.from(image.data.slice(0, 3))).toEqual([1, 2, 3]); // no top padding
  });
});

describe("preprocess", () => {
  it("produces a [1,3,S,S] RGB CHW float tensor in 0..1", () => {
    const { tensor, shape } = preprocess(solid(64, 64, [0, 51, 255]), 64); // B=0, G=51, R=255
    expect(shape).toEqual([1, 3, 64, 64]);
    expect(tensor.length).toBe(3 * 64 * 64);
    const plane = 64 * 64;
    expect(tensor[0]).toBeCloseTo(1.0, 6); // R
    expect(tensor[plane]).toBeCloseTo(0.2, 6); // G
    expect(tensor[2 * plane]).toBeCloseTo(0.0, 6); // B
  });
});

describe("convertToDetectionResult", () => {
  // rows: [batch, x1, y1, x2, y2, class, score]
  const rows = [0, 110, 120, 210, 170, 0, 0.9, 0, 10, 10, 20, 20, 0, 0.1];

  it("drops low scores and undoes the letterbox", () => {
    const results = convertToDetectionResult(rows, 2, ["License Plate"], [2, 2], [10, 20], 0.4);
    expect(results).toEqual([
      {
        label: "License Plate",
        confidence: 0.9,
        boundingBox: { x1: 50, y1: 50, x2: 100, y2: 75 },
      },
    ]);
  });

  it("truncates toward zero like Python's int()", () => {
    const r = convertToDetectionResult([0, 5, 5, 29.9, 29.9, 0, 0.9], 1, ["p"], [1, 1], [10, 10], 0.4);
    expect(r[0]!.boundingBox).toEqual({ x1: -5, y1: -5, x2: 19, y2: 19 });
  });

  it("falls back to the class id when there is no label", () => {
    const r = convertToDetectionResult([0, 0, 0, 1, 1, 3, 0.9], 1, ["only"], [1, 1], [0, 0], 0.4);
    expect(r[0]!.label).toBe("3");
  });
});
