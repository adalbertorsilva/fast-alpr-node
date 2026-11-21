import { beforeAll, describe, expect, it } from "vitest";
import { initCv } from "../src/cv.js";
import { parsePlateConfig } from "../src/ocr/config.js";
import { postprocessOutput, resizeImage } from "../src/ocr/process.js";
import type { Image } from "../src/types.js";

beforeAll(async () => {
  await initCv();
});

describe("parsePlateConfig", () => {
  it("parses a config with regions and defaults", () => {
    const cfg = parsePlateConfig(`
max_plate_slots: 10
alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
pad_char: "_"
img_height: 64
img_width: 128
plate_regions: [Argentina, Brazil, Unknown]
`);
    expect(cfg.maxPlateSlots).toBe(10);
    expect(cfg.padChar).toBe("_");
    expect(cfg.imageColorMode).toBe("grayscale");
    expect(cfg.interpolation).toBe("linear");
    expect(cfg.keepAspectRatio).toBe(false);
    expect(cfg.paddingColor).toEqual([114, 114, 114]);
    expect(cfg.plateRegions).toEqual(["Argentina", "Brazil", "Unknown"]);
  });

  it("accepts scalar or list padding_color and ignores unknown keys", () => {
    const a = parsePlateConfig(
      "max_plate_slots: 1\nalphabet: AB\npad_char: B\nimg_height: 1\nimg_width: 1\npadding_color: 7\nextra: x\n",
    );
    expect(a.paddingColor).toEqual([7, 7, 7]);
    const b = parsePlateConfig(
      "max_plate_slots: 1\nalphabet: AB\npad_char: B\nimg_height: 1\nimg_width: 1\npadding_color: [1, 2, 3]\n",
    );
    expect(b.paddingColor).toEqual([1, 2, 3]);
  });

  it("rejects a config with missing keys", () => {
    expect(() => parsePlateConfig("alphabet: AB\n")).toThrow(/max_plate_slots/);
  });
});

describe("postprocessOutput", () => {
  // alphabet "AB_" (pad "_"), 4 slots, vocab 3
  const base = { maxPlateSlots: 4, alphabet: "AB_", padChar: "_" };
  // slot argmax: A, B, _, _   -> "AB" after stripping pad
  const plateOutput = [
    0.9, 0.05, 0.05, //
    0.1, 0.8, 0.1,
    0.0, 0.1, 0.9,
    0.2, 0.1, 0.7,
  ];

  it("decodes argmax per slot and strips trailing pad", () => {
    const [p] = postprocessOutput({ ...base, plateOutput });
    expect(p!.plate).toBe("AB");
    expect(p!.charProbs).toBeNull();
    expect(p!.region).toBeNull();
  });

  it("keeps the pad when removePadChar is false", () => {
    const [p] = postprocessOutput({ ...base, plateOutput, removePadChar: false });
    expect(p!.plate).toBe("AB__");
  });

  it("returns per-slot confidences that are not trimmed", () => {
    const [p] = postprocessOutput({ ...base, plateOutput, returnConfidence: true });
    expect(p!.charProbs).toEqual([0.9, 0.8, 0.9, 0.7]);
  });

  it("does not strip pad characters that are not trailing", () => {
    // A, _, B, _  -> "A_B"
    const out = [0.9, 0, 0, 0, 0, 0.9, 0, 0.9, 0, 0, 0, 0.9];
    const [p] = postprocessOutput({ ...base, plateOutput: out });
    expect(p!.plate).toBe("A_B");
  });

  it("decodes region with its probability only when confidence is requested", () => {
    const regionOutput = [0.1, 0.7, 0.2];
    const labels = ["X", "Y", "Z"];
    const [a] = postprocessOutput({ ...base, plateOutput, regionOutput, regionLabels: labels });
    expect(a!.region).toBe("Y");
    expect(a!.regionProb).toBeNull();
    const [b] = postprocessOutput({ ...base, plateOutput, regionOutput, regionLabels: labels, returnConfidence: true });
    expect(b!.regionProb).toBe(0.7);
  });

  it("handles a batch", () => {
    const two = [...plateOutput, ...plateOutput];
    expect(postprocessOutput({ ...base, plateOutput: two }).map((p) => p.plate)).toEqual(["AB", "AB"]);
  });

  it("rejects an output with an unexpected size", () => {
    expect(() => postprocessOutput({ ...base, plateOutput: [1, 2, 3, 4] })).toThrow();
  });
});

describe("resizeImage", () => {
  const gray = (w: number, h: number, v: number): Image => ({
    data: new Uint8Array(w * h).fill(v),
    width: w,
    height: h,
    channels: 1,
  });
  const cfg = (over = {}) => ({
    maxPlateSlots: 1,
    alphabet: "A",
    padChar: "A",
    imgHeight: 64,
    imgWidth: 128,
    keepAspectRatio: false,
    interpolation: "linear" as const,
    imageColorMode: "grayscale" as const,
    paddingColor: [114, 114, 114] as [number, number, number],
    plateRegions: null,
    ...over,
  });

  it("stretches to the model size by default", () => {
    const out = resizeImage(gray(30, 30, 200), cfg());
    expect([out.width, out.height, out.channels]).toEqual([128, 64, 1]);
  });

  it("letterboxes with the padding colour when keepAspectRatio is set", () => {
    const out = resizeImage(gray(64, 64, 200), cfg({ keepAspectRatio: true }));
    expect([out.width, out.height]).toEqual([128, 64]);
    expect(out.data[0]).toBe(114); // left padding
    expect(out.data[32 * 128 + 64]).toBe(200); // centre keeps the content
  });
});
