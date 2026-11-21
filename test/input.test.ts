import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import {
  ALPR,
  isImage,
  readImage,
  readImageFromBase64,
  readImageFromBuffer,
  resolveImage,
} from "../src/index.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const imagePath = join(root, "assets", "test_image.png");

let fileBytes: Buffer;
let base64: string;
let dataUrl: string;
let reference: Awaited<ReturnType<typeof readImage>>;

beforeAll(async () => {
  fileBytes = await readFile(imagePath);
  base64 = fileBytes.toString("base64");
  dataUrl = `data:image/png;base64,${base64}`;
  reference = await readImage(imagePath);
});

describe("resolveImage", () => {
  it("detects an already-decoded Image", () => {
    expect(isImage(reference)).toBe(true);
    expect(isImage(fileBytes)).toBe(false);
    expect(isImage({ data: fileBytes, width: 1, height: 1, channels: 3 })).toBe(true);
  });

  it("loads a file path", async () => {
    const img = await resolveImage(imagePath);
    expect(img.width).toBe(reference.width);
    expect(img.height).toBe(reference.height);
    expect(img.channels).toBe(3);
    expect(img.data).toEqual(reference.data);
  });

  it("decodes a Buffer / Uint8Array / ArrayBuffer", async () => {
    const arrayBuffer = fileBytes.buffer.slice(
      fileBytes.byteOffset,
      fileBytes.byteOffset + fileBytes.byteLength,
    ) as ArrayBuffer;
    for (const input of [fileBytes, new Uint8Array(fileBytes), arrayBuffer]) {
      const img = await resolveImage(input);
      expect(img.data).toEqual(reference.data);
    }
  });

  it("decodes raw base64 and a data URL", async () => {
    // PNG base64 contains `/`; that must not be mistaken for a file path.
    expect(base64.includes("/")).toBe(true);
    expect((await resolveImage(base64)).data).toEqual(reference.data);
    expect((await resolveImage(dataUrl)).data).toEqual(reference.data);
    expect((await readImageFromBase64(`  ${base64}  \n`)).data).toEqual(reference.data);
  });

  it("concatenates an async byte stream before decoding", async () => {
    const mid = Math.floor(fileBytes.length / 2);
    const stream = Readable.from([fileBytes.subarray(0, mid), fileBytes.subarray(mid)]);
    expect((await resolveImage(stream)).data).toEqual(reference.data);
  });

  it("accepts a Node file ReadStream", async () => {
    expect((await resolveImage(createReadStream(imagePath))).data).toEqual(reference.data);
  });

  it("passes an Image through unchanged", async () => {
    expect(await resolveImage(reference)).toBe(reference);
  });

  it("rejects empty / invalid inputs", async () => {
    await expect(readImageFromBase64("")).rejects.toThrow(/empty string/);
    await expect(readImageFromBuffer(Buffer.alloc(0))).rejects.toThrow(/Failed to load image from buffer/);
    await expect(resolveImage(Readable.from([]))).rejects.toThrow(/no bytes/);
    await expect(resolveImage("not-a-real-file.png")).rejects.toThrow(/Failed to load image from path/);
  });

  it("treats short extension-less names as paths, not base64", async () => {
    await expect(resolveImage("foto")).rejects.toThrow(/Failed to load image from path/);
  });
});

describe("ALPR.predict with alternate inputs", () => {
  let alpr: ALPR;

  beforeAll(async () => {
    alpr = await ALPR.create();
  });

  it("returns the same plate for path, buffer, base64 and stream", async () => {
    const [fromPath] = await alpr.predict(imagePath);
    const [fromBuffer] = await alpr.predict(fileBytes);
    const [fromBase64] = await alpr.predict(dataUrl);
    const [fromStream] = await alpr.predict(createReadStream(imagePath));

    expect(fromPath?.ocr?.text).toBe("5AU5341");
    expect(fromBuffer?.ocr?.text).toBe(fromPath?.ocr?.text);
    expect(fromBase64?.ocr?.text).toBe(fromPath?.ocr?.text);
    expect(fromStream?.ocr?.text).toBe(fromPath?.ocr?.text);
    expect(fromBuffer?.detection.boundingBox).toEqual(fromPath?.detection.boundingBox);
  });
});
