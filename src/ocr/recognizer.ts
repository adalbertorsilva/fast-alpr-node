/**
 * License plate OCR. Port of `LicensePlateRecognizer` (fast-plate-ocr) combined with
 * fast-alpr's `DefaultOCR`.
 */
import { access } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { bgrToGray, bgrToRgb } from "../cv.js";
import { downloadOcrModel, type OcrModel } from "../hub.js";
import type { BaseOCR, ExecutionProvider, Image, OcrResult } from "../types.js";
import { hasRegionRecognition, loadPlateConfig, numChannels, type PlateConfig } from "./config.js";
import { postprocessOutput, resizeImage, type PlatePrediction } from "./process.js";

export interface PlateRecognizerOptions {
  modelPath: string;
  configPath: string;
  providers?: ExecutionProvider[];
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

export class LicensePlateRecognizer {
  private constructor(
    readonly config: PlateConfig,
    private readonly session: ort.InferenceSession,
    private readonly inputName: string,
    private readonly plateOutputName: string,
    private readonly regionOutputName: string | null,
  ) {}

  static async create(options: PlateRecognizerOptions): Promise<LicensePlateRecognizer> {
    for (const path of [options.modelPath, options.configPath]) {
      await access(path).catch(() => {
        throw new Error(`Missing model/config file: ${path}`);
      });
    }

    const config = await loadPlateConfig(options.configPath);
    const session = await ort.InferenceSession.create(options.modelPath, {
      executionProviders: (options.providers ?? ["cpu"]) as ort.InferenceSession.ExecutionProviderConfig[],
      ...options.sessionOptions,
    });

    const outputNames = session.outputNames;
    const plateOutputName = outputNames.includes("plate") ? "plate" : outputNames[0]!;
    const regionOutputName = outputNames.includes("region") ? "region" : null;
    const inputName = session.inputNames.includes("input") ? "input" : session.inputNames[0]!;

    if (regionOutputName === null && hasRegionRecognition(config)) {
      console.warn("Plate config declares regions but the loaded model has no 'region' output. Region predictions will be disabled.");
    }
    if (regionOutputName !== null && !hasRegionRecognition(config)) {
      console.warn("Model exports a 'region' output but the plate config has no region list. Region predictions will be disabled.");
    }

    return new LicensePlateRecognizer(config, session, inputName, plateOutputName, regionOutputName);
  }

  private get hasRegionHead(): boolean {
    return this.regionOutputName !== null && hasRegionRecognition(this.config);
  }

  /**
   * Runs OCR on one image that is already in the colour mode of the model
   * (grayscale or RGB, see `PlateConfig.imageColorMode`).
   */
  async runOne(image: Image, returnConfidence = false, removePadChar = true): Promise<PlatePrediction> {
    const cfg = this.config;
    const expected = numChannels(cfg);
    if (image.channels !== expected) {
      throw new Error(`Model expects ${expected} channel(s), got ${image.channels}`);
    }

    const resized = resizeImage(image, cfg);
    // The model normalizes pixel values internally: feed raw uint8 NHWC.
    const input = new ort.Tensor("uint8", resized.data, [1, cfg.imgHeight, cfg.imgWidth, expected]);

    const fetches = this.hasRegionHead ? [this.plateOutputName, this.regionOutputName!] : [this.plateOutputName];
    const outputs = await this.session.run({ [this.inputName]: input }, fetches);

    const plate = outputs[this.plateOutputName]!;
    const region = this.hasRegionHead ? outputs[this.regionOutputName!]! : null;

    const [prediction] = postprocessOutput({
      maxPlateSlots: cfg.maxPlateSlots,
      alphabet: cfg.alphabet,
      padChar: cfg.padChar,
      removePadChar,
      returnConfidence,
      plateOutput: plate.data as Float32Array,
      regionOutput: region ? (region.data as Float32Array) : null,
      regionLabels: this.hasRegionHead ? cfg.plateRegions : null,
    });
    if (!prediction) throw new Error("Expected exactly 1 result, got 0.");
    return prediction;
  }
}

/** fast-alpr's `DefaultOCR`: takes a BGR crop and handles the colour conversion. */
export class DefaultOCR implements BaseOCR {
  private constructor(private readonly recognizer: LicensePlateRecognizer) {}

  static async create(options: PlateRecognizerOptions): Promise<DefaultOCR> {
    return new DefaultOCR(await LicensePlateRecognizer.create(options));
  }

  async predict(croppedPlate: Image): Promise<OcrResult | null> {
    if (!croppedPlate || croppedPlate.width === 0 || croppedPlate.height === 0) return null;

    const mode = this.recognizer.config.imageColorMode;
    const image = mode === "grayscale" ? bgrToGray(croppedPlate) : bgrToRgb(croppedPlate);

    const prediction = await this.recognizer.runOne(image, true);
    return {
      text: prediction.plate,
      confidence: prediction.charProbs ?? 0.0,
      region: prediction.region,
      regionConfidence: prediction.regionProb,
    };
  }
}

export interface DefaultOcrOptions {
  /** OCR model name from the hub. Ignored when `modelPath` and `configPath` are given. */
  modelName?: OcrModel;
  modelPath?: string;
  configPath?: string;
  providers?: ExecutionProvider[];
  sessionOptions?: ort.InferenceSession.SessionOptions;
  forceDownload?: boolean;
}

/** Downloads (if needed) a hub model, or uses a custom model, and builds the OCR. */
export async function createDefaultOcr(options: DefaultOcrOptions = {}): Promise<DefaultOCR> {
  let { modelPath, configPath } = options;
  if (!modelPath || !configPath) {
    const downloaded = await downloadOcrModel(options.modelName ?? "cct-xs-v2-global-model", {
      forceDownload: options.forceDownload,
    });
    modelPath = downloaded.modelPath;
    configPath = downloaded.configPath;
  }
  return DefaultOCR.create({
    modelPath,
    configPath,
    providers: options.providers,
    sessionOptions: options.sessionOptions,
  });
}
