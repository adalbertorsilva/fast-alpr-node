/**
 * YOLOv9 license plate detector (ONNX, end2end NMS). Port of `YoloV9Detector` and
 * fast-alpr's `DefaultDetector`.
 */
import { access } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import {
  DETECTOR_CLASS_LABELS,
  downloadDetectorModel,
  type PlateDetectorModel,
} from "../hub.js";
import type { BaseDetector, DetectionResult, ExecutionProvider, Image } from "../types.js";
import { convertToDetectionResult, END2END_COLUMNS } from "./postprocess.js";
import { preprocess } from "./preprocess.js";

export interface YoloV9DetectorOptions {
  /** Path to an ONNX model. */
  modelPath: string;
  classLabels?: readonly string[];
  /** Confidence threshold. Defaults to 0.25 (fast-alpr's ALPR overrides it to 0.4). */
  confThresh?: number;
  providers?: ExecutionProvider[];
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

export class YoloV9Detector implements BaseDetector {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly inputName: string,
    private readonly outputName: string,
    readonly imgSize: number,
    private readonly classLabels: readonly string[],
    readonly confThresh: number,
  ) {}

  static async create(options: YoloV9DetectorOptions): Promise<YoloV9Detector> {
    const { modelPath, classLabels = DETECTOR_CLASS_LABELS, confThresh = 0.25 } = options;
    await access(modelPath).catch(() => {
      throw new Error(`ONNX model not found at '${modelPath}'`);
    });

    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: (options.providers ?? ["cpu"]) as ort.InferenceSession.ExecutionProviderConfig[],
      ...options.sessionOptions,
    });

    const inputName = session.inputNames[0]!;
    const outputName = session.outputNames[0]!;
    const imgSize = inputImageSize(session);
    return new YoloV9Detector(session, inputName, outputName, imgSize, classLabels, confThresh);
  }

  async predict(frame: Image): Promise<DetectionResult[]> {
    if (frame.channels !== 3) throw new Error("Detector expects a 3-channel BGR image");

    const { tensor, shape, ratio, padding } = preprocess(frame, this.imgSize);

    let output: ort.Tensor | undefined;
    try {
      const outputs = await this.session.run(
        { [this.inputName]: new ort.Tensor("float32", tensor, shape) },
        [this.outputName],
      );
      output = outputs[this.outputName];
    } catch (e) {
      // CoreML cannot handle the empty-output case of end2end NMS, see
      // https://github.com/microsoft/onnxruntime/issues/20372. The Python library also
      // swallows this and returns no detections.
      console.warn("An error occurred during model inference:", e);
      return [];
    }
    if (!output) return [];

    const dims = output.dims;
    const last = dims[dims.length - 1];
    if (last === undefined || last < END2END_COLUMNS) {
      throw new Error(`Unexpected YOLOv9 output shape: [${dims.join(", ")}]`);
    }
    // [N, 7] (batch index in column 0) or [1, N, 7]. We always run with batch size 1.
    const rows = dims.length === 3 ? dims[1]! : dims[0]!;
    if (dims.length === 3 && dims[0] !== 1) {
      throw new Error(`Unexpected YOLOv9 output shape: [${dims.join(", ")}]`);
    }

    return convertToDetectionResult(
      output.data as Float32Array,
      rows,
      this.classLabels,
      ratio,
      padding,
      this.confThresh,
    );
  }
}

/** Reads the (square) input size from the model's `[batch, 3, H, W]` input. */
function inputImageSize(session: ort.InferenceSession): number {
  const meta = session.inputMetadata[0];
  const shape = meta && meta.isTensor ? meta.shape : [];
  const h = shape[2];
  const w = shape[3];
  if (typeof h !== "number" || typeof w !== "number") {
    throw new Error(`Could not determine model input size from shape [${shape.join(", ")}]`);
  }
  if (h !== w) {
    throw new Error(`Model only supports square images, but received shape: ${h}x${w}`);
  }
  return h;
}

export interface DefaultDetectorOptions {
  /** Detector model name from the hub. */
  modelName?: PlateDetectorModel;
  confThresh?: number;
  providers?: ExecutionProvider[];
  sessionOptions?: ort.InferenceSession.SessionOptions;
  forceDownload?: boolean;
}

/** Downloads (if needed) a hub model and builds the detector. Equivalent to `DefaultDetector`. */
export async function createDefaultDetector(options: DefaultDetectorOptions = {}): Promise<YoloV9Detector> {
  const { modelName = "yolo-v9-t-384-license-plate-end2end", forceDownload = false } = options;
  const modelPath = await downloadDetectorModel(modelName, { forceDownload });
  return YoloV9Detector.create({
    modelPath,
    confThresh: options.confThresh ?? 0.4,
    providers: options.providers,
    sessionOptions: options.sessionOptions,
  });
}
