/**
 * Automatic License Plate Recognition. Port of `fast_alpr/alpr.py`.
 */
import type * as ort from "onnxruntime-node";
import { crop, initCv, resolveImage } from "./cv.js";
import { createDefaultDetector } from "./detector/yolov9.js";
import type { OcrModel, PlateDetectorModel } from "./hub.js";
import { createDefaultOcr } from "./ocr/recognizer.js";
import type {
  ALPRResult,
  BaseDetector,
  BaseOCR,
  ExecutionProvider,
  PredictInput,
} from "./types.js";

export interface ALPROptions {
  /** Custom detector. When omitted, the default YOLOv9 detector is used. */
  detector?: BaseDetector;
  /** Custom OCR. When omitted, the default `fast-plate-ocr` model is used. */
  ocr?: BaseOCR;
  /** Detector model name from the hub. Default: "yolo-v9-t-384-license-plate-end2end". */
  detectorModel?: PlateDetectorModel;
  /** Detector confidence threshold. Default: 0.4. */
  detectorConfThresh?: number;
  detectorProviders?: ExecutionProvider[];
  detectorSessionOptions?: ort.InferenceSession.SessionOptions;
  /** OCR model name from the hub. Default: "cct-xs-v2-global-model". */
  ocrModel?: OcrModel;
  ocrProviders?: ExecutionProvider[];
  ocrSessionOptions?: ort.InferenceSession.SessionOptions;
  /** Custom OCR model file; requires `ocrConfigPath`. */
  ocrModelPath?: string;
  ocrConfigPath?: string;
  ocrForceDownload?: boolean;
}

export class ALPR {
  private constructor(
    readonly detector: BaseDetector,
    readonly ocr: BaseOCR,
  ) {}

  /** Loads the models (downloading them on first use) and returns a ready-to-use instance. */
  static async create(options: ALPROptions = {}): Promise<ALPR> {
    await initCv();

    const detector =
      options.detector ??
      (await createDefaultDetector({
        modelName: options.detectorModel,
        confThresh: options.detectorConfThresh ?? 0.4,
        providers: options.detectorProviders,
        sessionOptions: options.detectorSessionOptions,
      }));

    const ocr =
      options.ocr ??
      (await createDefaultOcr({
        modelName: options.ocrModel,
        modelPath: options.ocrModelPath,
        configPath: options.ocrConfigPath,
        providers: options.ocrProviders,
        sessionOptions: options.ocrSessionOptions,
        forceDownload: options.ocrForceDownload,
      }));

    return new ALPR(detector, ocr);
  }

  /**
   * Runs plate detection and OCR on an image.
   *
   * @param frame File path, BGR `Image`, encoded bytes (`Buffer` / `Uint8Array`),
   *   base64 / data URL, or a stream of encoded bytes.
   * @returns One `ALPRResult` per detected plate.
   */
  async predict(frame: PredictInput): Promise<ALPRResult[]> {
    const img = await resolveImage(frame);

    const detections = await this.detector.predict(img);
    const results: ALPRResult[] = [];

    for (const detection of detections) {
      const { x1, y1, x2, y2 } = detection.boundingBox;
      const cx1 = Math.max(x1, 0);
      const cy1 = Math.max(y1, 0);
      const cx2 = Math.min(x2, img.width);
      const cy2 = Math.min(y2, img.height);

      // A degenerate box gives an empty crop; the OCR cannot read it.
      const croppedPlate = crop(img, cx1, cy1, cx2, cy2);
      const ocr = croppedPlate.width > 0 && croppedPlate.height > 0 ? await this.ocr.predict(croppedPlate) : null;

      results.push({ detection, ocr });
    }

    return results;
  }
}
