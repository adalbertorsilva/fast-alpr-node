/**
 * YOLOv9 end2end postprocessing. Port of
 * `open_image_models/detection/core/yolo_v9/postprocess.py`.
 */
import type { DetectionResult } from "../types.js";

/** Columns of an end2end row: [batch_index, x1, y1, x2, y2, class_id, score]. */
export const END2END_COLUMNS = 7;

/**
 * Converts the end2end output (N rows x 7 columns, row-major) into detections in the
 * coordinate space of the original image.
 *
 * Rows are already NMS-filtered by the model; here we only apply the score threshold and
 * undo the letterbox (subtract padding, divide by ratio). Coordinates are truncated toward
 * zero, like Python's `int()`.
 */
export function convertToDetectionResult(
  predictions: ArrayLike<number>,
  rows: number,
  classLabels: readonly string[],
  ratio: [number, number],
  padding: [number, number],
  scoreThreshold = 0.5,
): DetectionResult[] {
  const results: DetectionResult[] = [];

  for (let row = 0; row < rows; row++) {
    const o = row * END2END_COLUMNS;
    const score = predictions[o + 6]!;
    if (score < scoreThreshold) continue;

    const x1 = (predictions[o + 1]! - padding[0]) / ratio[0];
    const y1 = (predictions[o + 2]! - padding[1]) / ratio[1];
    const x2 = (predictions[o + 3]! - padding[0]) / ratio[0];
    const y2 = (predictions[o + 4]! - padding[1]) / ratio[1];

    const classId = Math.trunc(predictions[o + 5]!);
    results.push({
      label: classLabels[classId] ?? String(classId),
      confidence: score,
      boundingBox: {
        x1: Math.trunc(x1),
        y1: Math.trunc(y1),
        x2: Math.trunc(x2),
        y2: Math.trunc(y2),
      },
    });
  }

  return results;
}
