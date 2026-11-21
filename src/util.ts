/**
 * Round half to even, like Python's built-in `round()`.
 * `Math.round` rounds halves up, which would shift letterbox padding by one pixel
 * in some cases (e.g. round(2.5) is 2 in Python but 3 in JavaScript).
 */
export function roundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
