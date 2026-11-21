import { describe, expect, it } from "vitest";
import { roundHalfEven } from "../src/util.js";

describe("roundHalfEven", () => {
  it("matches Python's round() on halves", () => {
    // Python: round(0.5)=0, round(1.5)=2, round(2.5)=2, round(3.5)=4, round(-0.5)=-0, round(-1.5)=-2
    expect(roundHalfEven(0.5)).toBe(0);
    expect(roundHalfEven(1.5)).toBe(2);
    expect(roundHalfEven(2.5)).toBe(2);
    expect(roundHalfEven(3.5)).toBe(4);
    expect(roundHalfEven(-1.5)).toBe(-2);
    expect(roundHalfEven(-2.5)).toBe(-2);
  });

  it("rounds normally away from halves", () => {
    expect(roundHalfEven(2.4)).toBe(2);
    expect(roundHalfEven(2.6)).toBe(3);
    expect(roundHalfEven(7)).toBe(7);
    expect(roundHalfEven(-2.6)).toBe(-3);
  });

  it("differs from Math.round exactly where it should", () => {
    expect(Math.round(2.5)).toBe(3);
    expect(roundHalfEven(2.5)).toBe(2);
  });
});
