import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // The integration test downloads models on first run.
    testTimeout: 180_000,
    hookTimeout: 180_000,
  },
});
