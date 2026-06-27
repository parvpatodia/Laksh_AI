/**
 * Vitest configuration for web/lib/realtime unit tests.
 *
 * Tests are pure TypeScript — no DOM, no React, no MediaPipe WASM.
 * @mediapipe/tasks-vision is imported via `import type` only in the
 * production code, so no mocking is required.
 */
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["lib/realtime/__tests__/**/*.test.ts"],
  },
});
