import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const css = fs.readFileSync(new URL("../public/style.css", import.meta.url), "utf8");
const summary = fs.readFileSync(new URL("../public/live-summary.js", import.meta.url), "utf8");

test("GPU cards keep two columns until the large-screen breakpoint", () => {
  assert.match(css, /\.gpu-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(css, /@media \(min-width: 1900px\)\s*\{\s*\.gpu-grid\s*\{\s*grid-template-columns:\s*repeat\(4,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(css, /@media \(max-width: 1050px\)\s*\{\s*\.gpu-grid\s*\{\s*grid-template-columns:\s*1fr/);
  const denseStart = css.lastIndexOf(".gpu-section-head");
  const dense = denseStart < 0 ? "" : css.slice(denseStart).match(/\.gpu-grid\s*\{([\s\S]*?)\}/)?.[1] || "";
  assert.doesNotMatch(dense, /grid-template-columns/, "dense monitor block must not override the shared grid");
});

test("BenchLocal category cards collapse before the console sidebar eats their width", () => {
  assert.match(css, /@media\s*\(max-width:1200px\)\s*\{\s*\.bl-categories\s*\{\s*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
});

test("header summary yields width instead of pushing action buttons off-screen", () => {
  assert.match(css, /#appHeader #liveSummary\s*\{\s*flex:0 1 auto;\s*min-width:0;\s*overflow:hidden;\s*\}/);
  assert.match(css, /@media \(max-width:1100px\)\s*\{\s*#appHeader #workspaceToolbar\s*\{\s*min-width:0;\s*\}/);
  assert.match(css, /@media \(max-width:700px\)\s*\{\s*#appHeader #liveSummary\s*\{\s*display:none;\s*\}/);
  assert.match(summary, /@media \(max-width:1450px\)\s*\{\s*\.item\[data-metric=kv\],\.item\[data-metric=cacheHit\]\s*\{\s*display:none/);
  assert.match(summary, /@media \(max-width:1050px\)\s*\{\s*\.item\[data-metric=prefill\],\.item\[data-metric=decode\]\s*\{\s*display:none/);
});
