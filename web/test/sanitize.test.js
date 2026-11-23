import assert from "node:assert/strict";
import { test } from "node:test";
import createDOMPurify from "dompurify";
import { JSDOM } from "jsdom";
import { reportHtml } from "../src/sanitize.js";

const purifier = createDOMPurify(new JSDOM("").window);
const render = (md) => reportHtml(md, purifier);

test("a javascript: link is not rendered as a link", () => {
  const html = render("[open](javascript:alert(1))");
  assert.ok(!/javascript:/i.test(html), html);
});

test("an unclosed tag cannot carry an event handler", () => {
  const html = render('text <img src=x onerror=alert(1) and more');
  assert.ok(!/<img[^>]*onerror/i.test(html), html);
});

test("raw script and iframe tags are gone", () => {
  const html = render("hi <script>alert(1)</script> <iframe src=//evil></iframe>");
  assert.ok(!/<script|<iframe/i.test(html), html);
});

test("an image with a javascript source is neutralised", () => {
  const html = render("![x](javascript:alert(1))");
  assert.ok(!/javascript:/i.test(html), html);
});

test("normal Markdown keeps its structure and https links", () => {
  const html = render("## Title\n\n- **one**\n\n[site](https://example.com)");
  assert.match(html, /<h2[^>]*>Title<\/h2>/);
  assert.match(html, /<strong>one<\/strong>/);
  assert.match(html, /href="https:\/\/example\.com"/);
});
