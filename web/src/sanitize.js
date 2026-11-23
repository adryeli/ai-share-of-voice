import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({ gfm: true, breaks: false });

// The report text can quote model answers and web pages, so it is untrusted. Raw HTML is dropped
// before Markdown is parsed, and the result goes through DOMPurify, which also blocks
// javascript: links and event handlers that a tag-stripping regex cannot see.
export function reportHtml(markdown, purifier = DOMPurify) {
  const html = marked.parse(String(markdown).replace(/<[^>]*>/g, ""));
  return purifier.sanitize(html, { USE_PROFILES: { html: true }, FORBID_ATTR: ["style"] });
}
