#!/usr/bin/env node

// Minimal PDF text extractor for unencrypted, non-object-stream PDFs.
// It decodes Flate streams and the embedded ToUnicode CMaps used by the
// Google Docs-rendered System Cards in this research set.

const fs = require("node:fs");
const zlib = require("node:zlib");

const input = process.argv[2];
const output = process.argv[3];
if (!input) {
  console.error("usage: pdf_text_extract.js input.pdf [output.txt]");
  process.exit(2);
}

const pdf = fs.readFileSync(input);
const objects = new Map();
const objectRe = /(?:^|[\r\n])(\d+)\s+(\d+)\s+obj\b/g;
let match;
while ((match = objectRe.exec(pdf.toString("latin1")))) {
  const start = match.index;
  const bodyStart = objectRe.lastIndex;
  const end = pdf.indexOf(Buffer.from("endobj"), bodyStart);
  if (end < 0) break;
  objects.set(Number(match[1]), { start, bodyStart, end, body: pdf.subarray(bodyStart, end) });
  objectRe.lastIndex = end + 6;
}

function dictText(body) {
  const text = body.toString("latin1");
  const stream = text.indexOf("stream");
  return stream < 0 ? text : text.slice(0, stream);
}

function refIn(text, key) {
  const m = new RegExp(`/${key}\\s+(\\d+)\\s+0\\s+R`).exec(text);
  return m ? Number(m[1]) : null;
}

function refsIn(text, key) {
  const m = new RegExp(`/${key}\\s+(\\[[^\\]]+\\]|\\d+\\s+0\\s+R)`).exec(text);
  if (!m) return [];
  return [...m[1].matchAll(/(\d+)\s+0\s+R/g)].map((x) => Number(x[1]));
}

function streamBytes(id) {
  const obj = objects.get(id);
  if (!obj) return null;
  const header = dictText(obj.body);
  const marker = Buffer.from("stream");
  const markerAt = obj.body.indexOf(marker);
  if (markerAt < 0) return null;
  let start = markerAt + marker.length;
  if (obj.body[start] === 13 && obj.body[start + 1] === 10) start += 2;
  else if (obj.body[start] === 10 || obj.body[start] === 13) start += 1;
  const lengthMatch = /\/Length\s+(\d+)/.exec(header);
  const length = lengthMatch ? Number(lengthMatch[1]) : (() => {
    const end = obj.body.indexOf(Buffer.from("endstream"), start);
    return end < 0 ? 0 : end - start;
  })();
  let bytes = obj.body.subarray(start, start + length);
  if (/\/FlateDecode/.test(header)) bytes = zlib.inflateSync(bytes);
  return bytes;
}

function hexToBytes(hex) {
  const clean = hex.replace(/\s+/g, "");
  const even = clean.length % 2 ? `0${clean}` : clean;
  return Buffer.from(even, "hex");
}

function hexCode(hex) {
  return Number.parseInt(hex.replace(/\s+/g, ""), 16);
}

function utf16be(bytes) {
  let result = "";
  for (let i = 0; i + 1 < bytes.length; i += 2) result += String.fromCharCode(bytes.readUInt16BE(i));
  return result;
}

function cmapFor(id) {
  const bytes = streamBytes(id);
  if (!bytes) return null;
  const text = bytes.toString("latin1");
  const map = new Map();
  const widths = new Set();

  const charBlock = /beginbfchar([\s\S]*?)endbfchar/g;
  let block;
  while ((block = charBlock.exec(text))) {
    for (const line of block[1].split(/\r?\n/)) {
      const m = /<([0-9A-Fa-f\s]+)>\s*<([0-9A-Fa-f\s]+)>/.exec(line);
      if (!m) continue;
      const source = m[1].replace(/\s+/g, "").toUpperCase();
      map.set(source, utf16be(hexToBytes(m[2])));
      widths.add(source.length / 2);
    }
  }

  const rangeBlock = /beginbfrange([\s\S]*?)endbfrange/g;
  while ((block = rangeBlock.exec(text))) {
    for (const line of block[1].split(/\r?\n/)) {
      const range = /<([0-9A-Fa-f\s]+)>\s+<([0-9A-Fa-f\s]+)>\s+(.+)/.exec(line);
      if (!range) continue;
      const firstHex = range[1].replace(/\s+/g, "").toUpperCase();
      const lastHex = range[2].replace(/\s+/g, "").toUpperCase();
      const first = hexCode(firstHex);
      const last = hexCode(lastHex);
      const tail = range[3].trim();
      const array = tail.match(/^\[(.*)\]$/);
      widths.add(firstHex.length / 2);
      for (let n = first; n <= last; n += 1) {
        const source = n.toString(16).padStart(firstHex.length, "0").toUpperCase();
        let value;
        if (array) {
          const entries = [...array[1].matchAll(/<([0-9A-Fa-f\s]+)>/g)].map((x) => x[1]);
          const entry = entries[n - first];
          if (!entry) break;
          value = utf16be(hexToBytes(entry));
        } else {
          const destination = /<([0-9A-Fa-f\s]+)>/.exec(tail);
          if (!destination) break;
          const raw = hexToBytes(destination[1]);
          if (raw.length < 2) value = String.fromCodePoint(hexCode(destination[1]));
          else {
            const code = raw.readUInt16BE(0) + (n - first);
            value = String.fromCharCode(code);
          }
        }
        map.set(source, value);
      }
    }
  }
  return { map, widths: [...widths].sort((a, b) => b - a) };
}

const cmapCache = new Map();
function fontCmap(fontId) {
  if (cmapCache.has(fontId)) return cmapCache.get(fontId);
  const id = refIn(dictText(objects.get(fontId)?.body || Buffer.alloc(0)), "ToUnicode");
  const cmap = id ? cmapFor(id) : null;
  cmapCache.set(fontId, cmap);
  return cmap;
}

function decodeString(raw, cmap) {
  if (!cmap) return raw.toString("latin1");
  let result = "";
  for (let i = 0; i < raw.length;) {
    let found = false;
    for (const width of cmap.widths) {
      const slice = raw.subarray(i, i + width).toString("hex").toUpperCase();
      if (slice.length === width * 2 && cmap.map.has(slice)) {
        result += cmap.map.get(slice);
        i += width;
        found = true;
        break;
      }
    }
    if (!found) {
      result += raw[i] >= 32 && raw[i] < 127 ? String.fromCharCode(raw[i]) : "�";
      i += 1;
    }
  }
  return result;
}

function parseLiteral(bytes, position) {
  let i = position + 1;
  let depth = 1;
  const out = [];
  while (i < bytes.length && depth) {
    const ch = bytes[i++];
    if (ch === 92) {
      if (i >= bytes.length) break;
      const escaped = bytes[i++];
      const simple = { 110: 10, 114: 13, 116: 9, 98: 8, 102: 12 };
      if (simple[escaped] !== undefined) out.push(simple[escaped]);
      else if (escaped === 10) {}
      else if (escaped === 13) { if (bytes[i] === 10) i += 1; }
      else if (escaped >= 48 && escaped <= 55) {
        let octal = String.fromCharCode(escaped);
        while (octal.length < 3 && bytes[i] >= 48 && bytes[i] <= 55) octal += String.fromCharCode(bytes[i++]);
        out.push(Number.parseInt(octal, 8));
      } else out.push(escaped);
    } else if (ch === 40) { depth += 1; out.push(ch); }
    else if (ch === 41) { depth -= 1; if (depth) out.push(ch); }
    else out.push(ch);
  }
  return { value: Buffer.from(out), end: i };
}

function tokens(bytes) {
  const result = [];
  let i = 0;
  while (i < bytes.length) {
    const ch = bytes[i];
    if (ch <= 32) { i += 1; continue; }
    if (ch === 37) { while (i < bytes.length && bytes[i] !== 10 && bytes[i] !== 13) i += 1; continue; }
    if (ch === 40) { const value = parseLiteral(bytes, i); result.push({ type: "string", value: value.value }); i = value.end; continue; }
    if (ch === 60 && bytes[i + 1] === 60) { result.push({ type: "word", value: "<<" }); i += 2; continue; }
    if (ch === 62 && bytes[i + 1] === 62) { result.push({ type: "word", value: ">>" }); i += 2; continue; }
    if (ch === 60) {
      const end = bytes.indexOf(62, i + 1);
      if (end < 0) break;
      result.push({ type: "string", value: hexToBytes(bytes.subarray(i + 1, end).toString("latin1")) });
      i = end + 1;
      continue;
    }
    if (ch === 91 || ch === 93) { result.push({ type: "bracket", value: String.fromCharCode(ch) }); i += 1; continue; }
    let end = i + 1;
    while (end < bytes.length && bytes[end] > 32 && ![40, 41, 60, 62, 91, 93, 37].includes(bytes[end])) end += 1;
    result.push({ type: "word", value: bytes.subarray(i, end).toString("latin1") });
    i = end;
  }
  return result;
}

function textForContent(contentId, fontMap) {
  const bytes = streamBytes(contentId);
  if (!bytes) return "";
  const ts = tokens(bytes);
  let currentFont = null;
  let currentY = 0;
  let lastTextY = null;
  const out = [];
  const emit = (value) => {
    if (!value) return;
    if (lastTextY !== null && Math.abs(currentY - lastTextY) > 0.5) out.push("\n");
    out.push(value);
    lastTextY = currentY;
  };
  for (let i = 0; i < ts.length; i += 1) {
    const token = ts[i];
    if (token.type === "word" && token.value === "Tm" && i >= 6) {
      const y = Number(ts[i - 1].value);
      if (Number.isFinite(y)) currentY = y;
    }
    if (token.type === "word" && token.value === "Td" && i >= 2) {
      const y = Number(ts[i - 1].value);
      if (Number.isFinite(y)) currentY += y;
    }
    if (token.type === "word" && token.value === "Tf" && i >= 2) {
      const name = ts[i - 2];
      if (name?.type === "word") currentFont = fontMap.get(name.value) || null;
    }
    if (token.type === "word" && ["Tj", "'", '"'].includes(token.value) && i >= 1) {
      const value = ts[i - 1];
      if (value?.type === "string") emit(decodeString(value.value, fontCmap(currentFont)));
    }
    if (token.type === "word" && token.value === "TJ" && i >= 1) {
      const start = (() => { let j = i - 1; while (j >= 0 && ts[j].value !== "[") j -= 1; return j; })();
      if (start >= 0) {
        for (let j = start + 1; j < i; j += 1) if (ts[j].type === "string") emit(decodeString(ts[j].value, fontCmap(currentFont)));
      }
    }
  }
  return `${out.join("")}\n`;
}

function pageText(pageId) {
  const page = dictText(objects.get(pageId)?.body || Buffer.alloc(0));
  const resourcesId = refIn(page, "Resources");
  const resources = dictText(objects.get(resourcesId)?.body || Buffer.alloc(0));
  const resourceText = resourcesId ? resources : page;
  const fontId = refIn(resourceText, "Font");
  const directFont = /\/Font\s*<<(.*?)>>/s.exec(resourceText)?.[1] || "";
  const fontDict = fontId ? dictText(objects.get(fontId)?.body || Buffer.alloc(0)) : directFont;
  const fonts = new Map();
  for (const m of fontDict.matchAll(/\/(F\w+)\s+(\d+)\s+0\s+R/g)) fonts.set(`/${m[1]}`, Number(m[2]));
  const contentIds = refsIn(page, "Contents");
  return contentIds.map((id) => textForContent(id, fonts)).join("");
}

const pages = [];
for (const [id, obj] of objects) if (/\/Type\s*\/Page\b/.test(dictText(obj.body))) pages.push(id);
const chunks = pages.map((id, index) => `\n\n===== PAGE ${index + 1} (object ${id}) =====\n${pageText(id)}`);
const text = chunks.join("").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n");
if (output) fs.writeFileSync(output, text);
else process.stdout.write(text);
console.error(`pages=${pages.length} bytes=${Buffer.byteLength(text)}`);
