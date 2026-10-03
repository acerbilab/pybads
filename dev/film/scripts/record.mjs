// Record film.html through its hooks: a clip of one shot as an MP4 (badsFilm.render(k, t)), stills of it at chosen
// times as PNGs, the whole film on its timeline (badsFilm.frame(T)), or the film's events (badsFilm.events()).
//
//   node scripts/record.mjs OUT.mp4 --shot K [--fps N] [--size WxH] [--from S] [--to S] [--crf N]   one shot, t from S to S
//   node scripts/record.mjs DIR --shot K --times T1,T2,...                                          stills at those times, DIR/sK_T.png
//   node scripts/record.mjs OUT.mp4 --film [--audio WAV] [--from S] [--to S] ...                    the whole film on its timeline
//   node scripts/record.mjs OUT.json --film                                                         the film's events, for the score
//
// K counts shots from 1, as film.html?shot=K does. The clip runs from --from (default 0) to --to (default the shot's
// dur, the draft length of its line) at --fps (default 25), and holds its last frame for --hold seconds (default 0.6).
// With --film the recording steps the film (film_timeline.js, written by voice.py) through badsFilm.frame(T) from --from
// (default 0) to --to (default its end), with no hold, and --audio muxes the narration in.
// The script serves the film's folder (the parent of scripts/) itself on a free port. Chrome is CHROME or its default
// install location on Windows; ffmpeg is FFMPEG, or ffmpeg on the PATH (`pip install --target DIR imageio-ffmpeg` puts
// one under DIR/imageio_ffmpeg/binaries/). Needs Node 22 or later (global WebSocket).
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pos = [], opt = {};
for (const argv = process.argv.slice(2); argv.length;) { const a = argv.shift(); if (a === "--film") opt.film = true; else if (a.startsWith("--")) opt[a.slice(2)] = argv.shift(); else pos.push(a); }
if (pos.length !== 1 || !(opt.shot || opt.film)) { console.error("usage: node scripts/record.mjs OUT.mp4|DIR|OUT.json (--shot K | --film) [--fps N] [--size WxH] [--from S] [--to S] [--times T1,T2] [--hold S] [--crf N] [--audio WAV]"); process.exit(2); }
if (!opt.times && !(opt.film && pos[0].endsWith(".json")) && !pos[0].endsWith(".mp4")) { console.error(`a recording is written as an MP4: ${pos[0]} does not end in .mp4`); process.exit(2); }
const out = resolve(pos[0]), K = Number(opt.shot), FPS = Number(opt.fps || 25), [W, H] = (opt.size || "1280x720").split("x").map(Number);
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const FFMPEG = process.env.FFMPEG || "ffmpeg";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8" };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  try { if (!path.startsWith(ROOT + sep) || !statSync(path).isFile()) throw new Error(); res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" }).end(readFileSync(path)); }
  catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/film.html?` + (opt.film ? "film=1" : `shot=${K}`);

const profile = mkdtempSync(join(tmpdir(), "bads-film-record-"));
const port = 9300 + Math.floor(Math.random() * 600);
const chrome = spawn(CHROME, ["--headless=new", "--enable-unsafe-swiftshader", "--hide-scrollbars", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
let target;
for (let k = 0; k < 100 && !target; k++) { try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch { await sleep(200); } }
if (!target) throw new Error(`Chrome (${CHROME}) did not open its DevTools port`);
const sock = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r, j) => { sock.addEventListener("open", r); sock.addEventListener("error", j); });
let nextId = 0; const pending = new Map();
sock.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; pending.set(id, r); sock.send(JSON.stringify({ id, method, params })); });
async function evaluate(expression) {
  const m = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (m.result?.exceptionDetails) throw new Error(`page: ${m.result.exceptionDetails.exception?.description || expression}`);
  return m.result?.result?.value;
}
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
let ready = false;
for (let k = 0; k < 600 && !ready; k++) { ready = await evaluate("document.title.startsWith('ready')").catch(() => false); if (!ready) await sleep(100); }
if (!ready) throw new Error(`${url} did not start: ${await evaluate("document.title").catch(() => "?")}`);
const dur = await evaluate(opt.film ? "badsFilm.seconds" : `badsFilm.shots[${K - 1}].dur`);
const draw = opt.film ? (t) => `badsFilm.frame(${t}), true` : (t) => `badsFilm.render(${K - 1}, ${t}), true`;
const frame = async (t) => { await evaluate(draw(t)); return Buffer.from((await send("Page.captureScreenshot", { format: "png" })).result.data, "base64"); };

const started = Date.now();
if (opt.film && out.endsWith(".json")) {
  writeFileSync(out, await evaluate("JSON.stringify(badsFilm.events())"));
  console.log(`wrote ${out}`);
} else if (opt.times) {
  mkdirSync(out, { recursive: true });
  for (const t of opt.times.split(",").map(Number)) writeFileSync(join(out, `s${String(K).padStart(2, "0")}_${t.toFixed(2)}.png`), await frame(t));
  console.log(`wrote ${opt.times.split(",").length} stills of shot ${K} to ${out}`);
} else {
  const T0 = Number(opt.from || 0), T1 = opt.to !== undefined ? Number(opt.to) : dur, hold = Number(opt.hold ?? (opt.film ? 0 : 0.6));
  const n = Math.max(1, Math.round((T1 - T0) * FPS) + 1), nHold = Math.round(hold * FPS);
  const audio = opt.audio ? ["-ss", String(T0), "-i", resolve(opt.audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"] : [];
  const ff = spawn(FFMPEG, ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-", ...audio,
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", String(opt.crf || 20), "-preset", "slow", "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((r, j) => { ff.on("close", (c) => (c ? j(new Error(`ffmpeg exited with ${c}`)) : r())); ff.on("error", j); });
  const write = (png) => new Promise((r) => (ff.stdin.write(png) ? r() : ff.stdin.once("drain", r)));
  let last;
  for (let i = 0; i < n; i++) {
    last = await frame(Math.min(T1, T0 + i / FPS)); await write(last);
    if (opt.film && i % 250 === 0) console.log(`frame ${i} of ${n}, T = ${(T0 + i / FPS).toFixed(1)} s (${((Date.now() - started) / 1000).toFixed(0)} s)`);
  }
  for (let i = 0; i < nHold; i++) await write(last);
  ff.stdin.end(); await done;
  console.log(`wrote ${out}: ${opt.film ? "the film" : "shot " + K}, t ${T0} to ${T1} s, ${n + nHold} frames at ${FPS} fps, in ${((Date.now() - started) / 1000).toFixed(0)} s`);
}
sock.close(); chrome.kill(); server.close();
await sleep(500);
try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold it on Windows */ }
