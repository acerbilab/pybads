// Check the film's line changes: a label or an overlay on screen when a line starts must stay, or fade out over the
// crossfade, not vanish in one frame or blink off and back.
//
//   node scripts/transitions.mjs
//
// Steps film.html's film (film_timeline.js) through badsFilm.frame(T) from 0.04 s before each line starts to 0.76 s
// after, and reads what each overlay shows: the labels (by class and text), the HUD, each SVG over the frame (scene 4's
// diagram, scene 7's tiles and charts, by its first text) and the footnote (by its text). It lists each that vanishes
// in one frame ("cut") or drops and comes back ("blink"), and each label that two consecutive shots share but place
// apart, and exits with status 1 when it lists any. The script serves the film's folder (the parent of scripts/)
// itself on a free port; Chrome is CHROME or its default install location on Windows. Needs Node 22 or later.
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DT = [-0.04, 0.001, 0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76];

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8" };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  try { if (!path.startsWith(ROOT + sep) || !statSync(path).isFile()) throw new Error(); res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" }).end(readFileSync(path)); }
  catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));

const profile = mkdtempSync(join(tmpdir(), "bads-film-transitions-"));
const port = 9300 + Math.floor(Math.random() * 600);
const chrome = spawn(CHROME, ["--headless=new", "--enable-unsafe-swiftshader", "--hide-scrollbars", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
let sock, chromeError, problems = 0;
chrome.on("error", (e) => { chromeError = e; });
try {
  let target;
  for (let k = 0; k < 100 && !target; k++) {
    if (chromeError) throw new Error(`Chrome (${CHROME}) did not start: ${chromeError.message}`);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch { /* not listening yet */ }
    if (!target) await sleep(200);
  }
  if (!target) throw new Error(`Chrome (${CHROME}) did not open its DevTools port`);
  sock = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { sock.addEventListener("open", r); sock.addEventListener("error", j); });
  let nextId = 0; const pending = new Map();
  sock.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; pending.set(id, r); sock.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const m = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (m.result?.exceptionDetails) throw new Error(`page: ${m.result.exceptionDetails.exception?.description || expression}`);
    return m.result?.result?.value;
  };
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  const url = `http://127.0.0.1:${server.address().port}/film.html?film=1`;
  await send("Page.navigate", { url });
  let ready = false;
  for (let k = 0; k < 600 && !ready; k++) { ready = await evaluate("document.title.startsWith('ready') || document.title.startsWith('ERROR')").catch(() => false); if (!ready) await sleep(100); }
  const title = await evaluate("document.title").catch(() => "?");
  if (!ready || title.startsWith("ERROR")) throw new Error(`${url} did not start: ${title}`);

  // What is on screen at T: the opacity of each overlay, keyed by what it shows.
  const state = (T) => evaluate(`(() => { badsFilm.frame(${T});
    const op = (e) => { const s = getComputedStyle(e); return s.display === "none" ? 0 : +s.opacity; }, out = {}, dia = document.getElementById("diagram");
    for (const svg of dia.querySelectorAll(":scope > svg")) {
      if (!svg.querySelector("rect, path, text, circle, line")) continue;
      const g = svg.querySelector(":scope > g[opacity]"), t = svg.querySelector("text");
      out["svg " + (t ? t.textContent.slice(0, 16) : "")] = op(dia) * (g ? +g.getAttribute("opacity") : 1);
    }
    out["HUD"] = op(document.getElementById("mode"));
    const f = document.getElementById("foot"); if (f.textContent) out["footnote " + f.textContent.slice(0, 16)] = op(f);
    for (const e of document.querySelectorAll("#labels .lab")) { const k = "label " + (e.textContent.slice(0, 20) || "[bracket]"); out[k] = Math.max(out[k] || 0, e.style.opacity === "" ? 1 : +e.style.opacity); }
    return out; })()`);
  const starts = await evaluate("window.BADS_FILM.scenes.flatMap((sc) => sc.lines.map((ln) => [ln.id, sc.start + ln.start]))");
  for (const [id, T] of starts.slice(1)) {
    const s = []; for (const d of DT) s.push(await state(T + d));
    for (const k of new Set(s.flatMap((x) => Object.keys(x)))) {
      const v = s.map((x) => x[k] || 0), before = v[0], first = v[1], low = Math.min(...v.slice(1)), after = v[v.length - 1];
      const cut = before > 0.3 && first < 0.05 && after < 0.05, blink = before > 0.3 && low < before - 0.3 && after > low + 0.2;
      if (cut || blink) { problems++; console.log(`${id} at ${T.toFixed(3)} s: ${cut ? "cut  " : "blink"} ${k.padEnd(30)} ${v.map((x) => x.toFixed(2)).join(" ")}`); }
    }
  }
  // A label that both shots show stays as the new shot draws it, so it must be where the shot before left it.
  const moved = await evaluate(`(() => { const out = [], at = () => [...document.querySelectorAll("#labels .lab")].map((e) => { const r = e.getBoundingClientRect(); return [e.className + "|" + e.innerHTML, r.left, r.top]; });
    for (let k = 1; k < badsFilm.shots.length; k++) {
      badsFilm.render(k - 1, badsFilm.shots[k - 1].dur || undefined); const a = at();
      badsFilm.render(k, 0);
      for (const [key, x, y] of at()) { const q = a.find((p) => p[0] === key); if (q && Math.hypot(x - q[1], y - q[2]) > 2) out.push([badsFilm.shots[k].id, key, Math.hypot(x - q[1], y - q[2])]); }
    } return out; })()`);
  for (const [id, key, d] of moved) { problems++; console.log(`${id}: the label ${key.split("|").pop().replace(/<[^>]+>/g, "") || "[bracket]"} moves ${d.toFixed(1)} px from the shot before`); }
  console.log(`${problems ? problems + " problems" : "no problem"} at ${starts.length - 1} line changes`);
} finally {
  try { sock?.close(); } catch { /* already closed */ }
  server.close();
  if (!chromeError && chrome.exitCode === null) {                    // Chrome holds its profile until it has exited
    const exited = new Promise((r) => chrome.once("exit", r));
    chrome.kill();
    await Promise.race([exited, sleep(5000)]);
  }
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); } catch { /* still held: a temp folder */ }
}
process.exitCode = problems ? 1 : 0;
