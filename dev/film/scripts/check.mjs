// Syntax-check the last inline script of film.html, or of the page given, with node.
//
//   node scripts/check.mjs [PAGE]
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const page = resolve(process.argv[2] || join(dirname(fileURLToPath(import.meta.url)), "..", "film.html"));
const scripts = [...readFileSync(page, "utf-8").matchAll(/<script>\r?\n([\s\S]*?)<\/script>/g)];
if (!scripts.length) { console.error(`no inline script in ${page}`); process.exit(1); }
const dir = mkdtempSync(join(tmpdir(), "bads-film-check-")), file = join(dir, "page.js");
writeFileSync(file, scripts[scripts.length - 1][1]);
const r = spawnSync(process.execPath, ["--check", file], { stdio: "inherit" });
rmSync(dir, { recursive: true, force: true });
if (r.status === 0) console.log("script ok");
process.exit(r.status ?? 1);
