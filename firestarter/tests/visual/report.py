"""
Builds ``test_artifacts/report.html`` (and ``contact_sheet.png``) from a test run:

* the status of every run_tests.sh stage (JUnit XML written by pytest),
* every playback script with its gameplay assertions, sounds heard and frame timings,
* every screenshot with its image metrics (visual_metrics.json).

    python -m tests.visual.report [artifacts_dir]
"""

import glob
import html
import json
import os
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _font(size):
    for p in (os.path.join(ROOT, "assets", "fonts", "Poppins-Medium.ttf"), os.path.join(ROOT, "assets", "fonts", "DejaVuSansMono.ttf")):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def contact_sheet(shots, out_path, cols=4, tile=(480, 270)):
    if not shots:
        return None
    rows = (len(shots) + cols - 1) // cols
    W, H = tile
    label_h = 24
    sheet = Image.new("RGB", (cols * W, rows * (H + label_h)), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f = _font(14)
    for i, p in enumerate(shots):
        im = Image.open(p).convert("RGB").resize(tile, Image.LANCZOS)
        x, y = (i % cols) * W, (i // cols) * (H + label_h)
        sheet.paste(im, (x, y))
        d.text((x + 6, y + H + 4), os.path.basename(p)[:-4].replace("__", " / "), font=f, fill=(225, 215, 200))
    sheet.save(out_path, optimize=True)
    return out_path


def junit_summary(path):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tot = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    failed = []
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            tot[k] += int(s.get(k, 0))
        tot["time"] += float(s.get("time", 0.0))
        for case in s.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed.append(f"{case.get('classname')}::{case.get('name')}")
    tot["failed_cases"] = failed
    return tot


def build(art):
    art = os.path.abspath(art)
    shots = sorted(glob.glob(os.path.join(art, "playback", "*.png")))
    sheet = contact_sheet(shots, os.path.join(art, "contact_sheet.png"))
    metrics = {}
    mp = os.path.join(art, "visual_metrics.json")
    if os.path.exists(mp):
        metrics = json.load(open(mp))
    reports = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(art, "playback", "*__report.json")))]
    stages = []
    for p in sorted(glob.glob(os.path.join(art, "junit-*.xml"))):
        s = junit_summary(p)
        if s:
            stages.append((os.path.basename(p)[6:-4], s))
    esc = html.escape
    out = ["<!doctype html><html><head><meta charset='utf-8'><title>Firestarter test report</title><style>",
           "body{font-family:system-ui,sans-serif;background:#141416;color:#e8e2d6;margin:24px}h1,h2{color:#ffb060}",
           "table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #333;padding:4px 10px;font-size:13px}",
           ".ok{color:#8fd18f}.bad{color:#ff7a6b}.grid{display:flex;flex-wrap:wrap;gap:14px}",
           ".card{background:#1e1e22;border-radius:8px;padding:8px;width:420px}.card img{width:420px;border-radius:4px}",
           "code{color:#c9b98f}small{color:#9a9488}</style></head><body>",
           "<h1>Firestarter - test report</h1>"]
    if stages:
        out.append("<h2>Test stages</h2><table><tr><th>stage</th><th>tests</th><th>failed</th><th>skipped</th><th>time (s)</th></tr>")
        for name, s in stages:
            bad = s["failures"] + s["errors"]
            out.append(f"<tr><td>{esc(name)}</td><td>{s['tests']}</td><td class='{'bad' if bad else 'ok'}'>{bad}</td>"
                       f"<td>{s['skipped']}</td><td>{s['time']:.1f}</td></tr>")
            for c in s["failed_cases"]:
                out.append(f"<tr><td colspan=5 class='bad'><code>{esc(c)}</code></td></tr>")
        out.append("</table>")
    if reports:
        out.append("<h2>Playback scripts</h2><table><tr><th>script</th><th>result</th><th>checks</th><th>shots</th>"
                   "<th>sim time (s)</th><th>render ms (mean / p95)</th><th>sounds heard</th></tr>")
        for r in reports:
            n_ok = sum(1 for x in r["results"] if x.get("passed"))
            out.append(f"<tr><td>{esc(r['name'])}</td><td class='{'ok' if r['passed'] else 'bad'}'>{'PASS' if r['passed'] else 'FAIL'}</td>"
                       f"<td>{n_ok}/{len(r['results'])}</td><td>{len(r['screenshots'])}</td><td>{r['sim_time_s']}</td>"
                       f"<td>{r['render_ms_mean']} / {r['render_ms_p95']}</td><td><small>{esc(', '.join(r.get('sounds_played', [])[:18]))}</small></td></tr>")
        out.append("</table>")
        for r in reports:
            out.append(f"<h3>{esc(r['name'])}</h3><table><tr><th>step</th><th>check</th><th>result</th><th>value</th></tr>")
            for x in r["results"]:
                label = x.get("message") or x.get("expr", "")
                out.append(f"<tr><td>{x.get('step', '')}</td><td>{esc(str(label))}<br><small><code>{esc(str(x.get('expr', '')))}</code></small></td>"
                           f"<td class='{'ok' if x.get('passed') else 'bad'}'>{'pass' if x.get('passed') else 'FAIL'}</td>"
                           f"<td><small>{esc(str(x.get('value', x.get('elapsed', ''))))}</small></td></tr>")
            out.append("</table>")
    if sheet:
        out.append("<h2>Contact sheet</h2><img src='contact_sheet.png' style='max-width:100%'>")
    if shots:
        out.append("<h2>Screenshots</h2><div class='grid'>")
        for p in shots:
            name = os.path.basename(p)
            m = metrics.get(name, {})
            ms = " &middot; ".join(f"{k} {v}" for k, v in m.items())
            out.append(f"<div class='card'><a href='playback/{esc(name)}'><img src='playback/{esc(name)}'></a>"
                       f"<div>{esc(name[:-4])}</div><small>{ms}</small></div>")
        out.append("</div>")
    out.append("</body></html>")
    path = os.path.join(art, "report.html")
    with open(path, "w") as f:
        f.write("\n".join(out))
    return path


if __name__ == "__main__":
    art = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIRESTARTER_ARTIFACTS", os.path.join(ROOT, "test_artifacts"))
    print(build(art))
