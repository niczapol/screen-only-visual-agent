from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any

import cv2

from scripts.build_terrain_coverage_route import colorize_terrain
from scripts.build_topographic_route import zone_from_manifest
from vision_bot.coords import coord_to_xy
from vision_bot.terrain_routing import (
    TerrainCostConfig,
    build_terrain_cost_map,
    build_terrain_raster,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_ROUTE = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v13_rail.json"
NEXT_ROUTE = ROOT / "data/routes/generated/tanaris_terrain_coverage_cycle_v13_rail.json"
MANIFEST = ROOT / "data/extracted_client_data/tanaris_corrected/manifest.json"
ADT_DIR = ROOT / "data/extracted_client_data/tanaris_corrected/world/maps/Kalimdor"
RUNS = (
    (
        ROOT / "data/live_v0821_run1_20260813_031055",
        "v0.8.21 RUN1 · mining calibration",
        "var(--viz-series-3)",
    ),
    (
        ROOT / "data/live_v0821_run2b_20260813_034238",
        "v0.8.21 RUN2 · southern V13 rail",
        "var(--viz-series-5)",
    ),
    (
        ROOT / "data/live_v0821_run3b_20260813_042536",
        "v0.8.21 RUN3 · final validation",
        "var(--viz-series-1)",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Tanaris route/live telemetry visualization fragment."
    )
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def point(coord: int) -> dict[str, float]:
    x, y = coord_to_xy(int(coord))
    return {"x": round(float(x), 4), "y": round(float(y), 4)}


def compact_track(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    last: tuple[float, float] | None = None
    for row in rows:
        coord = row.get("coord")
        if coord is None:
            continue
        current = coord_to_xy(int(coord))
        if last is not None and math.hypot(current[0] - last[0], current[1] - last[1]) < 0.025:
            continue
        result.append(
            {
                "x": round(float(current[0]), 4),
                "y": round(float(current[1]), 4),
                "i": int(row.get("index", 0)),
            }
        )
        last = current
    return result


def track_length(track: list[dict[str, float | int]]) -> float:
    return sum(
        math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
        for a, b in zip(track, track[1:])
    )


def run_events(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    by_index = {int(row.get("index", -1)): row for row in rows}
    events: list[dict[str, Any]] = []
    combat_active = False
    tooltip_active = False
    stuck_active = False
    skip_active = False
    for row in rows:
        coord = row.get("coord")
        if coord is None:
            continue
        x, y = coord_to_xy(int(coord))
        index = int(row.get("index", 0))
        combat = bool((row.get("combat") or {}).get("active"))
        if combat and not combat_active:
            events.append({"x": x, "y": y, "i": index, "kind": "combat", "label": "Combat"})
        combat_active = combat

        tooltip = row.get("minimap_ore_tooltip") or {}
        tooltip_now = bool(tooltip)
        if tooltip_now and not tooltip_active:
            ore_type = tooltip.get("ore_type") or (row.get("mining") or {}).get("confirmed_ore_type")
            events.append(
                {
                    "x": x,
                    "y": y,
                    "i": index,
                    "kind": "tooltip",
                    "label": f"Tooltip: {ore_type or 'ore'}",
                }
            )
        tooltip_active = tooltip_now

        action = str(row.get("action") or "")
        stuck = action.startswith(("recover_", "local_avoid_"))
        if stuck and not stuck_active:
            events.append(
                {"x": x, "y": y, "i": index, "kind": "stuck", "label": action}
            )
        stuck_active = stuck

        skipped = action == "target_blocked_cycle_next"
        if skipped and not skip_active:
            events.append(
                {
                    "x": x,
                    "y": y,
                    "i": index,
                    "kind": "skip",
                    "label": "route window skipped while stationary",
                }
            )
        skip_active = skipped

    for outcome in summary.get("mining_outcomes", []):
        index = int(outcome["index"])
        row = by_index.get(index)
        if row is None or row.get("coord") is None:
            continue
        x, y = coord_to_xy(int(row["coord"]))
        events.append(
            {
                "x": x,
                "y": y,
                "i": index,
                "kind": "mine",
                "label": str(outcome.get("reason") or "mining outcome"),
            }
        )
    if not summary.get("mining_outcomes"):
        for row in rows:
            action = str(row.get("action") or "")
            if not action.startswith("mining_failed_resume_route:"):
                continue
            coord = row.get("coord")
            if coord is None:
                continue
            x, y = coord_to_xy(int(coord))
            events.append(
                {
                    "x": x,
                    "y": y,
                    "i": int(row.get("index", 0)),
                    "kind": "mine",
                    "label": action.split(":", 2)[1],
                }
            )
    return events


def topography_data_url() -> tuple[str, dict[str, float]]:
    manifest = load_json(MANIFEST)
    zone = zone_from_manifest(manifest)
    raster = build_terrain_raster(sorted(ADT_DIR.glob("*.adt")))
    costs = build_terrain_cost_map(
        raster,
        TerrainCostConfig(soft_slope_degrees=15.0, hard_slope_degrees=35.0),
    )
    image = colorize_terrain(raster, costs)
    ok, encoded = cv2.imencode(
        ".webp",
        image,
        [cv2.IMWRITE_WEBP_QUALITY, 72],
    )
    if not ok:
        raise OSError("Unable to encode Tanaris topography")
    row0, col0 = raster.ui_to_grid(0.0, 0.0, zone)
    row100, col100 = raster.ui_to_grid(100.0, 100.0, zone)
    height, width = image.shape[:2]
    bounds = {
        "x": round((0.0 - col0) / (col100 - col0) * 100.0, 6),
        "y": round((0.0 - row0) / (row100 - row0) * 100.0, 6),
        "width": round(width / (col100 - col0) * 100.0, 6),
        "height": round(height / (row100 - row0) * 100.0, 6),
    }
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/webp;base64,{payload}", bounds


def build_payload() -> tuple[dict[str, Any], str, dict[str, float]]:
    full = load_json(FULL_ROUTE)
    next_route = load_json(NEXT_ROUTE)
    all_nodes = [
        {
            "x": round(float(node["x"]), 4),
            "y": round(float(node["y"]), 4),
            "ore": str(node.get("ore_type") or "Ore"),
            "status": str(node.get("coverage_status") or "unknown"),
            "index": int(node.get("index", -1)),
        }
        for node in full["route_nodes"]
    ]

    runs: list[dict[str, Any]] = []
    for path, label, color in RUNS:
        rows = load_jsonl(path / "metadata.jsonl")
        summary_path = path / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        if not summary:
            actions = [str(row.get("action") or "") for row in rows]
            summary = {
                "steps": len(rows),
                "completed_targets": max(
                    (int(row.get("completed_targets", 0)) for row in rows),
                    default=0,
                ),
                "stuck_events": sum(
                    action.startswith(("recover_", "local_avoid_"))
                    and (index == 0 or actions[index - 1] != action)
                    for index, action in enumerate(actions)
                ),
                "mining_attempts": max(
                    (int(row.get("mining_outcomes", 0)) for row in rows),
                    default=0,
                ),
                "mining_successes": 0,
                "mining_outcomes": [],
            }
        track = compact_track(rows)
        runs.append(
            {
                "id": path.name,
                "label": label,
                "color": color,
                "track": track,
                "events": run_events(rows, summary),
                "stats": {
                    "steps": int(summary.get("steps", 0)),
                    "completed": int(summary.get("completed_targets", 0)),
                    "stuck": int(summary.get("stuck_events", 0)),
                    "attempts": int(summary.get("mining_attempts", 0)),
                    "successes": int(summary.get("mining_successes", 0)),
                    "length": round(track_length(track), 2),
                },
            }
        )

    background, bounds = topography_data_url()
    return (
        {
            "nodes": all_nodes,
            "fullRoute": [
                {"x": round(float(item["x"]), 4), "y": round(float(item["y"]), 4)}
                for item in full["route_loop"]
            ],
            "nextRoute": [
                {"x": round(float(item["x"]), 4), "y": round(float(item["y"]), 4)}
                for item in next_route["route_loop"]
            ],
            "runs": runs,
            "summary": {
                "allNodes": len(all_nodes),
                "runtimeNodes": len(next_route["route_nodes"]),
                "fullWaypoints": len(full["route_loop"]),
                "nextWaypoints": len(next_route["route_loop"]),
            },
        },
        background,
        bounds,
    )


FRAGMENT = r'''
<style>
  .tz-wrap{color:var(--foreground);display:flex;flex-direction:column;gap:12px;min-height:720px;min-width:0;width:100%}
  .tz-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}
  .tz-title{font-weight:500;margin:0;white-space:normal;overflow-wrap:anywhere}.tz-sub{color:var(--muted-foreground);margin-top:4px;white-space:normal;overflow-wrap:anywhere}
  .tz-wrap .viz-controls{display:flex;flex-wrap:wrap;min-width:0;max-width:100%}.tz-wrap .form-check{min-width:0;white-space:normal}
  .tz-stats{display:flex;gap:12px;flex-wrap:wrap}.tz-stat{min-width:92px}.tz-stat b{display:block;font-weight:500}.tz-stat span{color:var(--muted-foreground)}
  .tz-main{display:grid;grid-template-columns:minmax(0,1fr) 240px;gap:16px;flex:1;min-height:580px}
  .tz-map-card{position:relative;min-height:580px;overflow:hidden}.tz-map{width:100%;height:100%;min-height:580px;display:block;touch-action:none;cursor:grab}.tz-map.dragging{cursor:grabbing}
  .tz-grid{stroke:var(--border);stroke-width:.08;vector-effect:non-scaling-stroke}.tz-axis{fill:var(--muted-foreground);font-size:1.35px}
  .tz-side{display:flex;flex-direction:column;gap:14px}.tz-side h3{font-weight:500;margin:0 0 6px}.tz-legend{display:grid;gap:7px}.tz-leg{display:flex;align-items:center;gap:8px}.tz-swatch{width:18px;height:4px}.tz-dot{width:9px;height:9px;border-radius:50%}
  .tz-run{padding:7px 0;border-bottom:1px solid var(--border)}.tz-run b{display:block;font-weight:500}.tz-note{color:var(--muted-foreground)}
  .tz-tip{position:absolute;pointer-events:none;display:none;z-index:4;background:var(--popover);color:var(--popover-foreground);border:1px solid var(--border);padding:7px 9px;max-width:230px}
  .tz-route{fill:none;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}.tz-node{vector-effect:non-scaling-stroke}.tz-event{vector-effect:non-scaling-stroke;cursor:pointer}
  @media(max-width:850px){.tz-main{grid-template-columns:1fr}.tz-side{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}.tz-map-card{min-height:0;height:auto;aspect-ratio:1}.tz-map{min-height:0;height:100%}}
  @media(max-width:520px){.tz-side{grid-template-columns:1fr}}
</style>
<div class="tz-wrap" id="tanaris-v13-live-map">
  <div class="tz-head">
    <div><h2 class="tz-title">Tanaris · маршрут V13 и фактические треки</h2><div class="tz-sub">v0.8.18–v0.8.20 · добыча, Dunemaul и западное препятствие · координаты UI 0–100</div></div>
    <div class="tz-stats" id="tzStats"></div>
  </div>
  <div class="viz-controls">
    <label class="form-check" for="tzNodes"><input class="form-check-input" id="tzNodes" type="checkbox" data-layer="allNodes" checked><span class="form-check-label">Рудные точки</span></label>
    <label class="form-check" for="tzPlan"><input class="form-check-input" id="tzPlan" type="checkbox" data-layer="fullRoute" checked><span class="form-check-label">План V13</span></label>
    <label class="form-check" for="tzRun0"><input class="form-check-input" id="tzRun0" type="checkbox" data-layer="run0" checked><span class="form-check-label">Добыча v0.8.18</span></label>
    <label class="form-check" for="tzRun1"><input class="form-check-input" id="tzRun1" type="checkbox" data-layer="run1" checked><span class="form-check-label">Dunemaul</span></label>
    <label class="form-check" for="tzRun2"><input class="form-check-input" id="tzRun2" type="checkbox" data-layer="run2" checked><span class="form-check-label">Камень</span></label>
    <label class="form-check" for="tzRun3"><input class="form-check-input" id="tzRun3" type="checkbox" data-layer="run3" checked><span class="form-check-label">Повтор препятствия</span></label>
    <label class="form-check" for="tzEvents"><input class="form-check-input" id="tzEvents" type="checkbox" data-layer="events" checked><span class="form-check-label">События</span></label>
    <button class="btn btn-ghost" id="tzReset" type="button">Сбросить масштаб</button>
  </div>
  <div class="tz-main">
    <div class="tz-map-card" id="tzMapCard"><svg class="tz-map" id="tzMap" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Плановый маршрут V13, рудные точки и четыре фактических трека"><title>Tanaris: V13 и фактическое движение</title><desc>Интерактивная карта координат 0–100 с масштабированием и переключаемыми слоями.</desc><g id="tzViewport"></g></svg><div class="tz-tip" id="tzTip" role="tooltip"></div></div>
    <aside class="tz-side">
      <section><h3>Слои</h3><div class="tz-legend"><div class="tz-leg"><span class="tz-dot" style="background:var(--viz-series-2)"></span>Рудные точки</div><div class="tz-leg"><span class="tz-swatch" style="background:var(--muted-foreground)"></span>План V13</div><div class="tz-leg"><span class="tz-swatch" style="background:var(--viz-series-3)"></span>Добыча v0.8.18</div><div class="tz-leg"><span class="tz-swatch" style="background:var(--viz-series-5)"></span>Dunemaul</div><div class="tz-leg"><span class="tz-swatch" style="background:var(--viz-series-1)"></span>Камень</div><div class="tz-leg"><span class="tz-swatch" style="background:var(--viz-series-2)"></span>Повтор препятствия</div></div></section>
      <section id="tzRuns"></section>
      <section><h3>События</h3><div class="tz-legend"><div class="tz-leg">◈ Tooltip</div><div class="tz-leg">× Mining outcome</div><div class="tz-leg">⚔ Combat entry</div><div class="tz-leg">△ Recovery</div><div class="tz-leg">↻ Route-window skip</div></div></section>
      <div class="tz-note">Колесо — масштаб, drag — перемещение. Наведи на точку или событие для координат и типа. Topography построена из локального ADT height/slope raster; это offline-фон, не скрытое runtime-состояние.</div>
    </aside>
  </div>
</div>
<script>
(()=>{
const D=__DATA__, bg=__BACKGROUND__, B=__BOUNDS__;
const root=document.getElementById('tanaris-v13-live-map');
const svg=root.querySelector('#tzMap'), vp=root.querySelector('#tzViewport'), tip=root.querySelector('#tzTip'), card=root.querySelector('#tzMapCard');
const NS='http://www.w3.org/2000/svg'; const el=(n,a={})=>{const x=document.createElementNS(NS,n);Object.entries(a).forEach(([k,v])=>x.setAttribute(k,String(v)));return x};
vp.appendChild(el('rect',{x:0,y:0,width:100,height:100,fill:'var(--muted)'})); vp.appendChild(el('image',{href:bg,x:B.x,y:B.y,width:B.width,height:B.height,opacity:.68,preserveAspectRatio:'none'}));
const grid=el('g',{'data-layer':'grid'}); for(let n=10;n<100;n+=10){grid.append(el('line',{x1:n,y1:0,x2:n,y2:100,class:'tz-grid'}),el('line',{x1:0,y1:n,x2:100,y2:n,class:'tz-grid'}));const tx=el('text',{x:n+.3,y:99,class:'tz-axis'});tx.textContent=n;grid.append(tx);const ty=el('text',{x:.5,y:n-.3,class:'tz-axis'});ty.textContent=n;grid.append(ty)}vp.appendChild(grid);
const pathData=p=>p.map((q,i)=>(i?'L':'M')+q.x+','+q.y).join(' ');
const fullData=pathData([...D.fullRoute,D.fullRoute[0]]);
const fullHalo=el('path',{d:fullData,class:'tz-route',stroke:'var(--background)','stroke-width':5.5,opacity:.94,'data-layer':'fullRoute'});
const full=el('path',{d:fullData,class:'tz-route',stroke:'var(--foreground)','stroke-width':2.8,'stroke-dasharray':'8 5',opacity:1,'data-layer':'fullRoute'});vp.append(fullHalo,full);
const allG=el('g',{'data-layer':'allNodes'}); const oreColor={Iron:'var(--viz-series-4)',Mithril:'var(--viz-series-2)','Small Thorium':'var(--viz-series-3)',Thorium:'var(--viz-series-3)',Truesilver:'var(--viz-series-6)',Gold:'var(--yellow)'};
const showTip=(e,html)=>{tip.innerHTML=html;tip.style.display='block';const r=card.getBoundingClientRect();tip.style.left=(e.clientX-r.left+12)+'px';tip.style.top=(e.clientY-r.top+12)+'px'}; const hideTip=()=>tip.style.display='none';
 D.nodes.forEach(n=>{const c=el('circle',{cx:n.x,cy:n.y,r:.30,fill:oreColor[n.ore]||'var(--viz-series-2)',opacity:n.status==='covered'?.92:.55,class:'tz-node'});c.addEventListener('pointermove',e=>showTip(e,`<b>${n.ore}</b><br>${n.x.toFixed(2)}, ${n.y.toFixed(2)}<br>${n.status}`));c.addEventListener('pointerleave',hideTip);allG.appendChild(c)});vp.append(allG);
const symbols={tooltip:'◆',mine:'×',combat:'⚔',stuck:'△',skip:'↻'}; D.runs.forEach((run,ri)=>{const g=el('g',{'data-layer':'run'+ri});const trackData=pathData(run.track);const halo=el('path',{d:trackData,class:'tz-route',stroke:'var(--background)','stroke-width':6.4,opacity:.92});const p=el('path',{d:trackData,class:'tz-route',stroke:run.color,'stroke-width':3.4,opacity:1});g.append(halo,p);if(run.track.length){const a=run.track[0],b=run.track.at(-1);g.append(el('circle',{cx:a.x,cy:a.y,r:.7,fill:run.color,stroke:'var(--foreground)','stroke-width':.18}),el('circle',{cx:b.x,cy:b.y,r:.7,fill:'var(--background)',stroke:run.color,'stroke-width':.28}))}vp.appendChild(g);const eg=el('g',{'data-layer':'events'});run.events.forEach(ev=>{const t=el('text',{x:ev.x,y:ev.y,class:'tz-event',fill:run.color,'font-size':ev.kind==='combat'?1.65:1.45,'font-weight':500,'text-anchor':'middle','paint-order':'stroke',stroke:'var(--background)','stroke-width':.55});t.textContent=symbols[ev.kind]||'•';t.addEventListener('pointermove',e=>showTip(e,`<b>${ev.label}</b><br>${ev.x.toFixed(2)}, ${ev.y.toFixed(2)} · frame ${ev.i}<br>${run.label}`));t.addEventListener('pointerleave',hideTip);eg.appendChild(t)});vp.appendChild(eg)});
root.querySelector('#tzStats').innerHTML=[[D.summary.allNodes,'рудных точек'],[D.summary.fullWaypoints,'waypoints V13'],[D.runs.length,'live-трека']].map(([v,l])=>`<div class="tz-stat"><b>${v}</b><span>${l}</span></div>`).join('');
root.querySelector('#tzRuns').innerHTML='<h3>Live-run</h3>'+D.runs.map(r=>`<div class="tz-run" style="border-left:4px solid ${r.color}"><b>${r.label}</b>${r.stats.completed} целей · ${r.stats.stuck} stuck<br>${r.stats.attempts} mining / ${r.stats.successes} success<br>длина трека ${r.stats.length} UI</div>`).join('');
root.querySelectorAll('[data-layer]').forEach(input=>{if(input.tagName!=='INPUT')return;const apply=()=>{root.querySelectorAll(`[data-layer="${input.dataset.layer}"]`).forEach(x=>{if(x!==input)x.style.display=input.checked?'':'none'})};input.addEventListener('change',apply);apply()});
let vb={x:0,y:0,w:100,h:100},drag=null;const applyView=()=>svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);root.querySelector('#tzReset').onclick=()=>{vb={x:0,y:0,w:100,h:100};applyView()};svg.addEventListener('wheel',e=>{e.preventDefault();const f=e.deltaY>0?1.18:.84,nw=Math.max(16,Math.min(100,vb.w*f)),nh=nw;const r=svg.getBoundingClientRect(),mx=vb.x+(e.clientX-r.left)/r.width*vb.w,my=vb.y+(e.clientY-r.top)/r.height*vb.h;vb.x=Math.max(0,Math.min(100-nw,mx-(mx-vb.x)*nw/vb.w));vb.y=Math.max(0,Math.min(100-nh,my-(my-vb.y)*nh/vb.h));vb.w=nw;vb.h=nh;applyView()},{passive:false});svg.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,vx:vb.x,vy:vb.y};svg.setPointerCapture(e.pointerId);svg.classList.add('dragging')});svg.addEventListener('pointermove',e=>{if(!drag)return;const r=svg.getBoundingClientRect();vb.x=Math.max(0,Math.min(100-vb.w,drag.vx-(e.clientX-drag.x)/r.width*vb.w));vb.y=Math.max(0,Math.min(100-vb.h,drag.vy-(e.clientY-drag.y)/r.height*vb.h));applyView()});svg.addEventListener('pointerup',()=>{drag=null;svg.classList.remove('dragging')});
})();
</script>
'''


def main() -> int:
    args = parse_args()
    payload, background, bounds = build_payload()
    html = (
        FRAGMENT.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__BACKGROUND__", json.dumps(background))
        .replace("__BOUNDS__", json.dumps(bounds, separators=(",", ":")))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "nodes": payload["summary"]["allNodes"],
                "runs": [run["stats"] for run in payload["runs"]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
