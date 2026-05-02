#!/usr/bin/env python3
"""
Interactive graph visualization tool.

Usage:
    python visualize_graph.py graph_no_drop.json
    python visualize_graph.py graph_no_drop.json -o my_viz.html
    python visualize_graph.py graph_no_drop.json --open

Generates a self-contained HTML file with:
  - Force-directed node-link diagram (nodes colored by scene)
  - Click any node to inspect its landmarks and connections
  - Scene filter toggles
  - Adjacency matrix heatmap
  - Summary statistics and degree distribution
"""

import argparse
import json
import math
import os
import sys
import webbrowser
from collections import Counter


def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    return data


def extract_scene(node, scenes):
    img = node["info"].get("image", "")
    for i, s in enumerate(scenes):
        if s in img:
            return i
    return -1


def compute_layout(nodes, scenes_list, scene_indices):
    """Arrange nodes in clusters per scene around a circle."""
    positions = {}
    scene_groups = {}
    for node in nodes:
        si = scene_indices[node["node_id"]]
        scene_groups.setdefault(si, []).append(node["node_id"])

    n_scenes = len(scenes_list)
    scene_radius = 300
    node_base_radius = 60

    for si, nids in scene_groups.items():
        if si < 0:
            cx, cy = 0, 0
        else:
            angle = 2 * math.pi * si / max(n_scenes, 1)
            cx = scene_radius * math.cos(angle)
            cy = scene_radius * math.sin(angle)

        n = len(nids)
        r = max(node_base_radius, n * 8)
        for j, nid in enumerate(nids):
            a = 2 * math.pi * j / max(n, 1)
            positions[nid] = (cx + r * math.cos(a), cy + r * math.sin(a))

    return positions


def build_html(data, graph_path):
    scenes = data["scenes"]
    nodes = data["nodes"]

    scene_indices = {}
    for node in nodes:
        scene_indices[node["node_id"]] = extract_scene(node, scenes)

    positions = compute_layout(nodes, scenes, scene_indices)

    short_scenes = []
    for s in scenes:
        parts = s.split("_")
        short_scenes.append("_".join(parts[-2:]) if len(parts) >= 2 else s)

    js_nodes = []
    for node in nodes:
        nid = node["node_id"]
        x, y = positions[nid]
        landmarks = node["info"].get("landmarks", [])
        conns = [c["other_node_id"] for c in node["connections"]]
        si = scene_indices[nid]
        img = node["info"].get("image", "")
        js_nodes.append({
            "id": nid,
            "x": round(x, 1),
            "y": round(y, 1),
            "scene": si,
            "landmarks": landmarks,
            "connections": conns,
            "degree": len(conns),
            "image": img,
        })

    total_edges = sum(len(n["connections"]) for n in nodes)
    degrees = [len(n["connections"]) for n in nodes]
    avg_deg = sum(degrees) / len(degrees) if degrees else 0

    scene_counts = Counter(scene_indices.values())
    intra = Counter()
    inter = Counter()
    for node in nodes:
        ns = scene_indices[node["node_id"]]
        for conn in node["connections"]:
            os_ = scene_indices[conn["other_node_id"]]
            if ns == os_:
                intra[ns] += 1
            else:
                inter[(ns, os_)] += 1

    adj_matrix = []
    id_list = sorted(positions.keys())
    id_to_idx = {nid: i for i, nid in enumerate(id_list)}
    for node in nodes:
        nid = node["node_id"]
        row_set = set(c["other_node_id"] for c in node["connections"])
        adj_matrix.append(sorted(id_to_idx[c] for c in row_set if c in id_to_idx))

    scene_stats = []
    for si, sname in enumerate(short_scenes):
        count = scene_counts.get(si, 0)
        intra_count = intra.get(si, 0)
        scene_stats.append({"name": sname, "full": scenes[si], "nodes": count, "intraEdges": intra_count})

    deg_hist = Counter()
    bucket = 10
    for d in degrees:
        b = (d // bucket) * bucket
        deg_hist[b] = deg_hist.get(b, 0) + 1
    deg_hist_data = [{"bucket": k, "count": v} for k, v in sorted(deg_hist.items())]

    graph_name = os.path.basename(graph_path)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Graph Visualizer — {graph_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
       background: #1a1a2e; color: #e0e0e0; overflow: hidden; height: 100vh; }}
.app {{ display: flex; height: 100vh; }}
.sidebar {{ width: 340px; min-width: 340px; background: #16213e; border-right: 1px solid #2a2a4a;
            display: flex; flex-direction: column; overflow-y: auto; }}
.sidebar h1 {{ font-size: 14px; padding: 12px 16px; border-bottom: 1px solid #2a2a4a;
               color: #7ec8e3; letter-spacing: 0.5px; }}
.section {{ padding: 12px 16px; border-bottom: 1px solid #2a2a4a; }}
.section-title {{ font-size: 11px; text-transform: uppercase; color: #888; letter-spacing: 1px;
                  margin-bottom: 8px; }}
.stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
.stat {{ background: #1a1a2e; border-radius: 4px; padding: 8px; }}
.stat-val {{ font-size: 18px; font-weight: 700; color: #7ec8e3; }}
.stat-label {{ font-size: 10px; color: #888; margin-top: 2px; }}
.scene-list {{ display: flex; flex-direction: column; gap: 4px; }}
.scene-btn {{ display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 4px;
              cursor: pointer; border: none; background: none; color: #e0e0e0; font-size: 12px;
              text-align: left; }}
.scene-btn:hover {{ background: #2a2a4a; }}
.scene-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.scene-info {{ font-size: 10px; color: #888; margin-left: auto; }}
.node-detail {{ padding: 12px 16px; }}
.node-detail h3 {{ font-size: 13px; color: #7ec8e3; margin-bottom: 8px; }}
.landmark {{ font-size: 11px; color: #ccc; padding: 3px 0; border-bottom: 1px solid #2a2a4a; }}
.conn-list {{ font-size: 11px; color: #999; margin-top: 8px; line-height: 1.6; }}
.main {{ flex: 1; display: flex; flex-direction: column; }}
.toolbar {{ display: flex; align-items: center; gap: 12px; padding: 8px 16px;
            background: #16213e; border-bottom: 1px solid #2a2a4a; }}
.toolbar label {{ font-size: 11px; color: #888; }}
.toolbar select, .toolbar input {{ background: #1a1a2e; border: 1px solid #2a2a4a; color: #e0e0e0;
                                   padding: 4px 8px; border-radius: 4px; font-size: 11px; }}
.tab-bar {{ display: flex; gap: 0; }}
.tab {{ padding: 6px 16px; font-size: 12px; cursor: pointer; border: none;
        background: none; color: #888; border-bottom: 2px solid transparent; }}
.tab.active {{ color: #7ec8e3; border-bottom-color: #7ec8e3; }}
.tab:hover {{ color: #e0e0e0; }}
.canvas-wrap {{ flex: 1; position: relative; overflow: hidden; background: #0f0f23; }}
canvas {{ display: block; }}
.tooltip {{ position: absolute; background: #16213e; border: 1px solid #2a2a4a; border-radius: 4px;
            padding: 8px 12px; font-size: 11px; pointer-events: none; display: none;
            max-width: 280px; z-index: 10; }}
.tooltip .tt-title {{ color: #7ec8e3; font-weight: 600; margin-bottom: 4px; }}
.tooltip .tt-body {{ color: #ccc; }}
.matrix-wrap {{ flex: 1; overflow: auto; display: none; background: #0f0f23; padding: 16px; }}
.matrix-wrap canvas {{ image-rendering: pixelated; }}
.hist-wrap {{ flex: 1; overflow: auto; display: none; background: #0f0f23; padding: 24px; }}
.hist-wrap svg {{ display: block; margin: 0 auto; }}
</style>
</head>
<body>
<div class="app">
<div class="sidebar">
    <h1>{graph_name}</h1>
    <div class="section">
        <div class="section-title">Summary</div>
        <div class="stat-grid">
            <div class="stat"><div class="stat-val">{len(nodes)}</div><div class="stat-label">Nodes</div></div>
            <div class="stat"><div class="stat-val">{total_edges:,}</div><div class="stat-label">Directed Edges</div></div>
            <div class="stat"><div class="stat-val">{avg_deg:.1f}</div><div class="stat-label">Avg Degree</div></div>
            <div class="stat"><div class="stat-val">{data['sparsification_steps']}</div><div class="stat-label">Sparsification</div></div>
        </div>
    </div>
    <div class="section">
        <div class="section-title">Scenes</div>
        <div class="scene-list" id="scene-list"></div>
    </div>
    <div class="section" id="node-detail-section" style="display:none">
        <div class="node-detail" id="node-detail"></div>
    </div>
</div>
<div class="main">
    <div class="toolbar">
        <div class="tab-bar">
            <button class="tab active" data-view="graph">Graph</button>
            <button class="tab" data-view="matrix">Adjacency Matrix</button>
            <button class="tab" data-view="hist">Degree Distribution</button>
        </div>
        <div style="flex:1"></div>
        <label>Edge mode:
            <select id="edge-mode">
                <option value="selected">Selected node</option>
                <option value="hover">Hover node</option>
                <option value="mutual">Mutual edges only</option>
                <option value="none">Hide all</option>
            </select>
        </label>
        <label>Node size:
            <input type="range" id="node-size" min="3" max="20" value="8">
        </label>
    </div>
    <div class="canvas-wrap" id="graph-view">
        <canvas id="graph-canvas"></canvas>
        <div class="tooltip" id="tooltip"></div>
    </div>
    <div class="matrix-wrap" id="matrix-view">
        <canvas id="matrix-canvas"></canvas>
    </div>
    <div class="hist-wrap" id="hist-view"></div>
</div>
</div>

<script>
const NODES = {json.dumps(js_nodes)};
const SCENES = {json.dumps(short_scenes)};
const SCENE_FULL = {json.dumps(scenes)};
const SCENE_STATS = {json.dumps(scene_stats)};
const DEG_HIST = {json.dumps(deg_hist_data)};
const SCENE_COLORS = ['#e74c3c','#3498db','#2ecc71','#f1c40f','#9b59b6','#e67e22','#1abc9c','#e84393'];

const nodeById = {{}};
NODES.forEach(n => nodeById[n.id] = n);

const connSets = {{}};
NODES.forEach(n => connSets[n.id] = new Set(n.connections));

// Scene filter state
const sceneVisible = new Array(SCENES.length).fill(true);
let selectedNode = null;
let hoverNode = null;

// Build scene list
const sceneListEl = document.getElementById('scene-list');
SCENES.forEach((s, i) => {{
    const btn = document.createElement('button');
    btn.className = 'scene-btn';
    btn.innerHTML = `<span class="scene-dot" style="background:${{SCENE_COLORS[i % SCENE_COLORS.length]}}"></span>
        <span>${{s}}</span>
        <span class="scene-info">${{SCENE_STATS[i].nodes}} nodes</span>`;
    btn.addEventListener('click', () => {{
        sceneVisible[i] = !sceneVisible[i];
        btn.style.opacity = sceneVisible[i] ? 1 : 0.3;
        drawGraph();
    }});
    sceneListEl.appendChild(btn);
}});

// Tab switching
const tabs = document.querySelectorAll('.tab');
const views = {{ graph: document.getElementById('graph-view'),
                 matrix: document.getElementById('matrix-view'),
                 hist: document.getElementById('hist-view') }};
tabs.forEach(tab => {{
    tab.addEventListener('click', () => {{
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const v = tab.dataset.view;
        Object.values(views).forEach(el => el.style.display = 'none');
        views[v].style.display = v === 'graph' ? '' : (v === 'matrix' ? '' : '');
        if (v === 'graph') {{ views.graph.style.display = ''; drawGraph(); }}
        else if (v === 'matrix') {{ views.matrix.style.display = ''; drawMatrix(); }}
        else {{ views.hist.style.display = ''; drawHist(); }}
    }});
}});

// ─── Graph View ─────────────────────────────────────────
const gCanvas = document.getElementById('graph-canvas');
const gCtx = gCanvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const edgeModeEl = document.getElementById('edge-mode');
const nodeSizeEl = document.getElementById('node-size');

let cam = {{ x: 0, y: 0, zoom: 1 }};
let dragging = false, dragStart = {{ x: 0, y: 0 }}, camStart = {{ x: 0, y: 0 }};

function resizeGraph() {{
    const wrap = document.getElementById('graph-view');
    gCanvas.width = wrap.clientWidth;
    gCanvas.height = wrap.clientHeight;
}}

function worldToScreen(wx, wy) {{
    return {{
        x: (wx - cam.x) * cam.zoom + gCanvas.width / 2,
        y: (wy - cam.y) * cam.zoom + gCanvas.height / 2
    }};
}}

function screenToWorld(sx, sy) {{
    return {{
        x: (sx - gCanvas.width / 2) / cam.zoom + cam.x,
        y: (sy - gCanvas.height / 2) / cam.zoom + cam.y
    }};
}}

function drawGraph() {{
    resizeGraph();
    const ctx = gCtx;
    const w = gCanvas.width, h = gCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const nodeSize = parseInt(nodeSizeEl.value);
    const edgeMode = edgeModeEl.value;

    const visible = NODES.filter(n => sceneVisible[n.scene] || n.scene < 0);
    const visSet = new Set(visible.map(n => n.id));

    // Draw edges
    const edgeNode = edgeMode === 'selected' ? selectedNode :
                     edgeMode === 'hover' ? (hoverNode || selectedNode) : null;

    if (edgeMode !== 'none' && edgeNode && visSet.has(edgeNode.id)) {{
        ctx.lineWidth = 0.5;
        edgeNode.connections.forEach(cid => {{
            if (!visSet.has(cid)) return;
            const other = nodeById[cid];
            const isMutual = connSets[cid].has(edgeNode.id);
            if (edgeMode === 'mutual' && !isMutual) return;

            const s = worldToScreen(edgeNode.x, edgeNode.y);
            const e = worldToScreen(other.x, other.y);
            ctx.strokeStyle = isMutual ? 'rgba(126,200,227,0.3)' : 'rgba(126,200,227,0.12)';
            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(e.x, e.y);
            ctx.stroke();
        }});
    }}

    // Draw nodes
    visible.forEach(n => {{
        const s = worldToScreen(n.x, n.y);
        const color = SCENE_COLORS[n.scene % SCENE_COLORS.length] || '#888';
        const isSelected = selectedNode && n.id === selectedNode.id;
        const isHover = hoverNode && n.id === hoverNode.id;
        const isConnected = (edgeNode && connSets[edgeNode.id].has(n.id));

        ctx.beginPath();
        const r = isSelected ? nodeSize + 3 : (isHover ? nodeSize + 2 : nodeSize);
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);

        if (isSelected) {{
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.fill();
            ctx.stroke();
        }} else if (isConnected) {{
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.9;
            ctx.fill();
            ctx.globalAlpha = 1;
        }} else {{
            ctx.fillStyle = color;
            ctx.globalAlpha = edgeNode ? 0.25 : 0.8;
            ctx.fill();
            ctx.globalAlpha = 1;
        }}
    }});

    // Node labels for large zoom
    if (cam.zoom > 2.5) {{
        ctx.font = '10px monospace';
        ctx.fillStyle = '#ccc';
        ctx.textAlign = 'center';
        visible.forEach(n => {{
            const s = worldToScreen(n.x, n.y);
            ctx.fillText(n.id, s.x, s.y - nodeSize - 4);
        }});
    }}
}}

function findNodeAt(sx, sy) {{
    const w = screenToWorld(sx, sy);
    const nodeSize = parseInt(nodeSizeEl.value);
    const threshold = (nodeSize + 4) / cam.zoom;
    let best = null, bestDist = Infinity;
    NODES.forEach(n => {{
        if (!sceneVisible[n.scene] && n.scene >= 0) return;
        const dx = n.x - w.x, dy = n.y - w.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < threshold && d < bestDist) {{ best = n; bestDist = d; }}
    }});
    return best;
}}

function showNodeDetail(node) {{
    const sec = document.getElementById('node-detail-section');
    const det = document.getElementById('node-detail');
    if (!node) {{ sec.style.display = 'none'; return; }}
    sec.style.display = '';
    const sceneName = node.scene >= 0 ? SCENES[node.scene] : '?';
    const lm = node.landmarks.map(l => `<div class="landmark">${{l}}</div>`).join('');
    const mutualCount = node.connections.filter(c => connSets[c] && connSets[c].has(node.id)).length;
    det.innerHTML = `
        <h3>Node ${{node.id}}</h3>
        <div style="font-size:11px;color:#888;margin-bottom:6px">${{sceneName}}</div>
        <div style="font-size:11px;margin-bottom:8px">
            Degree: <span style="color:#7ec8e3">${{node.degree}}</span> out /
            Mutual: <span style="color:#2ecc71">${{mutualCount}}</span>
        </div>
        ${{lm ? '<div class="section-title" style="margin-top:8px">Landmarks</div>' + lm : ''}}
        <div class="conn-list">
            <div class="section-title" style="margin-top:8px">Connections (${{node.connections.length}})</div>
            ${{node.connections.slice(0, 30).map(c => {{
                const cn = nodeById[c];
                const cs = cn ? SCENES[cn.scene] : '?';
                return `<span style="color:${{SCENE_COLORS[cn?.scene % SCENE_COLORS.length]}}">${{c}}</span>`;
            }}).join(', ')}}${{node.connections.length > 30 ? ' ...' : ''}}
        </div>`;
}}

gCanvas.addEventListener('mousedown', e => {{
    dragging = true;
    dragStart = {{ x: e.clientX, y: e.clientY }};
    camStart = {{ x: cam.x, y: cam.y }};
}});

gCanvas.addEventListener('mousemove', e => {{
    if (dragging) {{
        cam.x = camStart.x - (e.clientX - dragStart.x) / cam.zoom;
        cam.y = camStart.y - (e.clientY - dragStart.y) / cam.zoom;
        drawGraph();
        return;
    }}
    const rect = gCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const node = findNodeAt(mx, my);
    hoverNode = node;
    gCanvas.style.cursor = node ? 'pointer' : 'grab';

    if (node) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 16) + 'px';
        tooltip.style.top = (my - 8) + 'px';
        const sceneName = node.scene >= 0 ? SCENES[node.scene] : '?';
        tooltip.innerHTML = `<div class="tt-title">Node ${{node.id}}</div>
            <div class="tt-body">${{sceneName}}<br>Degree: ${{node.degree}}<br>
            ${{node.landmarks.length ? node.landmarks[0] : ''}}</div>`;
    }} else {{
        tooltip.style.display = 'none';
    }}
    drawGraph();
}});

gCanvas.addEventListener('mouseup', e => {{
    if (dragging) {{
        const dx = e.clientX - dragStart.x, dy = e.clientY - dragStart.y;
        if (Math.abs(dx) < 3 && Math.abs(dy) < 3) {{
            const rect = gCanvas.getBoundingClientRect();
            const node = findNodeAt(e.clientX - rect.left, e.clientY - rect.top);
            selectedNode = (selectedNode && node && selectedNode.id === node.id) ? null : node;
            showNodeDetail(selectedNode);
        }}
    }}
    dragging = false;
    drawGraph();
}});

gCanvas.addEventListener('wheel', e => {{
    e.preventDefault();
    const rect = gCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const before = screenToWorld(mx, my);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    cam.zoom = Math.max(0.1, Math.min(50, cam.zoom * factor));
    const after = screenToWorld(mx, my);
    cam.x -= (after.x - before.x);
    cam.y -= (after.y - before.y);
    drawGraph();
}}, {{ passive: false }});

edgeModeEl.addEventListener('change', drawGraph);
nodeSizeEl.addEventListener('input', drawGraph);

// ─── Adjacency Matrix View ─────────────────────────────
function drawMatrix() {{
    const mc = document.getElementById('matrix-canvas');
    const n = NODES.length;
    const cellSize = Math.max(3, Math.min(6, Math.floor(800 / n)));
    const size = n * cellSize;
    mc.width = size;
    mc.height = size;
    mc.style.width = size + 'px';
    mc.style.height = size + 'px';
    const ctx = mc.getContext('2d');
    ctx.fillStyle = '#0f0f23';
    ctx.fillRect(0, 0, size, size);

    const sortedIds = [...NODES].sort((a, b) => a.scene - b.scene || a.id - b.id);
    const idToPos = {{}};
    sortedIds.forEach((n, i) => idToPos[n.id] = i);

    // Scene bands
    let prevScene = -1, bandStart = 0;
    sortedIds.forEach((n, i) => {{
        if (n.scene !== prevScene && prevScene >= 0) {{
            ctx.strokeStyle = 'rgba(126,200,227,0.2)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(size, i * cellSize);
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, size);
            ctx.stroke();
        }}
        prevScene = n.scene;
    }});

    sortedIds.forEach((node, row) => {{
        const conns = connSets[node.id];
        sortedIds.forEach((other, col) => {{
            if (conns.has(other.id)) {{
                const mutual = connSets[other.id].has(node.id);
                const color = SCENE_COLORS[node.scene % SCENE_COLORS.length];
                ctx.globalAlpha = mutual ? 0.8 : 0.35;
                ctx.fillStyle = color;
                ctx.fillRect(col * cellSize, row * cellSize, cellSize - 0.5, cellSize - 0.5);
            }}
        }});
    }});
    ctx.globalAlpha = 1;
}}

// ─── Degree Distribution ────────────────────────────────
function drawHist() {{
    const wrap = document.getElementById('hist-view');
    if (DEG_HIST.length === 0) return;
    const svgW = 600, svgH = 300;
    const margin = {{ t: 20, r: 20, b: 40, l: 50 }};
    const w = svgW - margin.l - margin.r;
    const h = svgH - margin.t - margin.b;

    const maxCount = Math.max(...DEG_HIST.map(d => d.count));
    const barW = w / DEG_HIST.length;

    let bars = '';
    DEG_HIST.forEach((d, i) => {{
        const bh = (d.count / maxCount) * h;
        const x = margin.l + i * barW;
        const y = margin.t + h - bh;
        bars += `<rect x="${{x}}" y="${{y}}" width="${{barW - 2}}" height="${{bh}}" fill="#3498db" opacity="0.8"/>`;
        bars += `<text x="${{x + barW / 2}}" y="${{svgH - margin.b + 16}}" fill="#888" font-size="10" text-anchor="middle">${{d.bucket}}</text>`;
        bars += `<text x="${{x + barW / 2}}" y="${{y - 4}}" fill="#ccc" font-size="9" text-anchor="middle">${{d.count}}</text>`;
    }});

    // Axes
    const axisLine = `<line x1="${{margin.l}}" y1="${{margin.t + h}}" x2="${{margin.l + w}}" y2="${{margin.t + h}}" stroke="#555" stroke-width="1"/>`;
    const yLabel = `<text x="${{12}}" y="${{margin.t + h / 2}}" fill="#888" font-size="11" text-anchor="middle" transform="rotate(-90, 12, ${{margin.t + h / 2}})">Count</text>`;
    const xLabel = `<text x="${{margin.l + w / 2}}" y="${{svgH - 4}}" fill="#888" font-size="11" text-anchor="middle">Out-degree (bucketed by 10)</text>`;

    wrap.innerHTML = `<h2 style="color:#7ec8e3;font-size:14px;margin-bottom:16px">Degree Distribution</h2>
        <svg width="${{svgW}}" height="${{svgH}}">${{axisLine}}${{bars}}${{yLabel}}${{xLabel}}</svg>
        <div style="margin-top:24px">
            <h3 style="color:#7ec8e3;font-size:13px;margin-bottom:8px">Per-Scene Statistics</h3>
            <table style="border-collapse:collapse;font-size:12px;width:100%;max-width:500px">
                <tr style="border-bottom:1px solid #2a2a4a"><th style="text-align:left;padding:4px 8px;color:#888">Scene</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">Nodes</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">Intra-edges</th></tr>
                ${{SCENE_STATS.map((s, i) => `<tr style="border-bottom:1px solid #1a1a2e">
                    <td style="padding:4px 8px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{SCENE_COLORS[i]}};margin-right:6px"></span>${{s.name}}</td>
                    <td style="text-align:right;padding:4px 8px">${{s.nodes}}</td>
                    <td style="text-align:right;padding:4px 8px">${{s.intraEdges}}</td>
                </tr>`).join('')}}
            </table>
        </div>`;
}}

// Init
window.addEventListener('resize', drawGraph);
drawGraph();
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Visualize a navigation graph JSON file.")
    parser.add_argument("graph_json", help="Path to the graph JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file (default: <input>_viz.html)")
    parser.add_argument("--open", action="store_true", help="Open in browser after generating")
    args = parser.parse_args()

    data = load_graph(args.graph_json)
    html = build_html(data, args.graph_json)

    out_path = args.output or args.graph_json.replace(".json", "_viz.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Visualization written to {out_path}")
    print(f"  Nodes: {len(data['nodes'])}, Edges: {sum(len(n['connections']) for n in data['nodes']):,}")

    if args.open:
        webbrowser.open("file://" + os.path.abspath(out_path))


if __name__ == "__main__":
    main()
