#!/usr/bin/env python3
"""Interactive drill-down benchmark dashboard.

Drill order: Overview → SNR → Topology → Sparsity. Methods always on x-axis.

Usage:
    python -m sparselink.bench.dashboard
    python -m sparselink.bench.dashboard -i results.json -o benchmark_dashboard.html
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = ["auroc", "aupr", "f1", "mcc", "r2", "fdr"]
METRIC_LABELS = {
    "auroc": "AUROC", "aupr": "AUPR", "mcc": "MCC",
    "fdr": "FDR", "r2": "R²", "f1": "F1",
}


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    rows = [r for r in data if r.get("error") is None]
    df = pd.DataFrame(rows)
    if "f1" not in df.columns:
        p, r = df["precision"], df["recall"]
        df["f1"] = (2 * p * r / (p + r)).fillna(0)
    if "sparsity" not in df.columns:
        df["sparsity"] = "—"
    if "dataset_idx" not in df.columns:
        df["dataset_idx"] = df.get("dataset_name", range(len(df)))
    return df


def _fig_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def _make_comparison_grid(df: pd.DataFrame, metric: str, label: str, snr_val: float) -> str:
    """Grid of box plots: rows=topology, cols=sparsity (or just topology if no sparsity)."""
    topos = sorted(df["topology"].unique())
    sparsities = sorted(df["sparsity"].unique())
    methods = sorted(df["method"].unique())
    sub = df[df["snr"] == snr_val]
    has_sparsity = len(sparsities) > 1
    n_r = len(topos)
    n_c = len(sparsities) if has_sparsity else 1
    titles = ([f"{t} / sp={sp}" for t in topos for sp in sparsities]
              if has_sparsity else [t for t in topos])
    fig = make_subplots(rows=n_r, cols=n_c, subplot_titles=titles,
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    colors = px.colors.qualitative.Set2
    cmap = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    for ri, t in enumerate(topos, 1):
        cols_iter = list(enumerate(sparsities, 1)) if has_sparsity else [(1, sparsities[0])]
        for ci, sp in cols_iter:
            chunk = sub[(sub["topology"] == t) & (sub["sparsity"] == sp)]
            for m in methods:
                fig.add_trace(go.Box(
                    y=chunk[chunk["method"] == m][metric], name=m,
                    marker_color=cmap[m],
                    showlegend=(ri == 1 and ci == 1), legendgroup=m,
                ), row=ri, col=ci)
            fig.update_yaxes(title_text=label if ci == 1 else "", row=ri, col=ci)
    fig.update_layout(height=280 * n_r + 80, boxmode="group",
                      legend=dict(orientation="h", y=-0.04), margin=dict(t=40, b=40))
    return _fig_html(fig, f"static_{metric}_snr{snr_val}")


def build_dashboard(df: pd.DataFrame) -> str:
    methods = sorted(df["method"].unique())
    topologies = sorted(df["topology"].unique())
    sparsities = sorted(df["sparsity"].unique())
    snrs = sorted(df["snr"].unique())
    n_runs = len(df)

    has_sparsity = len(sparsities) > 1

    # ── Aggregated data for JS drill-down ────────────────────────────────────
    # L1: overview — methods grouped by SNR
    l1_snr = {}
    for s in snrs:
        agg = df[df["snr"] == s].groupby("method")[METRICS].mean().reindex(methods).round(4)
        l1_snr[str(s)] = agg.to_dict(orient="index")

    # L2: SNR → topology — methods grouped by topology
    l2 = {}
    for s in snrs:
        l2[str(s)] = {}
        for t in topologies:
            agg = df[(df["snr"] == s) & (df["topology"] == t)].groupby("method")[METRICS].mean().reindex(methods).round(4)
            l2[str(s)][t] = agg.to_dict(orient="index")

    # L3: SNR → topology → sparsity — methods grouped by sparsity
    l3 = {}
    if has_sparsity:
        for s in snrs:
            l3[str(s)] = {}
            for t in topologies:
                l3[str(s)][t] = {}
                for sp in sparsities:
                    agg = df[(df["snr"] == s) & (df["topology"] == t) & (df["sparsity"] == sp)].groupby("method")[METRICS].mean().reindex(methods).round(4)
                    l3[str(s)][t][str(sp)] = agg.to_dict(orient="index")

    # ── Static plots (collapsible) ───────────────────────────────────────────
    static_sections = ""
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        snr_plots = ""
        for s in snrs:
            plot_html = _make_comparison_grid(df, metric, label, s)
            snr_plots += f"""
        <button class="toggle" onclick="toggleNext(this)">▶ SNR={s}</button>
        <div class="collapsible"><div class="card">{plot_html}</div></div>"""
        static_sections += f"""
      <button class="toggle" onclick="toggleNext(this)">▶ {label} — full comparison grids</button>
      <div class="collapsible">{snr_plots}</div>"""

    # ── Table ────────────────────────────────────────────────────────────────
    cols = ["method", "topology", "sparsity", "snr", "dataset_idx",
            "auroc", "aupr", "f1", "mcc", "fdr", "r2", "elapsed_sec"]
    tcols = [c for c in cols if c in df.columns]
    trows = df[tcols].round(4).values.tolist()

    snr_colors = {}
    palette = ['#3fb950', '#d29922', '#f85149', '#bc8cff', '#58a6ff']
    for i, s in enumerate(snrs):
        snr_colors[str(s)] = palette[i % len(palette)]
    topo_colors = {}
    tpal = ['#58a6ff', '#3fb950', '#d29922', '#f85149']
    for i, t in enumerate(topologies):
        topo_colors[t] = tpal[i % len(tpal)]
    sp_colors = {}
    spal = ['#58a6ff', '#bc8cff', '#f85149']
    for i, sp in enumerate(sparsities):
        sp_colors[str(sp)] = spal[i % len(spal)]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sparselink Benchmark</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 1400px; margin: 0 auto; }}
  h1 {{ text-align: center; margin: 20px 0 5px; font-size: 1.8em; color: #58a6ff; }}
  .sub {{ text-align: center; color: #8b949e; margin-bottom: 20px; font-size: 0.95em; }}
  .stats {{ display: flex; gap: 14px; justify-content: center; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 20px; text-align: center; }}
  .stat .n {{ font-size: 1.4em; color: #58a6ff; font-weight: bold; }}
  .stat .l {{ font-size: 0.8em; color: #8b949e; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .breadcrumb {{ margin-bottom: 12px; font-size: 0.9em; }}
  .breadcrumb span {{ color: #58a6ff; cursor: pointer; }}
  .breadcrumb span:hover {{ text-decoration: underline; }}
  .breadcrumb .sep {{ color: #484f58; margin: 0 6px; cursor: default; }}
  .metric-picker {{ margin-bottom: 12px; display: flex; gap: 6px; flex-wrap: wrap; }}
  .metric-picker button {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85em; }}
  .metric-picker button.active {{ background: #58a6ff; color: #0d1117; }}
  .hint {{ color: #484f58; font-size: 0.8em; margin-top: 6px; }}
  .section-title {{ color: #58a6ff; font-size: 1.1em; margin: 24px 0 10px; font-weight: 600; }}
  .toggle {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    padding: 10px 16px; border-radius: 6px; cursor: pointer; width: 100%; text-align: left;
    font-size: 0.95em; margin-bottom: 6px; }}
  .toggle:hover {{ background: #1c2128; }}
  .collapsible {{ display: none; margin-bottom: 6px; }}
  .collapsible.open {{ display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
  th, td {{ padding: 5px 8px; text-align: left; border-bottom: 1px solid #21262d; }}
  th {{ background: #0d1117; color: #58a6ff; position: sticky; top: 0; cursor: pointer; }}
  tr:hover {{ background: #1c2128; }}
  .table-wrap {{ max-height: 500px; overflow-y: auto; }}
  input {{ background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
           padding: 8px 12px; border-radius: 6px; width: 300px; margin-bottom: 10px; }}
</style>
</head>
<body>
<h1>sparselink Benchmark</h1>
<p class="sub">{n_runs} runs · click bars to drill: SNR → topology → sparsity</p>

<div class="stats">
  <div class="stat"><div class="n">{n_runs}</div><div class="l">Runs</div></div>
  <div class="stat"><div class="n">{len(methods)}</div><div class="l">Methods</div></div>
  <div class="stat"><div class="n">{len(snrs)}</div><div class="l">SNR Levels</div></div>
  <div class="stat"><div class="n">{len(topologies)}</div><div class="l">Topologies</div></div>
  <div class="stat"><div class="n">{len(sparsities)}</div><div class="l">Sparsities</div></div>
</div>

<div class="card">
  <div class="breadcrumb" id="bc"><span class="active">Overview</span></div>
  <div class="metric-picker" id="mp"></div>
  <div id="mainPlot"></div>
  <p class="hint" id="hint">Click a bar to drill into that SNR level</p>
</div>

<p class="section-title">Detailed Comparisons</p>
{static_sections}

<button class="toggle" onclick="toggleNext(this)">▶ Raw Data Table ({n_runs} rows)</button>
<div class="collapsible">
  <div class="card">
    <input type="text" id="tf" placeholder="Filter..." oninput="filterT()">
    <div class="table-wrap">
      <table id="rt"><thead><tr>{"".join(f'<th onclick="sortT({i})">{c}</th>' for i,c in enumerate(tcols))}</tr></thead>
      <tbody>{"".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in trows)}</tbody></table>
    </div>
  </div>
</div>

<script>
const L1 = {json.dumps(l1_snr)};
const L2 = {json.dumps(l2)};
const L3 = {json.dumps(l3)};
const METHODS = {json.dumps(methods)};
const SNRS = {json.dumps([str(s) for s in snrs])};
const TOPOS = {json.dumps(topologies)};
const SPARSITIES = {json.dumps([str(s) for s in sparsities])};
const HAS_SP = {'true' if has_sparsity else 'false'};
const METRICS = {json.dumps(METRICS)};
const ML = {json.dumps(METRIC_LABELS)};
const C = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#79c0ff','#56d364','#e3b341'];
const SNR_C = {json.dumps(snr_colors)};
const TOPO_C = {json.dumps(topo_colors)};
const SP_C = {json.dumps(sp_colors)};
const LP = {{paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:'#c9d1d9'}}}};

let cm = 'auroc', state = {{l:1}};

function initPicker() {{
  const el = document.getElementById('mp');
  METRICS.forEach(m => {{
    const b = document.createElement('button');
    b.textContent = ML[m]; b.id = 'mp_'+m;
    b.onclick = () => {{ cm = m; render(); updP(); }};
    if (m===cm) b.classList.add('active');
    el.appendChild(b);
  }});
}}
function updP() {{ METRICS.forEach(m => document.getElementById('mp_'+m).classList.toggle('active', m===cm)); }}
function render() {{
  if (state.l===1) L1view();
  else if (state.l===2) L2view(state.snr);
  else if (state.l===3) L3view(state.snr, state.topo);
  else if (state.l===4) L4view(state.snr, state.topo, state.sp);
}}

// Level 1: methods grouped by SNR
function L1view() {{
  state = {{l:1}};
  const traces = SNRS.map(s => ({{
    x: METHODS, y: METHODS.map(m => L1[s][m] ? L1[s][m][cm] : 0),
    name: 'SNR='+s, type: 'bar', marker: {{color: SNR_C[s]}},
    hovertemplate: '%{{x}} (SNR='+s+')<br>'+ML[cm]+': %{{y:.3f}}<extra></extra>'
  }}));
  const layout = {{...LP, title: ML[cm]+' — all methods by SNR', barmode:'group',
    yaxis:{{title:ML[cm]}}, height:440, margin:{{t:50,b:80}}, xaxis:{{tickangle:-30}},
    legend:{{orientation:'h',y:1.1}} }};
  Plotly.newPlot('mainPlot', traces, layout).then(() => {{
    document.getElementById('mainPlot').on('plotly_click', d => {{
      if (d.points[0]) L2view(d.points[0].data.name.replace('SNR=',''));
    }});
  }});
  document.getElementById('bc').innerHTML = '<span class="active">Overview</span>';
  document.getElementById('hint').textContent = 'Click a bar to drill into that SNR level';
}}

// Level 2: methods grouped by topology (for one SNR)
function L2view(snr) {{
  state = {{l:2, snr}};
  const traces = TOPOS.map(t => ({{
    x: METHODS, y: METHODS.map(m => L2[snr] && L2[snr][t] && L2[snr][t][m] ? L2[snr][t][m][cm] : 0),
    name: t, type: 'bar', marker: {{color: TOPO_C[t]}},
    hovertemplate: '%{{x}} ('+t+')<br>'+ML[cm]+': %{{y:.3f}}<extra></extra>'
  }}));
  const layout = {{...LP, title: ML[cm]+' — SNR='+snr+' — methods by topology', barmode:'group',
    yaxis:{{title:ML[cm]}}, height:440, margin:{{t:50,b:80}}, xaxis:{{tickangle:-30}},
    legend:{{orientation:'h',y:1.1}} }};
  Plotly.newPlot('mainPlot', traces, layout).then(() => {{
    document.getElementById('mainPlot').on('plotly_click', d => {{
      if (d.points[0]) {{
        const topo = d.points[0].data.name;
        if (HAS_SP) L3view(snr, topo);
        else L4view(snr, topo, SPARSITIES[0]);
      }}
    }});
  }});
  document.getElementById('bc').innerHTML =
    '<span onclick="L1view()">Overview</span><span class="sep">›</span><span class="active">SNR='+snr+'</span>';
  document.getElementById('hint').textContent = HAS_SP ? 'Click a bar to drill into that topology' : 'Click a bar to see method comparison for that topology';
}}

// Level 3: methods grouped by sparsity (for one SNR + topology)
function L3view(snr, topo) {{
  state = {{l:3, snr, topo}};
  const traces = SPARSITIES.map(sp => ({{
    x: METHODS, y: METHODS.map(m => L3[snr] && L3[snr][topo] && L3[snr][topo][sp] && L3[snr][topo][sp][m] ? L3[snr][topo][sp][m][cm] : 0),
    name: 'sp='+sp, type: 'bar', marker: {{color: SP_C[sp]}},
    hovertemplate: '%{{x}} (sp='+sp+')<br>'+ML[cm]+': %{{y:.3f}}<extra></extra>'
  }}));
  const layout = {{...LP, title: ML[cm]+' — SNR='+snr+' / '+topo+' — methods by sparsity', barmode:'group',
    yaxis:{{title:ML[cm]}}, height:440, margin:{{t:50,b:80}}, xaxis:{{tickangle:-30}},
    legend:{{orientation:'h',y:1.1}} }};
  Plotly.newPlot('mainPlot', traces, layout).then(() => {{
    document.getElementById('mainPlot').on('plotly_click', d => {{
      const sp = d.points[0].data.name.replace('sp=','');
      L4view(snr, topo, sp);
    }});
  }});
  document.getElementById('bc').innerHTML =
    '<span onclick="L1view()">Overview</span><span class="sep">›</span>' +
    '<span onclick="L2view(\\''+snr+'\\')">SNR='+snr+'</span><span class="sep">›</span>' +
    '<span class="active">'+topo+'</span>';
  document.getElementById('hint').textContent = 'Click a bar to see that exact condition';
}}

// Level 4: single condition — one bar per method
function L4view(snr, topo, sp) {{
  state = {{l:4, snr, topo, sp}};
  let d;
  if (HAS_SP && L3[snr] && L3[snr][topo] && L3[snr][topo][sp]) {{
    d = L3[snr][topo][sp];
  }} else {{
    d = L2[snr] && L2[snr][topo] ? L2[snr][topo] : {{}};
  }}
  const vals = METHODS.map(m => d[m] ? d[m][cm] : 0);
  const colors = METHODS.map((_,i) => C[i%C.length]);
  const title = HAS_SP ? ML[cm]+' — SNR='+snr+' / '+topo+' / sp='+sp : ML[cm]+' — SNR='+snr+' / '+topo;
  const trace = {{x:METHODS, y:vals, type:'bar', marker:{{color:colors}},
    hovertemplate:'%{{x}}<br>'+ML[cm]+': %{{y:.3f}}<extra></extra>'}};
  const layout = {{...LP, title:title,
    yaxis:{{title:ML[cm]}}, height:440, margin:{{t:50,b:80}}, xaxis:{{tickangle:-30}} }};
  Plotly.newPlot('mainPlot', [trace], layout);
  let bc = '<span onclick="L1view()">Overview</span><span class="sep">›</span>' +
    '<span onclick="L2view(\\''+snr+'\\')">SNR='+snr+'</span><span class="sep">›</span>';
  if (HAS_SP) {{
    bc += '<span onclick="L3view(\\''+snr+'\\',\\''+topo+'\\')">'+topo+'</span><span class="sep">›</span>' +
      '<span class="active">sp='+sp+'</span>';
  }} else {{
    bc += '<span class="active">'+topo+'</span>';
  }}
  document.getElementById('bc').innerHTML = bc;
  document.getElementById('hint').textContent = 'Exact condition. Use breadcrumbs or metric picker to navigate.';
}}

function toggleNext(btn) {{
  const el = btn.nextElementSibling;
  const opening = !el.classList.contains('open');
  el.classList.toggle('open');
  btn.textContent = btn.textContent.replace(/^[▶▼]/, opening ? '▼' : '▶');
  if (opening) window.dispatchEvent(new Event('resize'));
}}
function filterT() {{
  const q = document.getElementById('tf').value.toLowerCase();
  document.querySelectorAll('#rt tbody tr').forEach(r => {{
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
let sd = {{}};
function sortT(c) {{
  const tb = document.querySelector('#rt tbody');
  const rows = Array.from(tb.rows);
  sd[c] = !sd[c];
  rows.sort((a,b) => {{
    let av=a.cells[c].textContent, bv=b.cells[c].textContent;
    let an=parseFloat(av), bn=parseFloat(bv);
    if (!isNaN(an)&&!isNaN(bn)) return sd[c]?an-bn:bn-an;
    return sd[c]?av.localeCompare(bv):bv.localeCompare(av);
  }});
  rows.forEach(r => tb.appendChild(r));
}}

initPicker();
L1view();
</script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard")
    parser.add_argument("-i", "--input", default="benchmark_results.json")
    parser.add_argument("-o", "--output", default="benchmark_dashboard.html")
    args = parser.parse_args(argv)

    df = load(args.input)
    print(f"Loaded {len(df)} successful runs from {args.input}")

    html = build_dashboard(df)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Dashboard written to {args.output}")


if __name__ == "__main__":
    main()
