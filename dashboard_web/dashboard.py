# app.py
# -----------------------------------------------------------------------------
# Flask app that:
# 1) Lazy-samples a Parquet file with Polars, uploads to SQLite.
# 2) Serves API endpoints to read from SQLite.
# 3) Provides a minimal web page to hit the API and render charts (Plotly).
#
# Run:
#   pip install -r requirements.txt
#   python app.py
#   open http://127.0.0.1:5000
#
# Configure PARQUET_PATH etc. via environment variables or inline defaults below.

from __future__ import annotations

import os
import io
import sqlite3
from typing import List, Optional, Dict, Any, Tuple

from flask import Flask, jsonify, Response, request

import numpy as np
import pandas as pd
import polars as pl

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PARQUET_PATH   = os.getenv("PARQUET_PATH", "data/sample.parquet")  # set your file
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "regimes.db")
TABLE_NAME     = os.getenv("TABLE_NAME", "sampled_regimes")
LABEL_COL      = os.getenv("LABEL_COL", "cluster")
SAMPLE_FRACTION = float(os.getenv("SAMPLE_FRACTION", "0.20"))  # 0..1
STRATIFY_COL    = os.getenv("STRATIFY_COL", "") or None
# Optional explicit column pruning: comma-separated list; leave empty for all
COLUMNS_CSV     = os.getenv("COLUMNS", "")
PRAGMA_JOURNAL  = os.getenv("SQLITE_JOURNAL_MODE", "WAL")
PRAGMA_SYNCHRONOUS = os.getenv("SQLITE_SYNCHRONOUS", "NORMAL")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    # Pragmas for decent performance
    try:
        conn.execute(f"PRAGMA journal_mode={PRAGMA_JOURNAL}")
        conn.execute(f"PRAGMA synchronous={PRAGMA_SYNCHRONOUS}")
    except Exception:
        pass
    return conn

def lazy_sample_parquet(
    path: str,
    fraction: float = 0.05,
    *,
    columns: Optional[List[str]] = None,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Low-memory(ish) sampling loader for Parquet.

    Strategy:
    - Read schema (0 rows) for column pruning.
    - If stratify_col provided and present:
        read full pruned DF and groupwise sample each category by fraction.
      (One pass; simpler and robust for medium/large data.)
    - Else:
        read pruned DF once, sample Bernoulli-style by fraction.
    """
    schema = pl.read_parquet(path, n_rows=0).schema
    available = list(schema.keys())

    if columns:
        columns = [c for c in columns if c in available]
        if not columns:
            raise ValueError("None of the requested columns exist in the Parquet file.")
    else:
        columns = available

    df = pl.read_parquet(path, columns=columns)

    if stratify_col and stratify_col in df.columns:
        parts = []
        for val in df.select(pl.col(stratify_col)).unique().to_series().to_list():
            sub = df.filter(pl.col(stratify_col) == val)
            if sub.height == 0:
                continue
            take_n = max(1, int(round(sub.height * fraction)))
            parts.append(sub.sample(n=take_n, with_replacement=False, seed=seed))
        if parts:
            out = pl.concat(parts, how="vertical")
        else:
            take_n = max(1, int(round(df.height * fraction)))
            out = df.sample(n=take_n, with_replacement=False, seed=seed)
    else:
        if fraction >= 1.0:
            out = df
        else:
            take_n = max(1, int(round(df.height * fraction)))
            out = df.sample(n=take_n, with_replacement=False, seed=seed)

    return out

def upload_polars_to_sqlite(
    df: pl.DataFrame,
    sqlite_db_path: str,
    table: str,
    *,
    if_exists: str = "replace",
    chunksize: int = 50_000,
) -> None:
    """Upload a Polars DataFrame to SQLite via pandas.to_sql."""
    pdf = df.to_pandas(use_pyarrow_extension_array=False)

    # Normalize categoricals to strings for SQLite
    for c in pdf.select_dtypes(include=["category"]).columns:
        pdf[c] = pdf[c].astype("string")

    with _connect_sqlite() as conn:
        pdf.to_sql(
            table,
            conn,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
        )

def detect_categories_sqlite(conn: sqlite3.Connection, table: str, label_col: str) -> List[str]:
    try:
        cur = conn.execute(f'SELECT DISTINCT "{label_col}" FROM "{table}" LIMIT 50')
        vals = [str(r[0]) for r in cur.fetchall()]
        return vals
    except Exception:
        return []

def sqlite_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [row[1] for row in cur.fetchall()]

def df_to_js_records(pdf: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    dt_cols = [c for c in pdf.columns if np.issubdtype(pdf[c].dtype, np.datetime64)]
    rows: List[Dict[str, Any]] = []
    for _, r in pdf.iterrows():
        rec: Dict[str, Any] = {}
        for c, v in r.items():
            if pd.isna(v):
                rec[c] = None
            elif c in dt_cols:
                rec[c] = pd.to_datetime(v).isoformat()
            elif isinstance(v, (np.integer, np.floating, np.bool_)):
                rec[c] = v.item()
            else:
                rec[c] = v
        rows.append(rec)
    return rows, dt_cols

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------

app = Flask(__name__)

@app.get("/")
def index() -> Response:
    # Minimal SPA that calls /bootstrap (optional) and /api/data
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Regime Dashboard — SQLite API</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; }}
    .row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:end; }}
    .card {{ border:1px solid #e5e7eb; border-radius:10px; padding:12px; margin-bottom:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2, minmax(300px,1fr)); gap:12px; }}
    input, select {{ padding:6px; }}
    .btn {{ padding:8px 12px; background:#111827; color:#fff; border:none; border-radius:8px; cursor:pointer; }}
    .title {{ font-weight:600; font-size:18px; margin-bottom:8px; }}
    .muted {{ color:#666; font-size:12px; }}
  </style>
</head>
<body>
  <div class="title">Regime Dashboard — SQLite API</div>

  <div class="card row">
    <div>
      <label>Parquet path (server)</label><br/>
      <input id="parquet" value="{PARQUET_PATH}" size="40"/>
    </div>
    <div>
      <label>Fraction</label><br/>
      <input id="fraction" type="number" min="0" max="1" step="0.01" value="{SAMPLE_FRACTION}"/>
    </div>
    <div>
      <label>Stratify column (optional)</label><br/>
      <input id="strat" value="{STRATIFY_COL or ''}"/>
    </div>
    <div>
      <label>Table</label><br/>
      <input id="table" value="{TABLE_NAME}"/>
    </div>
    <div>
      <button class="btn" id="bootstrap">Bootstrap DB</button>
    </div>
    <div style="flex:1">
      <div id="status" class="muted">Pointed at SQLite: {SQLITE_DB_PATH}</div>
    </div>
  </div>

  <div class="card row">
    <div>
      <label>Label column</label><br/>
      <input id="labelCol" value="{LABEL_COL}"/>
    </div>
    <div>
      <label>Windows</label><br/>
      <input id="windows" type="number" min="1" max="8" value="4"/>
    </div>
    <div>
      <label>Marker</label><br/>
      <input id="marker" type="number" min="5" max="80" value="20"/>
    </div>
    <div>
      <label>Alpha</label><br/>
      <input id="alpha" type="number" min="0.05" max="1" step="0.05" value="0.6"/>
    </div>
    <div>
      <label>Rotate X°</label><br/>
      <input id="rotate" type="number" min="0" max="90" step="5" value="30"/>
    </div>
    <div>
      <button class="btn" id="load">Load Data</button>
    </div>
  </div>

  <div class="card row">
    <div style="min-width:260px">
      <label>X/Y selectors</label>
      <div id="xy-box"></div>
    </div>
    <div style="flex:1">
      <label>Charts</label>
      <div id="charts" class="grid"></div>
    </div>
  </div>

<script>
let GLOBAL = {{
  data: [],
  columns: [],
  categories: [],
  labelCol: "{LABEL_COL}",
  axisOptions: [],
  windows: 4,
  xys: [],
  marker: 20,
  alpha: 0.6,
  rotate: 30,
  suptitle: "Regime Characteristics Over Time",
}};

function detectTypes(rows, exclude) {{
  if (!rows.length) return {{ numeric: [], datetime: [] }};
  const cols = Object.keys(rows[0]).filter(c => !exclude.has(c));
  const numeric = [], datetime = [];
  cols.forEach(c => {{
    let n=0, d=0;
    for (let i=0; i<Math.min(rows.length,200); i++) {{
      const v = rows[i][c];
      if (typeof v === 'number' && Number.isFinite(v)) n++;
      else if (v != null) {{
        const asNum = Number(v); if (Number.isFinite(asNum)) n++;
        const asDate = new Date(String(v)); if (!isNaN(asDate.valueOf())) d++;
      }}
    }}
    if (d > n && d > 3) datetime.push(c);
    else if (n > 0) numeric.push(c);
  }});
  return {{ numeric, datetime }};
}}

function buildXYSelectors() {{
  const box = document.getElementById('xy-box');
  box.innerHTML = '';
  const k = Math.max(1, Math.min(GLOBAL.windows, 8));
  GLOBAL.xys = GLOBAL.xys.slice(0, k);
  while (GLOBAL.xys.length < k) {{
    const i = GLOBAL.xys.length;
    const x = GLOBAL.axisOptions[i % GLOBAL.axisOptions.length] || GLOBAL.axisOptions[0];
    const y = GLOBAL.axisOptions[(i+1) % GLOBAL.axisOptions.length] || GLOBAL.axisOptions[0];
    GLOBAL.xys.push({{ x, y }});
  }}
  GLOBAL.xys.forEach((pair, i) => {{
    const wrap = document.createElement('div'); wrap.style.display='flex'; wrap.style.gap='6px'; wrap.style.marginBottom='6px';
    const xSel = document.createElement('select'); const ySel = document.createElement('select');
    xSel.style.minWidth='180px'; ySel.style.minWidth='180px';
    GLOBAL.axisOptions.forEach(c => {{
      const ox = document.createElement('option'); ox.value=c; ox.textContent=c; if (c===pair.x) ox.selected=true; xSel.appendChild(ox);
      const oy = document.createElement('option'); oy.value=c; oy.textContent=c; if (c===pair.y) oy.selected=true; ySel.appendChild(oy);
    }});
    xSel.addEventListener('change', () => {{ GLOBAL.xys[i].x = xSel.value; }});
    ySel.addEventListener('change', () => {{ GLOBAL.xys[i].y = ySel.value; }});
    wrap.appendChild(xSel); wrap.appendChild(ySel); box.appendChild(wrap);
  }});
}}

function renderCharts() {{
  const cont = document.getElementById('charts');
  cont.innerHTML = '';
  const k = Math.max(1, Math.min(GLOBAL.windows, 8));
  const cats = GLOBAL.categories;
  const colors10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
  const colors20 = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5'];
  const base = cats.length <= 10 ? colors10 : colors20; cats.forEach((c,i) => colorMap[c] = base[i % base.length]);

  for (let i=0; i<k; i++) {{
    const div = document.createElement('div'); div.className='card';
    const title = document.createElement('div'); title.textContent = `${{GLOBAL.xys[i].x}} vs ${{GLOBAL.xys[i].y}}`; title.style.marginBottom='6px'; title.style.fontWeight='600';
    const plot = document.createElement('div'); plot.style.height='360px';
    div.appendChild(title); div.appendChild(plot); cont.appendChild(div);

    const traces = [];
    cats.forEach(cat => {{
      const xs = []; const ys = [];
      GLOBAL.data.forEach(r => {{
        if (String(r[GLOBAL.labelCol]) !== String(cat)) return;
        const xv = r[GLOBAL.xys[i].x]; const yv = r[GLOBAL.xys[i].y];
        if (xv == null || yv == null) return;
        xs.push(xv); ys.push(yv);
      }});
      traces.push({{
        x: xs, y: ys, mode: 'markers', type: 'scattergl', name: String(cat),
        marker: {{ size: GLOBAL.marker, opacity: GLOBAL.alpha, color: colorMap[cat] }},
      }});
    }});

    const layout = {{
      margin: {{ l: 40, r: 10, t: 10, b: 40 }},
      xaxis: {{ title: GLOBAL.xys[i].x, tickangle: Number(GLOBAL.rotate) }},
      yaxis: {{ title: GLOBAL.xys[i].y }},
      showlegend: true,
    }};
    Plotly.newPlot(plot, traces, layout, {{responsive: true, displayModeBar: false}});
  }}
}}

async function bootstrap() {{
  const parquet = document.getElementById('parquet').value;
  const fraction = Number(document.getElementById('fraction').value || {SAMPLE_FRACTION});
  const strat = document.getElementById('strat').value;
  const table = document.getElementById('table').value || '{TABLE_NAME}';
  const status = document.getElementById('status');
  status.textContent = 'Bootstrapping...';
  const res = await fetch('/bootstrap', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ parquet, fraction, stratify_col: strat, table }})
  }});
  if (!res.ok) {{
    status.textContent = 'Bootstrap failed: ' + await res.text();
    return;
  }}
  const payload = await res.json();
  status.textContent = `Bootstrapped ${'{'}payload.rows{'}'} rows into table ${'{'}payload.table{'}'}.`;
}}

async function loadData() {{
  const labelCol = document.getElementById('labelCol').value || '{LABEL_COL}';
  const res = await fetch(`/api/data?label_col=${{encodeURIComponent(labelCol)}}&limit=100000`);
  if (!res.ok) {{
    document.getElementById('status').textContent = 'Load failed: ' + await res.text();
    return;
  }}
  const payload = await res.json();
  GLOBAL.data = payload.rows || [];
  GLOBAL.columns = payload.columns || [];
  GLOBAL.labelCol = payload.label_col;
  GLOBAL.categories = payload.categories || [];

  const exclude = new Set([GLOBAL.labelCol]);
  const types = (function(rows, exclude) {{
    if (!rows.length) return {{ numeric: [], datetime: [] }};
    const cols = Object.keys(rows[0]).filter(c => !exclude.has(c));
    const numeric = [], datetime = [];
    cols.forEach(c => {{
      let n=0, d=0;
      for (let i=0; i<Math.min(rows.length,200); i++) {{
        const v = rows[i][c];
        if (typeof v === 'number' && Number.isFinite(v)) n++;
        else if (v != null) {{
          const asNum = Number(v); if (Number.isFinite(asNum)) n++;
          const asDate = new Date(String(v)); if (!isNaN(asDate.valueOf())) d++;
        }}
      }}
      if (d > n && d > 3) datetime.push(c);
      else if (n > 0) numeric.push(c);
    }});
    return {{ numeric, datetime }};
  }})(GLOBAL.data, exclude);

  GLOBAL.axisOptions = [...types.datetime, ...types.numeric];
  if (!GLOBAL.axisOptions.length) {{
    document.getElementById('status').textContent = 'No numeric or datetime columns detected.';
    return;
  }}

  const k = Math.max(1, Math.min(8, Math.floor(GLOBAL.axisOptions.length / 2)));
  GLOBAL.windows = k;
  GLOBAL.xys = [];
  for (let i=0; i<k; i++) {{
    GLOBAL.xys.push({{
      x: GLOBAL.axisOptions[i % GLOBAL.axisOptions.length],
      y: GLOBAL.axisOptions[(i+1) % GLOBAL.axisOptions.length],
    }});
  }}
  document.getElementById('windows').value = String(k);
  buildXYSelectors();
  renderCharts();
  document.getElementById('status').textContent = `Loaded ${{GLOBAL.data.length.toLocaleString()}} rows.`;
}}

document.getElementById('bootstrap').addEventListener('click', bootstrap);
document.getElementById('load').addEventListener('click', () => {{
  GLOBAL.windows = Math.max(1, Math.min(8, Number(document.getElementById('windows').value || 4)));
  GLOBAL.marker  = Number(document.getElementById('marker').value || 20);
  GLOBAL.alpha   = Number(document.getElementById('alpha').value || 0.6);
  GLOBAL.rotate  = Number(document.getElementById('rotate').value || 30);
  buildXYSelectors();
  loadData();
}});

// Auto-load right away (assumes DB already exists); otherwise click Bootstrap first.
loadData();
</script>
</body>
</html>
    """
    return Response(html, mimetype="text/html")

@app.post("/bootstrap")
def bootstrap() -> Response:
    """
    Bootstrap the SQLite DB table:
    - Lazy-sample the Parquet file (fraction, optional stratify_col, optional column pruning)
    - Upload to SQLite table (replace)
    """
    payload = request.get_json(force=True, silent=True) or {}
    parquet = payload.get("parquet", PARQUET_PATH)
    fraction = float(payload.get("fraction", SAMPLE_FRACTION))
    stratify_col = payload.get("stratify_col") or STRATIFY_COL
    table = payload.get("table", TABLE_NAME)

    cols = [c.strip() for c in COLUMNS_CSV.split(",") if c.strip()] if COLUMNS_CSV else None

    try:
        df = lazy_sample_parquet(
            parquet,
            fraction=max(0.0, min(1.0, fraction)),
            columns=cols,
            stratify_col=stratify_col,
        )
    except Exception as e:
        return Response(f"Sampling error: {e}", status=400)

    try:
        upload_polars_to_sqlite(df, SQLITE_DB_PATH, table, if_exists="replace")
    except Exception as e:
        return Response(f"SQLite upload error: {e}", status=500)

    return jsonify({"table": table, "rows": int(df.height)})

@app.get("/api/meta")
def api_meta() -> Response:
    with _connect_sqlite() as conn:
        cols = sqlite_columns(conn, TABLE_NAME)
        cats = detect_categories_sqlite(conn, TABLE_NAME, LABEL_COL)
    return jsonify({"db": SQLITE_DB_PATH, "table": TABLE_NAME, "columns": cols, "label_col": LABEL_COL, "categories": cats})

@app.get("/api/data")
def api_data() -> Response:
    """
    Serve data from SQLite with optional limit/offset. Defaults are generous for client-side plotting.
    """
    label_col = request.args.get("label_col", LABEL_COL)
    try:
        limit = int(request.args.get("limit", "100000"))
        limit = max(1, min(limit, 1_000_000))
    except Exception:
        limit = 100000
    try:
        offset = int(request.args.get("offset", "0"))
        offset = max(0, offset)
    except Exception:
        offset = 0

    with _connect_sqlite() as conn:
        cols = sqlite_columns(conn, TABLE_NAME)
        # read page
        q = f'SELECT * FROM "{TABLE_NAME}" LIMIT ? OFFSET ?'
        pdf = pd.read_sql_query(q, conn, params=(limit, offset))
        # total rows
        total = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{TABLE_NAME}"', conn).iloc[0]["n"]
        cats = detect_categories_sqlite(conn, TABLE_NAME, label_col)

    rows_js, dt_cols = df_to_js_records(pdf)
    return jsonify({
        "columns": cols,
        "label_col": label_col,
        "categories": cats,
        "rows": rows_js,
        "total": int(total),
        "datetime_cols": dt_cols,
    })

if __name__ == "__main__":
    # Ensure DB file exists; no-op if already there.
    if not os.path.exists(SQLITE_DB_PATH):
        # create empty DB file
        open(SQLITE_DB_PATH, "ab").close()
    app.run(host="127.0.0.1", port=5000, debug=True)
