

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any, Tuple

from flask import Flask, jsonify, Response, request
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import dotenv

# Load .env if present
dotenv.load_dotenv()

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    # Fallback only for local dev; prefer setting POSTGRES_URI in .env
    "postgresql+psycopg2://user:password@host:5432/dbname"
)

SCHEMA = os.getenv("SCHEMA", None)
TABLE_NAME = os.getenv("TABLE_NAME", "sampled_regimes")
LABEL_COL  = os.getenv("LABEL_COL", "cluster")

# Maximum rows the API will serve at once (safety cap)
API_ROWS_CAP = int(os.getenv("API_ROWS_CAP", "50000"))

def _append_conn_params(uri: str) -> str:
    # keepalives improve stability behind proxies; sslmode is required on many hosted PGs
    sep = "&" if "?" in uri else "?"
    if "sslmode=" not in uri:
        uri += sep + "sslmode=require"
        sep = "&"
    if "keepalives=" not in uri:
        uri += sep + "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
    return uri

def make_engine() -> Engine:
    """Create a SQLAlchemy engine with SSL + keepalives."""
    uri = _append_conn_params(POSTGRES_URI)
    return create_engine(uri, pool_pre_ping=True)

ENGINE: Engine = make_engine()

def fq_table(table: str, schema: Optional[str] = SCHEMA) -> str:
    """Fully-qualified table name with optional schema, safe-quoted."""
    return f'"{schema}"."{table}"' if schema else f'"{table}"'

def pg_columns(engine: Engine, table: str, schema: Optional[str] = SCHEMA) -> List[str]:
    """Return column names for a Postgres table (schema-aware)."""
    sql = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table
          AND (:schema IS NULL OR table_schema = :schema)
        ORDER BY ordinal_position
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"table": table, "schema": schema}).fetchall()
    return [r[0] for r in rows]

def detect_categories_postgres(engine: Engine, table: str, label_col: str, schema: Optional[str] = SCHEMA) -> List[str]:
    """Fetch up to 50 distinct values for label_col (schema-aware)."""
    target = fq_table(table, schema)
    sql = text(f'SELECT DISTINCT "{label_col}" FROM {target} LIMIT 50')
    try:
        with engine.begin() as conn:
            rows = conn.execute(sql).fetchall()
        return [str(r[0]) for r in rows]
    except Exception:
        return []

def df_to_js_records(pdf: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert pandas DataFrame to JSON-serializable records; return rows + list of datetime columns."""
    dt_cols = [c for c in pdf.columns if np.issubdtype(pdf[c].dtype, np.datetime64)]
    rows: List[Dict[str, Any]] = []
    it = pdf.itertuples(index=False, name=None)
    for row in it:
        rec: Dict[str, Any] = {}
        for c, v in zip(pdf.columns, row):
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

# Flask app

app = Flask(__name__)

@app.get("/")
def index() -> Response:
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Regime Dashboard </title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {{
      --gap: 12px;
      --card-border: #e5e7eb;
    }}
    body {{ font-family: system-ui, sans-serif; margin: 16px; }}
    .row {{ display:flex; gap:var(--gap); flex-wrap:wrap; align-items:end; }}
    .card {{ border:1px solid var(--card-border); border-radius:10px; padding:12px; margin-bottom:12px; background:#fff; }}
    .title {{ font-weight:600; font-size:18px; margin-bottom:8px; }}
    .muted {{ color:#666; font-size:12px; }}
    input, select {{ padding:6px; }}
    .btn {{ padding:8px 12px; background:#111827; color:#fff; border:none; border-radius:8px; cursor:pointer; }}

    /* selectors first, charts below */
    .stack {{ display: grid; gap: var(--gap); }}

    /* sticky so selectors remain visible while scrolling charts */
    .sticky {{ position: sticky; top: 8px; z-index: 2; background: #fff; }}

    /* XY selectors: 2 columns × 4 rows (max 8 pairs) */
    .xy-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(220px, 1fr));
      gap: 12px;
    }}
    .xy-cell {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      align-items: center;
      border: 1px solid #cbd5e1;     /* clearer border around a pair */
      border-radius: 8px;
      padding: 10px;
      background: #fafafa;
    }}
    .xy-label {{
      grid-column: 1 / -1;
      font-size: 12px;
      color: #374151;
      margin-bottom: 2px;
      font-weight: 600;
    }}

    /* charts grid */
    .charts-grid {{ display:grid; grid-template-columns:repeat(2, minmax(300px,1fr)); gap:var(--gap); }}
    @media (max-width: 1100px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
      .xy-grid {{ grid-template-columns: 1fr; }}  /* stack selectors on narrow screens */
    }}
  </style>
</head>
<body>
  <div class="title">Regime Dashboard</div>

  <div class="card row">
    <div>
      <label>Label column</label><br/>
      <input id="labelCol" value="{LABEL_COL}"/>
    </div>
    <div>
      <label>Limit</label><br/>
      <input id="limit" type="number" min="1" max="{API_ROWS_CAP}" value="{API_ROWS_CAP}"/>
    </div>
    <div>
      <label>Offset</label><br/>
      <input id="offset" type="number" min="0" value="0"/>
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
    <div style="flex:1">
      <div id="status" class="muted">Connected to Postgres. Table: {(SCHEMA + "." if SCHEMA else "") + TABLE_NAME}</div>
    </div>
  </div>

  <div class="stack">
    <!-- selectors on top -->
    <section class="card sticky">
      <div style="font-weight:600; margin-bottom:6px;">X/Y selectors</div>
      <div id="xy-grid" class="xy-grid"></div>
    </section>

    <!-- charts below -->
    <section class="card">
      <div style="font-weight:600; margin-bottom:6px;">Charts</div>
      <div id="charts" class="charts-grid"></div>
    </section>
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

/* Default axis helpers */
function findPCAColumns(cols) {{
  // Return PCA columns sorted by trailing index if possible (pca_component_1, pca_component_2, ...)
  const pcs = cols.filter(c => /^pca_component_\\d+$/i.test(c));
  return pcs.sort((a,b) => {{
    const ia = parseInt(a.split("_").pop() || "0", 10);
    const ib = parseInt(b.split("_").pop() || "0", 10);
    return ia - ib;
  }});
}}

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

function pickDefaultPair(i) {{
  const cols = GLOBAL.axisOptions;
  const hasImpl = cols.includes('impl_volatility');
  const pcas = findPCAColumns(cols);
  const yPCA = (pcas[i] || pcas[0] || cols[(i+1) % cols.length] || cols[0]);
  const x = hasImpl ? 'impl_volatility' : (cols[i % cols.length] || cols[0]);
  const y = yPCA;
  return {{ x, y }};
}}

function buildXYSelectors() {{
  const grid = document.getElementById('xy-grid');
  grid.innerHTML = '';
  const maxPairs = 8;
  const k = Math.max(1, Math.min(Number(GLOBAL.windows) || 1, maxPairs));
  GLOBAL.xys = GLOBAL.xys.slice(0, k);

  while (GLOBAL.xys.length < k) {{
    const i = GLOBAL.xys.length;
    const defPair = pickDefaultPair(i);
    GLOBAL.xys.push(defPair);
  }}

  for (let i=0; i<k; i++) {{
    const cell = document.createElement('div');
    cell.className = 'xy-cell';

    const label = document.createElement('div');
    label.className = 'xy-label';
    label.textContent = `Pair #${{i+1}}`;

    const xSel = document.createElement('select');
    const ySel = document.createElement('select');

    GLOBAL.axisOptions.forEach(c => {{
      const ox = document.createElement('option'); ox.value=c; ox.textContent=c; if (c===GLOBAL.xys[i].x) ox.selected=true; xSel.appendChild(ox);
      const oy = document.createElement('option'); oy.value=c; oy.textContent=c; if (c===GLOBAL.xys[i].y) oy.selected=true; ySel.appendChild(oy);
    }});

    xSel.addEventListener('change', () => {{ GLOBAL.xys[i].x = xSel.value; }});
    ySel.addEventListener('change', () => {{ GLOBAL.xys[i].y = ySel.value; }});

    cell.appendChild(label);
    cell.appendChild(xSel);
    cell.appendChild(ySel);
    grid.appendChild(cell);
  }}
}}

function renderCharts() {{
  const cont = document.getElementById('charts');
  cont.innerHTML = '';
  const k = Math.max(1, Math.min(Number(GLOBAL.windows) || 1, 8));
  const cats = GLOBAL.categories;
  const colors10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
  const colors20 = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5'];
  const base = cats.length <= 10 ? colors10 : colors20;
  const colorMap = {{}}; cats.forEach((c,i) => colorMap[c] = base[i % base.length]);

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

async function loadData() {{
  const labelCol = document.getElementById('labelCol').value || '{LABEL_COL}';
  const limit = Math.min(Number(document.getElementById('limit').value || {API_ROWS_CAP}), {API_ROWS_CAP});
  const offset = Number(document.getElementById('offset').value || 0);
  const status = document.getElementById('status');
  status.textContent = 'Loading...';
  const res = await fetch(`/api/data?label_col=${{encodeURIComponent(labelCol)}}&limit=${{encodeURIComponent(limit)}}&offset=${{encodeURIComponent(offset)}}`);
  if (!res.ok) {{
    status.textContent = 'Load failed: ' + await res.text();
    return;
  }}
  const payload = await res.json();
  GLOBAL.data = payload.rows || [];
  GLOBAL.columns = payload.columns || [];
  GLOBAL.labelCol = payload.label_col;
  GLOBAL.categories = payload.categories || [];

  const exclude = new Set([GLOBAL.labelCol]);
  const types = detectTypes(GLOBAL.data, exclude);
  GLOBAL.axisOptions = [...types.datetime, ...types.numeric];
  if (!GLOBAL.axisOptions.length) {{
    status.textContent = 'No numeric or datetime columns detected.';
    return;
  }}

  const k = Math.max(1, Math.min(8, Math.floor(GLOBAL.axisOptions.length / 2)));
  GLOBAL.windows = k;
  GLOBAL.xys = []; // rebuild pairs using defaults
  for (let i=0; i<k; i++) {{
    GLOBAL.xys.push(pickDefaultPair(i));
  }}
  document.getElementById('windows').value = String(k);

  buildXYSelectors();
  renderCharts();
  status.textContent = `Loaded ${{GLOBAL.data.length.toLocaleString()}} rows.`;
}}

document.getElementById('load').addEventListener('click', () => {{
  GLOBAL.windows = Math.max(1, Math.min(8, Number(document.getElementById('windows').value || 4)));
  GLOBAL.marker  = Number(document.getElementById('marker').value || 20);
  GLOBAL.alpha   = Number(document.getElementById('alpha').value || 0.6);
  GLOBAL.rotate  = Number(document.getElementById('rotate').value || 30);
  buildXYSelectors();
  renderCharts();
}});

// init
buildXYSelectors();
loadData();
</script>
</body>
</html>
    """
    return Response(html, mimetype="text/html")

@app.get("/api/data")
def api_data() -> Response:
    """
    Serve data from Postgres with limit/offset (schema-aware).
    """
    label_col = request.args.get("label_col", LABEL_COL)
    try:
        limit = int(request.args.get("limit", str(API_ROWS_CAP)))
        limit = max(1, min(limit, API_ROWS_CAP))
    except Exception:
        limit = API_ROWS_CAP
    try:
        offset = int(request.args.get("offset", "0"))
        offset = max(0, offset)
    except Exception:
        offset = 0

    cols = pg_columns(ENGINE, TABLE_NAME, SCHEMA)
    target = fq_table(TABLE_NAME, SCHEMA)

    q = text(f"SELECT * FROM {target} LIMIT :limit OFFSET :offset")
    q_count = text(f"SELECT COUNT(*) AS n FROM {target}")

    with ENGINE.begin() as conn:
        pdf = pd.read_sql_query(q, conn, params={"limit": limit, "offset": offset})
        total = conn.execute(q_count).scalar_one()
        cats = detect_categories_postgres(ENGINE, TABLE_NAME, label_col, SCHEMA)

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
    # Optional: explicitly point to a non-default .env path:
    # dotenv.load_dotenv(".env")  # already loaded above; uncomment to force reload
    app.run(host="127.0.0.1", port=5000, debug=True)
