from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import MaxNLocator
import ipywidgets as widgets
from IPython.display import display, clear_output
from .kscan import KScanResult

def to_pandas_minimal(df, cols: List[str], label_col: str) -> pd.DataFrame:
    cols = list(dict.fromkeys(cols))
    out = df.select(cols).to_pandas(use_pyarrow_extension_array=False)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    if label_col in out.columns:
        out[label_col] = pd.Categorical(out[label_col])
    return out

def build_plot_cols(feature_set: List[str], label_col: str = "cluster", max_feats: int = 8) -> List[str]:
    feats = [c for c in feature_set if c != label_col][:max_feats]
    return [label_col] + feats

def regime_dashboard_windows(
    df: pd.DataFrame,
    label_col: str,
    candidate_columns: Optional[List[str]] = None,
    max_charts: int = 8,
    title_prefix: str = "Regime Characteristics Over Time",
    palette=None,
    include_object_date_guess: bool = False,
):
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in dataframe.")
    if candidate_columns is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]", "datetimetz"]).columns.tolist()
        obj_date_like = []
        if include_object_date_guess:
            for c in df.select_dtypes(include=["object"]).columns.tolist():
                try:
                    pd.to_datetime(df[c].dropna().astype(str).head(5), errors="raise")
                    obj_date_like.append(c)
                except Exception:
                    pass
        candidate_columns = list(dict.fromkeys(dt_cols + obj_date_like + numeric))
    axis_options = [c for c in candidate_columns if c != label_col]
    if not axis_options:
        raise ValueError("No numeric/datetime columns available for axes (after excluding label column).")

    regimes = pd.Categorical(df[label_col]); cats = list(regimes.categories)
    if palette is None:
        base = plt.cm.tab10.colors if len(cats) <= 10 else plt.cm.tab20.colors
        color_map = {cat: base[i % len(base)] for i, cat in enumerate(cats)}
    else:
        base = plt.cm.tab10.colors
        color_map = {cat: palette.get(cat, base[i % len(base)]) for i, cat in enumerate(cats)}

    n_charts = widgets.IntSlider(value=min(4, max(1, len(axis_options)//2)),
                                 min=1, max=max(1, max_charts), step=1,
                                 description="Windows", continuous_update=False)
    width_slider  = widgets.IntSlider(value=15, min=8, max=24, step=1, description="Fig width",  continuous_update=False)
    height_slider = widgets.IntSlider(value=10, min=6, max=20, step=1, description="Fig height", continuous_update=False)
    size_slider   = widgets.IntSlider(value=20, min=5, max=80, step=1, description="Marker size", continuous_update=False)
    alpha_slider  = widgets.FloatSlider(value=0.6, min=0.05, max=1.0, step=0.05, description="Alpha", continuous_update=False)
    title_text    = widgets.Text(value=title_prefix, description="Suptitle", continuous_update=False)
    max_xticks    = widgets.IntSlider(value=6, min=3, max=15, step=1, description="Max X ticks", continuous_update=False)
    rotate_xticks = widgets.IntSlider(value=30, min=0, max=90, step=5, description="Rotate X°", continuous_update=False)
    concise_dates = widgets.Checkbox(value=True, description="Concise datetime format")
    selectors_box = widgets.VBox()

    def _nice_grid(n):
        if n <= 1: return (1, 1)
        cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
        return rows, cols

    def make_xy_row(i):
        return widgets.HBox([
            widgets.Dropdown(options=axis_options, value=axis_options[min(i, len(axis_options)-1)],
                             description=f"X{i+1}", layout=widgets.Layout(width="50%")),
            widgets.Dropdown(options=axis_options, value=axis_options[(i+1) % len(axis_options)],
                             description=f"Y{i+1}", layout=widgets.Layout(width="50%")),
        ])

    def refresh_selectors(*_):
        selectors_box.children = [make_xy_row(i) for i in range(n_charts.value)]

    n_charts.observe(lambda change: refresh_selectors(), names="value")
    refresh_selectors()
    out = widgets.Output()

    def _is_datetime_series(s: pd.Series) -> bool:
        return pd.api.types.is_datetime64_any_dtype(s)

    def _format_axes(ax, x_series: pd.Series):
        if _is_datetime_series(x_series):
            locator = mdates.AutoDateLocator(minticks=3, maxticks=max_xticks.value)
            formatter = mdates.ConciseDateFormatter(locator) if concise_dates.value else mdates.AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks.value, prune='both'))
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotate_xticks.value)
            tick.set_horizontalalignment("right" if rotate_xticks.value else "center")

    def render(*_):
        with out:
            clear_output(wait=True)
            k = n_charts.value
            rows, cols = _nice_grid(k)
            fig, axes = plt.subplots(rows, cols, figsize=(width_slider.value, height_slider.value))
            axes_list = [axes] if rows*cols == 1 else np.array(axes).reshape(-1).tolist()
            for i in range(k):
                ax = axes_list[i]
                x_col = selectors_box.children[i].children[0].value
                y_col = selectors_box.children[i].children[1].value
                cols_sel = list(dict.fromkeys([x_col, y_col, label_col]))
                valid = df[cols_sel].dropna()
                lab = valid[label_col]
                if isinstance(lab, pd.DataFrame):
                    lab = lab.iloc[:, 0]
                for cat in cats:
                    mask = (lab == cat)
                    if not mask.any(): continue
                    sub = valid.loc[mask]
                    ax.scatter(sub[x_col].values, sub[y_col].values,
                               s=size_slider.value, alpha=alpha_slider.value,
                               c=[color_map[cat]], label=str(cat))
                ax.set_title(f"{x_col} vs {y_col}"); ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.grid(True, alpha=0.3)
                _format_axes(ax, valid[x_col])
            for j in range(k, rows*cols): axes_list[j].axis("off")
            fig.suptitle(f"{title_text.value} — {label_col}", y=0.995)
            handles, labels = [], []
            for ax in axes_list[:k]:
                h, l = ax.get_legend_handles_labels(); handles += h; labels += l
            if handles:
                seen, h_u, l_u = set(), [], []
                for h, l in zip(handles, labels):
                    if l not in seen: seen.add(l); h_u.append(h); l_u.append(l)
                fig.legend(h_u, l_u, title=str(label_col), loc="upper right", ncol=min(len(l_u), 5), bbox_to_anchor=(1, 1))
            fig.tight_layout(rect=(0, 0, 1, 0.97)); plt.subplots_adjust(bottom=0.12); plt.show()

    btn = widgets.Button(description="Render", button_style="primary", icon="refresh")
    btn.on_click(lambda _: render())
    ui = widgets.VBox([
        widgets.HBox([n_charts, size_slider, alpha_slider]),
        widgets.HBox([max_xticks, rotate_xticks, concise_dates]),
        widgets.HBox([width_slider, height_slider, title_text]),
        selectors_box, btn, out
    ])
    display(ui)
    return ui
