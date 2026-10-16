"""Photon buildup-factor calculator - Streamlit front end for Part II of the ANS-6.4.3 notebook.

Run locally:   streamlit run app.py
Needs:         buildup_bundle.joblib next to this file (exported by section 25 of the notebook)
"""
from __future__ import annotations

import hmac
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import buildup_core
from buildup_core import ELEMENTS, BuildupModel

CORE_VERSION = getattr(buildup_core, "CORE_VERSION", "unknown")

HERE = Path(__file__).parent
BUNDLE_PATH = Path(os.environ.get("BUILDUP_BUNDLE", HERE / "buildup_bundle.joblib"))

# ---------------------------------------------------------------- design tokens
INK, INK2, MUTED, RULE, PAPER = "#13293d", "#4a5d70", "#7c8d9c", "#d5dde5", "#f3f5f7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
RESP_COLOR = {"EBF": SERIES[0], "EABF": SERIES[1]}
WARN = "#fab219"
BLUES = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
RESP_NAME = {"EBF": "Exposure buildup (EBF)", "EABF": "Energy-absorption buildup (EABF)"}

st.set_page_config(page_title="Photon buildup factors", page_icon="◐", layout="wide")

st.markdown(f"""
<style>
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select, label {{
  font-family: 'Archivo', system-ui, sans-serif; font-variation-settings: 'wdth' 100;
}}
.block-container {{ padding-top: 3.6rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.app-title {{ font-size: 2.35rem; line-height: 1.05; font-weight: 760; font-variation-settings: 'wdth' 118;
             margin: 0 0 .35rem 0; color: {INK}; }}
.app-lede {{ color: {INK2}; font-size: 1.02rem; max-width: 72ch; margin-bottom: 1.1rem; }}
.section-h {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: 1.2rem 0 .1rem 0; }}
.section-sub {{ color: {MUTED}; font-size: .9rem; margin: 0 0 .4rem 0; }}
.readout {{ border-left: 5px solid {INK}; padding: .3rem 0 .4rem 1.1rem; margin-top: .6rem; }}
.readout .what {{ color: {INK2}; font-size: .95rem; line-height: 1.35; }}
.readout .value {{ font-variation-settings: 'wdth' 72; font-weight: 640; font-size: 4.6rem; line-height: 1;
                  color: {INK}; font-variant-numeric: tabular-nums; margin: .35rem 0 .25rem 0; }}
.readout .value small {{ font-size: 1.5rem; font-weight: 500; color: {INK2}; margin-right: .35rem;
                         font-variation-settings: 'wdth' 100; }}
.readout .band {{ color: {INK2}; font-size: 1rem; font-variant-numeric: tabular-nums; }}
.route {{ display: inline-block; margin-top: .7rem; padding: .18rem .6rem; border: 1px solid {RULE};
          border-radius: 3px; color: {INK}; font-size: .86rem; background: #fff; }}
.route.table {{ border-color: {INK}; }}
.note {{ margin-top: .7rem; font-size: .9rem; color: {INK}; line-height: 1.4; }}
.note.warn {{ border-left: 4px solid {WARN}; padding: .35rem .7rem; background: #fff8e6; }}
.twin {{ color: {INK2}; font-size: .9rem; margin-top: .8rem; font-variant-numeric: tabular-nums; }}
.demo-bar {{ background: #fff8e6; border-left: 4px solid {WARN}; padding: .55rem .9rem; color: {INK};
            font-size: .92rem; margin-bottom: 1rem; }}
.kv {{ color: {INK2}; font-size: .93rem; line-height: 1.6; }}
.kv b {{ color: {INK}; font-weight: 620; }}
div[data-baseweb="tab-list"] button p {{ font-size: 1rem; }}
@media (max-width: 640px) {{ .readout .value {{ font-size: 3.2rem; }} .app-title {{ font-size: 1.8rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading the model")
def load_model(path: str, core_version: str, bundle_mtime: float) -> BuildupModel:
    # core_version and bundle_mtime are part of the cache key, so replacing buildup_core.py or the bundle while the
    # server is running loads a fresh model instead of reusing an object built from the old code
    return BuildupModel.load(path)


# ---------------------------------------------------------------- password gate
def configured_password() -> str | None:
    """The app password, from .streamlit/secrets.toml (app_password = "...") or the APP_PASSWORD environment variable."""
    try:
        pw = st.secrets.get("app_password")
    except Exception:          # no secrets file at all
        pw = None
    pw = pw or os.environ.get("APP_PASSWORD")
    return str(pw) if pw else None


def password_gate() -> bool:
    if st.session_state.get("unlocked"):
        return True
    expected = configured_password()
    if not expected:
        st.error("This app is locked because no password has been set. Add a line `app_password = \"your password\"` "
                 "to `.streamlit/secrets.toml` (or the Secrets box of your hosting service), or set the APP_PASSWORD "
                 "environment variable, then reload the page.")
        return False
    with st.form("login", clear_on_submit=True):
        c1, _ = st.columns([1, 2])
        with c1:
            entered = st.text_input("Password", type="password", placeholder="Enter the app password")
            go_in = st.form_submit_button("Unlock", type="primary", width="stretch")
    if go_in:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["unlocked"] = True
            st.rerun()
        time.sleep(1.5)        # slows down guessing
        st.error("That password is not correct.")
    return False


st.markdown('<div class="app-title">Photon buildup factors</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-lede">Buildup factor B(E, x) for point-isotropic gamma sources, 0.015–15 MeV and 1–40 mean free '
    'paths, for any element from beryllium to uranium plus water, air and concrete. Tabulated ANSI/ANS-6.4.3 values are '
    'returned exactly; everything else is a validated estimate with its typical error shown.</div>',
    unsafe_allow_html=True)

if not password_gate():
    st.stop()

if not BUNDLE_PATH.exists():
    st.error(f"No model file found at `{BUNDLE_PATH}`. Run section 25 of the notebook to export "
             "`buildup_bundle.joblib`, then place it next to `app.py` (or set the BUILDUP_BUNDLE environment variable).")
    st.stop()

model = load_model(str(BUNDLE_PATH), CORE_VERSION, BUNDLE_PATH.stat().st_mtime)
if not hasattr(model, "identify"):
    st.error("`buildup_core.py` is older than `app.py`. Copy both files from the same download, then stop and "
             "restart Streamlit.")
    st.stop()
META = model.meta
if META.get("demo"):
    st.markdown(f'<div class="demo-bar"><b>Demo model.</b> {META.get("data_source", "")}. Export '
                '<code>buildup_bundle.joblib</code> from the notebook with your own source workbook before relying on '
                'these numbers or deploying.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def fmt_b(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:,.1f}"
    if v >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def material_options() -> list[str]:
    listed = model.listed.sort_values("Z")
    names = list(listed.material)
    listed_z = {int(round(z)) for z, c in zip(listed.Z, listed.is_compound) if c == 0}
    others = [n for i, (_, n) in enumerate(ELEMENTS)
              if model.Z_MIN <= i + 1 <= model.Z_MAX and (i + 1) not in listed_z]
    return names + others


LISTED_NAMES = set(model.listed.material)


def mat_label(name: str) -> str:
    return f"{name}  (tabulated)" if name in LISTED_NAMES else name


def log_ticks(fig: go.Figure, axis: str) -> dict:
    """Readable ticks for a log axis: 1-2-5 for short ranges, 1-3 for medium, decades only for long ones."""
    vals = []
    for tr in fig.data:
        v = getattr(tr, axis, None)
        if v is not None and tr.type != "heatmap":
            v = pd.to_numeric(pd.Series(list(v)), errors="coerce")
            vals.append(v[v > 0].to_numpy())
    vals = np.concatenate(vals) if vals else np.array([1.0, 10.0])
    span = np.log10(vals.max() / vals.min()) if len(vals) else 1
    mult = (1, 2, 5) if span <= 3.2 else (1, 3) if span <= 4.5 else (1,)
    ticks = [m * 10.0 ** p for p in range(-3, 7) for m in mult]
    return dict(tickvals=ticks, ticktext=[f"{v:,.0f}" if v >= 1 else f"{v:g}" for v in ticks])


def base_layout(fig: go.Figure, height=380, xtitle="", ytitle="", xlog=False, ylog=False, legend=True,
                hovermode="x unified"):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Archivo, system-ui, sans-serif", size=13, color=INK2),
        hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(color=INK, family="Archivo")),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                       font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        hovermode=hovermode,
    )
    ax = dict(showline=True, linecolor=RULE, gridcolor="#e3e8ed", zeroline=False, ticks="outside",
              tickcolor=RULE, title_font=dict(color=INK2, size=13))
    fig.update_xaxes(title_text=xtitle, type="log" if xlog else "-", **ax)
    fig.update_yaxes(title_text=ytitle, type="log" if ylog else "-", **ax)
    if ylog:
        fig.update_yaxes(**log_ticks(fig, "y"))
    if xlog:
        fig.update_xaxes(**log_ticks(fig, "x"))
    return fig


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="section-h">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=512)
def cached_predict(material, energies: tuple, depths: tuple, response, method):
    return model.predict(material, np.array(energies), np.array(depths), response, method)


DEPTH_GRID = tuple(np.round(np.geomspace(model.X_MIN, model.X_MAX, 80), 5))
ENERGY_GRID = tuple(np.round(np.geomspace(model.E_MIN, model.E_MAX, 180), 7))
METHOD_HELP = {"auto": "Validated best (recommended)", "interp": "Interpolation only",
               "ml": "Extra Trees only", "blend": "Blend of both"}

tab_single, tab_batch, tab_compare, tab_identify, tab_model = st.tabs(
    ["Single point", "Batch", "Compare materials", "Identify material", "Model and data"])

# ================================================================ SINGLE POINT
def render_single_point():
    opts = material_options()
    c1, c2, c3, c4 = st.columns([2.2, 1.6, 1.2, 1.2])
    with c1:
        material = st.selectbox("Material", opts, index=opts.index("Lead") if "Lead" in opts else 0,
                                format_func=mat_label, help="Tabulated materials are listed first; any other "
                                "element from Be to U is estimated.")
    with c2:
        response = st.segmented_control("Response", ["EBF", "EABF"], default="EBF",
                                        help="EBF: exposure (air kerma). EABF: energy absorbed in the medium.")
        response = response or "EBF"
    with c3:
        energy = st.number_input("Energy (MeV)", min_value=float(model.E_MIN), max_value=float(model.E_MAX),
                                 value=0.662, step=0.1, format="%.4g")
    with c4:
        depth = st.number_input("Depth (mfp)", min_value=float(model.X_MIN), max_value=float(model.X_MAX),
                                value=10.0, step=1.0, format="%.3g")
    with st.expander("Prediction method"):
        method = st.radio("Used wherever no tabulated value exists", list(METHOD_HELP), format_func=METHOD_HELP.get,
                          horizontal=True, label_visibility="visible")
        _nm = {"ml": "Extra Trees", "interp": "interpolation", "blend": "the blend"}
        st.caption(f"The validated best route is {_nm[model.routing['energy']]} for energies between tabulated ones "
                   f"and {_nm[model.routing['element']]} for unlisted elements (see Model and data). Tabulated values "
                   "are always used where they exist.")

    try:
        point = model.predict(material, energy, depth, response, method).iloc[0]
        other_resp = "EABF" if response == "EBF" else "EBF"
        twin = model.predict(material, energy, depth, other_resp, method).iloc[0]
    except ValueError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        B = point.predicted_B
        f = point.expected_error_factor
        band = ("Tabulated value from the ANS-6.4.3 coefficients"
                if np.isnan(f) else
                f"Typical range {fmt_b(point.B_low)} – {fmt_b(point.B_high)} (±{100*(f-1):.0f} %)")
        route_cls = "route table" if point.gap == "table" else "route"
        route_txt = {"table": "From the tabulated G-P coefficients",
                     "energy": f"Energy gap: {point.source}",
                     "element": f"Unlisted element: {point.source}"}[point.gap]
        notes = ""
        if point.near_K_edge:
            ek = float(model.k_edge(point.Z)) * 1000
            notes += (f'<div class="note warn">⚠ Just above the K-edge of {point.material} ({ek:.1f} keV). The tables '
                      'have little or no clean data here, so treat this value as low confidence.</div>')
        if point.guard_correction_log10 > 0.01:
            notes += (f'<div class="note">The raw model output decreased with depth here and was raised by '
                      f'{100*(10**point.guard_correction_log10-1):.0f} % to keep B non-decreasing.</div>')
        st.markdown(f"""
<div class="readout">
  <div class="what">{RESP_NAME[response]} in {point.material} (Z = {point.Z:g})<br>at {energy:g} MeV and {depth:g} mfp</div>
  <div class="value"><small>B =</small>{fmt_b(B)}</div>
  <div class="band">{band}</div>
  <div class="{route_cls}">{route_txt}</div>
  {notes}
  <div class="twin">{other_resp} at the same point: {fmt_b(twin.predicted_B)}<br>log₁₀ B = {point.predicted_log10B:.4f}</div>
</div>""", unsafe_allow_html=True)

    with right:
        section(f"Growth with depth at {energy:g} MeV",
                "Both responses; the shaded band is the typical error of the route used.")
        fig = go.Figure()
        for resp in (response, other_resp):
            c = cached_predict(material, (energy,), DEPTH_GRID, resp, method)
            col = RESP_COLOR[resp]
            if resp == response and c.expected_error_factor.notna().any():
                fig.add_trace(go.Scatter(x=list(c.depth_mfp) + list(c.depth_mfp[::-1]),
                                         y=list(c.B_high) + list(c.B_low[::-1]), fill="toself",
                                         fillcolor="rgba(42,120,214,0.13)" if resp == "EBF" else "rgba(235,104,52,0.13)",
                                         line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=c.depth_mfp, y=c.predicted_B, name=resp, mode="lines",
                                     line=dict(color=col, width=2.4 if resp == response else 2,
                                               dash="solid" if resp == response else "dot"),
                                     hovertemplate=f"{resp} %{{y:.4g}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[depth], y=[B], mode="markers", name="This point",
                                 marker=dict(size=11, color=RESP_COLOR[response], line=dict(color="white", width=2)),
                                 hovertemplate="B = %{y:.4g} at %{x:g} mfp<extra></extra>"))
        base_layout(fig, 360, "depth (mfp)", "buildup factor B", ylog=True)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    cA, cB = st.columns(2, gap="large")
    with cA:
        section(f"Energy dependence at {depth:g} mfp",
                "Dots are tabulated coefficient sets that passed the data screens.")
        e = cached_predict(material, ENERGY_GRID, (depth,), response, method)
        fig = go.Figure()
        nk = e[e.near_K_edge]
        if len(nk):
            fig.add_vrect(x0=nk.energy_MeV.min(), x1=nk.energy_MeV.max(), fillcolor=WARN, opacity=0.16,
                          line_width=0, annotation_text="⚠ K-edge window", annotation_position="top left",
                          annotation_font=dict(color=INK, size=12))
        fig.add_trace(go.Scatter(x=e.energy_MeV, y=e.predicted_B, mode="lines", name="Prediction",
                                 line=dict(color=RESP_COLOR[response], width=2.2),
                                 customdata=e.source.tolist(), hovertemplate="%{y:.4g} at %{x:.4g} MeV<br>%{customdata}"
                                                                   "<extra></extra>"))
        if point.element_status == "listed":
            tp = model.table_points
            tp = tp[(tp.material == point.material) & (tp.response == response)]
            if len(tp):
                from buildup_core import gp_buildup
                yb = [float(gp_buildup(depth, r.b, r.c, r.a, r.d, r.Xk_mfp)) for r in tp.itertuples()]
                fig.add_trace(go.Scatter(x=tp.energy_MeV, y=yb, mode="markers", name="Tabulated",
                                         marker=dict(size=8, color="white", line=dict(color=INK, width=1.6)),
                                         hovertemplate="table %{y:.4g} at %{x:g} MeV<extra></extra>"))
        fig.add_trace(go.Scatter(x=[energy], y=[B], mode="markers", name="This point",
                                 marker=dict(size=11, color=RESP_COLOR[response], line=dict(color="white", width=2)),
                                 hovertemplate="B = %{y:.4g}<extra></extra>"))
        base_layout(fig, 340, "photon energy (MeV)", "B", xlog=True, ylog=True)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    with cB:
        section("Buildup surface", "log₁₀ B over energy and depth; the cross marks this point.")
        hE = tuple(np.round(np.geomspace(model.E_MIN, model.E_MAX, 70), 7))
        hX = tuple(np.round(np.linspace(model.X_MIN, model.X_MAX, 40), 4))
        s = cached_predict(material, hE, hX, response, method)
        Zm = s.predicted_log10B.values.reshape(len(hE), len(hX)).T
        fig = go.Figure(go.Heatmap(x=np.array(hE), y=np.array(hX), z=Zm, colorscale=BLUES,
                                   colorbar=dict(title=dict(text="log₁₀ B", side="right"), thickness=12,
                                                 outlinewidth=0),
                                   hovertemplate="%{x:.4g} MeV, %{y:.3g} mfp<br>log₁₀ B = %{z:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[energy], y=[depth], mode="markers", showlegend=False,
                                 marker=dict(symbol="cross-thin", size=18, line=dict(color="white", width=3)),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[energy], y=[depth], mode="markers", showlegend=False,
                                 marker=dict(symbol="cross-thin", size=18, line=dict(color=INK, width=1.5)),
                                 hoverinfo="skip"))
        base_layout(fig, 340, "photon energy (MeV)", "depth (mfp)", xlog=True, legend=False)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    section(f"Across the periodic table at {energy:g} MeV and {depth:g} mfp",
            "Filled dots are tabulated elements, the line covers every element from Be to U, and the ring marks "
            "your selection.")

    @st.cache_data(show_spinner=False, max_entries=64)
    def across_z(energy, depth, response, method):
        rows = []
        for z in range(model.Z_MIN, model.Z_MAX + 1):
            r = model.predict(z, energy, depth, response, method).iloc[0]
            rows.append({"Z": z, "name": r.material, "B": r.predicted_B, "status": r.element_status,
                         "source": r.source})
        return pd.DataFrame(rows)

    az = across_z(energy, depth, response, method)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=az.Z, y=az.B, mode="lines", name="All elements", line=dict(color=MUTED, width=1.6),
                             customdata=np.c_[az.name.tolist(), az.source.tolist()],
                             hovertemplate="%{customdata[0]} (Z=%{x})<br>B = %{y:.4g}<br>%{customdata[1]}"
                                           "<extra></extra>"))
    lz = az[az.status == "listed"]
    fig.add_trace(go.Scatter(x=lz.Z, y=lz.B, mode="markers", name="Tabulated elements",
                             marker=dict(size=8, color=RESP_COLOR[response], line=dict(color="white", width=1.5)),
                             customdata=lz.name.tolist(), hovertemplate="%{customdata} (Z=%{x})<br>B = %{y:.4g}<extra></extra>"))
    if point.Z == int(point.Z) and point.material.lower() not in ("water", "air", "concrete"):
        fig.add_trace(go.Scatter(x=[point.Z], y=[B], mode="markers+text", name=point.material,
                                 text=[point.material], textposition="top center",
                                 textfont=dict(color=INK, size=13),
                                 marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=INK, width=2)),
                                 hoverinfo="skip", showlegend=False))
    base_layout(fig, 320, "atomic number Z", "B", ylog=True)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


with tab_single:
    render_single_point()

# ================================================================ BATCH
with tab_batch:
    section("Predict many points at once",
            "Upload a CSV or Excel file with one point per row, or build a grid below. Columns: material "
            "(name, symbol or Z), energy_MeV, depth_mfp, and optionally response (EBF or EABF; EBF if blank).")
    template = pd.DataFrame({"material": ["Lead", "Iron", "Ni", "79", "Water", "Concrete"],
                             "energy_MeV": [0.662, 1.25, 0.35, 0.1, 2.0, 6.0],
                             "depth_mfp": [5, 10, 17.5, 3, 20, 40],
                             "response": ["EBF", "EABF", "EBF", "EBF", "EABF", "EBF"]})
    b1, b2 = st.columns([3, 1])
    with b1:
        up = st.file_uploader("Points file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    with b2:
        st.download_button("Download template", template.to_csv(index=False).encode(), "buildup_batch_template.csv",
                           "text/csv", width="stretch")

    with st.expander("Or build a grid of points"):
        g1, g2 = st.columns(2)
        with g1:
            gmats = st.multiselect("Materials", material_options(), default=["Water", "Iron", "Nickel", "Lead"])
            gresp = st.multiselect("Responses", ["EBF", "EABF"], default=["EBF"])
        with g2:
            gE = st.text_input("Energies (MeV), comma-separated", "0.1, 0.5, 1, 2, 5")
            gX = st.text_input("Depths (mfp), comma-separated", "1, 5, 10, 20, 40")
        use_grid = st.toggle("Use this grid instead of an uploaded file", value=up is None)

    bmethod = st.radio("Method wherever no tabulated value exists", list(METHOD_HELP), format_func=METHOD_HELP.get,
                       horizontal=True, key="bmethod")

    def parse_list(txt):
        vals = []
        for tok in txt.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
        return vals

    rows, problem = None, None
    try:
        if use_grid:
            Es, Xs = parse_list(gE), parse_list(gX)
            if not (gmats and gresp and Es and Xs):
                problem = "Choose at least one material and response, and enter at least one energy and depth."
            else:
                rows = pd.DataFrame([(m, e, x, r) for m in gmats for r in gresp for e in Es for x in Xs],
                                    columns=["material", "energy_MeV", "depth_mfp", "response"])
        elif up is not None:
            rows = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            rows.columns = [c.strip().lower() for c in rows.columns]
            rename = {"energy": "energy_MeV", "energy_mev": "energy_MeV", "e_mev": "energy_MeV",
                      "depth": "depth_mfp", "depth_mfp": "depth_mfp", "x_mfp": "depth_mfp", "element": "material"}
            rows = rows.rename(columns={c: rename.get(c, c) for c in rows.columns})
            missing = [c for c in ("material", "energy_MeV", "depth_mfp") if c not in rows]
            if missing:
                problem = f"The file has no column named {', '.join(missing)}. Download the template to see the layout."
                rows = None
        else:
            problem = "Upload a file or switch on the grid builder to get started."
    except ValueError:
        problem = "Energies and depths must be numbers separated by commas."
    except Exception as exc:  # unreadable file
        problem = f"The file could not be read ({exc})."

    if problem:
        st.info(problem)
    elif rows is not None:
        if len(rows) > 50000:
            st.warning(f"{len(rows):,} rows - only the first 50,000 are predicted.")
            rows = rows.head(50000)
        rows["energy_MeV"] = pd.to_numeric(rows["energy_MeV"], errors="coerce")
        with st.spinner(f"Predicting {len(rows):,} points"):
            res = model.predict_rows(rows, bmethod)
        ok = res[res.error == ""]
        bad = res[res.error != ""]
        st.markdown(
            f'<div class="kv"><b>{len(ok):,}</b> of {len(res):,} points predicted'
            + (f', <b>{len(bad):,}</b> could not be (reasons below)' if len(bad) else "")
            + (f'. <b>{int(ok.near_K_edge.sum()):,}</b> lie in a K-edge window and are low confidence'
               if len(ok) and ok.near_K_edge.any() else "")
            + (f'. <b>{int((ok.gap == "table").sum()):,}</b> are exact table values.' if len(ok) else ".")
            + "</div>", unsafe_allow_html=True)

        show = ["input_material", "input_energy_MeV", "input_depth_mfp", "input_response", "material", "Z",
                "element_status", "predicted_B", "B_low", "B_high", "expected_error_factor", "source", "near_K_edge",
                "error"]
        st.dataframe(res[[c for c in show if c in res]], width="stretch", height=330, hide_index=True,
                     column_config={
                         "predicted_B": st.column_config.NumberColumn("B", format="%.4g"),
                         "B_low": st.column_config.NumberColumn("B low", format="%.4g"),
                         "B_high": st.column_config.NumberColumn("B high", format="%.4g"),
                         "expected_error_factor": st.column_config.NumberColumn("error factor", format="%.3f"),
                         "near_K_edge": st.column_config.CheckboxColumn("K-edge window"),
                     })
        d1, d2, _ = st.columns([1, 1, 3])
        with d1:
            st.download_button("Download CSV", res.to_csv(index=False).encode(), "buildup_predictions.csv",
                               "text/csv", width="stretch")
        with d2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                res.to_excel(xw, sheet_name="Predictions", index=False)
                pd.DataFrame({"item": ["source", "model", "created", "method"],
                              "value": [META.get("data_source"), META.get("model"), META.get("created"),
                                        METHOD_HELP[bmethod]]}).to_excel(xw, sheet_name="About", index=False)
            st.download_button("Download Excel", buf.getvalue(), "buildup_predictions.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               width="stretch")

        if len(ok):
            v1, v2 = st.columns([2, 1], gap="large")
            with v1:
                section("Predictions with their typical error",
                        "Each point is one row; bars span the typical error of its route. Colour is the material, "
                        "marker shape the response (circle EBF, diamond EABF); points are nudged sideways so bars "
                        "at the same depth stay readable.")
                mats = list(dict.fromkeys(ok.material))
                top = mats[:7]
                fig = go.Figure()
                groups = top + (["Other"] if len(mats) > 7 else [])
                span = max(float(ok.depth_mfp.max() - ok.depth_mfp.min()), 1.0)
                dodge = (np.arange(len(groups)) - (len(groups) - 1) / 2) * span * 0.012   # small horizontal offset
                for k, m in enumerate(groups):
                    sub = ok[ok.material == m] if m != "Other" else ok[~ok.material.isin(top)]
                    for resp, sym in (("EBF", "circle"), ("EABF", "diamond")):
                        s2 = sub[sub.response == resp]
                        if s2.empty:
                            continue
                        fig.add_trace(go.Scatter(
                            x=s2.depth_mfp + dodge[k], y=s2.predicted_B, mode="markers", name=f"{m} {resp}",
                            marker=dict(size=9, symbol=sym, color=SERIES[k] if m != "Other" else MUTED,
                                        line=dict(color="white", width=1.2)),
                            error_y=dict(type="data", symmetric=False, array=s2.B_high - s2.predicted_B,
                                         arrayminus=s2.predicted_B - s2.B_low, thickness=1.2, width=0,
                                         color=SERIES[k] if m != "Other" else MUTED),
                            customdata=np.c_[s2.energy_MeV.to_numpy(), s2.source.to_numpy(dtype=object), s2.depth_mfp.to_numpy()],
                            hovertemplate=f"{m} {resp}<br>%{{customdata[0]:g}} MeV, %{{customdata[2]:g}} mfp<br>"
                                          "B = %{y:.4g}<br>%{customdata[1]}<extra></extra>"))
                base_layout(fig, 380, "depth (mfp)", "B", ylog=True)
                fig.update_layout(hovermode="closest")
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
            with v2:
                section("How each value was obtained")
                cnt = ok.source.value_counts().sort_values()
                fig = go.Figure(go.Bar(x=cnt.values.tolist(), y=cnt.index.tolist(), orientation="h", marker=dict(color=SERIES[0]),
                                       text=cnt.values.tolist(), textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x} points<extra></extra>"))
                base_layout(fig, 380, "points", "", legend=False)
                fig.update_layout(hovermode="closest", bargap=0.45)
                fig.update_yaxes(showgrid=False, ticks="")
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        if len(bad):
            section("Rows that could not be predicted")
            st.dataframe(bad[["input_material", "input_energy_MeV", "input_depth_mfp", "input_response", "error"]],
                         width="stretch", hide_index=True)

# ================================================================ COMPARE
with tab_compare:
    section("Compare materials", "Solid lines are tabulated materials, dashed lines are estimated (unlisted) elements.")
    k1, k2, k3 = st.columns([3, 1.3, 2])
    with k1:
        cmats = st.multiselect("Materials (up to 8)", material_options(), max_selections=8,
                               default=["Water", "Aluminum", "Iron", "Nickel", "Lead"])
    with k2:
        cresp = st.segmented_control("Response", ["EBF", "EABF"], default="EBF", key="cresp") or "EBF"
    with k3:
        axis = st.segmented_control("Plot against", ["Energy", "Depth"], default="Energy", key="caxis") or "Energy"
    if axis == "Energy":
        fixed = st.select_slider("Depth (mfp)", options=[1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40], value=10)
    else:
        fixed = st.select_slider("Energy (MeV)", options=[0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2,
                                                         0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                         8.0, 10.0, 15.0], value=1.0)
    if not cmats:
        st.info("Pick at least one material to compare.")
    else:
        fig = go.Figure()
        table, ends = {}, []
        for k, m in enumerate(cmats):
            if axis == "Energy":
                c = cached_predict(m, ENERGY_GRID, (float(fixed),), cresp, "auto")
                xv = c.energy_MeV
            else:
                c = cached_predict(m, (float(fixed),), DEPTH_GRID, cresp, "auto")
                xv = c.depth_mfp
            unl = c.element_status.iloc[0] == "unlisted"
            fig.add_trace(go.Scatter(x=xv, y=c.predicted_B, mode="lines", name=m + (" (estimated)" if unl else ""),
                                     line=dict(color=SERIES[k], width=2.2, dash="dash" if unl else "solid"),
                                     hovertemplate=f"{m}: %{{y:.4g}}<extra></extra>"))
            ends.append((np.log10(c.predicted_B.iloc[-1]), m, np.log10(xv.iloc[-1]) if axis == "Energy" else xv.iloc[-1]))
            table[m] = c.predicted_B.values
        # direct labels at the line ends, spread apart so they never overlap
        allv = np.log10(np.concatenate([v for v in table.values()]))
        gap = max(allv.max() - allv.min(), 0.5) * 0.045
        ends.sort()
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        shift = max(0.0, ys[-1] - max(e[0] for e in ends)) / 2 if ys else 0
        for (y0, m, x0), y in zip(ends, ys):
            fig.add_annotation(x=x0, y=y - shift, text=m, showarrow=False, xanchor="left", xshift=6,
                               font=dict(color=INK2, size=12))
        xtitle = "photon energy (MeV)" if axis == "Energy" else "depth (mfp)"
        base_layout(fig, 470, xtitle, f"{cresp}", xlog=axis == "Energy", ylog=True)
        fig.update_layout(margin=dict(r=90))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        with st.expander("Show the numbers"):
            idx = np.array(ENERGY_GRID if axis == "Energy" else DEPTH_GRID)
            tdf = pd.DataFrame(table, index=pd.Index(idx, name="energy_MeV" if axis == "Energy" else "depth_mfp"))
            st.dataframe(tdf.iloc[::4].style.format("{:.4g}"), width="stretch")

# ================================================================ IDENTIFY MATERIAL
with tab_identify:
    section("Identify the material from buildup values",
            "Enter buildup factors you know (measured, simulated or quoted) with their energy and depth. Every "
            "element from Be to U, plus water, air and concrete, is tried in both EBF and EABF, and ranked by how "
            "well its curve passes through your points, allowing for your measurement uncertainty and the model's "
            "own typical error.")

    EXAMPLES = {
        "Iron, EBF at 1 MeV": ("Iron", [1.0], [2, 5, 10, 20], "EBF"),
        "An unlisted element (antimony), EABF at 0.5 MeV": ("Antimony", [0.5], [1, 4, 10, 25], "EABF"),
        "Lead, EBF at two energies": ("Lead", [0.5, 2.0], [2, 6, 15], "EBF"),
        "Water, EABF at 0.3 MeV": ("Water", [0.3], [1, 3, 7, 15], "EABF"),
        "Start empty": None,
    }

    @st.cache_data(show_spinner=False)
    def example_points(key):
        ex = EXAMPLES[key]
        if ex is None:
            return pd.DataFrame({"energy_MeV": [np.nan] * 3, "depth_mfp": [np.nan] * 3, "B": [np.nan] * 3})
        mat, Es, xs, resp = ex
        rng = np.random.default_rng(11)
        rows = []
        for E in Es:
            p = model.predict(mat, E, xs, resp)
            for x, b in zip(p.depth_mfp, p.predicted_B):
                rows.append((E, x, round(float(b * np.exp(rng.normal(0, 0.035))), 4)))
        return pd.DataFrame(rows, columns=["energy_MeV", "depth_mfp", "B"])

    i_left, i_right = st.columns([1, 1.35], gap="large")
    with i_left:
        ex_key = st.selectbox("Start from", list(EXAMPLES), help="Examples are generated by the model with a few "
                              "per cent of random scatter; replace them with your own values.")
        pts = st.data_editor(example_points(ex_key), num_rows="dynamic", width="stretch", hide_index=True,
                             key=f"pts_{ex_key}",
                             column_config={
                                 "energy_MeV": st.column_config.NumberColumn("Energy (MeV)", min_value=model.E_MIN,
                                                                             max_value=model.E_MAX, format="%.4g"),
                                 "depth_mfp": st.column_config.NumberColumn("Depth (mfp)", min_value=model.X_MIN,
                                                                            max_value=model.X_MAX, format="%.3g"),
                                 "B": st.column_config.NumberColumn("Buildup factor B", min_value=1.0, format="%.4g")})
        unc = st.slider("Uncertainty of your values (%)", 1, 30, 5,
                        help="One standard deviation. Use your measurement or simulation uncertainty.")
        o1, o2 = st.columns(2)
        with o1:
            iresp = st.segmented_control("Response", ["Unknown", "EBF", "EABF"], default="Unknown",
                                         key="iresp") or "Unknown"
        with o2:
            iscope = st.segmented_control("Candidates", ["All", "Tabulated only"], default="All",
                                          key="iscope") or "All"
        st.caption("Depth must be in mean free paths, not centimetres. More depths, and points at more than one "
                   "energy, separate similar materials better.")

    result, ierr = None, None
    try:
        result = model.identify(pts, meas_unc=unc / 100,
                                responses=("EBF", "EABF") if iresp == "Unknown" else (iresp,),
                                scope="listed" if iscope == "Tabulated only" else "all")
    except ValueError as exc:
        ierr = str(exc)

    with i_right:
        if ierr:
            st.info(ierr)
        else:
            R = result["ranking"]
            best = result["best"]
            z_lo, z_hi = result["z_range"]

            def zname(z):
                for nm, zc in model.compound_z.items():
                    if np.isclose(z, zc):
                        return nm.capitalize()
                return ELEMENTS[int(round(z)) - 1][1]

            resp_line = {
                "EBF": f"The values behave like exposure buildup (EBF): {100*result['p_response'].get('EBF', 0):.0f} % "
                       "of the match weight.",
                "EABF": f"The values behave like energy-absorption buildup (EABF): "
                        f"{100*result['p_response'].get('EABF', 0):.0f} % of the match weight.",
                "undecided": f"EBF and EABF fit about equally well ({100*result['p_response'].get('EBF', 0):.0f} % vs "
                             f"{100*result['p_response'].get('EABF', 0):.0f} %); these points cannot tell them apart.",
            }.get(result["response_verdict"], f"Response fixed to {iresp}.")
            warn = ""
            if result["poor_fit"]:
                warn = ('<div class="note warn">⚠ No candidate fits well: the best one is off by '
                        f'{100*best["typical_deviation"]:.0f} % on average against your stated {unc} %. Check '
                        'the units (depth in mfp), the energy, and whether the material could be a mixture.</div>')
            st.markdown(f"""
<div class="readout">
  <div class="what">Effective atomic number from {result['n_points']} points
  at {', '.join(f'{e:g}' for e in result['energies'])} MeV</div>
  <div class="value"><small>Z ≈</small>{result['z_mean']:.0f}</div>
  <div class="band">90 % range Z = {z_lo:g} – {z_hi:g} ({zname(z_lo)} to {zname(z_hi)})</div>
  <div class="route">Most likely: {best['material']}, {best['response']} ({100*best['probability']:.0f} %)</div>
  <div class="note">{resp_line}</div>
  {warn}
</div>""", unsafe_allow_html=True)

            top = R.head(10).copy()
            top["probability"] = 100 * top.probability
            top["typical_deviation"] = 100 * top.typical_deviation
            top["kind"] = np.where(top.compound, "compound", np.where(top.tabulated, "tabulated", "estimated"))
            st.dataframe(top[["material", "Z", "response", "probability", "typical_deviation", "kind"]],
                         hide_index=True, width="stretch", height=250,
                         column_config={
                             "Z": st.column_config.NumberColumn("Z", format="%.3g", width="small"),
                             "probability": st.column_config.ProgressColumn("match weight", format="%.0f %%",
                                                                            min_value=0, max_value=100),
                             "typical_deviation": st.column_config.NumberColumn("typical misfit", format="%.1f %%"),
                         })

    if result is not None:
        R = result["ranking"]
        P = result["points"]
        f1, f2 = st.columns(2, gap="large")
        with f1:
            Es_in = list(result["energies"])
            e_show = Es_in[0] if len(Es_in) == 1 else st.selectbox(
                "Energy to plot", Es_in, format_func=lambda e: f"{e:g} MeV", key="ie_show")
            section(f"Your points against the three best matches at {e_show:g} MeV",
                    "Error bars are your stated uncertainty. Solid lines are EBF, dotted lines EABF.")
            fig = go.Figure()
            for k, row in enumerate(R.head(3).itertuples()):
                c = cached_predict(row.material, (float(e_show),), DEPTH_GRID, row.response, "auto")
                fig.add_trace(go.Scatter(x=c.depth_mfp, y=c.predicted_B, mode="lines",
                                         name=f"{row.material} {row.response} ({100*row.probability:.0f} %)",
                                         line=dict(color=SERIES[k], width=2.2,
                                                   dash="solid" if row.response == "EBF" else "dot"),
                                         hovertemplate=f"{row.material} {row.response}: %{{y:.4g}}<extra></extra>"))
            sp = P[np.isclose(P.energy_MeV, e_show)]
            fig.add_trace(go.Scatter(x=sp.depth_mfp, y=sp.B, mode="markers", name="Your values",
                                     marker=dict(size=10, color=INK, line=dict(color="white", width=1.5)),
                                     error_y=dict(type="data", symmetric=False, array=sp.B * unc / 100,
                                                  arrayminus=sp.B * (1 - 1 / (1 + unc / 100)), color=INK,
                                                  thickness=1.3, width=4),
                                     hovertemplate="your B = %{y:.4g} at %{x:g} mfp<extra></extra>"))
            base_layout(fig, 380, "depth (mfp)", "B", ylog=True, hovermode="closest")
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        with f2:
            section("Match weight across the periodic table",
                    "How much of the total match weight each atomic number receives; the band is the 90 % range.")
            fig = go.Figure()
            fig.add_vrect(x0=result["z_range"][0] - 0.5, x1=result["z_range"][1] + 0.5, fillcolor=SERIES[0],
                          opacity=0.08, line_width=0)
            el = R[~R.compound]
            for resp in ("EBF", "EABF"):
                s = el[el.response == resp].sort_values("Z")
                if s.empty:
                    continue
                fig.add_trace(go.Scatter(x=s.Z, y=100 * s.probability, mode="lines", name=resp,
                                         line=dict(color=RESP_COLOR[resp], width=2, shape="hvh"),
                                         customdata=s.material.tolist(),
                                         hovertemplate="%{customdata} (Z=%{x}) " + resp + ": %{y:.1f} %<extra></extra>"))
            cp = R[R.compound]
            if len(cp):
                fig.add_trace(go.Scatter(x=cp.Z, y=100 * cp.probability, mode="markers", name="Compounds",
                                         marker=dict(size=9, symbol="diamond", color="white",
                                                     line=dict(color=INK, width=1.5)),
                                         customdata=np.c_[cp.material.tolist(), cp.response.tolist()],
                                         hovertemplate="%{customdata[0]} %{customdata[1]} (Zeff %{x}): "
                                                       "%{y:.1f} %<extra></extra>"))
            base_layout(fig, 380, "atomic number Z (effective for compounds)", "match weight (%)",
                        hovermode="closest")
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    with st.expander("Identify many samples from one file"):
        st.markdown('<div class="kv">One row per point, with a <b>sample</b> column naming which curve it belongs '
                    'to: sample, energy_MeV, depth_mfp, B. Each sample is identified separately with the settings '
                    'above.</div>', unsafe_allow_html=True)
        tmpl = pd.concat([example_points(k).assign(sample=k.split(",")[0])
                          for k in list(EXAMPLES)[:4]])[["sample", "energy_MeV", "depth_mfp", "B"]]
        u1, u2 = st.columns([3, 1])
        with u1:
            iup = st.file_uploader("Samples file", type=["csv", "xlsx", "xls"], key="iup", label_visibility="collapsed")
        with u2:
            st.download_button("Download template", tmpl.to_csv(index=False).encode(), "identify_template.csv",
                               "text/csv", width="stretch")
        if iup is not None:
            try:
                sdf = pd.read_csv(iup) if iup.name.lower().endswith(".csv") else pd.read_excel(iup)
                sdf.columns = [str(c).strip() for c in sdf.columns]
                if "sample" not in sdf.columns:
                    raise ValueError("The file needs a column named sample.")
                with st.spinner("Identifying samples"):
                    ires = model.identify_groups(sdf, meas_unc=unc / 100,
                                                 responses=("EBF", "EABF") if iresp == "Unknown" else (iresp,),
                                                 scope="listed" if iscope == "Tabulated only" else "all")
                st.dataframe(ires, hide_index=True, width="stretch",
                             column_config={"probability": st.column_config.NumberColumn(format="%.2f"),
                                            "runner_up_probability": st.column_config.NumberColumn(format="%.2f"),
                                            "P_EBF": st.column_config.NumberColumn(format="%.2f"),
                                            "typical_deviation": st.column_config.NumberColumn(format="%.3f")})
                st.download_button("Download results", ires.to_csv(index=False).encode(),
                                   "identification_results.csv", "text/csv")
            except Exception as exc:
                st.info(f"The file could not be used: {exc}")

    with st.expander("How reliable is identification?"):
        st_path = HERE / "identify_selftest.json"
        stt = None
        if st_path.exists():
            import json
            stt = json.loads(st_path.read_text())
            if stt.get("bundle_created") != META.get("created"):
                stt = None
        if stt is None:
            st.markdown('<div class="kv">Run <code>python identify_selftest.py</code> next to the app after '
                        'exporting a new model file; the results appear here.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="kv">In {stt['trials']} trials, a real tabulated curve was read at
{stt['points_per_trial']} random depths with {100*stt['scatter']:.0f} % random scatter, and the answer was hidden:<br>
<b>{100*stt['true_Z_inside_90pct_range']:.0f} %</b> of the time the true atomic number was inside the 90 % range,
and the effective-Z estimate was typically off by <b>{stt['median_abs_Z_error']:.1f}</b>
({100*stt['median_rel_Z_error']:.0f} % of Z).<br>
The exact material and response were ranked first <b>{100*stt['exact_match_top1']:.0f} %</b> of the time and in the
top five <b>{100*stt['exact_match_top5']:.0f} %</b>; neighbouring elements often have near-identical curves, which is
why the range matters more than the single best name.<br>
Response: called correctly <b>{100*stt['response_correct']:.0f} %</b>, called wrongly
<b>{100*stt['response_wrong']:.0f} %</b>, left undecided <b>{100*stt['response_undecided']:.0f} %</b>. EBF and EABF are
often too close to separate, and the app says so rather than guessing.<br>
Match weights treat each point's error as independent; real model errors at neighbouring depths move together, so
read them as a ranking rather than exact odds.</div>""", unsafe_allow_html=True)

# ================================================================ MODEL & DATA
with tab_model:
    tables = model.bundle["tables"]
    name_of = {"ml": "Extra Trees", "interp": "Interpolation", "blend": "Blend (ET + interp)"}
    m1, m2 = st.columns([1.15, 1], gap="large")
    with m1:
        section("How a value is produced")
        fid = META.get("lookup_fidelity", {})
        st.markdown(f"""
<div class="kv">
<b>Tabulated material at a tabulated energy:</b> the ANS-6.4.3 geometric-progression formula of that coefficient set,
exact at any depth.<br>
<b>Tabulated material, any other energy</b> (or a tabulated energy whose set is missing or failed the screens):
{name_of[model.routing['energy']]}.<br>
<b>Element not in the tables:</b> {name_of[model.routing['element']]}.<br>
These routes won the held-out comparison in section 18 of the notebook; they are recomputed whenever the notebook is
re-run. Every result is kept at B ≥ 1 and non-decreasing in depth.<br><br>
<b>Data:</b> {META.get('data_source', '')}. {META.get('n_sets_after_s4', '?')} coefficient sets passed the basic
checks, {META.get('n_flagged_s15', '?')} more were removed by the physical screens, leaving
{META.get('n_clean_sets', '?')}.<br>
<b>Model:</b> {META.get('model', '')}. Lookup vs the model itself: median difference
{100*(10**fid.get('median_abs_log10', 0)-1):.2f} %, 95th percentile {100*(10**fid.get('p95_abs_log10', 0)-1):.1f} %
in B.<br>
<b>Exported:</b> {META.get('created', '')} with scikit-learn {META.get('sklearn_version', '')}.
</div>""", unsafe_allow_html=True)
    with m2:
        section("Typical error by depth", "From data the model never saw. Error grows with depth because deep "
                "buildup amplifies small coefficient differences.")
        xs = np.geomspace(model.X_MIN, model.X_MAX, 60)
        fig = go.Figure()
        for k, (g, lbl) in enumerate([("energy", "Energy gap"), ("element", "Unlisted element")]):
            fac = model.expected_factor(g, xs)
            fig.add_trace(go.Scatter(x=xs, y=100 * (fac - 1), mode="lines", name=lbl,
                                     line=dict(color=SERIES[k], width=2.2),
                                     hovertemplate=f"{lbl}: ±%{{y:.1f}} % at %{{x:.3g}} mfp<extra></extra>"))
        base_layout(fig, 300, "depth (mfp)", "typical error in B (%)")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    section("Limits")
    st.markdown("""<div class="kv">Z outside 4–92 (H, He, Li, and anything heavier than U) is refused; compounds other
than water, air and concrete are not supported. Values for unlisted elements are estimates with no table value to
check against. Energies just above heavy-element K-edges are flagged: the source data there is sparse or rejected.
The source coefficients come from a 1988 preliminary draft of ANSI/ANS-6.4.3, so verify licensing-grade work against
the final standard. The small steps some curves show at 1.022 MeV come from the model's pair-production feature,
which switches on at that threshold; real buildup is continuous there.</div>""", unsafe_allow_html=True)

    with st.expander("Method comparison on held-out data"):
        st.dataframe(tables["choice"], width="stretch", hide_index=True)
        st.dataframe(tables["validation"], width="stretch", hide_index=True)
    with st.expander("Typical error by depth, every method"):
        st.dataframe(tables["by_depth"], width="stretch", hide_index=True)
    with st.expander("Rebuilding a missing tabulated set"):
        st.dataframe(tables["imputation"], width="stretch", hide_index=True)
    with st.expander(f"Coefficient sets removed by the screens ({len(tables['screen_flags'])})"):
        st.dataframe(tables["screen_flags"], width="stretch", hide_index=True)
    with st.expander("K-edge diagnostic"):
        st.dataframe(tables["kedge_diag"], width="stretch", hide_index=True)

st.divider()
if st.button("Lock the app", help="Ask for the password again in this browser tab."):
    st.session_state.pop("unlocked", None)
    st.rerun()
