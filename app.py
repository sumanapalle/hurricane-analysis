# app.py — Hurricane Dashboard (Dash) for Hugging Face Spaces (Docker)

import os
import re, requests, io, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from dash import dash_table
from dash.dash_table import FormatTemplate
from urllib.parse import urlparse, parse_qs

# Filter out the specific openpyxl UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')

# ----------------------------- Config / Sources ------------------------------ #
AOML_URL = "https://www.aoml.noaa.gov/hrd/hurdat/SeasonalVerification.html"

METRIC_LABELS = {
    "NS":  "Named Storms",
    "HU":  "Hurricanes",
    "MH":  "Major Hurricanes",
    "ACE": "Accumulated Cyclone Energy"
}

OBS_COLOR = "#d62728"
ISSUER_COLORS = {"NOAA": "#1f77b4", "CSU": "#2c7fb8", "TSR": "#74add1"}

# ------------------------- CERF allocations (with storm info) ---------------- #
# NOTE: Amounts stored as integers for currency formatting performance.
CERF_ROWS = [
    {"Application Code":"14-RR-HTI-9285","Country":"Haiti","Window":"RR","Emergency Type":"Cholera","Emergency Group for Global Reporting":"Disease Outbreak","Year":2014,"Amount Approved":2668206,"Region":"Latin America and the Caribbean","Storm":"","Storm Type":"Hurricane Season","Category":""},
    {"Application Code":"14-UFE-HTI-7984","Country":"Haiti","Window":"UF","Emergency Type":"Cholera","Emergency Group for Global Reporting":"Disease Outbreak","Year":2014,"Amount Approved":6205232,"Region":"Latin America and the Caribbean","Storm":"","Storm Type":"Hurricane Season","Category":""},
    {"Application Code":"16-RR-HTI-23486","Country":"Haiti","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2016,"Amount Approved":3544711,"Region":"Latin America and the Caribbean","Storm":"Matthew","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"16-RR-HTI-22873","Country":"Haiti","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2016,"Amount Approved":6838529,"Region":"Latin America and the Caribbean","Storm":"Matthew","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"16-RR-CUB-22839","Country":"Cuba","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2016,"Amount Approved":5352736,"Region":"Latin America and the Caribbean","Storm":"Matthew","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"17-RR-DMA-27733","Country":"Dominica","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2017,"Amount Approved":3011838,"Region":"Latin America and the Caribbean","Storm":"Maria","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"17-RR-ATG-27500","Country":"Antigua and Barbuda","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2017,"Amount Approved":2154461,"Region":"Latin America and the Caribbean","Storm":"Irma","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"17-RR-CUB-27383","Country":"Cuba","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2017,"Amount Approved":7999469,"Region":"Latin America and the Caribbean","Storm":"Irma","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"18-UF-HTI-28521","Country":"Haiti","Window":"UF","Emergency Type":"Multiple Emergencies","Emergency Group for Global Reporting":"Multiple Emergencies","Year":2018,"Amount Approved":8985177,"Region":"Latin America and the Caribbean","Storm":"Matthew","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"19-RR-BHS-38922","Country":"Bahamas","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2019,"Amount Approved":1002151,"Region":"Latin America and the Caribbean","Storm":"Dorian","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"20-RR-NIC-46275","Country":"Nicaragua","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2020,"Amount Approved":2000000,"Region":"Latin America and the Caribbean","Storm":"Eta, Iota","Storm Type":"Hurricane","Category":"4, 4"},
    {"Application Code":"20-RR-HND-45959","Country":"Honduras","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2020,"Amount Approved":3901926,"Region":"Latin America and the Caribbean","Storm":"Iota","Storm Type":"Hurricane","Category":"4"},
    {"Application Code":"20-RR-SLV-43848","Country":"El Salvador","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2020,"Amount Approved":2999884,"Region":"Latin America and the Caribbean","Storm":"Amanda, Cristóbal","Storm Type":"Tropical Storm","Category":""},
    {"Application Code":"21-RR-COL-49434","Country":"Colombia","Window":"RR","Emergency Type":"Flood","Emergency Group for Global Reporting":"Natural Disaster","Year":2021,"Amount Approved":2006312,"Region":"Latin America and the Caribbean","Storm":"","Storm Type":"Hurricane Season","Category":""},
    {"Application Code":"21-RR-HTI-48843","Country":"Haiti","Window":"RR","Emergency Type":"Earthquake","Emergency Group for Global Reporting":"Natural Disaster","Year":2021,"Amount Approved":7872943,"Region":"Latin America and the Caribbean","Storm":"","Storm Type":"Hurricane Season","Category":""},
    {"Application Code":"21-RR-HTI-48351","Country":"Haiti","Window":"RR","Emergency Type":"Violence/Clashes","Emergency Group for Global Reporting":"Conflict-related","Year":2021,"Amount Approved":998966,"Region":"Latin America and the Caribbean","Storm":"","Storm Type":"Hurricane Season","Category":""},
    {"Application Code":"22-RR-CUB-55712","Country":"Cuba","Window":"RR","Emergency Type":"Storm","Emergency Group for Global Reporting":"Natural Disaster","Year":2022,"Amount Approved":7827734,"Region":"Latin America and the Caribbean","Storm":"Ian","Storm Type":"Hurricane","Category":"5"},
    {"Application Code":"22-UF-HND-51277","Country":"Honduras","Window":"UF","Emergency Type":"Economic Disruption","Emergency Group for Global Reporting":"Unspecified Emergency","Year":2022,"Amount Approved":4994779,"Region":"Latin America and the Caribbean","Storm":"Eta, Iota","Storm Type":"Hurricane","Category":"4, 4"},
]
CERF_DF = pd.DataFrame(CERF_ROWS)

# ---------------------------- AOML (forecasts/obs) --------------------------- #
def _clean_multiindex(mi):
    tuples = []
    for a, b, c in mi.to_list():
        def norm(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): return ""
            s = str(x).strip()
            return "" if s.startswith("Unnamed") else s
        tuples.append((norm(a), norm(b), norm(c)))
    return pd.MultiIndex.from_tuples(tuples, names=["issuer", "lead", "metric"])

def load_aoml():
    df = pd.read_html(AOML_URL, header=[0, 1, 2])[0]
    df.columns = _clean_multiindex(df.columns)
    year_col = [c for c in df.columns if str(c[0]).strip().lower() == "year"][0]
    long = (df.set_index([year_col])
              .stack(level=[0, 1, 2], future_stack=True)
              .rename_axis(["year", "issuer", "lead", "metric"])
              .reset_index(name="value"))
    def norm_issuer(x: str) -> str:
        s = (x or "").replace("\n"," ").strip().lower()
        if s.startswith("observed"): return "Observed"
        if "colorado state" in s: return "CSU"
        if "noaa" in s: return "NOAA"
        if "tropical storm risk" in s or "tsr" in s: return "TSR"
        return x or "Observed"
    def norm_lead(x: str) -> str:
        m = re.search(r"(Early|Late)", x or "", flags=re.I)
        return (m.group(1).title() if m else x) or ""
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["issuer"] = long["issuer"].map(norm_issuer)
    long["lead"] = long["lead"].map(norm_lead)
    long.loc[long["issuer"]=="Observed","lead"]="Observed"
    long["metric"] = long["metric"].astype(str).str.strip()
    long = long[long["metric"].ne("")]
    long["is_delta"] = long["metric"].str.startswith("Δ")
    long["metric"] = (long["metric"].str.replace("Δ","",regex=False)
                                   .str.replace("%ACE","ACE_pct",regex=False).str.strip())
    long["value"] = pd.to_numeric(long["value"].replace({"—":"","-":""}), errors="coerce")
    obs_long = long[(long["issuer"]=="Observed") & (~long["is_delta"])].dropna(subset=["metric","value"]).copy()
    obs_long = obs_long.groupby(["year","metric"], as_index=False)["value"].mean()
    observed = (obs_long.pivot(index="year", columns="metric", values="value")
                        .reset_index().rename_axis(None,axis=1).sort_values("year")
                        .filter(["year","NS","HU","MH","ACE"]))
    forecasts = long[(long["issuer"].isin(["CSU","NOAA","TSR"])) & (~long["is_delta"])].copy()
    forecasts["metric"] = pd.Categorical(forecasts["metric"], categories=["NS","HU","MH","ACE","ACE_pct"])
    return forecasts, observed

def build_master():
    return load_aoml()

forecasts, observed = build_master()
YEARS = sorted(observed["year"].dropna().astype(int).tolist())
MINY, MAXY = YEARS[0], YEARS[-1]
DEFAULT_START = max(MINY, 2000)
DEFAULT_END   = min(MAXY, 2025)

# -------- Frameworks (Active / Under Development) ingestion & shaping -------- #
def extract_src_from_office_online_url(office_online_url):
    parsed = urlparse(office_online_url)
    q = parse_qs(parsed.query)
    return q.get("src", [None])[0]

frameworks_dev_url = "https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.anticipation-hub.org%2FDocuments%2FReports%2FOverview-report-2024%2FTable_A3._Frameworks_under_development_in_2024_FINAL_TABLE.xlsx&wdOrigin=BROWSELINK"
frameworks_active_url = "https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.anticipation-hub.org%2FDocuments%2FReports%2FOverview-report-2024%2FTable_A2._Active_frameworks_in_2024_FINAL_TABLE.xlsx&wdOrigin=BROWSELINK"

direct_dev = extract_src_from_office_online_url(frameworks_dev_url)
direct_act = extract_src_from_office_online_url(frameworks_active_url)

df_frameworks_combined = pd.DataFrame()
df_country_framework_status = pd.DataFrame()

if direct_dev and direct_act:
    try:
        r_dev = requests.get(direct_dev, timeout=60); r_dev.raise_for_status()
        r_act = requests.get(direct_act, timeout=60); r_act.raise_for_status()
        df_frameworks_dev = pd.read_excel(io.BytesIO(r_dev.content), engine="openpyxl")
        df_frameworks_act = pd.read_excel(io.BytesIO(r_act.content), engine="openpyxl")

        def clean_cols(cols):
            out=[]
            for c in cols:
                s = re.sub(r"[^\w_]", "", str(c).lower().strip().replace(" ","_").replace(".","_"))
                s = re.sub(r"^\d+_","",s)
                out.append(s)
            return out

        df_frameworks_dev.columns = clean_cols(df_frameworks_dev.columns)
        df_frameworks_act.columns = clean_cols(df_frameworks_act.columns)
        df_frameworks_dev["status"] = "Under Development"
        df_frameworks_act["status"] = "Active"
        df_frameworks_combined = pd.concat([df_frameworks_dev, df_frameworks_act], ignore_index=True)

        cerf_countries = CERF_DF["Country"].unique()
        hurricane_countries = ['Colombia', 'Cuba', 'Haiti', 'Honduras']
        hnrp_countries = ['Colombia', 'El Salvador', 'Haiti', 'Honduras']

        rows=[]
        for country in cerf_countries:
            act = df_frameworks_combined[(df_frameworks_combined.get("_country")==country) & (df_frameworks_combined["status"]=="Active")].copy()
            dev = df_frameworks_combined[(df_frameworks_combined.get("_country")==country) & (df_frameworks_combined["status"]=="Under Development")].copy()

            act_list=[]
            for _,r in act.iterrows():
                hazard = r.get("_hazard","N/A Hazard")
                org = r.get("_coordinating_organizations", "N/A Org")
                act_list.append(f"{hazard} ({org})")

            dev_list=[]
            for _,r in dev.iterrows():
                hazard = r.get("_hazard","N/A Hazard")
                org = r.get("_implementing_organizations","N/A Org")
                dev_list.append(f"{hazard} ({org})")

            rows.append({
                "Country": country,
                "Active Framework in 2024": ", ".join(act_list) if act_list else "No",
                "Framework Under Development": ", ".join(dev_list) if dev_list else "No",
                "Hurricane or Related": "Yes" if country in hurricane_countries else "No",
                "HNRP status": "Yes" if country in hnrp_countries else "No",
            })
        df_country_framework_status = pd.DataFrame(rows)
    except Exception as e:
        print("Framework ingest error:", e)

# ----------------------- ENSO (ONI v5 -> bands & helpers) --------------------
ONI_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
SEASONS = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
SEASON_TO_MONTHS = {
    "DJF": (12, 1, 2), "JFM": (1, 2, 3),  "FMA": (2, 3, 4),  "MAM": (3, 4, 5),
    "AMJ": (4, 5, 6),  "MJJ": (5, 6, 7),  "JJA": (6, 7, 8),  "JAS": (7, 8, 9),
    "ASO": (8, 9, 10), "SON": (9, 10, 11),"OND": (10, 11, 12),"NDJ": (11, 12, 1),
}

# --- Coarse (run-peak) intensity for main chart ---
def _intensity_bucket(peak_abs):
    if peak_abs >= 2.0: return "very strong"
    if peak_abs >= 1.5: return "strong"
    if peak_abs >= 1.0: return "moderate"
    return "weak"

ENSO_FILL = {
    ("El Niño", "weak"):        "rgba(255,165,0,0.15)",
    ("El Niño", "moderate"):    "rgba(255,140,0,0.22)",
    ("El Niño", "strong"):      "rgba(255,99, 0,0.30)",
    ("El Niño", "very strong"): "rgba(200,60,  0,0.38)",
    ("La Niña", "weak"):        "rgba(100,149,237,0.15)",
    ("La Niña", "moderate"):    "rgba(30, 144,255,0.22)",
    ("La Niña", "strong"):      "rgba(0,  114,206,0.30)",
    ("La Niña", "very strong"): "rgba(0,   76,160,0.38)",
}
ENSO_TEXT = {"El Niño": "rgb(150,70,0)", "La Niña": "rgb(10,80,160)"}

# --- Detailed (monthly-intensity) for lazy chart ---
def _intensity_bucket_month(abs_oni: float) -> str:
    if abs_oni >= 2.0:  return "very strong"
    if abs_oni >= 1.5:  return "strong"
    if abs_oni >= 1.0:  return "moderate"
    if abs_oni >= 0.8:  return "weak"
    return "very weak"  # 0.5–0.7

ENSO_FILL_DETAILED = {
    ("El Niño", "very weak"):   "rgba(255,190,120,0.12)",
    ("El Niño", "weak"):        "rgba(255,165,  0,0.18)",
    ("El Niño", "moderate"):    "rgba(255,140,  0,0.24)",
    ("El Niño", "strong"):      "rgba(255, 99,  0,0.32)",
    ("El Niño", "very strong"): "rgba(200, 60,  0,0.40)",
    ("La Niña", "very weak"):   "rgba(173,216,230,0.12)",
    ("La Niña", "weak"):        "rgba(100,149,237,0.18)",
    ("La Niña", "moderate"):    "rgba( 30,144,255,0.24)",
    ("La Niña", "strong"):      "rgba(  0,114,206,0.32)",
    ("La Niña", "very strong"): "rgba(  0, 76,160,0.40)",
}

# --- ONI loader (common) ---
_ONI_CACHE = None
def _load_oni_table():
    """Fetch & parse CPC ONI v5 table into tidy long form (year, season, oni)."""
    global _ONI_CACHE
    if _ONI_CACHE is not None:
        return _ONI_CACHE
    r = requests.get(ONI_URL, timeout=30); r.raise_for_status()
    tables = pd.read_html(r.text, flavor="bs4", header=0)
    oni = None
    for t in tables:
        cols = [str(c).strip().upper() for c in t.columns]
        if "YEAR" in cols and all(s in cols for s in SEASONS):
            oni = t.copy(); break
    if oni is None:
        raise RuntimeError("Could not find ONI table on CPC page.")
    oni = oni.rename(columns={"Year":"year"})[["year"] + SEASONS]
    long = oni.melt(id_vars="year", var_name="season", value_name="oni")
    long["oni"] = pd.to_numeric(long["oni"], errors="coerce")
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long = long.dropna(subset=["year"])
    long["sidx"] = long["season"].map({s:i for i,s in enumerate(SEASONS)})
    long = long.sort_values(["year","sidx"]).reset_index(drop=True)
    _ONI_CACHE = long
    return long

# --- Derive coarse global bands (main chart) ---
def _derive_enso_bands_global(long, thr_el=0.5, thr_la=-0.5, min_run=5):
    df = long.copy()
    df["phase"] = None
    df.loc[df["oni"] >= thr_el, "phase"] = "El Niño"
    df.loc[df["oni"] <= thr_la, "phase"] = "La Niña"
    df["run_id"] = (df["phase"].ne(df["phase"].shift())).cumsum()

    qualified = []
    for _, g in df.groupby("run_id", sort=False):
        ph = g["phase"].iloc[0]
        if ph in ("El Niño","La Niña") and len(g) >= min_run:
            peak_abs = float(np.nanmax(np.abs(g["oni"].values)))
            intensity = _intensity_bucket(peak_abs)
            gg = g.copy(); gg["intensity"] = intensity
            qualified.append((ph, gg, intensity))
    if not qualified:
        return []

    rows = []
    for phase, g, intensity in qualified:
        for _, row in g.iterrows():
            y = int(row["year"]); season = row["season"]
            for m in SEASON_TO_MONTHS[season]:
                yy = y
                if season == "DJF" and m == 12: yy = y - 1
                if season == "NDJ" and m == 1:  yy = y + 1
                rows.append((yy, int(m), phase, intensity))

    mm = (pd.DataFrame(rows, columns=["year","month","phase","intensity"])
            .dropna(subset=["year","month"]).drop_duplicates())
    mm = mm[mm["month"].between(1,12)]
    if mm.empty:
        return []

    mm["tidx"] = mm["year"] * 12 + (mm["month"] - 1)
    mm = mm.sort_values("tidx").reset_index(drop=True)

    bands = []
    start_idx = prev_idx = None
    cur_phase = cur_int = None
    for _, r in mm.iterrows():
        tidx, ph, inten = int(r["tidx"]), r["phase"], r["intensity"]
        if start_idx is None:
            start_idx = prev_idx = tidx; cur_phase = ph; cur_int = inten; continue
        if (ph == cur_phase) and (inten == cur_int) and (tidx == prev_idx + 1):
            prev_idx = tidx
        else:
            bands.append((start_idx, prev_idx, cur_phase, cur_int))
            start_idx = prev_idx = tidx; cur_phase = ph; cur_int = inten
    bands.append((start_idx, prev_idx, cur_phase, cur_int))

    out = []
    for s, e, phase, inten in bands:
        sy, sm0 = divmod(s, 12); ey, em0 = divmod(e, 12)
        x0 = sy + sm0/12.0
        x1 = ey + (em0+1)/12.0
        out.append({"x0": x0, "x1": x1, "phase": phase, "intensity": inten})
    return out

def _add_month_bands_global(fig, global_bands, y0, y1, show_labels=True):
    """Draw global run-peak bands (neutral omitted); optional vertical labels."""
    view_min = float(y0)
    view_max = float(y1) + 1.0
    for b in global_bands:
        x0, x1, phase, intensity = b["x0"], b["x1"], b["phase"], b["intensity"]
        if x1 <= view_min or x0 >= view_max:
            continue
        cx0, cx1 = max(x0, view_min), min(x1, view_max)
        fill = ENSO_FILL.get((phase, intensity), "rgba(0,0,0,0.12)")
        tcol = ENSO_TEXT.get(phase, "#333")
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=cx0, x1=cx1, y0=0, y1=1,
            fillcolor=fill, line_width=0, layer="below"
        )
        if show_labels:
            fig.add_annotation(
                x=(cx0+cx1)/2.0, y=1.02, xref="x", yref="paper",
                text=f"{phase} ({intensity})",
                showarrow=False, textangle=-90,
                font=dict(size=10, color=tcol),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.15)", borderwidth=1, borderpad=1,
            )

# --- Derive and draw detailed monthly-intensity bands (for lazy chart) ---
def _derive_enso_bands_monthly_intensity(long, thr_el=0.5, thr_la=-0.5, min_run=5):
    df = long.copy()
    df["phase"] = None
    df.loc[df["oni"] >= thr_el, "phase"] = "El Niño"
    df.loc[df["oni"] <= thr_la, "phase"] = "La Niña"
    df["run_id"] = (df["phase"].ne(df["phase"].shift())).cumsum()

    qualified = []
    for _, g in df.groupby("run_id", sort=False):
        ph = g["phase"].iloc[0]
        if ph in ("El Niño", "La Niña") and len(g) >= min_run:
            qualified.append((ph, g.copy()))
    if not qualified:
        return []

    rows = []
    for phase, g in qualified:
        for _, row in g.iterrows():
            y = int(row["year"]); season = row["season"]; oni = float(row["oni"])
            for m in SEASON_TO_MONTHS[season]:
                yy = y
                if season == "DJF" and m == 12: yy = y - 1
                if season == "NDJ" and m == 1:  yy = y + 1
                rows.append((yy, int(m), phase, oni))

    mm = (pd.DataFrame(rows, columns=["year","month","phase","oni"])
            .dropna(subset=["year","month"]).drop_duplicates())
    mm = mm[mm["month"].between(1,12)]
    if mm.empty:
        return []

    mm["intensity"] = mm["oni"].abs().apply(_intensity_bucket_month)
    mm["tidx"] = mm["year"] * 12 + (mm["month"] - 1)
    mm = mm.sort_values("tidx").reset_index(drop=True)

    # Compress contiguous months with same (phase,intensity)
    bands = []
    start_idx = prev_idx = None
    cur_phase = cur_int = None
    for _, r in mm.iterrows():
        tidx, ph, inten = int(r["tidx"]), r["phase"], r["intensity"]
        if start_idx is None:
            start_idx = prev_idx = tidx; cur_phase = ph; cur_int = inten; continue
        if (ph == cur_phase) and (inten == cur_int) and (tidx == prev_idx + 1):
            prev_idx = tidx
        else:
            bands.append((start_idx, prev_idx, cur_phase, cur_int))
            start_idx = prev_idx = tidx; cur_phase = ph; cur_int = inten
    bands.append((start_idx, prev_idx, cur_phase, cur_int))

    out = []
    for s, e, phase, inten in bands:
        sy, sm0 = divmod(s, 12); ey, em0 = divmod(e, 12)
        x0 = sy + sm0/12.0
        x1 = ey + (em0+1)/12.0
        out.append({"x0": x0, "x1": x1, "phase": phase, "intensity": inten})
    return out

def _add_month_bands_detailed(fig, monthly_bands, y0, y1):
    view_min = float(y0)
    view_max = float(y1) + 1.0
    for b in monthly_bands:
        x0, x1, phase, intensity = b["x0"], b["x1"], b["phase"], b["intensity"]
        if x1 <= view_min or x0 >= view_max:
            continue
        cx0, cx1 = max(x0, view_min), min(x1, view_max)
        fill = ENSO_FILL_DETAILED.get((phase, intensity), "rgba(0,0,0,0.12)")
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=cx0, x1=cx1, y0=0, y1=1,
            fillcolor=fill, line_width=0, layer="below"
        )

def _add_enso_legend_traces(fig):
    """Simplified legend: red/orange = El Niño, blue = La Niña."""
    legend_items = [("El Niño", "#ff7f0e"), ("La Niña", "#1f77b4")]
    for name, col in legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=col, line=dict(width=0)),
            name=name, hoverinfo="skip", showlegend=True,
        ))

# --- Pre-compute ONI -> bands once ---
try:
    _ONI_LONG = _load_oni_table()
    _ENSO_BANDS_GLOBAL  = _derive_enso_bands_global(_ONI_LONG, thr_el=0.5, thr_la=-0.5, min_run=5)
    _ENSO_BANDS_MONTHLY = _derive_enso_bands_monthly_intensity(_ONI_LONG, thr_el=0.5, thr_la=-0.5, min_run=5)
except Exception as _e:
    print("ONI ingest error:", _e)
    _ENSO_BANDS_GLOBAL, _ENSO_BANDS_MONTHLY = [], []

# --------------------------------- App Layout -------------------------------- #
app = Dash(__name__, title="Seasonal Hurricanes – Forecasts vs Observed")
server = app.server  # <-- required for gunicorn/Spaces

money = FormatTemplate.money(0)

app.layout = html.Div([
    html.H1("Seasonal Hurricanes — Forecasts vs Observed"),

    # Controls
    html.Div([
        html.Div([html.Label("Metric"),
            dcc.Dropdown(id="metric",
                options=[{"label": METRIC_LABELS[m], "value": m} for m in ["NS","HU","MH","ACE"]],
                value="NS", clearable=False)], style={"width":"28%","display":"inline-block","padding":"0 8px"}),

        html.Div([html.Label("Forecast Lead"),
            dcc.RadioItems(id="lead",
                options=[{"label":"Early (pre-season)","value":"Early"},
                         {"label":"Late (Aug update)","value":"Late"}],
                value="Early", inline=True)], style={"width":"34%","display":"inline-block","padding":"0 8px"}),

        html.Div([html.Label("Issuers"),
            dcc.Checklist(id="issuers",
                options=[{"label": lab, "value": lab} for lab in ["NOAA","CSU","TSR"]],
                value=["NOAA","CSU","TSR"], inline=True)], style={"width":"38%","display":"inline-block","padding":"0 8px"}),
    ], style={"background":"#f7f7f9","borderRadius":"12px","padding":"10px","marginBottom":"10px"}),

    # Year range
    html.Div([
        html.Label("Year Range"),
        dcc.RangeSlider(id="year_range", min=MINY, max=MAXY, step=1,
                        value=[DEFAULT_START, DEFAULT_END],
                        marks={y: str(y) for y in range(MINY, MAXY+1)})
    ], style={"padding":"8px 8px 0", "marginBottom":"10px"}),

    # Main chart
    dcc.Graph(id="forecast_vs_observed", config={"displayModeBar": False}),

    # ---- Accuracy toggle ----
    html.Div([
        dcc.Checklist(
            id="show_accuracy",
            options=[{"label": " Show Forecast Accuracy (table + deltas)", "value": "on"}],
            value=[],
            inputStyle={"marginRight":"6px"},
            style={"margin":"8px 0"}
        )
    ]),

    # ---- Accuracy summary table ----
    html.Div(id="accuracy_section", children=[
        html.H3("Forecast Accuracy Summary", style={"marginBottom":"6px"}),
        dash_table.DataTable(
            id="accuracy_table",
            columns=[
                {"name":"Issuer", "id":"issuer"},
                {"name":"Mean Absolute Error (MAE) ⓘ", "id":"mae", "type":"numeric"},
                {"name":"Root Mean Squared Error (RMSE) ⓘ", "id":"rmse", "type":"numeric"},
                {"name":"Bias (Mean Error) ⓘ", "id":"me", "type":"numeric"},
                {"name":"Correlation (Pearson r) ⓘ", "id":"r", "type":"numeric"},
                {"name":"n (years)", "id":"n", "type":"numeric"}
            ],
            data=[],
            sort_action="native",
            style_table={"overflowX":"auto", "marginBottom":"10px"},
            style_cell={"fontFamily":"-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif","fontSize":"14px","padding":"6px"},
            style_header={"backgroundColor":"#f4f6f8","fontWeight":"600"},
            tooltip_header={
                "mae":  "MAE = (1/n) Σ |forecast − observed|\n\nLower is better.",
                "rmse": "RMSE = √[(1/n) Σ (forecast − observed)²]\n\nLower is better.",
                "me":   "ME = (1/n) Σ (forecast − observed)\n\nNegative = under-forecast; Positive = over-forecast.",
                "r":    "Pearson r = corr(forecast, observed)\n\nCloser to 1 is better."
            },
            tooltip_delay=0,
            tooltip_duration=None,
        )
    ], style={"display":"none"}),

    # ---- Delta plot ----
    html.Div(id="residual_section", children=[
        html.H3("Deltas by Year (Forecast − Observed)"),
        dcc.Graph(id="residual_graph", config={"displayModeBar": False})
    ], style={"display":"none"}),

    html.Hr(),

    # ---- ENSO Granularity (lazy) ----
    html.Div([
        html.Button("View ENSO Granularity", id="enso_detail_btn",
                    n_clicks=0,
                    style={"padding":"8px 12px","borderRadius":"8px","border":"1px solid #ccc","background":"#fff","cursor":"pointer"}),
        html.Div(id="enso_detail_section", style={"display":"none", "marginTop":"10px"}, children=[
            html.H3("View ENSO Granularity"),
            dcc.Loading(
                id="enso_detail_loading",
                type="default",
                children=dcc.Graph(id="enso_detail_graph", config={"displayModeBar": False})
            )
        ])
    ], style={"marginBottom":"16px"}),

    # ---- CERF table (hide Region; show Storm info) ----
    html.H3("CERF Allocations — Latin America & Caribbean"),
    dash_table.DataTable(
        id="cerf_table",
        columns=[
            {"name": "Year", "id": "Year", "type": "numeric"},
            {"name": "Application Code", "id": "Application Code"},
            {"name": "Country", "id": "Country"},
            {"name": "Window", "id": "Window"},
            {"name": "Emergency Type", "id": "Emergency Type"},
            {"name": "Emergency Group for Global Reporting", "id": "Emergency Group for Global Reporting"},
            {"name": "Storm", "id": "Storm"},
            {"name": "Storm Type", "id": "Storm Type"},
            {"name": "Category", "id": "Category"},
            {"name": "Amount Approved (USD)", "id": "Amount Approved", "type":"numeric", "format": money},
        ],
        data=CERF_DF.sort_values(["Year","Country"]).to_dict("records"),
        sort_action="native", filter_action="native", page_size=10,
        style_table={"overflowX": "auto", "marginBottom": "20px"},
        style_cell={"fontFamily":"-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif","fontSize":"14px","padding":"6px"},
        style_header={"backgroundColor":"#f4f6f8","fontWeight":"600"},
        style_data_conditional=[
            {"if":{"column_id":"Year"}, "textAlign":"center"},
            {"if":{"column_id":"Amount Approved"}, "textAlign":"right"},
        ],
    ),

    # ---- Country Framework Status Table ----
    html.H3("Other AA Frameworks"),
    dash_table.DataTable(
        id="country_framework_status_table",
        columns=[
            {"name": "Country", "id": "Country"},
            {"name": "Active Framework in 2024", "id": "Active Framework in 2024"},
            {"name": "Framework Under Development", "id": "Framework Under Development"},
            {"name": "Hurricane or Related", "id": "Hurricane or Related"},
            {"name": "HNRP status", "id": "HNRP status"}
        ],
        data=df_country_framework_status.to_dict("records"),
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily":"-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif",
            "fontSize":"14px",
            "padding":"6px",
            "textAlign": "left",
            "whiteSpace": "normal"
        },
        style_header={"backgroundColor":"#f4f6f8","fontWeight":"600"},
        style_cell_conditional=[
            {'if': {'column_id': c}, 'width': f'{100/5:.0f}%'}
            for c in ['Country','Active Framework in 2024','Framework Under Development','Hurricane or Related','HNRP status']
        ]
    ),

], style={"fontFamily":"-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif","padding":"10px 14px"})

# ----------------------------- CERF helpers ---------------------------------- #
def _event_key(row):
    """Key to dedupe 'events' per country-year: prefer Storm else Storm Type."""
    storm = (row.get("Storm") or "").strip()
    stype = (row.get("Storm Type") or "").strip()
    return storm.lower() if storm else stype.lower() if stype else "unspecified"

def _event_descriptor(row):
    """Human-readable descriptor: 'Storm, Category X' or fallback to Storm Type."""
    storm = (row.get("Storm") or "").strip()
    stype = (row.get("Storm Type") or "").strip()
    cat   = (row.get("Category") or "").strip()
    if storm:
        if cat:
            return f"{storm}, Category {cat}"
        elif stype:
            return f"{storm} ({stype})"
        else:
            return storm
    if stype:
        return stype
    if cat:
        return f"Category {cat}"
    return "Unspecified"

def _cerf_annotation_lines(cerf_year_df):
    """
    Return lines like:
      'Country (allocations_count) - unique storm descriptors'
    - Count is total allocations for that country-year (rows in CERF_DF)
    - Storm descriptors are unique by event key (Storm else Storm Type)
    """
    lines = []
    for country, g in cerf_year_df.groupby("Country"):
        n_alloc = int(len(g))  # number of allocations (rows)
        seen = {}
        for _, r in g.iterrows():
            key = _event_key(r)
            if key not in seen:
                seen[key] = _event_descriptor(r)
        descriptors = " | ".join(seen[k] for k in seen.keys())
        lines.append((country, n_alloc, f"{country} ({n_alloc}) - {descriptors}" if descriptors else f"{country} ({n_alloc})"))
    lines = [t[2] for t in sorted(lines, key=lambda x: (-x[1], x[0]))]
    return lines

# --------------------------------- Callbacks --------------------------------- #
@app.callback(
    Output("forecast_vs_observed","figure"),
    Input("metric","value"),
    Input("lead","value"),
    Input("issuers","value"),
    Input("year_range","value"),
    Input("forecast_vs_observed","hoverData"),
)
def update_chart(metric, lead, issuers, year_range, hoverData):
    y0, y1 = year_range

    obs = observed[(observed["year"]>=y0)&(observed["year"]<=y1)]
    fcs = (forecasts[(forecasts["lead"]==lead)&(forecasts["issuer"].isin(issuers))
                     & (forecasts["metric"]==metric)
                     & (forecasts["year"]>=y0)&(forecasts["year"]<=y1)]
                     .dropna(subset=["value"]))

    fig = go.Figure()

    # Observed
    fig.add_trace(go.Scatter(
        x=obs["year"], y=obs[metric],
        mode="lines+markers", name=f"Observed {METRIC_LABELS[metric]}",
        line=dict(width=3, color=OBS_COLOR),
        marker=dict(color=OBS_COLOR)
    ))

    # Forecasts
    for issuer, g in fcs.groupby("issuer"):
        fig.add_trace(go.Scatter(
            x=g["year"], y=g["value"],
            mode="lines+markers", name=f"{issuer} {lead}",
            line=dict(color=ISSUER_COLORS.get(issuer, "#1f77b4")),
            marker=dict(color=ISSUER_COLORS.get(issuer, "#1f77b4"))
        ))

    fig.update_xaxes(tickmode="linear", dtick=1)

    # ENSO GLOBAL BANDS (coarse run-peak, with labels)
    _add_month_bands_global(fig, _ENSO_BANDS_GLOBAL, y0, y1, show_labels=True)

    # Hover helper
    years_span = list(range(int(y0), int(y1)+1))
    if years_span:
        vals = pd.concat([obs[metric].dropna(), fcs["value"].dropna()], ignore_index=True)
        ymax = float(vals.max()) if not vals.empty and pd.notna(vals.max()) else 1.0
        label_y = ymax * 1.05

        fig.add_trace(go.Bar(
            x=years_span,
            y=[label_y]*len(years_span),
            marker=dict(color="rgba(0,0,0,0)"),
            opacity=0.003,
            hoverinfo="skip",
            showlegend=False,
            name="_hovercatch"
        ))
        fig.update_layout(barmode="overlay", bargap=0)

    hover_year = None
    if hoverData and "points" in hoverData and hoverData["points"]:
        xval = hoverData["points"][0].get("x", None)
        try:
            hover_year = int(round(float(xval)))
        except (TypeError, ValueError):
            hover_year = None

    # CERF allocation box (allocations count per country-year; unique storms listed)
    if hover_year is not None:
        cerf_year = CERF_DF[CERF_DF["Year"] == hover_year]
        if not cerf_year.empty:
            lines = _cerf_annotation_lines(cerf_year)
            box_text = f"<b>CERF Allocations - {hover_year}</b><br>" + "<br>".join(lines)
        else:
            box_text = f"<b>CERF Allocations - {hover_year}</b><br>No allocations"

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            xanchor="left", yanchor="top",
            align="left",
            text=box_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
            borderpad=6,
            font=dict(size=12, color="#111")
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Year", yaxis_title=METRIC_LABELS[metric],
        clickmode="event+select",
        hovermode="x",
        hoverdistance=100,
        spikedistance=100,
        plot_bgcolor="#f3f4f6"
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikedash="dot", spikecolor="#999"
    )

    _add_enso_legend_traces(fig)
    return fig

# --- CERF table updater (click to filter) ---
@app.callback(
    Output("cerf_table", "data"),
    Input("year_range", "value"),
    Input("forecast_vs_observed", "clickData"),
)
def update_cerf_table(year_range, clickData):
    y0, y1 = year_range
    df = CERF_DF[CERF_DF["Year"].between(y0, y1)].copy()
    click_year = None
    if clickData and "points" in clickData and clickData["points"]:
        xval = clickData["points"][0].get("x", None)
        try:
            click_year = int(round(float(xval)))
        except (TypeError, ValueError):
            click_year = None
    if click_year is not None:
        df = df[df["Year"] == click_year]
    return df.sort_values(["Year","Country","Application Code"]).to_dict("records")

# --- Country Framework Status table updater ---
@app.callback(
    Output("country_framework_status_table", "data"),
    Input("year_range", "value"),
)
def update_country_framework_status_table(year_range):
    return df_country_framework_status.sort_values('Country').to_dict("records")

# --- Accuracy toggle -> compute table + deltas ---
@app.callback(
    Output("accuracy_table", "data"),
    Output("accuracy_section", "style"),
    Output("residual_graph", "figure"),
    Output("residual_section", "style"),
    Input("show_accuracy", "value"),
    Input("metric", "value"),
    Input("lead", "value"),
    Input("issuers","value"),
    Input("year_range","value"),
)
def update_accuracy(toggle_vals, metric, lead, issuers, year_range):
    show = ("on" in toggle_vals)

    hidden = {"display":"none"}
    visible = {"display":"block"}

    y0, y1 = year_range
    obs = observed[(observed["year"]>=y0)&(observed["year"]<=y1)][["year", metric]].dropna()
    fcs = (forecasts[(forecasts["lead"]==lead)&(forecasts["issuer"].isin(issuers))
                     & (forecasts["metric"]==metric)
                     & (forecasts["year"]>=y0)&(forecasts["year"]<=y1)]
                     .dropna(subset=["value"]))[["year","issuer","value"]]

    if (not show) or obs.empty or fcs.empty:
        return [], hidden, go.Figure(), hidden

    acc_rows = []
    delta_fig = go.Figure()
    for issuer, gi in fcs.groupby("issuer"):
        merged = pd.merge(gi, obs, on="year", how="inner", validate="many_to_one")
        if merged.empty:
            continue
        merged["delta"] = merged["value"] - merged[metric]
        mae = float(merged["delta"].abs().mean())
        rmse = float(np.sqrt((merged["delta"]**2).mean()))
        me = float(merged["delta"].mean())
        r = float(pd.Series(merged["value"]).corr(pd.Series(merged[metric]))) if merged.shape[0] >= 2 else np.nan
        acc_rows.append({"issuer": issuer, "mae": round(mae,2), "rmse": round(rmse,2), "me": round(me,2), "r": round(r,2) if pd.notna(r) else None, "n": int(merged.shape[0])})

        delta_fig.add_trace(go.Scatter(
            x=merged["year"], y=merged["delta"],
            mode="lines+markers", name=f"{issuer} delta",
            line=dict(color=ISSUER_COLORS.get(issuer, "#1f77b4")),
            marker=dict(color=ISSUER_COLORS.get(issuer, "#1f77b4"))
        ))

    delta_fig.add_hline(y=0, line_dash="dot", line_color="#999")
    delta_fig.update_xaxes(tickmode="linear", dtick=1, title_text="Year")
    delta_fig.update_yaxes(title_text=f"Delta (Forecast − Observed) — {METRIC_LABELS[metric]}")
    delta_fig.update_layout(margin=dict(l=20,r=20,t=10,b=30), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    acc_rows = sorted(acc_rows, key=lambda r: r["rmse"])
    return acc_rows, visible, delta_fig, visible

# --- Lazy-load ENSO granularity chart (monthly-intensity bands) ---
@app.callback(
    Output("enso_detail_section", "style"),
    Output("enso_detail_graph", "figure"),
    Input("enso_detail_btn", "n_clicks"),
    Input("metric","value"),
    Input("lead","value"),
    Input("issuers","value"),
    Input("year_range","value"),
    prevent_initial_call=True
)
def update_enso_detail(n_clicks, metric, lead, issuers, year_range):
    if not n_clicks:
        return {"display":"none"}, go.Figure()

    y0, y1 = year_range
    obs = observed[(observed["year"]>=y0)&(observed["year"]<=y1)]
    fcs = (forecasts[(forecasts["lead"]==lead)&(forecasts["issuer"].isin(issuers))
                     & (forecasts["metric"]==metric)
                     & (forecasts["year"]>=y0)&(forecasts["year"]<=y1)]
                     .dropna(subset=["value"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs["year"], y=obs[metric],
        mode="lines+markers", name=f"Observed {METRIC_LABELS[metric]}",
        line=dict(width=3, color=OBS_COLOR),
        marker=dict(color=OBS_COLOR)
    ))
    for issuer, g in fcs.groupby("issuer"):
        fig.add_trace(go.Scatter(
            x=g["year"], y=g["value"],
            mode="lines+markers", name=f"{issuer} {lead}",
            line=dict(color=ISSUER_COLORS.get(issuer, "#1f77b4")),
            marker=dict(color=ISSUER_COLORS.get(issuer, "#1f77b4"))
        ))
    fig.update_xaxes(tickmode="linear", dtick=1)

    # Detailed monthly-intensity ENSO background (no labels)
    _add_month_bands_detailed(fig, _ENSO_BANDS_MONTHLY, y0, y1)

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Year", yaxis_title=METRIC_LABELS[metric],
        clickmode="event+select",
        hovermode="x",
        hoverdistance=100,
        spikedistance=100,
        plot_bgcolor="#f3f4f6"
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="#999")

    _add_enso_legend_traces(fig)
    return {"display":"block"}, fig

# ---- Run in Spaces/Docker ----
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
