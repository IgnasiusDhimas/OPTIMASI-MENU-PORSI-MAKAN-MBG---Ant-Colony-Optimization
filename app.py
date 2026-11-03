import time
import random
from copy import deepcopy
from html import escape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MBG NutriOpt — Simulasi ACO",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS — meniru desain Google Stitch (palet Material, Plus Jakarta Sans)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

.material-symbols-outlined {
  font-family: 'Material Symbols Outlined'; font-weight: normal; font-style: normal;
  font-size: 20px; line-height: 1; letter-spacing: normal; text-transform: none;
  display: inline-block; white-space: nowrap; direction: ltr;
  font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
}

/* ---------- Dasar ---------- */
.stApp { background: #f8f9ff; color: #0b1c30; font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp button, .stApp textarea, .stApp td, .stApp th {
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp, section[data-testid="stSidebar"] { color-scheme: light !important; }
header[data-testid="stHeader"] {
  position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000001;
  background: transparent !important; backdrop-filter: none !important; pointer-events: none;
}
[data-testid="stToolbar"] { background: transparent !important; backdrop-filter: none !important; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }

/* ---------- Tombol buka/tutup sidebar (di pojok kiri header) ---------- */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
  position: fixed !important; top: 12px !important; left: 16px !important; z-index: 1000002 !important;
  opacity: 1 !important; visibility: visible !important; pointer-events: auto !important;
}

/* Latar belakang tombol: Hijau */
[data-testid="stExpandSidebarButton"], 
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
  width: 40px !important; height: 40px !important; border-radius: 10px !important; 
  background: #006948 !important; border: none !important;
  display: flex !important; align-items: center !important; justify-content: center !important; 
  transition: background .15s; box-shadow: 0 2px 6px rgba(0,105,72,.3) !important;
  position: relative !important;
}

/* Efek Hover */
[data-testid="stExpandSidebarButton"]:hover, 
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover { 
  background: #00855d !important; 
}

/* 1. SEMBUNYIKAN total ikon bawaan Streamlit (yang bikin kotak) */
[data-testid="stExpandSidebarButton"] svg, 
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stExpandSidebarButton"] span, 
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="collapsedControl"] span {
  display: none !important;
}

/* 2. GANTI dengan panah CSS murni berwarna putih */
[data-testid="stExpandSidebarButton"]::after, 
[data-testid="stSidebarCollapseButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="collapsedControl"] button::after {
  content: '❯' !important; /* Karakter panah */
  color: #ffffff !important; /* Warna panah putih */
  font-size: 20px !important;
  font-family: Arial, sans-serif !important;
  font-weight: bold !important;
  display: block !important;
  line-height: 1 !important;
}

/* ---------- Header tetap di atas ---------- */
.app-header {
  position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000000;
  background: rgba(255,255,255,0.92); backdrop-filter: blur(16px);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  display: flex; align-items: center; justify-content: space-between; padding: 0 32px 0 72px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.logo {
  width: 34px; height: 34px; border-radius: 10px; font-size: 18px; color: #fff;
  background: linear-gradient(135deg, #006948, #00855d);
  display: flex; align-items: center; justify-content: center;
}
.brand-name { font-size: 18px; font-weight: 600; color: #006948; letter-spacing: -0.005em; margin-right: 8px; }
.brand-sub { font-size: 11px; color: #3d4a42; margin-top: 2px; }
.unit { text-align: right; line-height: 1.25; margin-right: 12px; }
.unit b { display: block; font-size: 14px; color: #0b1c30; font-weight: 600; }
.unit span { font-size: 11px; color: #3d4a42; }
.avatar {
  width: 32px; height: 32px; border-radius: 50%; background: #006948; color: #fff;
  display: flex; align-items: center; justify-content: center;
}
.pill {
  display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px;
  border-radius: 999px; font-size: 11px; font-weight: 600; vertical-align: middle;
}
.dot { width: 6px; height: 6px; border-radius: 50%; background: #006948; animation: pulse 1.6s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

/* ---------- Sidebar navigasi ---------- */
section[data-testid="stSidebar"] {
  background: #ffffff; border-right: none; box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"]:not([data-collapsed="true"]):not([aria-expanded="false"]) {
  width: 256px !important; min-width: 256px !important; max-width: 256px !important;
}
[data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebarHeader"] { height: 0 !important; min-height: 0 !important; padding: 0 !important; overflow: visible !important; }
[data-testid="stLogoSpacer"] { display: none !important; }
[data-testid="stSidebarUserContent"] { padding: 5.5rem 0.75rem 1rem 0.75rem !important; }
.nav-label { font-size: 11px; font-weight: 500; letter-spacing: .04em; color: #3d4a42; padding: 0 8px 6px 8px; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  width: 100%; margin: 0; padding: 10px 16px; border-radius: 8px; cursor: pointer; transition: background .15s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #dce9ff; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
  display: flex; align-items: center; gap: 12px; margin: 0;
  font-size: 14px; font-weight: 600; color: #3d4a42 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p::before {
  font-family: 'Material Symbols Outlined'; font-weight: normal; font-size: 20px; line-height: 1;
  font-feature-settings: 'liga';
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: #00855d; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #f5fff7 !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) p::before { content: 'hub'; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2) p::before { content: 'nutrition'; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3) p::before { content: 'calendar_month'; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4) p::before { content: 'monitoring'; }
.side-card { margin-top: 28px; padding: 12px; border-radius: 12px; background: #eff4ff; }
.side-row { display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; color: #006948; }
.side-row span:last-child { font-weight: 500; color: #3d4a42; }
.side-note { font-size: 11px; color: #3d4a42; margin-top: 6px; }

/* ---------- Kartu (st.container dengan key "card-...") ---------- */
[class*="st-key-card-"] {
  background: #ffffff; border-radius: 12px; padding: 24px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}

/* ---------- Tombol ---------- */
.stButton, .stDownloadButton, [data-testid="stFormSubmitButton"] { width: 100%; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
  width: 100%; border: none; border-radius: 12px; padding: 10px 16px; font-weight: 600; transition: all .15s;
}
button[kind="primary"], button[data-testid="stBaseButton-primary"] { background: #006948 !important; color: #ffffff !important; }
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #00855d !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { background: #e5eeff !important; color: #0b1c30 !important; }
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover { background: #dce9ff !important; }
.stButton button p, .stDownloadButton button p, [data-testid="stFormSubmitButton"] button p { color: inherit !important; }
.stButton button:disabled, .stDownloadButton button:disabled { opacity: .5; }

/* ---------- Input ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {
  background: #eff4ff !important; border-radius: 8px !important; border: none !important;
}
[data-testid="stWidgetLabel"] p { font-size: 12px; font-weight: 600; color: #0b1c30; }
[data-testid="stSlider"] div[role="slider"] { background-color: #006948 !important; }
[data-testid="stSliderThumbValue"] { color: #006948 !important; font-weight: 600; }
[data-testid="stExpander"] { border: none !important; background: #eff4ff; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #0b1c30 !important; }
.stApp input, .stApp textarea {
  color: #0b1c30 !important; -webkit-text-fill-color: #0b1c30 !important; caret-color: #006948;
}
.stApp input::placeholder, .stApp textarea::placeholder {
  color: #6d7a72 !important; -webkit-text-fill-color: #6d7a72 !important; opacity: 1;
}
[data-baseweb="select"] * { color: #0b1c30 !important; }
[data-baseweb="select"] svg { fill: #3d4a42 !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"] { background: #ffffff !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] { background: #ffffff !important; color: #0b1c30 !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] { background: #eff4ff !important; }
[data-testid="stNumberInput"] button { background: #dce9ff !important; color: #0b1c30 !important; border: none !important; }
[data-testid="stNumberInput"] button svg { fill: #0b1c30 !important; }
[data-testid="stFileUploaderDropzone"] { background: #eff4ff !important; border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] * { color: #0b1c30 !important; }
[data-testid="stForm"] { border: 1px solid #d3e4fe !important; border-radius: 12px; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #3d4a42 !important; }
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li { color: #0b1c30; }

/* ---------- Komponen HTML kustom ---------- */
.sec-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 4px; }
.sec-title { display: flex; align-items: center; gap: 8px; font-size: 16px; font-weight: 600; color: #0b1c30; }
.sec-title .material-symbols-outlined { color: #006948; font-size: 22px; }
.sec-sub { font-size: 11px; color: #3d4a42; }
.muted { font-size: 11px; color: #3d4a42; }
.h1 { font-size: 24px; line-height: 32px; font-weight: 600; letter-spacing: -.01em; color: #0b1c30; margin: 6px 0; }
.lead { font-size: 14px; line-height: 22px; color: #3d4a42; max-width: 760px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px;
  background: #e5eeff; font-size: 11px; font-weight: 500; color: #0b1c30;
}
.chip b { font-weight: 700; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
@media (max-width: 1200px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi {
  background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  display: flex; flex-direction: column; gap: 10px;
}
.kpi-top { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 600; color: #3d4a42; }
.kpi-ico { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
.kpi-val { font-size: 32px; line-height: 40px; font-weight: 700; letter-spacing: -.03em; color: #0b1c30; }
.kpi-val small { font-size: 14px; font-weight: 400; letter-spacing: 0; color: #3d4a42; }
.kpi-sub { font-size: 11px; color: #3d4a42; margin-top: -6px; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; width: fit-content; }
.b-ok { background: #85f8c4; color: #002114; }
.b-warn { background: #ffdcc3; color: #6e3900; }
.b-err { background: #ffdad6; color: #93000a; }
.b-neutral { background: #dce9ff; color: #3d4a42; }

.meal {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 12px; background: #eff4ff; border-radius: 12px; margin-bottom: 8px; transition: background .15s;
}
.meal:hover { background: #e5eeff; }
.meal-l { display: flex; align-items: center; gap: 12px; min-width: 0; }
.meal-ico {
  width: 40px; height: 40px; border-radius: 8px; background: #d3e4fe; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.meal-name { font-size: 14px; font-weight: 600; color: #0b1c30; }
.tag { font-size: 11px; font-weight: 500; padding: 1px 6px; border-radius: 4px; background: #dce9ff; color: #3d4a42; margin-left: 8px; }
.meal-r { display: flex; gap: 20px; text-align: right; }
.num-main { font-size: 12px; font-weight: 600; color: #0b1c30; }
.num-price { font-size: 14px; font-weight: 700; color: #006948; }
.total-box {
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
  padding: 16px; background: #e5eeff; border-radius: 12px; margin: 12px 0 16px 0;
}
.total-label { font-size: 12px; font-weight: 600; color: #3d4a42; letter-spacing: .02em; }
.total-line { font-size: 14px; font-weight: 600; color: #0b1c30; margin-top: 2px; }
.total-price { font-size: 18px; font-weight: 700; color: #006948; text-align: right; }
.bars { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 24px; margin-top: 8px; }
.bar-lbl { display: flex; justify-content: space-between; font-size: 11px; font-weight: 500; color: #0b1c30; }
.bar { height: 8px; background: #dce9ff; border-radius: 999px; overflow: hidden; margin-top: 4px; }
.bar > div { height: 100%; border-radius: 999px; }
.banner { display: flex; align-items: center; gap: 12px; padding: 16px; background: #e5eeff; border-radius: 12px; }
.banner-ico {
  width: 32px; height: 32px; border-radius: 50%; background: #d3e4fe; color: #006948; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
}
.banner b { display: block; font-size: 14px; color: #0b1c30; }
.banner span { font-size: 11px; color: #3d4a42; }
.empty { padding: 32px 16px; text-align: center; background: #eff4ff; border-radius: 12px; font-size: 13px; color: #3d4a42; }
.day-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.day { background: #eff4ff; border-radius: 12px; padding: 14px; }
.day-h { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.day-h b { font-size: 14px; color: #0b1c30; }
.day-h span { font-size: 12px; font-weight: 700; color: #006948; }
.day-i { font-size: 12px; color: #3d4a42; padding: 3px 0; display: flex; justify-content: space-between; gap: 8px; }

/* ---------- Responsif ---------- */
@media (max-width: 1100px) {
  .brand-sub { display: none; }
}
@media (max-width: 992px) {
  .block-container { padding: 5rem 1rem 2rem 1rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 1rem !important; }
  [data-testid="stColumn"], [data-testid="column"] { flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .bars { grid-template-columns: 1fr; }
  [class*="st-key-card-"] { padding: 16px; }
}
@media (max-width: 640px) {
  .app-header { padding: 0 12px 0 68px; }
  .unit, .pill-engine { display: none; }
  .brand-name { font-size: 16px; }
  .kpi-grid { grid-template-columns: 1fr; }
  .kpi-val { font-size: 26px; line-height: 34px; }
  .h1 { font-size: 20px; line-height: 28px; }
  .meal { flex-direction: column; align-items: flex-start; }
  .meal-r { width: 100%; justify-content: space-between; text-align: left; }
  .total-box { flex-direction: column; align-items: flex-start; }
  .total-price { text-align: left; }
}
</style>
"""


def md(s):
    """Render HTML ke Streamlit. Baris dirapatkan agar Markdown tidak
    menganggap baris berindentasi / baris kosong sebagai code block."""
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def rp(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def icon(name, size=None):
    style = f' style="font-size:{size}px"' if size else ""
    return f'<span class="material-symbols-outlined"{style}>{name}</span>'


md(CSS)

# ============================================================
# DATASET & PARAMETER DEFAULT
# ============================================================
DEFAULT_DATASET = {
    "Karbohidrat": [
        {"nama": "Nasi Putih",     "harga": 1500, "kalori": 175, "protein": 4,   "icon": "🍚"},
        {"nama": "Kentang Rebus",  "harga": 2000, "kalori": 150, "protein": 3,   "icon": "🥔"},
        {"nama": "Ubi Rebus",      "harga": 1800, "kalori": 160, "protein": 2,   "icon": "🍠"},
    ],
    "Protein": [
        {"nama": "Ayam Goreng",    "harga": 5000, "kalori": 250, "protein": 20,  "icon": "🍗"},
        {"nama": "Telur Dadar",    "harga": 3000, "kalori": 150, "protein": 12,  "icon": "🍳"},
        {"nama": "Daging Sapi",    "harga": 4500, "kalori": 300, "protein": 25,  "icon": "🥩"},
        {"nama": "Tempe Bacem",    "harga": 1200, "kalori": 100, "protein": 10,  "icon": "🟫"},
        {"nama": "Tahu Goreng",    "harga": 1000, "kalori": 80,  "protein": 8,   "icon": "⬜"},
    ],
    "Sayur": [
        {"nama": "Bayam / Bening", "harga": 800,  "kalori": 40,  "protein": 3,   "icon": "🥬"},
        {"nama": "Sup Wortel",     "harga": 1000, "kalori": 50,  "protein": 2,   "icon": "🥕"},
        {"nama": "Tumis Sawi",     "harga": 900,  "kalori": 45,  "protein": 2.5, "icon": "🥦"},
    ],
    "Buah": [
        {"nama": "Pisang",         "harga": 2000, "kalori": 105, "protein": 1.3, "icon": "🍌"},
        {"nama": "Apel",           "harga": 3000, "kalori": 95,  "protein": 0.5, "icon": "🍎"},
    ],
}

DEFAULT_PARAMS = {
    "p_semut": 60, "p_iterasi": 80, "p_alpha": 1.0, "p_beta": 2.0, "p_rho": 0.15, "p_q": 100.0,
    "p_min_kalori": 600, "p_min_protein": 20, "p_budget": 15000,
}
NAV = ["Simulasi & Optimasi ACO", "Katalog Bahan", "Perencanaan Mingguan", "Laporan & Evaluasi Biaya"]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]

# ============================================================
# SESSION STATE
# ============================================================
if "dataset" not in st.session_state:
    st.session_state.dataset = deepcopy(DEFAULT_DATASET)
for _k in ("hasil", "mingguan"):
    st.session_state.setdefault(_k, None)
# Menyalin ulang nilai parameter tiap run agar tidak hilang saat pindah halaman
# (Streamlit membuang state widget yang tidak dirender di halaman aktif).
for _k, _v in DEFAULT_PARAMS.items():
    st.session_state[_k] = st.session_state.get(_k, _v)


def get_params():
    s = st.session_state
    return {
        "semut": s.p_semut, "iterasi": s.p_iterasi, "alpha": s.p_alpha, "beta": s.p_beta,
        "rho": s.p_rho, "q": s.p_q, "min_kalori": s.p_min_kalori,
        "min_protein": s.p_min_protein, "budget": s.p_budget,
    }


# ============================================================
# ALGORITMA ACO (logika tidak diubah)
# ============================================================
def weighted_choice(items, weights):
    total = sum(weights)
    r = random.random() * total
    acc = 0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def run_aco(dataset, params):
    JUMLAH_SEMUT = params["semut"]
    ITERASI = params["iterasi"]
    ALFA = params["alpha"]
    BETA = params["beta"]
    EVAPORASI = params["rho"]
    Q = params["q"]
    MIN_KALORI = params["min_kalori"]
    MIN_PROTEIN = params["min_protein"]
    BUDGET = params["budget"]

    feromon = {}
    for kat, daftar in dataset.items():
        for m in daftar:
            feromon[m["nama"]] = 1.0

    riwayat_terbaik, riwayat_rata = [], []
    menu_terbaik, harga_terbaik = None, 0
    kategori_list = list(dataset.keys())

    for it in range(ITERASI):
        solusi = []
        total_harga_semut = 0

        for _ in range(JUMLAH_SEMUT):
            menu = []
            for kat in kategori_list:
                opsi = dataset[kat]
                if not opsi:
                    continue
                weights = []
                for m in opsi:
                    tau = feromon[m["nama"]] ** ALFA
                    eta = ((m["protein"] + m["kalori"] / 10) / max(m["harga"], 1)) ** BETA
                    weights.append(tau * eta)
                pilih = weighted_choice(opsi, weights)
                gram = random.randint(80, 250)
                f = gram / 100
                menu.append({
                    "nama": pilih["nama"], "gram": gram,
                    "harga": pilih["harga"] * f, "kalori": pilih["kalori"] * f,
                    "protein": pilih["protein"] * f,
                    "icon": pilih.get("icon", "🍽️"), "kategori": kat,
                })

            t_harga = sum(x["harga"] for x in menu)
            t_kalori = sum(x["kalori"] for x in menu)
            t_protein = sum(x["protein"] for x in menu)
            total_harga_semut += t_harga

            if t_kalori >= MIN_KALORI and t_protein >= MIN_PROTEIN and t_harga <= BUDGET:
                solusi.append({"menu": menu, "harga": t_harga, "fitness": t_harga})

        for k in feromon:
            feromon[k] *= (1 - EVAPORASI)

        if solusi:
            best = max(solusi, key=lambda x: x["harga"])
            for m in best["menu"]:
                feromon[m["nama"]] += Q * (best["fitness"] / 10000) * 0.1
            if best["harga"] > harga_terbaik:
                harga_terbaik = best["harga"]
                menu_terbaik = best["menu"]

        riwayat_terbaik.append(harga_terbaik if harga_terbaik > 0 else None)
        riwayat_rata.append(total_harga_semut / JUMLAH_SEMUT)

    return {
        "menu_terbaik": menu_terbaik, "harga_terbaik": harga_terbaik,
        "riwayat_terbaik": riwayat_terbaik, "riwayat_rata": riwayat_rata,
    }


# ============================================================
# CALLBACK TOMBOL
# ============================================================
def jalankan_aco():
    ds = st.session_state.dataset
    if sum(len(v) for v in ds.values()) == 0:
        st.session_state.hasil = {"kosong": True, "menu_terbaik": None}
        return
    params = get_params()
    t0 = time.perf_counter()
    hasil = run_aco(ds, params)
    hasil["waktu"] = time.perf_counter() - t0
    hasil["params"] = params
    final = hasil["harga_terbaik"]
    hasil["iter_konv"] = next(
        (i + 1 for i, v in enumerate(hasil["riwayat_terbaik"]) if v and v >= final),
        params["iterasi"],
    )
    st.session_state.hasil = hasil


def reset_params():
    for k, v in DEFAULT_PARAMS.items():
        st.session_state[k] = v


def reset_dataset():
    st.session_state.dataset = deepcopy(DEFAULT_DATASET)
    st.session_state.hasil = None
    st.session_state.mingguan = None


def susun_mingguan():
    params = get_params()
    ds = st.session_state.dataset
    rencana = []
    for h in HARI:
        r = run_aco(ds, params)
        rencana.append({"hari": h, "menu": r["menu_terbaik"], "harga": r["harga_terbaik"]})
    st.session_state.mingguan = {"rencana": rencana, "params": params}


# ============================================================
# KOMPONEN TAMPILAN
# ============================================================
def render_header():
    md(f"""
    <div class="app-header">
      <div class="brand">
        <div class="logo">🐜</div>
        <div>
          <div><span class="brand-name">MBG NutriOpt</span>
            <span class="pill pill-engine" style="background:#85f8c4;color:#002114"><span class="dot"></span>ACO Engine (Active)</span></div>
          <div class="brand-sub">Sistem Optimasi Menu &amp; Porsi Bergizi Gratis berbasis Ant Colony Optimization</div>
        </div>
      </div>
      <div style="display:flex;align-items:center">
        <div class="unit"><b>Unit Dapur SPPG 04</b><span>Satuan Pelayanan Program Gizi</span></div>
        <div class="avatar">{icon("person", 18)}</div>
      </div>
    </div>
    """)


def render_sidebar_card(hasil):
    s = st.session_state
    if hasil and hasil.get("menu_terbaik"):
        p = hasil["params"]
        pct = hasil["iter_konv"] / p["iterasi"] * 100
        semut, note = p["semut"], f"Konvergen di iterasi {hasil['iter_konv']} dari {p['iterasi']}"
    else:
        pct, semut, note = 0, s.p_semut, "Belum ada simulasi"
    md(f"""
    <div class="side-card">
      <div class="side-row"><span>Pheromone Node</span><span>{semut} Ants</span></div>
      <div class="bar" style="margin-top:8px"><div style="width:{pct:.0f}%;background:#006948"></div></div>
      <div class="side-note">{note}</div>
    </div>
    """)


def sec_head(ico, title, sub=None, right=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    md(f"""
    <div class="sec-head"><div><div class="sec-title">{icon(ico)}{title}</div>{sub_html}</div>{right}</div>
    """)


def kpi(title, ico, bg, fg, value, sub, b_cls, b_ico, b_txt):
    return f"""
    <div class="kpi">
      <div class="kpi-top"><span>{title}</span><div class="kpi-ico" style="background:{bg};color:{fg}">{icon(ico, 18)}</div></div>
      <div class="kpi-val">{value}</div>
      <div class="kpi-sub">{sub}</div>
      <span class="badge {b_cls}">{icon(b_ico, 14)}{b_txt}</span>
    </div>
    """


def render_kpis(hasil):
    if hasil and hasil.get("menu_terbaik"):
        p, menu, harga = hasil["params"], hasil["menu_terbaik"], hasil["harga_terbaik"]
        kal = sum(m["kalori"] for m in menu)
        prot = sum(m["protein"] for m in menu)
        n_kat = len({m["kategori"] for m in menu})
        sisa = p["budget"] - harga
        cards = [
            kpi("Total Energi / Kalori", "bolt", "#85f8c4", "#002114",
                f'{kal:.0f} <small>kkal</small>', f'Minimum: {p["min_kalori"]} kkal',
                "b-ok", "check_circle", f'Memenuhi Target ({kal / p["min_kalori"] * 100:.0f}%)'),
            kpi("Kandungan Protein", "egg_alt", "#dce9ff", "#006948",
                f'{prot:.1f} <small>gram</small>', f'Kebutuhan Min: {p["min_protein"]} gram',
                "b-ok", "verified", f'Memenuhi Target ({prot / p["min_protein"] * 100:.0f}%)'),
            kpi("Total Biaya Bahan", "payments", "#ffdcc3", "#2f1500",
                rp(harga), f'Pagu Anggaran: {rp(p["budget"])}',
                "b-ok", "savings", f'Sisa {rp(sisa)} ({sisa / p["budget"] * 100:.1f}%)'),
            kpi("Komponen Menu", "restaurant", "#dae2fd", "#131b2e",
                f'{len(menu)} <small>item</small>', f'Dari {n_kat} kategori bahan',
                "b-ok", "auto_awesome", "Solusi Valid"),
        ]
    else:
        gagal = bool(hasil)
        b_cls, b_txt = ("b-err", "Tidak ada solusi valid") if gagal else ("b-neutral", "Menunggu simulasi")
        cards = [
            kpi("Total Energi / Kalori", "bolt", "#85f8c4", "#002114", "—", "Belum ada data", b_cls, "hourglass_empty", b_txt),
            kpi("Kandungan Protein", "egg_alt", "#dce9ff", "#006948", "—", "Belum ada data", b_cls, "hourglass_empty", b_txt),
            kpi("Total Biaya Bahan", "payments", "#ffdcc3", "#2f1500", "—", "Belum ada data", b_cls, "hourglass_empty", b_txt),
            kpi("Komponen Menu", "restaurant", "#dae2fd", "#131b2e", "—", "Belum ada data", b_cls, "hourglass_empty", b_txt),
        ]
    md(f'<div class="kpi-grid">{"".join(cards)}</div>')


def bar(label, teks, pct, warna="#006948"):
    return f"""
    <div><div class="bar-lbl"><span>{label}</span><span style="font-weight:600;color:{warna}">{teks}</span></div>
    <div class="bar"><div style="width:{min(pct, 100):.0f}%;background:{warna}"></div></div></div>
    """


def grafik_konvergensi(hasil):
    p = hasil["params"]
    best = np.array([v if v else np.nan for v in hasil["riwayat_terbaik"]], dtype=float)
    avg = np.array(hasil["riwayat_rata"], dtype=float)
    x = np.arange(1, len(best) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor("#eff4ff")
    ax.set_facecolor("#eff4ff")
    ax.fill_between(x, 0, best, where=~np.isnan(best), color="#85f8c4", alpha=0.35)
    ax.plot(x, best, color="#006948", linewidth=3, label="Terbaik Global")
    ax.plot(x, avg, color="#565e74", linewidth=2, linestyle="--", label="Rata-rata Koloni")
    ax.axhline(p["budget"], color="#ba1a1a", linestyle=":", linewidth=1.5, label=f'Pagu {rp(p["budget"])}')
    k = hasil["iter_konv"]
    ax.scatter([k], [best[k - 1]], color="#006948", s=60, zorder=5)
    ax.annotate(f"Iterasi {k}", (k, best[k - 1]), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8, color="#006948", fontweight="bold")
    ax.set_xlim(1, len(x))
    ax.set_ylim(0, max(p["budget"], np.nanmax(best), avg.max()) * 1.08)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bccac0")
    ax.grid(axis="y", color="#d3e4fe")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors="#3d4a42", length=0)
    ax.set_xlabel("Iterasi", fontsize=9, color="#3d4a42")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def html_menu(menu, harga, p):
    rows = ""
    for m in menu:
        rows += f"""
        <div class="meal">
          <div class="meal-l"><div class="meal-ico">{escape(str(m["icon"]))}</div>
            <div><div><span class="meal-name">{escape(m["nama"])}</span><span class="tag">{escape(m["kategori"])}</span></div>
            <div class="muted">Takaran Porsi: <strong>{m["gram"]} gram</strong></div></div></div>
          <div class="meal-r">
            <div><div class="num-main">{m["kalori"]:.0f} kkal</div><div class="muted">Prot: {m["protein"]:.1f}g</div></div>
            <div style="min-width:84px"><div class="num-price">{rp(m["harga"])}</div>
            <div class="muted">{rp(m["harga"] / m["gram"])}/g</div></div>
          </div>
        </div>
        """
    kal = sum(m["kalori"] for m in menu)
    prot = sum(m["protein"] for m in menu)
    bars = (
        bar("Energi (min. " + str(p["min_kalori"]) + " kkal)", f'{kal:.0f} kkal ({kal / p["min_kalori"] * 100:.0f}%)',
            kal / p["min_kalori"] * 100)
        + bar("Protein (min. " + str(p["min_protein"]) + " g)", f'{prot:.1f} g ({prot / p["min_protein"] * 100:.0f}%)',
              prot / p["min_protein"] * 100)
        + bar("Pemakaian Pagu Anggaran", f'{rp(harga)} ({harga / p["budget"] * 100:.0f}%)',
              harga / p["budget"] * 100, "#006948" if harga <= p["budget"] else "#ba1a1a")
    )
    return f"""
    {rows}
    <div class="total-box">
      <div><div class="total-label">Subtotal Nutrisi Terpilih</div>
        <div class="total-line">{kal:.0f} kkal Energi • {prot:.1f} g Protein</div></div>
      <div><div class="muted" style="text-align:right">Total Biaya Bahan Baku</div>
        <div class="total-price">{rp(harga)}</div></div>
    </div>
    <div class="total-label">Analisis Pemenuhan Batasan</div>
    <div class="bars">{bars}</div>
    """


def page_intro(badge, title, desc):
    with st.container(key="card-intro"):
        md(f"""
        <span class="pill" style="background:#00855d;color:#f5fff7">{icon("scatter_plot", 14)}{badge}</span>
        <div class="h1">{title}</div><div class="lead">{desc}</div>
        """)


def empty_state(teks):
    md(f'<div class="empty">{teks}</div>')


# ============================================================
# HALAMAN 1 — SIMULASI & OPTIMASI ACO
# ============================================================
def page_simulasi():
    hasil = st.session_state.hasil
    ok = bool(hasil and hasil.get("menu_terbaik"))
    ds = st.session_state.dataset

    # ---- Bar kontrol atas ----
    with st.container(key="card-hero"):
        c_txt, c_btn = st.columns([3, 1.2], vertical_alignment="center")
        with c_txt:
            if ok:
                chips = (
                    f'<span class="chip"><span class="dot"></span><b style="color:#006948">Status Komputasi:</b> '
                    f'Konvergen (Iterasi {hasil["iter_konv"]}/{hasil["params"]["iterasi"]})</span>'
                    f'<span class="chip">{icon("trending_up", 15)}Best Fitness: <b>{rp(hasil["harga_terbaik"])}</b></span>'
                    f'<span class="chip">{icon("timer", 15)}Waktu Komputasi: <b>{hasil["waktu"]:.2f} dtk</b></span>'
                )
            else:
                chips = '<span class="chip"><b style="color:#3d4a42">Status Komputasi:</b> Menunggu dijalankan</span>'
            total_item = sum(len(v) for v in ds.values())
            md(f"""
            <span class="pill" style="background:#00855d;color:#f5fff7">{icon("scatter_plot", 14)}Modul Metaheuristik</span>
            <span class="muted"> • Dataset aktif: {total_item} bahan dalam {len(ds)} kategori</span>
            <div class="h1">Simulasi &amp; Optimasi Menu MBG (Ant Colony Optimization)</div>
            <div class="lead">Model komputasi metaheuristik untuk menentukan kombinasi bahan pangan dan takaran porsi
            yang memenuhi batas kalori, protein, dan pagu biaya.</div>
            <div class="chips">{chips}</div>
            """)
        with c_btn:
            st.button("▶ Jalankan Optimasi ACO", type="primary", on_click=jalankan_aco, key="btn_run")
            if ok:
                csv = pd.DataFrame(hasil["menu_terbaik"]).to_csv(index=False).encode("utf-8")
            else:
                csv = b""
            st.download_button("⬇ Unduh Menu (.CSV)", data=csv, file_name="menu_mbg_aco.csv",
                               mime="text/csv", disabled=not ok, key="btn_dl")
            st.button("↺ Reset Parameter", on_click=reset_params, key="btn_reset")

    empty_cats = [k for k, v in ds.items() if not v]
    if empty_cats:
        st.warning(f"Kategori berikut kosong dan akan dilewati: {', '.join(empty_cats)}")

    # ---- KPI ----
    render_kpis(hasil)

    # ---- Dua kolom: parameter | hasil ----
    col_l, col_r = st.columns([5, 7], gap="medium")

    with col_l:
        with st.container(key="card-params"):
            sec_head("tune", "Parameter Algoritma ACO", right='<span class="muted">Metaheuristic Hyperparameters</span>')
            st.slider("Jumlah Koloni Semut (Ant Population)", 10, 200, step=10, key="p_semut")
            st.slider("Maksimal Siklus / Iterasi", 10, 250, step=10, key="p_iterasi")
            st.slider("Bobot Jejak Feromon (Alpha / α)", 0.1, 5.0, step=0.1, key="p_alpha")
            st.slider("Bobot Heuristik Gizi-Biaya (Beta / β)", 0.5, 5.0, step=0.1, key="p_beta")
            st.slider("Koefisien Penguapan Feromon (Rho / ρ)", 0.05, 0.9, step=0.05, key="p_rho")
            st.number_input("Konstanta Deposit Jejak (Q)", min_value=10.0, max_value=1000.0, step=10.0, key="p_q")

        with st.container(key="card-batas"):
            sec_head("checklist", "Batasan Gizi &amp; Biaya")
            st.number_input("Minimum Kalori (kkal)", min_value=100, max_value=2000, step=50, key="p_min_kalori")
            st.number_input("Minimum Protein (gram)", min_value=5, max_value=100, step=1, key="p_min_protein")
            st.number_input("Pagu Biaya Bahan Mentah Maksimal (Rp / porsi)", min_value=5000, max_value=50000,
                            step=500, key="p_budget")

    with col_r:
        with st.container(key="card-chart"):
            sec_head("show_chart", "Grafik Konvergensi Iterasi ACO",
                     "Perkembangan biaya menu terbaik dan rata-rata koloni tiap iterasi")
            if ok:
                fig = grafik_konvergensi(hasil)
                st.pyplot(fig)
                plt.close(fig)
            elif hasil:
                empty_state("Belum ada grafik karena tidak ada solusi yang valid.")
            else:
                empty_state("Atur parameter lalu klik <b>Jalankan Optimasi ACO</b> untuk melihat grafik konvergensi.")

        with st.container(key="card-menu"):
            badge = f'<span class="badge b-ok">{len(hasil["menu_terbaik"])} Komponen Siap Saji</span>' if ok else ""
            sec_head("restaurant_menu", "Kompilasi Menu &amp; Rekomendasi Takaran Porsi", right=badge)
            if ok:
                md(html_menu(hasil["menu_terbaik"], hasil["harga_terbaik"], hasil["params"]))
            elif hasil:
                st.error("Semut gagal menemukan kombinasi yang valid.")
                st.markdown(
                    "- Turunkan **Minimum Kalori** atau **Minimum Protein**\n"
                    "- Naikkan **Pagu Biaya**\n"
                    "- Tambah bahan di tiap kategori pada halaman *Katalog Bahan*\n"
                    "- Naikkan **Jumlah Semut** atau **Iterasi**"
                )
            else:
                empty_state("Menu hasil optimasi akan tampil di sini.")

    # ---- Tabel detail ----
    if ok:
        with st.container(key="card-table"):
            sec_head("table_chart", "Detail Tabel Menu")
            df = pd.DataFrame([
                {"Kategori": m["kategori"], "Bahan": m["nama"], "Porsi (g)": m["gram"],
                 "Harga (Rp)": round(m["harga"]), "Kalori (kkal)": round(m["kalori"], 1),
                 "Protein (g)": round(m["protein"], 1)}
                for m in hasil["menu_terbaik"]
            ])
            st.dataframe(df, hide_index=True)

    md(f"""
    <div class="banner"><div class="banner-ico">{icon("verified_user", 18)}</div>
    <div><b>Hasil simulasi dapat diunduh sebagai CSV</b>
    <span>Gunakan sebagai dasar daftar belanja logistik dan verifikasikan dengan tenaga ahli gizi sebelum dipakai.</span></div></div>
    """)


# ============================================================
# HALAMAN 2 — KATALOG BAHAN (CRUD)
# ============================================================
def simpan_kategori(kat, edited, hapus_dicentang):
    new_items, removed = [], 0
    for _, row in edited.iterrows():
        if hapus_dicentang and row["hapus"]:
            removed += 1
            continue
        new_items.append({
            "nama": row["nama"], "harga": int(row["harga"]), "kalori": float(row["kalori"]),
            "protein": float(row["protein"]), "icon": row.get("icon", "🍽️") or "🍽️",
        })
    st.session_state.dataset[kat] = new_items
    st.session_state.pop(f"editor_{kat}", None)
    return removed


def page_katalog():
    ds = st.session_state.dataset
    page_intro("Basis Data Pangan", "Katalog Bahan Makanan",
               "Tambah, ubah, atau hapus bahan makanan. Semua nilai gizi dan harga dihitung per 100 gram. "
               "Perubahan tersimpan selama sesi ini berjalan.")

    # Ringkasan
    if ds:
        cells = "".join(
            f'<div class="kpi"><div class="kpi-top"><span>{escape(k)}</span></div>'
            f'<div class="kpi-val">{len(v)} <small>bahan</small></div></div>' for k, v in ds.items()
        )
        md(f'<div class="kpi-grid">{cells}</div>')

    # Form tambah
    with st.container(key="card-tambah"):
        sec_head("add_circle", "Tambah Bahan Baru")
        with st.form("form_tambah", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                kat_pilih = st.selectbox("Kategori", list(ds.keys())) if ds else None
                kat_baru = st.text_input("Atau buat kategori baru", placeholder="misal: Minuman (kosongkan jika memakai kategori di atas)")
                nama = st.text_input("Nama Bahan*", placeholder="misal: Ikan Kembung")
                ikon = st.text_input("Emoji Icon", value="🍽️", max_chars=4)
            with c2:
                harga = st.number_input("Harga per 100g (Rp)*", 0, 100000, 2000, 100)
                kalori = st.number_input("Kalori per 100g (kkal)*", 0.0, 2000.0, 100.0, 5.0)
                protein = st.number_input("Protein per 100g (g)*", 0.0, 200.0, 5.0, 0.5)
            if st.form_submit_button("💾 Simpan Bahan", type="primary"):
                target = kat_baru.strip() or kat_pilih
                if not nama.strip():
                    st.error("Nama bahan wajib diisi.")
                elif not target:
                    st.error("Pilih atau isi nama kategori.")
                elif nama.strip().lower() in [m["nama"].lower() for m in ds.get(target, [])]:
                    st.error(f"Bahan '{nama}' sudah ada di kategori '{target}'.")
                else:
                    ds.setdefault(target, []).append({
                        "nama": nama.strip(), "harga": int(harga), "kalori": float(kalori),
                        "protein": float(protein), "icon": ikon or "🍽️",
                    })
                    st.rerun()

    # Edit & hapus
    with st.container(key="card-edit"):
        sec_head("edit_note", "Edit &amp; Hapus Bahan",
                 "Ubah nilai langsung di tabel. Centang kolom 🗑️ lalu klik Hapus untuk membuang baris.")
        for kat in list(ds.keys()):
            items = ds[kat]
            with st.expander(f"{kat} — {len(items)} bahan", expanded=True):
                if not items:
                    st.info("Kategori ini kosong. Tambahkan bahan lewat form di atas.")
                    if st.button(f"Hapus kategori '{kat}'", key=f"delkat_{kat}"):
                        del ds[kat]
                        st.rerun()
                    continue
                df = pd.DataFrame(items)
                df.insert(0, "hapus", False)
                edited = st.data_editor(
                    df, key=f"editor_{kat}", hide_index=True, num_rows="fixed",
                    column_config={
                        "hapus": st.column_config.CheckboxColumn("🗑️", width="small"),
                        "icon": st.column_config.TextColumn("Icon", width="small", max_chars=4),
                        "nama": st.column_config.TextColumn("Nama Bahan", required=True),
                        "harga": st.column_config.NumberColumn("Harga/100g (Rp)", min_value=0, step=100, format="%d"),
                        "kalori": st.column_config.NumberColumn("Kalori/100g", min_value=0.0, step=1.0, format="%.1f"),
                        "protein": st.column_config.NumberColumn("Protein/100g (g)", min_value=0.0, step=0.5, format="%.1f"),
                    },
                )
                ca, cb, cc = st.columns([1, 1, 1])
                if ca.button("💾 Simpan Perubahan", key=f"save_{kat}"):
                    simpan_kategori(kat, edited, hapus_dicentang=False)
                    st.rerun()
                if cb.button("🗑️ Hapus yang Dicentang", key=f"delcheck_{kat}"):
                    simpan_kategori(kat, edited, hapus_dicentang=True)
                    st.rerun()
                if cc.button(f"❌ Hapus Kategori", key=f"delall_{kat}"):
                    del ds[kat]
                    st.rerun()

    # Backup / restore
    with st.container(key="card-backup"):
        sec_head("database", "Backup &amp; Restore Dataset")
        col_exp, col_imp = st.columns(2)
        with col_exp:
            rows = [{**m, "kategori": k} for k, v in ds.items() for m in v]
            if rows:
                df_all = pd.DataFrame(rows)[["kategori", "nama", "icon", "harga", "kalori", "protein"]]
                st.download_button("⬇ Unduh Dataset (.CSV)", data=df_all.to_csv(index=False).encode("utf-8"),
                                   file_name="dataset_bahan_mbg.csv", mime="text/csv", key="dl_dataset")
            st.button("↺ Reset Dataset ke Default", on_click=reset_dataset, key="btn_reset_ds")
        with col_imp:
            up = st.file_uploader("⬆ Upload Dataset (.CSV)", type=["csv"], key="upload_dataset")
            if up is not None:
                try:
                    df_in = pd.read_csv(up)
                    need = {"kategori", "nama", "harga", "kalori", "protein"}
                    if not need.issubset(df_in.columns):
                        st.error(f"CSV harus punya kolom: {sorted(need)}")
                    else:
                        new_ds = {}
                        for _, r in df_in.iterrows():
                            new_ds.setdefault(str(r["kategori"]), []).append({
                                "nama": str(r["nama"]), "harga": int(r["harga"]),
                                "kalori": float(r["kalori"]), "protein": float(r["protein"]),
                                "icon": str(r["icon"]) if "icon" in df_in.columns else "🍽️",
                            })
                        if st.button("✅ Terapkan Dataset dari CSV", key="apply_csv"):
                            st.session_state.dataset = new_ds
                            st.rerun()
                except Exception as e:
                    st.error(f"Gagal membaca CSV: {e}")
        with st.expander("Lihat dataset (JSON)"):
            st.json(ds)


# ============================================================
# HALAMAN 3 — PERENCANAAN MINGGUAN
# ============================================================
def page_mingguan():
    page_intro("Perencanaan", "Perencanaan Menu Mingguan",
               "Jalankan ACO sekali untuk tiap hari sekolah (Senin–Jumat) dengan parameter dan batasan yang sedang "
               "diatur di halaman Simulasi. Karena ACO bersifat stokastik, menu tiap hari bisa berbeda.")
    st.button("📅 Susun Rencana Mingguan", type="primary", on_click=susun_mingguan, key="btn_week")

    data = st.session_state.mingguan
    with st.container(key="card-week"):
        sec_head("calendar_month", "Rencana Lima Hari")
        if not data:
            empty_state("Klik <b>Susun Rencana Mingguan</b> untuk membuat menu Senin–Jumat.")
            return
        cards, total, rows = "", 0, []
        for d in data["rencana"]:
            if not d["menu"]:
                cards += f'<div class="day"><div class="day-h"><b>{d["hari"]}</b><span>—</span></div><div class="day-i">Tidak ada solusi valid</div></div>'
                continue
            total += d["harga"]
            items = "".join(
                f'<div class="day-i"><span>{escape(str(m["icon"]))} {escape(m["nama"])}</span><span>{m["gram"]} g</span></div>'
                for m in d["menu"]
            )
            cards += f'<div class="day"><div class="day-h"><b>{d["hari"]}</b><span>{rp(d["harga"])}</span></div>{items}</div>'
            rows += [{"Hari": d["hari"], "Kategori": m["kategori"], "Bahan": m["nama"], "Porsi (g)": m["gram"],
                      "Harga (Rp)": round(m["harga"]), "Kalori (kkal)": round(m["kalori"], 1),
                      "Protein (g)": round(m["protein"], 1)} for m in d["menu"]]
        md(f'<div class="day-grid">{cards}</div>')
        md(f"""
        <div class="total-box"><div><div class="total-label">Total Biaya Bahan Satu Minggu (per porsi)</div></div>
        <div class="total-price">{rp(total)}</div></div>
        """)
        if rows:
            df = pd.DataFrame(rows)
            st.download_button("⬇ Unduh Rencana Mingguan (.CSV)", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="rencana_mingguan_mbg.csv", mime="text/csv", key="dl_week")
            with st.expander("Lihat tabel lengkap"):
                st.dataframe(df, hide_index=True)


# ============================================================
# HALAMAN 4 — LAPORAN & EVALUASI BIAYA
# ============================================================
def page_laporan():
    page_intro("Evaluasi", "Laporan &amp; Evaluasi Biaya",
               "Rincian biaya dari hasil simulasi terakhir dan rencana mingguan (jika sudah disusun).")
    hasil, mg = st.session_state.hasil, st.session_state.mingguan
    ok = bool(hasil and hasil.get("menu_terbaik"))

    if not ok and not mg:
        with st.container(key="card-lap-kosong"):
            empty_state("Belum ada data. Jalankan optimasi di halaman <b>Simulasi &amp; Optimasi ACO</b> terlebih dahulu.")
        return

    if ok:
        p, menu, harga = hasil["params"], hasil["menu_terbaik"], hasil["harga_terbaik"]
        sisa = p["budget"] - harga
        cards = [
            kpi("Biaya Menu Harian", "payments", "#ffdcc3", "#2f1500", rp(harga), f'Pagu: {rp(p["budget"])}',
                "b-ok", "savings", f"Sisa {rp(sisa)}"),
            kpi("Pemakaian Pagu", "percent", "#85f8c4", "#002114", f'{harga / p["budget"] * 100:.1f}<small>%</small>',
                "Dari pagu per porsi", "b-ok", "check_circle", "Dalam anggaran"),
        ]
        if mg:
            valid = [d["harga"] for d in mg["rencana"] if d["menu"]]
            if valid:
                cards += [
                    kpi("Rata-rata Harian (Minggu)", "calendar_month", "#dae2fd", "#131b2e", rp(sum(valid) / len(valid)),
                        f"{len(valid)} hari tersusun", "b-neutral", "info", "Dari rencana mingguan"),
                    kpi("Total Satu Minggu", "account_balance_wallet", "#dce9ff", "#006948", rp(sum(valid)),
                        "Per porsi, Senin–Jumat", "b-neutral", "info", "Dari rencana mingguan"),
                ]
        md(f'<div class="kpi-grid">{"".join(cards)}</div>')

        with st.container(key="card-lap-kat"):
            sec_head("pie_chart", "Distribusi Biaya per Kategori", "Menu terbaik dari simulasi terakhir")
            per_kat = {}
            for m in menu:
                per_kat[m["kategori"]] = per_kat.get(m["kategori"], 0) + m["harga"]
            md(f'<div class="bars">{"".join(bar(escape(k), f"{rp(v)} ({v / harga * 100:.0f}%)", v / harga * 100) for k, v in per_kat.items())}</div>')
    else:
        with st.container(key="card-lap-info"):
            empty_state("Rincian per kategori tersedia setelah simulasi dijalankan di halaman Simulasi.")


# ============================================================
# RAKIT HALAMAN
# ============================================================
render_header()
with st.sidebar:
    md('<div class="nav-label">Navigasi Sistem</div>')
    st.radio("Navigasi", NAV, key="nav", label_visibility="collapsed")
    render_sidebar_card(st.session_state.hasil)

{
    NAV[0]: page_simulasi,
    NAV[1]: page_katalog,
    NAV[2]: page_mingguan,
    NAV[3]: page_laporan,
}[st.session_state.nav]()