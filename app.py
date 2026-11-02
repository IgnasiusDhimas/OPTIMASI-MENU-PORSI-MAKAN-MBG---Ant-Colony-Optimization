import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

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
# CUSTOM CSS (meniru gaya dashboard Tailwind)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Kartu metric */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e7e3;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { font-size: 13px !important; color: #3d4a42; }
[data-testid="stMetricValue"] { font-size: 28px !important; color: #006948; font-weight: 700; }

/* Header utama */
.main-header {
    background: linear-gradient(135deg, #006948 0%, #00855d 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 16px;
    margin-bottom: 20px;
}
.main-header h1 { margin: 0; font-size: 26px; font-weight: 700; }
.main-header p { margin: 6px 0 0 0; opacity: 0.9; font-size: 14px; }

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.badge-success { background: #d4f5e3; color: #006948; }
.badge-warning { background: #fff4e0; color: #8d4b00; }
.badge-danger  { background: #ffdad6; color: #ba1a1a; }

/* Card */
.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.card-title {
    font-size: 16px;
    font-weight: 600;
    color: #0b1c30;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Tombol utama */
.stButton > button {
    background: #006948;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00855d;
    transform: translateY(-1px);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8f9ff;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATASET DEFAULT (bisa diubah user)
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

# ============================================================
# SESSION STATE — dataset bisa diedit user
# ============================================================
if "dataset" not in st.session_state:
    st.session_state.dataset = deepcopy(DEFAULT_DATASET)

if "hasil" not in st.session_state:
    st.session_state.hasil = None

if "riwayat_terbaik" not in st.session_state:
    st.session_state.riwayat_terbaik = []

if "riwayat_rata" not in st.session_state:
    st.session_state.riwayat_rata = []

# ============================================================
# HELPER FUNCTIONS
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
    """Jalankan simulasi ACO dan kembalikan hasil."""
    JUMLAH_SEMUT = params["semut"]
    ITERASI = params["iterasi"]
    ALFA = params["alpha"]
    BETA = params["beta"]
    EVAPORASI = params["rho"]
    Q = params["q"]
    MIN_KALORI = params["min_kalori"]
    MIN_PROTEIN = params["min_protein"]
    BUDGET = params["budget"]

    # Init feromon
    feromon = {}
    for kat, daftar in dataset.items():
        for m in daftar:
            feromon[m["nama"]] = 1.0

    riwayat_terbaik = []
    riwayat_rata = []
    menu_terbaik = None
    harga_terbaik = 0

    # Pre-extract category list untuk efisiensi
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
                    "nama": pilih["nama"],
                    "gram": gram,
                    "harga": pilih["harga"] * f,
                    "kalori": pilih["kalori"] * f,
                    "protein": pilih["protein"] * f,
                    "icon": pilih.get("icon", "🍽️"),
                    "kategori": kat,
                })

            t_harga = sum(x["harga"] for x in menu)
            t_kalori = sum(x["kalori"] for x in menu)
            t_protein = sum(x["protein"] for x in menu)
            total_harga_semut += t_harga

            if t_kalori >= MIN_KALORI and t_protein >= MIN_PROTEIN and t_harga <= BUDGET:
                fitness = t_harga  # maximize cost within budget
                solusi.append({"menu": menu, "harga": t_harga, "fitness": fitness})

        # Evaporasi
        for k in feromon:
            feromon[k] *= (1 - EVAPORASI)

        # Deposit feromon dari solusi terbaik
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
        "menu_terbaik": menu_terbaik,
        "harga_terbaik": harga_terbaik,
        "riwayat_terbaik": riwayat_terbaik,
        "riwayat_rata": riwayat_rata,
    }


# ============================================================
# SIDEBAR — PARAMETER ACO
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Parameter ACO")
    semut = st.slider("Jumlah Koloni Semut", 10, 200, 60, 10)
    iterasi = st.slider("Maksimal Iterasi", 10, 250, 80, 10)
    alpha = st.slider("Alpha (α) — Bobot Feromon", 0.1, 5.0, 1.0, 0.1)
    beta = st.slider("Beta (β) — Bobot Heuristik", 0.5, 5.0, 2.0, 0.1)
    rho = st.slider("Rho (ρ) — Penguapan Feromon", 0.05, 0.9, 0.15, 0.05)
    q = st.number_input("Konstanta Deposit Q", 10.0, 1000.0, 100.0, 10.0)

    st.markdown("---")
    st.markdown("## 🎯 Batasan Gizi & Biaya")
    min_kalori = st.number_input("Min Kalori (kkal)", 100, 2000, 600, 50)
    min_protein = st.number_input("Min Protein (g)", 5, 100, 20, 1)
    budget = st.number_input("Budget Maks (Rp)", 5000, 50000, 15000, 500)

    st.markdown("---")
    if st.button("🔄 Reset Semua ke Default", use_container_width=True):
        st.session_state.dataset = deepcopy(DEFAULT_DATASET)
        st.session_state.hasil = None
        st.session_state.riwayat_terbaik = []
        st.session_state.riwayat_rata = []
        st.success("Dataset & parameter direset.")
        st.rerun()


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🐜 MBG NutriOpt — Simulasi ACO</h1>
    <p>Sistem Optimasi Menu & Porsi Bergizi Gratis berbasis Ant Colony Optimization</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["🚀 Simulasi & Optimasi", "📚 Kelola Bahan Makanan"])

# ============================================================
# TAB 1: SIMULASI
# ============================================================
with tab1:
    # Ringkasan dataset
    total_item = sum(len(v) for v in st.session_state.dataset.values())
    total_kategori = len(st.session_state.dataset)
    st.caption(f"📦 Dataset aktif: **{total_item} bahan** dalam **{total_kategori} kategori** "
               f"({', '.join(st.session_state.dataset.keys())})")

    # Tombol jalankan
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        run_btn = st.button("▶ Jalankan Optimasi ACO", type="primary", use_container_width=True)
    with col_btn2:
        if st.session_state.hasil:
            # Export CSV
            df_export = pd.DataFrame(st.session_state.hasil["menu_terbaik"])
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Unduh Menu (.CSV)",
                data=csv,
                file_name="menu_mbg_aco.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col_btn3:
        if st.button("📊 Ulangi dengan Seed Sama", use_container_width=True):
            if st.session_state.hasil:
                st.info("Simulasi ulang akan memberi hasil berbeda (ACO stokastik).")

    # Jalankan simulasi
    if run_btn:
        if total_item == 0:
            st.error("❌ Dataset kosong! Tambahkan minimal 1 bahan per kategori di tab 'Kelola Bahan Makanan'.")
        else:
            # Cek setiap kategori punya minimal 1 item
            empty_cats = [k for k, v in st.session_state.dataset.items() if len(v) == 0]
            if empty_cats:
                st.warning(f"⚠️ Kategori berikut kosong dan akan dilewati: {', '.join(empty_cats)}")

            with st.spinner("Menjalankan simulasi ACO..."):
                params = {
                    "semut": semut, "iterasi": iterasi,
                    "alpha": alpha, "beta": beta, "rho": rho, "q": q,
                    "min_kalori": min_kalori, "min_protein": min_protein,
                    "budget": budget,
                }
                hasil = run_aco(st.session_state.dataset, params)
                st.session_state.hasil = hasil
                st.session_state.riwayat_terbaik = hasil["riwayat_terbaik"]
                st.session_state.riwayat_rata = hasil["riwayat_rata"]

    # Tampilkan hasil
    hasil = st.session_state.hasil
    if hasil and hasil["menu_terbaik"]:
        menu = hasil["menu_terbaik"]
        total_kal = sum(m["kalori"] for m in menu)
        total_prot = sum(m["protein"] for m in menu)
        harga = hasil["harga_terbaik"]

        # KPI Cards
        st.markdown("### 📊 Ringkasan Hasil Optimasi")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Total Biaya", f"Rp {harga:,.0f}".replace(",", "."),
                  delta=f"{(budget - harga):,.0f} sisa".replace(",", "."))
        c2.metric("🔥 Total Kalori", f"{total_kal:.1f} kkal",
                  delta=f"+{total_kal - min_kalori:.0f} dari target")
        c3.metric("💪 Total Protein", f"{total_prot:.1f} g",
                  delta=f"+{total_prot - min_protein:.1f} dari target")
        c4.metric("🍽️ Jumlah Komponen", f"{len(menu)} item",
                  delta=f"{len(set(m['kategori'] for m in menu))} kategori")

        # Status badge
        if harga <= budget and total_kal >= min_kalori and total_prot >= min_protein:
            st.markdown('<span class="badge badge-success">✅ SOLUSI VALID — Memenuhi semua batasan</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-warning">⚠️ SOLUSI PARSIAL</span>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # Grafik + Menu side by side
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<div class="card-title">📈 Grafik Konvergensi ACO</div>', unsafe_allow_html=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

            # Grafik 1: terbaik global
            r_terbaik = hasil["riwayat_terbaik"]
            x = list(range(1, len(r_terbaik) + 1))
            ax1.plot(x, [v if v else np.nan for v in r_terbaik],
                     color="#006948", linewidth=2.5, label="Terbaik Global")
            ax1.axhline(y=budget, color="#e53935", linestyle="--",
                        linewidth=1.5, label=f"Budget (Rp {budget:,.0f})")
            ax1.set_title("Optimasi Nilai Terbaik Global", fontweight="bold", fontsize=11)
            ax1.set_xlabel("Iterasi", fontsize=9)
            ax1.set_ylabel("Rp", fontsize=9)
            ax1.legend(fontsize=8, loc="lower right")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.tick_params(labelsize=8)

            # Grafik 2: rata-rata koloni
            ax2.plot(x, hasil["riwayat_rata"], color="#66bb6a",
                     linewidth=2, label="Rata-rata Koloni")
            ax2.axhline(y=budget, color="#e53935", linestyle="--",
                        linewidth=1.5, label=f"Budget")
            ax2.set_title("Fluktuasi Rata-rata Koloni", fontweight="bold", fontsize=11)
            ax2.set_xlabel("Iterasi", fontsize=9)
            ax2.set_ylabel("Rp", fontsize=9)
            ax2.legend(fontsize=8, loc="lower right")
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.tick_params(labelsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.markdown('<div class="card-title">🍽️ Komposisi Isi Piringku</div>', unsafe_allow_html=True)
            for m in menu:
                pct_budget = (m["harga"] / harga * 100) if harga > 0 else 0
                st.markdown(f"""
                <div style="background:#f1f8e9; border-left:4px solid #43a047;
                            padding:12px 16px; border-radius:8px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:18px;">{m['icon']}</span>
                            <strong style="font-size:14px;">{m['nama']}</strong>
                            <span style="font-size:11px; color:#666; margin-left:6px;">
                                ({m['kategori']})
                            </span>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:#006948; font-weight:700; font-size:14px;">
                                Rp {m['harga']:,.0f}
                            </div>
                            <div style="font-size:11px; color:#666;">{pct_budget:.0f}% budget</div>
                        </div>
                    </div>
                    <div style="font-size:12px; color:#555; margin-top:4px;">
                        Porsi: <strong>{m['gram']} g</strong> &nbsp;|&nbsp;
                        {m['kalori']:.0f} kkal &nbsp;|&nbsp;
                        {m['protein']:.1f} g protein
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Total ringkas
            st.markdown(f"""
            <div style="background:#006948; color:white; padding:16px;
                        border-radius:12px; margin-top:12px; text-align:center;">
                <div style="font-size:12px; opacity:0.9;">TOTAL BIAYA BAHAN</div>
                <div style="font-size:26px; font-weight:700;">Rp {harga:,.0f}</div>
                <div style="font-size:12px; opacity:0.9;">
                    dari budget Rp {budget:,.0f} ({(harga/budget*100):.1f}%)
                </div>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)

        # Tabel detail + unduh
        st.markdown("---")
        st.markdown("### 📋 Detail Tabel Menu")
        df_menu = pd.DataFrame([
            {
                "Kategori": m["kategori"],
                "Bahan": m["nama"],
                "Porsi (g)": m["gram"],
                "Harga (Rp)": round(m["harga"]),
                "Kalori (kkal)": round(m["kalori"], 1),
                "Protein (g)": round(m["protein"], 1),
            }
            for m in menu
        ])
        st.dataframe(df_menu, use_container_width=True, hide_index=True)

    elif hasil and not hasil["menu_terbaik"]:
        st.error("❌ **Semut gagal menemukan kombinasi yang valid!**")
        st.markdown("""
        **Saran perbaikan:**
        - Turunkan **Min Kalori** atau **Min Protein**
        - Naikkan **Budget Maks**
        - Tambah lebih banyak **bahan** di tiap kategori
        - Naikkan **Jumlah Semut** atau **Iterasi**
        """)

    else:
        st.info("👆 Klik tombol **'Jalankan Optimasi ACO'** untuk memulai simulasi.")


# ============================================================
# TAB 2: KELOLA BAHAN MAKANAN (CRUD)
# ============================================================
with tab2:
    st.markdown("### 📚 Kelola Bahan Makanan")
    st.caption("Tambah, edit, atau hapus bahan makanan sesuai kebutuhan. Perubahan otomatis tersimpan di sesi ini.")

    # ---- Ringkasan per kategori ----
    st.markdown("#### 📦 Ringkasan Dataset")
    cols = st.columns(len(st.session_state.dataset) if st.session_state.dataset else 1)
    for i, (kat, items) in enumerate(st.session_state.dataset.items()):
        with cols[i % len(cols)]:
            st.metric(kat, f"{len(items)} bahan")

    st.markdown("---")

    # ---- FORM TAMBAH BAHAN ----
    with st.expander("➕ **Tambah Bahan Makanan Baru**", expanded=False):
        with st.form("form_tambah", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                kategori_baru = st.selectbox(
                    "Kategori",
                    list(st.session_state.dataset.keys()) + ["➕ Kategori Baru"],
                )
                if kategori_baru == "➕ Kategori Baru":
                    nama_kategori_baru = st.text_input(
                        "Nama Kategori Baru", placeholder="misal: Buah, Minuman"
                    )
                else:
                    nama_kategori_baru = None

                nama_bahan = st.text_input("Nama Bahan*", placeholder="misal: Ikan Kembung")
                icon_bahan = st.text_input("Emoji Icon", value="🍽️", max_chars=4)

            with col2:
                harga = st.number_input("Harga per 100g (Rp)*", 0, 100000, 2000, 100)
                kalori = st.number_input("Kalori per 100g (kkal)*", 0.0, 2000.0, 100.0, 5.0)
                protein = st.number_input("Protein per 100g (g)*", 0.0, 200.0, 5.0, 0.5)

            submitted = st.form_submit_button("💾 Simpan Bahan", type="primary", use_container_width=True)

            if submitted:
                target_kat = nama_kategori_baru.strip() if kategori_baru == "➕ Kategori Baru" else kategori_baru
                if not nama_bahan.strip():
                    st.error("❌ Nama bahan wajib diisi.")
                elif not target_kat:
                    st.error("❌ Nama kategori baru wajib diisi.")
                else:
                    # Cek duplikat
                    existing = [m["nama"].lower() for m in st.session_state.dataset.get(target_kat, [])]
                    if nama_bahan.strip().lower() in existing:
                        st.error(f"❌ Bahan '{nama_bahan}' sudah ada di kategori '{target_kat}'.")
                    else:
                        if target_kat not in st.session_state.dataset:
                            st.session_state.dataset[target_kat] = []
                        st.session_state.dataset[target_kat].append({
                            "nama": nama_bahan.strip(),
                            "harga": int(harga),
                            "kalori": float(kalori),
                            "protein": float(protein),
                            "icon": icon_bahan or "🍽️",
                        })
                        st.success(f"✅ '{nama_bahan}' ditambahkan ke kategori '{target_kat}'.")
                        st.rerun()

    # ---- EDIT & HAPUS PER KATEGORI ----
    st.markdown("#### ✏️ Edit & Hapus Bahan")
    st.caption("Ubah nilai langsung di tabel, lalu klik **Simpan Perubahan**. Atau centang baris lalu klik **Hapus**.")

    for kat in list(st.session_state.dataset.keys()):
        items = st.session_state.dataset[kat]
        with st.expander(f"**{kat}** — {len(items)} bahan", expanded=True):
            if not items:
                st.info("Kategori ini kosong. Tambahkan bahan di form atas.")
                # Tombol hapus kategori kosong
                if st.button(f"🗑️ Hapus Kategori '{kat}'", key=f"del_kat_{kat}"):
                    del st.session_state.dataset[kat]
                    st.success(f"Kategori '{kat}' dihapus.")
                    st.rerun()
                continue

            df = pd.DataFrame(items)
            df.insert(0, "hapus", False)
            edited = st.data_editor(
                df,
                key=f"editor_{kat}",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "hapus": st.column_config.CheckboxColumn("🗑️", width="small"),
                    "icon": st.column_config.TextColumn("Icon", width="small", max_chars=4),
                    "nama": st.column_config.TextColumn("Nama Bahan", required=True),
                    "harga": st.column_config.NumberColumn("Harga/100g (Rp)", min_value=0, step=100, format="%d"),
                    "kalori": st.column_config.NumberColumn("Kalori/100g", min_value=0.0, step=1.0, format="%.1f"),
                    "protein": st.column_config.NumberColumn("Protein/100g (g)", min_value=0.0, step=0.5, format="%.1f"),
                },
                num_rows="fixed",
            )

            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                if st.button(f"💾 Simpan Perubahan", key=f"save_{kat}", use_container_width=True):
                    new_items = []
                    for _, row in edited.iterrows():
                        if not row["hapus"]:
                            new_items.append({
                                "nama": row["nama"],
                                "harga": int(row["harga"]),
                                "kalori": float(row["kalori"]),
                                "protein": float(row["protein"]),
                                "icon": row.get("icon", "🍽️"),
                            })
                    st.session_state.dataset[kat] = new_items
                    st.success(f"✅ Perubahan '{kat}' disimpan.")
                    st.rerun()
            with col_b:
                if st.button(f"🗑️ Hapus yang Dicentang", key=f"delcheck_{kat}", use_container_width=True):
                    new_items = []
                    for _, row in edited.iterrows():
                        if not row["hapus"]:
                            new_items.append({
                                "nama": row["nama"],
                                "harga": int(row["harga"]),
                                "kalori": float(row["kalori"]),
                                "protein": float(row["protein"]),
                                "icon": row.get("icon", "🍽️"),
                            })
                    removed = len(items) - len(new_items)
                    st.session_state.dataset[kat] = new_items
                    if removed > 0:
                        st.success(f"🗑️ {removed} bahan dihapus dari '{kat}'.")
                    else:
                        st.info("Tidak ada baris yang dicentang.")
                    st.rerun()
            with col_c:
                # Tombol hapus seluruh kategori
                if st.button(f"❌ Hapus Seluruh Kategori '{kat}'", key=f"delall_{kat}", use_container_width=True):
                    del st.session_state.dataset[kat]
                    st.success(f"Kategori '{kat}' beserta {len(items)} bahan dihapus.")
                    st.rerun()

    # ---- EXPORT / IMPORT DATASET ----
    st.markdown("---")
    st.markdown("#### 💾 Backup / Restore Dataset")

    col_exp, col_imp = st.columns(2)

    with col_exp:
        # Export ke CSV (flatten)
        rows = []
        for kat, items in st.session_state.dataset.items():
            for m in items:
                rows.append({**m, "kategori": kat})
        if rows:
            df_all = pd.DataFrame(rows)[["kategori", "nama", "icon", "harga", "kalori", "protein"]]
            csv_all = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Unduh Dataset (.CSV)",
                data=csv_all,
                file_name="dataset_bahan_mbg.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col_imp:
        uploaded = st.file_uploader("⬆️ Upload Dataset (.CSV)", type=["csv"], key="upload_dataset")
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                required = {"kategori", "nama", "harga", "kalori", "protein"}
                if not required.issubset(df_in.columns):
                    st.error(f"❌ CSV harus punya kolom: {required}")
                else:
                    new_ds = {}
                    for _, r in df_in.iterrows():
                        kat = str(r["kategori"])
                        new_ds.setdefault(kat, []).append({
                            "nama": str(r["nama"]),
                            "harga": int(r["harga"]),
                            "kalori": float(r["kalori"]),
                            "protein": float(r["protein"]),
                            "icon": str(r.get("icon", "🍽️")) if "icon" in df_in.columns else "🍽️",
                        })
                    if st.button("✅ Terapkan Dataset dari CSV", use_container_width=True):
                        st.session_state.dataset = new_ds
                        st.success("Dataset berhasil diimpor!")
                        st.rerun()
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")

    # Preview JSON
    with st.expander("🔍 Lihat Dataset (JSON)"):
        st.json(st.session_state.dataset)