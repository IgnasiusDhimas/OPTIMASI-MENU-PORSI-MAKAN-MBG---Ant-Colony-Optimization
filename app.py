import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Simulasi ACO Menu MBG", page_icon="🐜", layout="wide")

st.title("🐜 Simulasi ACO Menu MBG")
st.caption("Optimasi Porsi Makanan Bergizi Seimbang Berdasarkan AKG Kemenkes")

# ============ SIDEBAR PARAMETER ============
with st.sidebar:
    st.header("⚙️ Parameter ACO")
    JUMLAH_SEMUT = st.slider("Jumlah Semut", 10, 200, 60)
    ITERASI = st.slider("Iterasi", 10, 200, 80)
    ALFA = st.slider("Alpha (feromon)", 0.1, 3.0, 1.0, 0.1)
    BETA = st.slider("Beta (heuristik)", 0.1, 5.0, 2.0, 0.1)
    EVAPORASI = st.slider("Evaporasi", 0.01, 0.5, 0.15, 0.01)
    st.divider()
    st.header("🎯 Target Gizi")
    MIN_KALORI = st.number_input("Min Kalori (kkal)", 100, 2000, 600, 50)
    MIN_PROTEIN = st.number_input("Min Protein (g)", 5, 100, 20, 1)
    BUDGET_MAKSIMAL = st.number_input("Budget Maks (Rp)", 5000, 50000, 15000, 500)

# ============ DATASET ============
dataset = {
    'Karbohidrat': [
        {"nama": "Nasi Putih",     "harga": 1500, "kalori": 175, "protein": 4},
        {"nama": "Kentang Rebus",  "harga": 2000, "kalori": 150, "protein": 3},
        {"nama": "Ubi Rebus",      "harga": 1800, "kalori": 160, "protein": 2}
    ],
    'Protein': [
        {"nama": "Ayam Goreng",    "harga": 5000, "kalori": 250, "protein": 20},
        {"nama": "Telur Dadar",    "harga": 3000, "kalori": 150, "protein": 12},
        {"nama": "Daging Sapi",    "harga": 4500, "kalori": 300, "protein": 25},
        {"nama": "Tempe Bacem",    "harga": 1200, "kalori": 100, "protein": 10},
        {"nama": "Tahu Goreng",    "harga": 1000, "kalori": 80,  "protein": 8}
    ],
    'Sayur': [
        {"nama": "Bayam / Bening", "harga": 800,  "kalori": 40,  "protein": 3},
        {"nama": "Sup Wortel",     "harga": 1000, "kalori": 50,  "protein": 2},
        {"nama": "Tumis Sawi",     "harga": 900,  "kalori": 45,  "protein": 2.5}
    ]
}

# ============ FUNGSI ACO ============
def run_aco():
    feromon = {m['nama']: 1.0 for kat in dataset for m in dataset[kat]}
    riwayat_terbaik, riwayat_rata = [], []
    menu_terbaik, harga_terbaik = None, 0

    progress = st.progress(0, text="Menjalankan simulasi...")
    for it in range(ITERASI):
        solusi = []
        total_harga_semut = 0

        for _ in range(JUMLAH_SEMUT):
            menu = []
            for kat in dataset:
                opsi = dataset[kat]
                weights = []
                for m in opsi:
                    tau = feromon[m['nama']] ** ALFA
                    eta = ((m['protein'] + m['kalori']/10) / m['harga']) ** BETA
                    weights.append(tau * eta)
                pilih = random.choices(opsi, weights=weights, k=1)[0]
                gram = random.randint(80, 250)
                f = gram / 100
                menu.append({
                    "nama": pilih["nama"], "gram": gram,
                    "harga": pilih["harga"]*f, "kalori": pilih["kalori"]*f,
                    "protein": pilih["protein"]*f
                })

            tH = sum(x['harga'] for x in menu)
            tK = sum(x['kalori'] for x in menu)
            tP = sum(x['protein'] for x in menu)
            total_harga_semut += tH
            if tK >= MIN_KALORI and tP >= MIN_PROTEIN and tH <= BUDGET_MAKSIMAL:
                solusi.append({"menu": menu, "harga": tH, "fitness": tH})

        for k in feromon: feromon[k] *= (1 - EVAPORASI)
        if solusi:
            best = max(solusi, key=lambda x: x['harga'])
            for m in best['menu']:
                feromon[m['nama']] += best['fitness'] * 0.001
            if best['harga'] > harga_terbaik:
                harga_terbaik = best['harga']
                menu_terbaik = best['menu']

        riwayat_terbaik.append(harga_terbaik if harga_terbaik else None)
        riwayat_rata.append(total_harga_semut / JUMLAH_SEMUT)
        progress.progress((it+1)/ITERASI, text=f"Iterasi {it+1}/{ITERASI}")

    progress.empty()
    return menu_terbaik, harga_terbaik, riwayat_terbaik, riwayat_rata

# ============ RUN ============
if st.button("▶ Jalankan Simulasi", type="primary", use_container_width=True):
    menu, harga, r_terbaik, r_rata = run_aco()

    if menu:
        total_kal = sum(m['kalori'] for m in menu)
        total_prot = sum(m['protein'] for m in menu)

        st.success("✅ Optimasi berhasil!")
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Harga", f"Rp {harga:.0f}")
        c2.metric("🔥 Total Kalori", f"{total_kal:.1f} kkal")
        c3.metric("💪 Total Protein", f"{total_prot:.1f} g")

        st.subheader("🍽️ Komposisi Isi Piringku Sehat")
        for m in menu:
            st.markdown(f"- **{m['nama']}** ({m['gram']} gram) — Rp {m['harga']:.0f}")

        st.subheader("📊 Grafik Konvergensi")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.plot(r_terbaik, color='darkgreen', lw=2.5, label='Terbaik Global')
        ax1.axhline(y=BUDGET_MAKSIMAL, color='red', ls='--', label='Budget')
        ax1.set_title('Optimasi Terbaik Global'); ax1.legend(); ax1.grid(True, ls=':', alpha=.6)
        ax1.set_xlabel('Iterasi'); ax1.set_ylabel('Rp')

        ax2.plot(r_rata, color='limegreen', lw=2, label='Rata-rata Koloni')
        ax2.axhline(y=BUDGET_MAKSIMAL, color='red', ls='--', label='Budget')
        ax2.set_title('Fluktuasi Rata-rata Koloni'); ax2.legend(); ax2.grid(True, ls=':', alpha=.6)
        ax2.set_xlabel('Iterasi'); ax2.set_ylabel('Rp')
        st.pyplot(fig)
    else:
        st.error("❌ Semut gagal menemukan kombinasi. Longgarkan parameter atau jalankan ulang.")