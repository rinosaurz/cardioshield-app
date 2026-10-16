"""
CardioShield v2 - Pusat Skrining & Tanggap Darurat Kardiovaskular (prototipe edukasi)

Jalankan di VS Code (terminal):
    pip install "streamlit>=1.32" matplotlib numpy
    streamlit run cardioshield_v2.py

Catatan penting: aplikasi ini ALAT EDUKASI & TRIASE AWAL, bukan alat diagnosis dan
bukan prediktor risiko tervalidasi (mis. WHO/ISH, Framingham). Bobot poin pada skrining
bersifat heuristik; validasi klinis adalah langkah pengembangan berikutnya.
Tidak ada data pengguna yang disimpan (hanya session Streamlit di memori).
"""
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="CardioShield v2", page_icon="❤️", layout="wide")

# ============================================================
# DATA & SUMBER (semua angka di bawah ini dirujuk ke sumbernya)
# ============================================================
SOURCES = [
    "[1] WHO. Cardiovascular diseases (CVDs) - Fact sheet. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "[2] Kemenkes RI, BKPK. Survei Kesehatan Indonesia (SKI) 2023 (ringkasan hipertensi usia muda): https://www.badankebijakan.kemkes.go.id/bahaya-hipertensi-mengintai-anak-muda-indonesia/",
    "[3] Kemenkes RI. Cek Kesehatan Gratis (CKG): https://ayosehat.kemkes.go.id/cek-kesehatan-gratis dan https://www.kemkes.go.id/id/kemenkes-perkuat-tindak-lanjut-cek-kesehatan-gratis-usai-tembus-73,8-juta-peserta",
    "[4] PERHI. Konsensus Penatalaksanaan Hipertensi 2019 (update 2021): http://faber.inash.or.id/upload/pdf/article_Update_konsensus_201939.pdf",
    "[5] Saver JL. Time is brain - quantified. Stroke. 2006;37(1):263-266. https://pubmed.ncbi.nlm.nih.gov/16339467/",
]

# SKI 2023: hasil PENGUKURAN vs DIAGNOSIS DOKTER (persen). Diabetes (>=15 th) dikutip dari
# artikel jurnal yang merujuk BKPK 2023 -> verifikasi ke laporan SKI sebelum dipakai di esai.
DETECTION_GAP = {
    "Hipertensi\n(≥18 th)": (30.8, 8.6),
    "Hipertensi\n18-24 th": (10.7, 0.4),
    "Hipertensi\n25-34 th": (17.4, 1.8),
    "Diabetes\n(≥15 th)": (11.7, 2.2),
}

NEURONS_PER_MIN_MILLION = 1.9  # Saver 2006: stroke iskemik pembuluh besar tipikal [5]

BP_LABELS = [  # klasifikasi tekanan darah klinik PERHI [4]
    "Optimal (<120/<80)",
    "Normal (120-129 / 80-84)",
    "Normal-tinggi (130-139 / 85-89)",
    "Hipertensi derajat 1 (140-159 / 90-99)",
    "Hipertensi derajat 2 (160-179 / 100-109)",
    "Hipertensi derajat 3 (≥180 / ≥110)",
]
BP_THRESHOLDS = [(120, 80), (130, 85), (140, 90), (160, 100), (180, 110)]
BP_POINTS = [0, 0, 1, 2, 3, 4]

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
<style>
    .box {padding:1rem 1.2rem;border-radius:10px;margin:.5rem 0;}
    .green {background:#dff0d8;color:#2d5a2d;border:1px solid #c3e0b5;}
    .yellow {background:#fcf8e3;color:#7a5d1e;border:1px solid #f0e0a8;}
    .red {background:#f7dcdc;color:#8f2a2a;border:1px solid #e8b4b4;}
    .emergency {background:#b71c1c;color:white;border:2px solid #7f0000;}
    .emergency h3, .emergency p {color:white;margin:.2rem 0;}
    div.stButton > button:first-child {background:#d9534f;color:white;border:none;
        border-radius:8px;font-weight:600;padding:.5rem 1.5rem;}
    div.stButton > button:first-child:hover {background:#c9302c;color:white;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOGIKA SKRINING
# ============================================================
def classify_bp(sys_bp: int, dia_bp: int) -> int:
    """Indeks kategori 0-5 menurut klasifikasi PERHI [4]."""
    cat = 0
    for k, (s, d) in enumerate(BP_THRESHOLDS, start=1):
        if sys_bp >= s or dia_bp >= d:
            cat = k
    return cat


def evaluate(p: dict) -> dict:
    """Hitung poin faktor risiko (transparan, heuristik) dan tentukan status triase."""
    pts: list[tuple[str, int]] = []
    reasons: list[str] = []
    unknown = 0
    high_marker = False

    # Usia
    age = p["age"]
    pts.append(("Usia", 0 if age < 40 else 1 if age < 55 else 2 if age < 65 else 3))

    # Tekanan darah
    bp_cat = None
    if p["bp_known"]:
        bp_cat = classify_bp(p["sys"], p["dia"])
        pts.append(("Tekanan darah", BP_POINTS[bp_cat]))
        if bp_cat >= 3:
            high_marker = True
            reasons.append(
                f"Tekanan darah {p['sys']}/{p['dia']} mmHg termasuk **{BP_LABELS[bp_cat]}** menurut PERHI. "
                "Diagnosis ditegakkan oleh tenaga kesehatan setelah pengukuran berulang."
            )
        elif bp_cat == 2:
            reasons.append(f"Tekanan darah {p['sys']}/{p['dia']} mmHg **normal-tinggi**: pantau rutin dan perbaiki gaya hidup.")
    else:
        unknown += 1
        pts.append(("Tekanan darah (tidak tahu)", 1))
        reasons.append("Tekanan darah belum diketahui. Hipertensi sering tanpa gejala; satu-satunya cara tahu adalah mengukurnya.")

    # Kolesterol
    if p["chol_known"]:
        hi, mid = (240, 200) if p["chol_type"] == "Kolesterol total" else (160, 130)
        v = p["chol_val"]
        pts.append(("Kolesterol", 2 if v >= hi else 1 if v >= mid else 0))
        if v >= hi:
            high_marker = True
            reasons.append(f"{p['chol_type']} {v} mg/dL tergolong **tinggi**.")
        elif v >= mid:
            reasons.append(f"{p['chol_type']} {v} mg/dL **batas tinggi**.")
    else:
        unknown += 1
        pts.append(("Kolesterol (tidak tahu)", 1))
        reasons.append("Kolesterol belum diketahui. Hanya tes darah (profil lipid) yang bisa mengukurnya; tanyakan ke petugas.")

    # Diabetes
    if p["diabetes"] == "Ya":
        high_marker = True
        pts.append(("Diabetes", 3))
        reasons.append("Diabetes meningkatkan risiko penyakit jantung dan stroke secara bermakna; kontrol rutin sangat penting.")
    elif p["diabetes"] == "Tidak yakin":
        unknown += 1
        pts.append(("Gula darah (tidak yakin)", 1))
        reasons.append("Status gula darah belum jelas; cek gula darah di fasilitas kesehatan.")
    else:
        pts.append(("Diabetes", 0))

    # Merokok
    smoke = p["smoking"]
    s_pts = 3 if smoke.startswith("Perokok aktif") else 1 if smoke.startswith("Perokok pasif") else 0
    pts.append(("Paparan rokok", s_pts))
    if s_pts == 3:
        reasons.append("Merokok aktif (termasuk rokok elektrik) adalah faktor risiko besar serangan jantung dan stroke yang **dapat diubah**.")
    elif s_pts == 1:
        reasons.append("Paparan asap rokok harian (perokok pasif) tetap meningkatkan risiko.")

    # Lingkar perut (cut-off Asia: pria >=90, wanita >=80 cm)
    waist = p["waist"]
    if waist > 0:
        central = waist >= (90 if p["sex"] == "Pria" else 80)
        pts.append(("Lingkar perut", 1 if central else 0))
        if central:
            reasons.append(f"Lingkar perut {waist} cm menunjukkan **obesitas sentral**.")
    # Riwayat keluarga
    if p["family"]:
        pts.append(("Riwayat keluarga", 1))
        reasons.append("Ada riwayat serangan jantung/stroke dini pada keluarga inti.")
    # Aktivitas, garam, alkohol
    if p["activity"] < 150:
        pts.append(("Kurang aktivitas", 1))
        reasons.append(f"Aktivitas {p['activity']} menit/minggu (< 150 menit yang dianjurkan).")
    if p["salt"].startswith("Sering"):
        pts.append(("Asin/gorengan sering", 1))
        reasons.append("Konsumsi makanan asin/olahan sering; WHO menganjurkan garam < 5 g/hari.")
    limit = 1.0 if p["sex"] == "Wanita" else 2.0
    if p["alcohol"] > limit:
        pts.append(("Alkohol", 1))
        reasons.append(f"Alkohol {p['alcohol']:.0f} porsi/hari melebihi batas anjuran (≤ {limit:.0f}).")

    total = sum(v for _, v in pts)
    if p["red_flags"]:
        status = "EMERGENCY"
    elif high_marker or total >= 7:
        status = "RED"
    elif total >= 3 or unknown:
        status = "YELLOW"
    else:
        status = "GREEN"
    return {"points": pts, "total": total, "reasons": reasons, "status": status, "bp_cat": bp_cat}


def bp_map(sys_bp: int, dia_bp: int):
    """Peta zona tekanan darah PERHI dengan posisi pengguna."""
    S, D = np.meshgrid(np.arange(80, 201), np.arange(40, 121))
    cat = np.zeros_like(S)
    for k, (s, d) in enumerate(BP_THRESHOLDS, start=1):
        cat[(S >= s) | (D >= d)] = k
    cmap = ListedColormap(["#c8e6c9", "#e6ee9c", "#fff59d", "#ffcc80", "#ef9a9a", "#e57373"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cat, origin="lower", extent=[40, 120, 80, 200], aspect="auto", cmap=cmap, vmin=0, vmax=5)
    ax.scatter([dia_bp], [sys_bp], color="black", s=90, zorder=5)
    ax.annotate("Anda", (dia_bp, sys_bp), textcoords="offset points", xytext=(8, 8), fontsize=9, weight="bold")
    ax.set_xlabel("Diastolik (mmHg)")
    ax.set_ylabel("Sistolik (mmHg)")
    ax.set_title("Posisi tekanan darah (klasifikasi PERHI)", fontsize=10)
    return fig


def points_chart(points: list[tuple[str, int]]):
    items = [(n, v) for n, v in points if v > 0] or [("Tidak ada faktor risiko", 0)]
    fig, ax = plt.subplots(figsize=(5, max(1.6, 0.45 * len(items))))
    ax.barh([n for n, _ in items][::-1], [v for _, v in items][::-1], color="#d9534f")
    ax.set_xlabel("Poin kontribusi (heuristik)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================
# VISUAL EDUKASI
# ============================================================
def artery_figure(stage: int):
    """Penampang arteri (1 sehat ... 4 plak masif + trombus). Deterministik."""
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.axhline(1.0, color="#c0504d", lw=10)
    ax.axhline(-1.0, color="#c0504d", lw=10)
    t = np.linspace(-1, 1, 200)
    top, bot, widths = [(0, 0, 0), (0.3, 0.1, 5), (0.65, 0.25, 3), (0.8, 0.4, 2.5)][stage - 1]
    if stage > 1:
        ax.fill_between(t * 1.5, 1 - top * np.exp(-widths * t**2), 1, color="#f0ad4e", label="Plak kolesterol")
        ax.fill_between(t * 1.5, -1, -1 + bot * np.exp(-widths * t**2), color="#f0ad4e")
    n = 35 - 5 * (stage - 1)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-0.55, 0.55, n) * (1 - 0.15 * (stage - 1))
    ax.scatter(x, y, color="#cc0000", s=90, alpha=0.6, label="Sel darah merah")
    if stage == 4:
        ax.scatter(rng.uniform(-0.3, 0.3, 14), rng.uniform(-0.3, 0.3, 14), color="#4a0000",
                   marker="X", s=140, zorder=5, label="Trombus (gumpalan)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    return fig


STAGES = {
    1: ("success", "Tahap 1 - Arteri sehat", "Dinding elastis dan halus; aliran darah bebas."),
    2: ("warning", "Tahap 2 - Plak awal", "LDL berlebih menembus lapisan dalam arteri; terbentuk garis lemak. Biasanya belum ada gejala."),
    3: ("warning", "Tahap 3 - Penyempitan", "Plak membesar dan lumen menyempit; bisa muncul nyeri dada saat aktivitas, tetapi sering masih tanpa gejala."),
    4: ("error", "Tahap 4 - Plak pecah & trombus", "Plak pecah, terbentuk gumpalan yang menyumbat total: penyebab utama serangan jantung dan stroke iskemik."),
}

# ============================================================
# GAME: data-driven (skenario -> langkah -> opsi)
# delay = menit yang terbuang; urutan opsi diacak tiap permainan
# ============================================================
def opt(text, ok, delay, fb):
    return {"text": text, "ok": ok, "delay": delay, "fb": fb}


SCENARIOS = {
    "stroke": {
        "title": "🧠 Stroke saat arisan keluarga",
        "intro": "Pak Hasan (65 tahun, riwayat hipertensi) tiba-tiba terlihat aneh saat berdiri mengambil piring. Anda berada di dekatnya.",
        "good": 10, "ok": 25, "metric": "neuron",
        "steps": [
            ("Memperhatikan", "Apa yang Anda lakukan?", [
                opt("Hampiri dan tanyakan keadaannya.", True, 1, "Tepat. Perubahan mendadak pada orang berisiko tinggi harus diperiksa."),
                opt("Biarkan saja, mungkin hanya lelah.", False, 15, "Stroke sering dimulai mendadak dan samar. Setiap menit menunda berarti jaringan otak hilang."),
                opt("Tunggu dulu, lihat apakah makin parah.", False, 10, "Menunggu memperburuk hasil; gejala ringan pun bisa tanda stroke."),
            ]),
            ("B - Balance (keseimbangan)", "Beliau tampak limbung dan berkata 'cuma tersandung'.", [
                opt("Perhatikan cara berdiri/berjalannya sambil dijaga agar tidak jatuh.", True, 1, "Hilangnya keseimbangan/koordinasi mendadak adalah tanda BE-FAST."),
                opt("Suruh duduk dan istirahat saja.", False, 5, "Tanda keseimbangan terlewat dan keputusan darurat tertunda."),
            ]),
            ("E - Eyes (penglihatan)", "Beliau berkedip-kedip dan bilang pandangannya kabur.", [
                opt("Tanya apakah penglihatan tiba-tiba kabur, ganda, atau hilang sebagian.", True, 1, "Gangguan penglihatan mendadak adalah tanda stroke."),
                opt("Ambilkan kacamata hitam karena silau.", False, 5, "Mengira silau menunda pengenalan gangguan penglihatan akibat stroke."),
            ]),
            ("F - Face (wajah)", "Anda perlu memeriksa wajahnya.", [
                opt("Minta tersenyum lebar; lihat apakah satu sisi wajah turun.", True, 1, "Wajah perot/asimetris adalah tanda khas."),
                opt("Beri segelas teh manis agar segar.", False, 5, "Jangan beri makan/minum: fungsi menelan bisa terganggu sehingga berisiko tersedak."),
            ]),
            ("A - Arm (lengan)", "Cara menilai kelemahan lengan?", [
                opt("Minta kedua lengan diangkat lurus ke depan ±10 detik; lihat apakah satu turun.", True, 1, "Lengan yang melorot satu sisi menandakan kelemahan sisi tubuh."),
                opt("Pijat lengannya atau tusuk ujung jari agar 'darah keluar'.", False, 8, "Mitos: tidak ada bukti manfaatnya dan menunda pertolongan medis."),
            ]),
            ("S - Speech (bicara)", "Beliau bicara pelo.", [
                opt("Minta mengulang kalimat sederhana ('Matahari terbit dari timur').", True, 1, "Tes pengulangan sederhana efektif menilai bicara pelo/afasia."),
                opt("Anggap pelo karena lelah, tunggu 15 menit lagi.", False, 10, "Bicara pelo mendadak = darurat. Menunggu menghabiskan waktu berharga."),
            ]),
            ("T - Time (bertindak)", "Semua tanda positif. Tindakan langsung Anda?", [
                opt("Telepon 119/112: sebut 'dugaan stroke', alamat, dan jam gejala mulai. Jangan beri makan/minum/obat.", True, 1, "Ambulans dapat memulai penilaian dan memberi tahu rumah sakit lebih dulu."),
                opt("Antar sendiri dengan mobil/motor ke klinik terdekat.", False, 15, "Tanpa ambulans tidak ada penilaian awal dan pemberitahuan rumah sakit; klinik terdekat belum tentu mampu menangani stroke."),
                opt("Beri aspirin dari kotak obat.", False, 6, "Stroke bisa akibat pendarahan; aspirin bisa memperburuk. Obat hanya atas arahan dokter setelah pencitraan."),
            ]),
            ("Menunggu ambulans", "Sambil menunggu, apa yang Anda lakukan?", [
                opt("Posisikan aman (duduk/baring miring), longgarkan pakaian, catat jam gejala mulai, siapkan daftar obat.", True, 1, "Jam gejala mulai menentukan pilihan terapi di rumah sakit."),
                opt("Ajak berjalan-jalan agar segar.", False, 3, "Berisiko jatuh dan memperburuk keadaan."),
            ]),
        ],
        "learn": [
            "BE-FAST: Balance, Eyes, Face, Arm, Speech, Time - telepon 119/112 begitu satu tanda muncul.",
            "Catat jam gejala mulai dan jangan beri makan/minum/obat.",
            "Stroke iskemik pembuluh besar tipikal kehilangan ≈1,9 juta neuron per menit tanpa terapi [5].",
        ],
    },
    "ami": {
        "title": "❤️ Serangan jantung 'tidak khas' di kantor",
        "intro": "Bu Ratna (52 tahun, diabetes) tampak pucat saat rapat: mual, berkeringat dingin, dan sesak. Ia bilang, 'Paling masuk angin.'",
        "good": 6, "ok": 25, "metric": "menit",
        "steps": [
            ("Mengenali", "Apa yang Anda lakukan?", [
                opt("Hampiri, tanyakan keluhannya dengan serius.", True, 1, "Benar. Gejala serangan jantung bisa tidak khas, terutama pada perempuan dan penderita diabetes."),
                opt("Biarkan, kemungkinan masuk angin.", False, 20, "'Masuk angin' adalah alasan umum keterlambatan ke rumah sakit."),
            ]),
            ("Menilai gejala", "Bagaimana Anda menggali keluhannya?", [
                opt("Tanya: nyeri/tekanan dada? menjalar ke rahang, lengan, atau punggung? sesak? keringat dingin? mual?", True, 1, "Kombinasi gejala ini menguatkan dugaan serangan jantung."),
                opt("Beri minyak angin dan kerok punggungnya.", False, 10, "Kerokan/minyak angin tidak mengobati sumbatan arteri dan hanya menunda."),
            ]),
            ("Memutuskan", "Gejalanya menguat. Keputusan Anda?", [
                opt("Telepon 119/112: sebut 'dugaan serangan jantung'; minta ia berhenti beraktivitas dan duduk.", True, 1, "Ambulans membawa EKG dan tenaga terlatih; waktu ke terapi sangat menentukan."),
                opt("Tunggu satu jam; kalau belum membaik baru ke rumah sakit.", False, 20, "Otot jantung terus rusak selama sumbatan; menunggu bisa fatal."),
                opt("Pesan ojek online ke rumah sakit.", False, 12, "Ia bisa kolaps di jalan tanpa pemantauan atau pertolongan."),
            ]),
            ("Sambil menunggu", "Apa yang Anda lakukan sebelum ambulans tiba?", [
                opt("Dudukkan setengah bersandar, tenangkan, ikuti arahan petugas (mis. aspirin hanya jika disarankan dan tidak alergi); siap CPR bila tidak sadar dan tidak bernapas normal.", True, 1, "Mengikuti arahan operator darurat adalah langkah paling aman."),
                opt("Beri obat apa saja dari kotak P3K tanpa bertanya.", False, 5, "Obat salah bisa berbahaya; tanyakan dulu ke petugas darurat."),
            ]),
        ],
        "learn": [
            "Gejala tidak khas (mual, sesak, nyeri rahang/punggung, keringat dingin) sering terjadi pada perempuan dan penderita diabetes.",
            "'Masuk angin' dan kerokan bukan pengobatan serangan jantung.",
            "Telepon 119/112 lebih dulu; jangan berkendara sendiri.",
        ],
    },
}


def game_reset(keep_screen: bool = False):
    ss = st.session_state
    ss.g_run = ss.get("g_run", 0) + 1
    ss.g_step, ss.g_elapsed, ss.g_log, ss.g_last, ss.g_order = 0, 0, [], None, {}
    if not keep_screen:
        ss.g_scn = None


def render_game():
    ss = st.session_state
    if "g_run" not in ss:
        game_reset()

    if ss.g_scn is None:
        st.write("Pilih skenario. Urutan jawaban diacak setiap permainan, jadi hafalan posisi tidak membantu.")
        cols = st.columns(len(SCENARIOS))
        for col, (key, sc) in zip(cols, SCENARIOS.items()):
            with col:
                st.subheader(sc["title"])
                st.caption(sc["intro"])
                if st.button("Mulai", key=f"start_{key}"):
                    game_reset(keep_screen=True)
                    ss.g_scn = key
                    st.rerun()
        return

    sc = SCENARIOS[ss.g_scn]
    steps = sc["steps"]
    st.subheader(sc["title"])

    if ss.g_step == 0 and not ss.g_log:
        st.info(sc["intro"])

    c1, c2 = st.columns(2)
    c1.metric("Waktu berlalu sejak gejala", f"{ss.g_elapsed} menit")
    if sc["metric"] == "neuron":
        c2.metric("Perkiraan neuron hilang", f"≈ {ss.g_elapsed * NEURONS_PER_MIN_MILLION:.1f} juta",
                  help="Rata-rata stroke iskemik pembuluh besar [5]; bukan prediksi individu.")

    if ss.g_last:
        (st.success if ss.g_last[0] else st.error)(ss.g_last[1])

    if ss.g_step < len(steps):
        title, question, options = steps[ss.g_step]
        okey = (ss.g_scn, ss.g_step)
        if okey not in ss.g_order:
            idx = list(range(len(options)))
            random.shuffle(idx)
            ss.g_order[okey] = idx
        shuffled = [options[i] for i in ss.g_order[okey]]

        st.markdown(f"#### 📍 Langkah {ss.g_step + 1}/{len(steps)}: {title}")
        st.progress(ss.g_step / len(steps))
        choice = st.radio(question, range(len(shuffled)), format_func=lambda k: shuffled[k]["text"],
                          index=None, key=f"r_{ss.g_run}_{ss.g_scn}_{ss.g_step}")
        if st.button("Konfirmasi", disabled=choice is None, key=f"b_{ss.g_run}_{ss.g_scn}_{ss.g_step}"):
            o = shuffled[choice]
            ss.g_elapsed += o["delay"]
            ss.g_log.append((o["ok"], title, o["text"], o["fb"], o["delay"]))
            ss.g_last = (o["ok"], ("✅ " if o["ok"] else f"❌ (+{o['delay']} menit) ") + o["fb"])
            ss.g_step += 1
            st.rerun()
        return

    # Hasil akhir
    st.markdown("### 🏁 Hasil")
    if ss.g_elapsed <= sc["good"]:
        st.success(f"Sangat baik! Total {ss.g_elapsed} menit: Anda mengenali tanda dan memanggil bantuan dengan cepat.")
    elif ss.g_elapsed <= sc["ok"]:
        st.info(f"Cukup baik ({ss.g_elapsed} menit), tetapi ada penundaan yang bisa dihindari.")
    else:
        st.warning(f"Banyak waktu terbuang ({ss.g_elapsed} menit). Pada kasus nyata, 'waktu adalah otak/otot jantung'.")
    if sc["metric"] == "neuron":
        lost = ss.g_elapsed * NEURONS_PER_MIN_MILLION
        st.caption(f"Perbandingan: {lost:.1f} juta neuron hilang pada permainan ini vs ≈ {180 * NEURONS_PER_MIN_MILLION:.0f} juta "
                   "bila menunggu 3 jam (estimasi rata-rata, rentang 1,1-3,2 juta/menit pada analisis sensitivitas [5]).")
    st.markdown("**Catatan langkah:**")
    for ok, title, text, fb, delay in ss.g_log:
        st.write(f"{'✅' if ok else '❌'} **{title}** - {text} *({fb})*")
    st.markdown("**Poin pembelajaran:**")
    for item in sc["learn"]:
        st.write(f"- {item}")
    b1, b2 = st.columns(2)
    if b1.button("Main lagi (acak ulang)"):
        game_reset(keep_screen=True)
        ss.g_scn = ss.g_scn
        st.rerun()
    if b2.button("Pilih skenario lain"):
        game_reset()
        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Tentang CardioShield")
    st.info(
        "Prototipe edukasi untuk membantu masyarakat mengenali faktor risiko penyakit kardiovaskular, "
        "memutuskan kapan perlu pemeriksaan, dan bertindak cepat saat stroke atau serangan jantung. "
        "Tanpa unduhan aplikasi dan tanpa menyimpan data."
    )
    st.error("**Darurat?** Telepon **119** (layanan darurat kesehatan) atau **112** (darurat nasional). Jangan menunggu hasil skrining.")
    st.warning("Alat ini **bukan diagnosis**. Skor bersifat heuristik dan belum divalidasi klinis; hasil harus dikonfirmasi tenaga kesehatan.")
    with st.expander("📚 Daftar sumber"):
        for s in SOURCES:
            st.markdown(f"- {s}")

st.title("❤️ CardioShield v2")
st.caption("Skrining risiko · edukasi patofisiologi · latihan tanggap darurat | merujuk WHO, Kemenkes RI, dan PERHI")

tab_data, tab_screen, tab_edu, tab_game = st.tabs(
    ["📊 Mengapa penting", "🔍 Skrining CardioCheck", "🩺 Edukasi", "⚡ Latihan Darurat"]
)

# ============================================================
# TAB 1: DATA
# ============================================================
with tab_data:
    st.header("Penyakit yang diam-diam: kesenjangan deteksi")
    a, b, c = st.columns(3)
    a.metric("Kematian CVD global (2022)", "19,8 juta", help="≈32% kematian global; 85% akibat serangan jantung dan stroke [1]")
    b.metric("Hipertensi terukur vs terdiagnosis", "30,8% vs 8,6%", help="Penduduk ≥18 th, SKI 2023 [2]")
    c.metric("Kasus baru hipertensi di CKG 2026", "≈1 juta", help="Dari 9,2 juta peserta yang tahun sebelumnya tercatat sehat [3]")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = list(DETECTION_GAP)
    measured = [v[0] for v in DETECTION_GAP.values()]
    diagnosed = [v[1] for v in DETECTION_GAP.values()]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, measured, 0.4, label="Terukur (pemeriksaan)", color="#d9534f")
    ax.bar(xs + 0.2, diagnosed, 0.4, label="Diketahui/terdiagnosis dokter", color="#5b9bd5")
    for x, m, d in zip(xs, measured, diagnosed):
        ax.text(x - 0.2, m + 0.5, f"{m}%", ha="center", fontsize=8)
        ax.text(x + 0.2, d + 0.5, f"{d}%", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Prevalensi (%)")
    ax.set_title("Banyak kasus belum terdiagnosis (SKI 2023)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.caption("Sumber: SKI 2023 [2]. Angka diabetes dikutip dari literatur sekunder yang merujuk BKPK 2023; verifikasi ke laporan primer.")

    st.markdown(
        "**Jembatan ke layanan yang sudah ada.** Pemerintah menyediakan *Cek Kesehatan Gratis* (CKG) untuk warga ber-NIK "
        "(termasuk non-peserta BPJS) berupa pemeriksaan tekanan darah, gula darah, status gizi, dan kebiasaan merokok di "
        "Puskesmas [3]. CardioShield **tidak menggantikan** CKG: tujuannya mengubah 'saya tidak tahu' menjadi 'saya perlu "
        "mengukur tekanan darah dan ikut CKG', serta mengajarkan tindakan saat darurat."
    )

# ============================================================
# TAB 2: SKRINING
# ============================================================
with tab_screen:
    st.header("🔍 Skrining Risiko CardioCheck")
    st.write("Isi sejujur mungkin. Data tidak disimpan. Hasil adalah panduan triase awal, bukan diagnosis.")

    st.subheader("⚠️ Langkah 0: Gejala darurat saat ini")
    red_flags = st.multiselect(
        "Apakah Anda atau orang di dekat Anda mengalami salah satu hal berikut SEKARANG?",
        [
            "Nyeri/tekanan di dada, atau nyeri menjalar ke rahang/lengan/punggung",
            "Sesak napas mendadak, keringat dingin, atau mual hebat",
            "Wajah perot, lengan lemah, atau bicara pelo mendadak",
            "Sakit kepala sangat hebat mendadak, atau penglihatan/keseimbangan hilang mendadak",
            "Pingsan",
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Data dasar & tanda vital")
        age = st.number_input("Usia (tahun)", 18, 100, 40)
        sex = st.radio("Jenis kelamin biologis", ["Wanita", "Pria"], horizontal=True)
        bp_known = st.radio("Apakah Anda tahu tekanan darah terakhir Anda?", ["Ya", "Tidak"], horizontal=True) == "Ya"
        sys_bp, dia_bp = 120, 80
        if bp_known:
            sys_bp = st.number_input("Sistolik (angka atas, mmHg)", 70, 260, 120)
            dia_bp = st.number_input("Diastolik (angka bawah, mmHg)", 40, 160, 80)
        chol_known = st.radio("Apakah Anda tahu kadar kolesterol Anda?", ["Ya", "Tidak"], horizontal=True) == "Ya"
        chol_type, chol_val = "Kolesterol total", 180
        if chol_known:
            chol_type = st.selectbox("Jenis", ["Kolesterol total", "LDL"])
            chol_val = st.number_input("Nilai (mg/dL)", 50, 500, 180)
        diabetes = st.selectbox("Pernah didiagnosis diabetes / gula darah tinggi?", ["Tidak", "Ya", "Tidak yakin"])
        waist = st.number_input("Lingkar perut (cm), isi 0 jika tidak tahu", 0, 200, 0)
    with col2:
        st.subheader("🍏 Gaya hidup & riwayat")
        smoking = st.selectbox("Rokok", [
            "Tidak pernah merokok",
            "Mantan perokok",
            "Perokok aktif (rokok biasa atau rokok elektrik)",
            "Perokok pasif (terpapar asap setiap hari)",
        ])
        alcohol = st.number_input("Alkohol (porsi per hari, rata-rata)", 0.0, 15.0, 0.0, step=1.0)
        activity = st.slider("Aktivitas fisik sedang-berat (menit/minggu)", 0, 600, 150)
        salt = st.selectbox("Seberapa sering makan makanan asin/gorengan/mi instan/kerupuk?", [
            "Jarang", "Kadang (1-3x/minggu)", "Sering (hampir setiap hari)"])
        family = st.checkbox("Ayah/ibu/saudara kandung pernah serangan jantung atau stroke dini "
                             "(laki-laki <55 th, perempuan <65 th)")

    if st.button("Evaluasi risiko saya"):
        res = evaluate(dict(
            age=age, sex=sex, bp_known=bp_known, sys=sys_bp, dia=dia_bp, chol_known=chol_known,
            chol_type=chol_type, chol_val=chol_val, diabetes=diabetes, waist=waist, smoking=smoking,
            alcohol=alcohol, activity=activity, salt=salt, family=family, red_flags=bool(red_flags),
        ))
        status = res["status"]
        st.divider()
        if status == "EMERGENCY":
            st.markdown(
                '<div class="box emergency"><h3>🚨 GEJALA DARURAT: telepon 119 / 112 SEKARANG</h3>'
                "<p>Jangan menunggu atau menyetir sendiri. Catat jam gejala mulai. Jangan beri makan, minum, atau obat "
                "kecuali diarahkan petugas.</p></div>", unsafe_allow_html=True)
        elif status == "RED":
            st.markdown('<div class="box red"><h3>🔴 Risiko tinggi: segera periksa ke Puskesmas/dokter</h3>'
                        "<p>Ada penanda risiko penting. Konfirmasi dengan pengukuran oleh tenaga kesehatan.</p></div>",
                        unsafe_allow_html=True)
        elif status == "YELLOW":
            st.markdown('<div class="box yellow"><h3>🟡 Risiko sedang / data belum lengkap</h3>'
                        "<p>Lengkapi data yang belum diketahui dan perbaiki kebiasaan berisiko.</p></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="box green"><h3>🟢 Risiko rendah berdasarkan data Anda</h3>'
                        "<p>Pertahankan gaya hidup sehat dan periksa rutin setiap tahun.</p></div>",
                        unsafe_allow_html=True)

        if status != "EMERGENCY":
            left, right = st.columns([3, 2])
            with left:
                st.markdown("#### Faktor yang teridentifikasi")
                for r in res["reasons"] or ["Tidak ada faktor risiko menonjol dari data yang diisi."]:
                    st.write(f"- {r}")
                st.info(
                    "**Langkah berikutnya:** ikut *Cek Kesehatan Gratis* di Puskesmas (tekanan darah, gula darah, status gizi) "
                    "[3]; tanyakan juga pemeriksaan kolesterol. Kurangi garam, berhenti merokok, aktif ≥150 menit/minggu."
                )
            with right:
                st.pyplot(points_chart(res["points"]))
                if res["bp_cat"] is not None:
                    st.pyplot(bp_map(sys_bp, dia_bp))
            summary = (f"CardioShield | usia {age}, {sex} | TD {sys_bp}/{dia_bp if bp_known else '?'} | "
                       f"status {status} | poin {res['total']} | faktor: "
                       + "; ".join(f"{n}={v}" for n, v in res["points"] if v))
            with st.expander("📝 Ringkasan untuk ditunjukkan ke petugas kesehatan"):
                st.code(summary)
            st.caption("Poin = tally faktor risiko heuristik, bukan estimasi risiko 10 tahun. Validasi terhadap WHO/ISH atau SCORE2 diperlukan.")

# ============================================================
# TAB 3: EDUKASI
# ============================================================
with tab_edu:
    st.header("🩺 Bagaimana faktor risiko merusak pembuluh darah")
    sub1, sub2, sub3 = st.tabs(["Aterosklerosis", "Jantung vs stroke", "Mitos & fakta"])

    with sub1:
        stage = st.slider("Tahap kesehatan arteri", 1, 4, 1)
        cv, cd = st.columns([2, 1])
        kind, head, body = STAGES[stage]
        with cd:
            getattr(st, kind)(f"**{head}**")
            st.write(body)
            st.caption("Merokok, hipertensi, LDL tinggi, dan gula darah tinggi merusak lapisan dalam arteri dan mempercepat proses ini.")
        with cv:
            st.pyplot(artery_figure(stage))

    with sub2:
        st.table({
            "": ["Penyebab", "Tanda utama", "Tindakan pertama"],
            "Serangan jantung": [
                "Sumbatan arteri koroner",
                "Nyeri/tekanan dada, menjalar ke rahang/lengan/punggung, sesak, keringat dingin, mual (sering tidak khas pada perempuan & diabetes)",
                "Telepon 119/112, duduk tenang, ikuti arahan petugas",
            ],
            "Stroke": [
                "Sumbatan (iskemik) atau pecahnya (hemoragik) pembuluh otak",
                "BE-FAST: keseimbangan, mata, wajah perot, lengan lemah, bicara pelo",
                "Telepon 119/112, catat jam gejala mulai, jangan beri makan/minum/obat",
            ],
        })

    with sub3:
        myths = [
            ("Tusuk ujung jari/telinga bisa menolong stroke", "Tidak ada bukti manfaat dan menunda pertolongan. Telepon 119/112."),
            ("Nyeri dada selalu ada pada serangan jantung", "Tidak. Mual, sesak, dan nyeri rahang/punggung bisa menjadi gejala utama, terutama pada perempuan."),
            ("Kalau tidak ada gejala, berarti sehat", "Hipertensi, kolesterol tinggi, dan diabetes biasanya tanpa gejala; hanya pengukuran yang bisa mendeteksinya."),
            ("'Masuk angin' dan kerokan membantu", "Keduanya tidak mengobati sumbatan arteri dan sering menunda ke rumah sakit."),
            ("Stroke hanya menyerang lansia", "Hipertensi terukur sudah 10,7% pada usia 18-24 th dan 17,4% pada 25-34 th [2]."),
        ]
        for m, f in myths:
            with st.expander(f"❌ Mitos: {m}"):
                st.write(f"✅ **Fakta:** {f}")
        st.caption("Mitos lokal (tusuk jari, kerokan) bersumber dari praktik umum di masyarakat; sertakan rujukan klinis tambahan di esai.")

# ============================================================
# TAB 4: GAME
# ============================================================
with tab_game:
    st.header("⚡ Latihan Tanggap Darurat")
    st.write("Simulasi keputusan nyata. Setiap pilihan yang salah menambah menit yang terbuang.")
    render_game()