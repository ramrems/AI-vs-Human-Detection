"""
BLM5121 – Web Mining Term Project
AI vs Human Content Detection — Streamlit GUI
Çalıştır: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Web Mining — AI vs Human",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigasyon ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Ekran Seç",
    ["📂 Dataset Manager", "⚙️ Algorithm Runner", "📈 Results & Analysis"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Proje:** BLM5121 Term Project  
**Konu:** AI vs Human Content Detection  
**Algoritmalar:** MNB · XGBoost · BERT  
**Veri:** DS1 (Global AI vs Human Content Dataset 2026) · DS2 (DAIGT V2 Train Dataset)
""")

# ── Sayfa yönlendirme ─────────────────────────────────────────────────────────
if page == "📂 Dataset Manager":
    from screens.screen1_dataset import show
    show()
elif page == "⚙️ Algorithm Runner":
    from screens.screen2_runner import show
    show()
elif page == "📈 Results & Analysis":
    from screens.screen3_results import show
    show()
