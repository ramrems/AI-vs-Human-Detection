"""
Ekran 1 — Dataset Manager
Veri seti istatistikleri, sınıf dağılımları, örnek gezgini, filtreleme
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_models


def show():
    st.title("📂 Dataset Manager")
    st.caption("Veri seti istatistikleri, sınıf dağılımları ve örnek gezgini")

    models, missing = load_models()

    if models is None:
        st.error(f"❌ Eksik model dosyaları: {missing}")
        st.info("Notebook'ta **Model Kaydetme** hücresini çalıştırın, ardından `saved_models/` klasörünü bu dizine kopyalayın.")
        return

    # ── Veri setleri ─────────────────────────────────────────────────────────
    X1_test = models['X1_test_text']
    y1_test = models['y1_test']
    X2_test = models['X2_test_text']
    y2_test = models['y2_test']
    X_comb  = models['X_comb_test']
    y_comb  = models['y_comb_test']

    ds_info = {
        'DS1 — Global AI vs Human': {
            'X_test': X1_test, 'y_test': y1_test,
            'source': 'Kaggle: asifxzaman/global-ai-vs-human-content-dataset-2026',
            'note'  : 'Kısa/kalıplaşmış AI metinleri — vocabulary 183 ile sınırlı',
        },
        'DS2 — DAIGT V2': {
            'X_test': X2_test, 'y_test': y2_test,
            'source': 'Kaggle: thedrcat/daigt-v2-train-dataset',
            'note'  : 'Gerçek öğrenci yazıları vs. GPT/Claude çıktıları',
        },
    }

    # ── Üst metrik kartları ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total = len(X1_test) + len(X2_test)
    ai_n  = int((y1_test == 1).sum() + (y2_test == 1).sum())
    hu_n  = int((y1_test == 0).sum() + (y2_test == 0).sum())

    col1.metric("Toplam Test Kaydı", f"{total:,}")
    col2.metric("AI Örnekleri", f"{ai_n:,}", f"%{ai_n/total*100:.1f}")
    col3.metric("Human Örnekleri", f"{hu_n:,}", f"%{hu_n/total*100:.1f}")
    col4.metric("Veri Seti Sayısı", "2")

    st.markdown("---")

    # ── Dataset kartları ──────────────────────────────────────────────────────
    for ds_name, info in ds_info.items():
        with st.expander(f"**{ds_name}**", expanded=True):
            X, y = info['X_test'], info['y_test']
            ai_c  = int((y == 1).sum())
            hu_c  = int((y == 0).sum())
            total_c = len(y)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Kayıtları", f"{total_c:,}")
            c2.metric("AI", f"{ai_c:,}", f"%{ai_c/total_c*100:.1f}")
            c3.metric("Human", f"{hu_c:,}", f"%{hu_c/total_c*100:.1f}")
            wc_mean = np.mean([len(str(t).split()) for t in X])
            c4.metric("Ort. Kelime", f"{wc_mean:.0f}")

            st.caption(f"📌 Kaynak: {info['source']}")
            st.caption(f"ℹ️ {info['note']}")

            # Sınıf dağılımı bar + kelime dağılımı
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))

            axes[0].bar(['Human', 'AI'], [hu_c, ai_c],
                        color=['#4C9BE8', '#FF6B6B'], edgecolor='white', linewidth=1.2)
            axes[0].set_title('Sınıf Dağılımı', fontsize=11)
            axes[0].set_ylabel('Kayıt Sayısı')
            for i, v in enumerate([hu_c, ai_c]):
                axes[0].text(i, v + total_c*0.01, str(v), ha='center', fontsize=10)

            wc_ai  = [len(str(t).split()) for t, l in zip(X, y) if l == 1]
            wc_hu  = [len(str(t).split()) for t, l in zip(X, y) if l == 0]
            axes[1].hist(wc_hu, bins=30, alpha=0.6, label='Human', color='#4C9BE8')
            axes[1].hist(wc_ai, bins=30, alpha=0.6, label='AI',    color='#FF6B6B')
            axes[1].set_title('Kelime Sayısı Dağılımı', fontsize=11)
            axes[1].set_xlabel('Kelime Sayısı')
            axes[1].legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ── Örnek Gezgini ─────────────────────────────────────────────────────────
    st.subheader("🔎 Örnek Gezgini")

    filt_col1, filt_col2, filt_col3 = st.columns(3)
    with filt_col1:
        sel_ds = st.selectbox("Veri Seti", ["DS1", "DS2", "DS1+DS2"])
    with filt_col2:
        sel_label = st.selectbox("Etiket", ["Tümü", "AI", "Human"])
    with filt_col3:
        n_samples = st.slider("Gösterilecek Kayıt", 5, 50, 10)

    # Veri toplama
    if sel_ds == "DS1":
        texts, labels = list(X1_test), list(y1_test)
        src_tag = ["DS1"] * len(texts)
    elif sel_ds == "DS2":
        texts, labels = list(X2_test), list(y2_test)
        src_tag = ["DS2"] * len(texts)
    else:
        texts  = list(X1_test) + list(X2_test)
        labels = list(y1_test) + list(y2_test)
        src_tag = ["DS1"]*len(X1_test) + ["DS2"]*len(X2_test)

    df_samples = pd.DataFrame({
        'Kaynak'      : src_tag,
        'Etiket'      : ['🤖 AI' if l == 1 else '🧑 Human' for l in labels],
        'Kelime Sayısı': [len(str(t).split()) for t in texts],
        'İçerik'      : [str(t)[:200] + '…' for t in texts],
    })

    if sel_label == "AI":
        df_samples = df_samples[df_samples['Etiket'] == '🤖 AI']
    elif sel_label == "Human":
        df_samples = df_samples[df_samples['Etiket'] == '🧑 Human']

    df_samples = df_samples.sample(min(n_samples, len(df_samples)),
                                    random_state=42).reset_index(drop=True)

    st.dataframe(df_samples, use_container_width=True, height=350)

    # Min/max kelime filtresi
    st.markdown("**Kelime Sayısı Filtresi:**")
    all_wc = df_samples['Kelime Sayısı'].tolist()
    if all_wc:
        wc_range = st.slider("Kelime aralığı",
                              int(min(all_wc)), max(int(max(all_wc)), 10),
                              (int(min(all_wc)), max(int(max(all_wc)), 10)))
        filtered = df_samples[
            (df_samples['Kelime Sayısı'] >= wc_range[0]) &
            (df_samples['Kelime Sayısı'] <= wc_range[1])
        ]
        st.write(f"Filtre sonrası: **{len(filtered)}** kayıt")
        st.dataframe(filtered, use_container_width=True, height=250)
