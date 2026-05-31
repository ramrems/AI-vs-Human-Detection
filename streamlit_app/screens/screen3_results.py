"""
Ekran 3 — Results & Analysis
Sıralı karşılaştırma, performance evaluation, tahmin örnekleri, dışa aktarım
"""

import io
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from utils import load_models, clean_text, ALGO_META, SCENARIO_LABELS


def show():
    st.title("📈 Results & Analysis")
    st.caption("Sıralı karşılaştırma, performans analizi ve dışa aktarım")

    models, missing = load_models()
    if models is None:
        st.error(f"❌ Eksik model dosyaları: {missing}")
        st.info("Notebook'ta **Model Kaydetme** hücresini çalıştırın.")
        return

    all_results = models.get('all_results', {})
    df_perf     = models.get('df_perf', None)

    if not all_results:
        st.warning("all_results boş. Notebook'un özet hücresini çalıştırın.")
        return

    # ── Sonuçları DataFrame'e dönüştür ───────────────────────────────────────
    rows = []
    for key, acc in all_results.items():
        parts = key.split('-', 1)
        if len(parts) == 2:
            algo, scenario = parts[0], parts[1]
        else:
            continue
        rows.append({'Algoritma': algo, 'Senaryo': scenario,
                     'Accuracy %': round(acc * 100, 2)})
    df_all = pd.DataFrame(rows).sort_values('Accuracy %', ascending=False)

    # ── Tab yapısı ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Sıralama",
        "📊 Performans Analizi",
        "🔢 Confusion Matrices",
        "🔍 Tahmin Örnekleri",
        "⬇️ Dışa Aktarım",
    ])

    # ── TAB 1: Sıralama ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("🏆 Tüm Modellerin Sıralı Karşılaştırması")

        medals = ['🥇','🥈','🥉'] + ['  '] * 30
        df_ranked = df_all.copy().reset_index(drop=True)
        df_ranked.insert(0, '#', [medals[i] for i in range(len(df_ranked))])
        df_ranked['Algoritma'] = df_ranked['Algoritma'].map(
            lambda x: f"{ALGO_META.get(x, {}).get('icon','📌')} {x}")

        st.dataframe(
            df_ranked.style.background_gradient(
                subset=['Accuracy %'], cmap='RdYlGn', vmin=40, vmax=100),
            use_container_width=True, hide_index=True
        )

        # Isı haritası
        st.markdown("**Algoritma × Senaryo Isı Haritası**")
        algos = df_all['Algoritma'].unique()
        scens = df_all['Senaryo'].unique()
        pivot = df_all.pivot_table(index='Senaryo', columns='Algoritma',
                                    values='Accuracy %', aggfunc='first')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=40, vmax=100, linewidths=0.5, ax=ax,
                    annot_kws={'size': 11})
        ax.set_title('Accuracy (%) — Algoritma × Senaryo', fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 2: Performance Analizi ────────────────────────────────────────────
    with tab2:
        st.subheader("📊 Performance Evaluation")

        if df_perf is not None:
            st.markdown("**Scalability — Farklı Veri Boyutlarında (%10 / %50 / %100)**")
            st.dataframe(df_perf.style.background_gradient(
                subset=['Accuracy','F1'], cmap='RdYlGn'),
                use_container_width=True, hide_index=True)

            # Zaman & bellek grafiği
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            metrics = ['Eğitim (s)', 'Tahmin (s)', 'Bellek (MB)']
            titles  = ['Eğitim Süresi (sn)', 'Tahmin Süresi (sn)', 'Peak Bellek (MB)']
            colors  = {'MNB':'#4C9BE8', 'XGB':'#FF9F1C', 'LR':'#56d364'}

            for ax, m, t in zip(axes, metrics, titles):
                for algo, grp in df_perf.groupby('Algoritma'):
                    ax.plot(grp['Boyut'], grp[m], marker='o', linewidth=2,
                            label=algo, color=colors.get(algo,'gray'))
                ax.set_title(t, fontsize=11)
                ax.set_xlabel('Veri Boyutu')
                ax.legend(); ax.grid(alpha=0.3)

            plt.suptitle('Scalability Analizi — DS2', fontsize=12, y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Radar chart
            st.markdown("**Radar Chart — %100 Veri, DS2**")
            df_100 = df_perf[df_perf['Boyut'] == '%100']
            cats   = ['Accuracy', 'F1', 'Precision', 'Recall']
            N      = len(cats)
            angles = [n/float(N)*2*np.pi for n in range(N)] + [0]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            for _, row in df_100.iterrows():
                vals = [row[c] for c in cats] + [row[cats[0]]]
                algo = row['Algoritma']
                ax.plot(angles, vals, linewidth=2, label=algo,
                        color=colors.get(algo, 'gray'))
                ax.fill(angles, vals, alpha=0.1, color=colors.get(algo, 'gray'))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(cats, fontsize=11)
            ax.set_ylim(0, 105)
            ax.set_title('Radar Chart — Algoritma Karşılaştırması',
                          fontsize=12, pad=15)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.info("Performance tablosu bulunamadı. Notebook'taki **Performance Evaluation** bölümünü çalıştırın.")

        # ROC — notebook sonuçlarından
        st.markdown("**ROC Eğrisi — Model B / DS2**")
        X2_test = models['X2_test_text']
        y2_test = models['y2_test']
        cleaned_x2 = [clean_text(t) for t in X2_test]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random')

        for algo, vec_key, model_key, is_mnb in [
            ('MNB', 'mnb_vec_2', 'mnb_B', True),
            ('XGB', 'tfidf_2',   'xgb_B', False),
        ]:
            try:
                vec   = models[vec_key]
                model = models[model_key]
                Xt    = vec.transform(cleaned_x2)
                if is_mnb:
                    Xt = normalize(Xt, norm='l1')
                y_score = model.predict_proba(Xt)[:, 1]
                fpr, tpr, _ = roc_curve(y2_test, y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=2,
                        color=ALGO_META[algo]['color'],
                        label=f'{algo} (AUC={roc_auc:.3f})')
            except Exception:
                pass

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Eğrisi — DS2 Test (Model B)', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 3: Confusion Matrices ─────────────────────────────────────────────
    with tab3:
        st.subheader("🔢 Confusion Matrix Izgarası")
        from sklearn.metrics import confusion_matrix as cm_fn

        X2_test = models['X2_test_text']
        y2_test = models['y2_test']
        Xc_test = models['X_comb_test']
        yc_test = models['y_comb_test']
        cleaned_x2 = [clean_text(t) for t in X2_test]
        cleaned_xc = [clean_text(t) for t in Xc_test]

        cm_configs = [
            ('MNB',  'mnb_B',  'mnb_vec_2', cleaned_x2, y2_test, True,  'DS2'),
            ('XGB',  'xgb_B',  'tfidf_2',   cleaned_x2, y2_test, False, 'DS2'),
            ('MNB',  'mnb_C',  'mnb_vec_C', cleaned_xc, yc_test, True,  'DS1+DS2'),
            ('XGB',  'xgb_C',  'tfidf_C',   cleaned_xc, yc_test, False, 'DS1+DS2'),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (algo, m_key, v_key, X, y, is_mnb, ds) in zip(
                axes.flatten(), cm_configs):
            try:
                model = models[m_key]; vec = models[v_key]
                Xt = vec.transform(X)
                if is_mnb:
                    Xt = normalize(Xt, norm='l1')
                y_pred = model.predict(Xt)
                cm = cm_fn(y, y_pred, labels=[0,1])
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                            xticklabels=['Human','AI'],
                            yticklabels=['Human','AI'], cbar=False)
                ax.set_title(f'{algo} — Train:{ds}',
                             color=ALGO_META[algo]['color'], fontsize=11)
                ax.set_ylabel('Gerçek'); ax.set_xlabel('Tahmin')
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center')
                ax.axis('off')

        plt.suptitle('Confusion Matrix — Tüm Modeller', fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 4: Tahmin Örnekleri ───────────────────────────────────────────────
    with tab4:
        st.subheader("🔍 Tahmin Örnekleri — En İyi Model (XGB / DS2)")

        X2_test = list(models['X2_test_text'])
        y2_test = list(models['y2_test'])
        cleaned = [clean_text(t) for t in X2_test]

        try:
            vec    = models['tfidf_2']
            model  = models['xgb_B']
            Xt     = vec.transform(cleaned)
            y_pred = list(model.predict(Xt))
            y_prob = model.predict_proba(Xt)

            import random
            random.seed(42)
            correct   = [i for i in range(len(y2_test)) if y2_test[i]==y_pred[i]]
            incorrect = [i for i in range(len(y2_test)) if y2_test[i]!=y_pred[i]]
            sel_idx   = (random.sample(correct, min(5, len(correct))) +
                         random.sample(incorrect, min(3, len(incorrect))))

            show_n = st.slider("Gösterilecek örnek", 3, min(15, len(sel_idx)), 8)
            for i in sel_idx[:show_n]:
                true_lbl  = "🤖 AI"    if y2_test[i]==1 else "🧑 Human"
                pred_lbl  = "🤖 AI"    if y_pred[i]==1  else "🧑 Human"
                conf      = y_prob[i][y_pred[i]] * 100
                ok        = "✅" if y2_test[i]==y_pred[i] else "❌"
                preview   = X2_test[i][:200]

                with st.expander(f"{ok} Gerçek: {true_lbl} | Tahmin: {pred_lbl} | Güven: %{conf:.1f}"):
                    st.write(preview + "…")
        except Exception as e:
            st.error(f"Tahmin örnekleri oluşturulamadı: {e}")

    # ── TAB 5: Dışa Aktarım ───────────────────────────────────────────────────
    with tab5:
        st.subheader("⬇️ Dışa Aktarım")

        col1, col2 = st.columns(2)

        # CSV dışa aktarım
        with col1:
            st.markdown("**📄 Sonuç Tablosu (CSV)**")
            csv_buf = io.StringIO()
            df_all.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ results.csv",
                data=csv_buf.getvalue().encode('utf-8'),
                file_name='web_mining_results.csv',
                mime='text/csv',
                use_container_width=True
            )

            if df_perf is not None:
                st.markdown("**📄 Performance Tablosu (CSV)**")
                csv_perf = io.StringIO()
                df_perf.to_csv(csv_perf, index=False)
                st.download_button(
                    label="⬇️ performance.csv",
                    data=csv_perf.getvalue().encode('utf-8'),
                    file_name='web_mining_performance.csv',
                    mime='text/csv',
                    use_container_width=True
                )

        # PNG grafik dışa aktarım
        with col2:
            st.markdown("**🖼️ Özet Grafik (PNG)**")
            if st.button("Grafik Oluştur ve İndir", use_container_width=True):
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Isı haritası
                try:
                    pivot = df_all.pivot_table(index='Senaryo', columns='Algoritma',
                                                values='Accuracy %', aggfunc='first')
                    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                                vmin=40, vmax=100, ax=axes[0], linewidths=0.5,
                                annot_kws={'size': 10})
                    axes[0].set_title('Accuracy (%) — Isı Haritası', fontsize=11)
                except Exception:
                    pass

                # Bar chart
                try:
                    x      = np.arange(len(df_all['Senaryo'].unique()))
                    width  = 0.25
                    scens  = df_all['Senaryo'].unique()
                    colors = {'MNB':'#4C9BE8','XGB':'#FF9F1C','BERT':'#9B59B6'}
                    for i, algo in enumerate(df_all['Algoritma'].unique()):
                        vals = df_all[df_all['Algoritma']==algo]['Accuracy %'].values
                        if len(vals) == len(x):
                            bars = axes[1].bar(x + i*width, vals, width,
                                               label=algo,
                                               color=colors.get(algo,'gray'),
                                               edgecolor='white')
                    axes[1].set_xticks(x + width)
                    axes[1].set_xticklabels(scens, rotation=20, ha='right', fontsize=8)
                    axes[1].set_ylabel('Accuracy (%)')
                    axes[1].set_title('Algoritma Karşılaştırması', fontsize=11)
                    axes[1].legend(); axes[1].set_ylim(0, 110)
                    axes[1].grid(axis='y', alpha=0.3)
                except Exception:
                    pass

                plt.suptitle('BLM5121 — Web Mining Term Project Results',
                              fontsize=13, y=1.02)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()

                st.download_button(
                    label="⬇️ results_chart.png",
                    data=buf,
                    file_name='web_mining_results_chart.png',
                    mime='image/png',
                    use_container_width=True
                )

        # Ekran görüntüsü notu
        st.markdown("---")
        st.info(
            "💡 **Rapor için:** Her sekmenin ekran görüntüsünü alın. "
            "Tarayıcıda `Ctrl+Shift+S` (tam sayfa) veya OS screenshot aracı kullanabilirsiniz."
        )
