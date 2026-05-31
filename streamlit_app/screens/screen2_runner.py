"""
Ekran 2 — Algorithm Runner & Comparator
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import normalize
from utils import load_models, predict_classic, predict_bert, load_bert_model, clean_text, ALGO_META


def run_eval(models, algo: str, ds_key: str) -> dict:
    if ds_key == 'DS1':
        X_test, y_test = models['X1_test_text'], models['y1_test']
    elif ds_key == 'DS2':
        X_test, y_test = models['X2_test_text'], models['y2_test']
    else:
        X_test, y_test = models['X_comb_test'], models['y_comb_test']

    suffix = {'DS1': 'A', 'DS2': 'B', 'DS1+DS2': 'C'}[ds_key]

    if algo == 'MNB':
        model = models[f'mnb_{suffix}']
        vec_k = 'mnb_vec_' + ('1' if suffix == 'A' else '2' if suffix == 'B' else 'C')
        vec   = models[vec_k]
    elif algo == 'XGB':
        model = models[f'xgb_{suffix}']
        vec_k = 'tfidf_' + ('1' if suffix == 'A' else '2' if suffix == 'B' else 'C')
        vec   = models[vec_k]
    elif algo == 'BERT':
        state = models.get(f'bert_{suffix}_state')
        if state is None:
            return None
        bert_model = load_bert_model(state, device='cpu')
        if bert_model is None:
            return None
        t0 = time.perf_counter()
        cleaned = [clean_text(t) for t in X_test]
        y_pred, y_prob_list = [], []
        for txt in cleaned:
            p, prob = predict_bert(bert_model, txt)
            if p is None:
                return None
            y_pred.append(p)
            y_prob_list.append(prob[1])
        elapsed = time.perf_counter() - t0
        y_pred = np.array(y_pred)
        y_prob_arr = np.array(y_prob_list)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])
        return {
            'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec,
            'elapsed': elapsed, 'cm': cm,
            'y_pred': y_pred, 'y_test': list(y_test),
            'y_prob': y_prob_arr,
        }
    else:
        return None

    t0 = time.perf_counter()
    cleaned = [clean_text(t) for t in X_test]
    Xt = vec.transform(cleaned)
    if algo == 'MNB':
        Xt = normalize(Xt, norm='l1')
    y_pred = model.predict(Xt)
    y_prob = model.predict_proba(Xt)
    elapsed = time.perf_counter() - t0

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec,
        'elapsed': elapsed, 'cm': cm,
        'y_pred': y_pred, 'y_test': list(y_test),
        'y_prob': y_prob[:, 1],
    }


def show():
    st.title("⚙️ Algorithm Runner & Comparator")
    st.caption("Algoritma seç, çalıştır ve sonuçları karşılaştır")

    models, missing = load_models()
    if models is None:
        st.error(f"❌ Eksik model dosyaları: {missing}")
        st.info("Notebook'ta **Model Kaydetme** hücresini çalıştırın.")
        return

    bert_available = any(
        models.get(f'bert_{s}_state') is not None
        for s in ['A', 'B', 'C']
    )
    algo_options = ['MNB', 'XGB'] + (['BERT'] if bert_available else [])

    # ── Kontrol paneli ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("🎛️ Çalıştırma Ayarları")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            sel_algos = st.multiselect(
                "Algoritmalar",
                options=algo_options,
                default=['MNB', 'XGB'],
                format_func=lambda x: f"{ALGO_META[x]['icon']} {ALGO_META[x]['label']}",
            )
        with col2:
            sel_ds = st.selectbox(
                "Dataset (Test)",
                options=['DS1', 'DS2', 'DS1+DS2'],
                index=1,
            )
        with col3:
            st.write("")
            st.write("")
            run_btn = st.button("▶ Çalıştır", type="primary", use_container_width=True)

        run_all = st.button("▶▶ Tüm Kombinasyonları Çalıştır", use_container_width=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if 'run_results' not in st.session_state:
        st.session_state.run_results = {}

    # ── Çalıştırma ────────────────────────────────────────────────────────────
    def execute(algos, datasets):
        results = {}
        total = len(algos) * len(datasets)
        prog  = st.progress(0, text="Başlıyor…")
        log   = st.empty()
        done  = 0
        for ds in datasets:
            for algo in algos:
                log.info(f"⏳ {ALGO_META[algo]['label']} / {ds} çalışıyor…")
                t0  = time.perf_counter()
                res = run_eval(models, algo, ds)
                elapsed = time.perf_counter() - t0
                if res:
                    key = f"{algo}|{ds}"
                    results[key] = res
                    log.success(
                        f"✅ {algo}/{ds} → Acc: %{res['acc']*100:.1f}  "
                        f"F1: %{res['f1']*100:.1f}  ({elapsed:.1f}s)"
                    )
                else:
                    log.warning(f"⚠️ {algo}/{ds} atlandı (model yüklenemedi)")
                done += 1
                prog.progress(done / total, text=f"{done}/{total} tamamlandı")
        prog.empty()
        log.empty()
        return results

    if run_btn and sel_algos:
        new = execute(sel_algos, [sel_ds])
        st.session_state.run_results.update(new)

    if run_all:
        algos_to_run = ['MNB', 'XGB'] + (['BERT'] if bert_available else [])
        new = execute(algos_to_run, ['DS1', 'DS2', 'DS1+DS2'])
        st.session_state.run_results.update(new)

    results = st.session_state.run_results

    # ── Sonuçlar varsa göster ─────────────────────────────────────────────────
    if results:
        st.markdown("---")
        st.subheader("📊 Karşılaştırma Tablosu")
        rows = []
        for tag, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
            algo, ds = tag.split('|')
            rows.append({
                'Algoritma'  : f"{ALGO_META[algo]['icon']} {algo}",
                'Dataset'    : ds,
                'Accuracy %' : round(r['acc']*100, 2),
                'F1 %'       : round(r['f1']*100,  2),
                'Precision %': round(r['prec']*100, 2),
                'Recall %'   : round(r['rec']*100,  2),
                'Süre (s)'   : round(r['elapsed'],  3),
            })

        df_cmp = pd.DataFrame(rows)
        st.dataframe(
            df_cmp.style.background_gradient(subset=['Accuracy %', 'F1 %'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📈 Bar Karşılaştırma", "🔢 Confusion Matrix", "📉 ROC Eğrisi"])

        with tab1:
            tags   = list(results.keys())
            accs   = [results[t]['acc']*100  for t in tags]
            f1s    = [results[t]['f1']*100   for t in tags]
            times  = [results[t]['elapsed']  for t in tags]
            clrs   = [ALGO_META[t.split('|')[0]]['color'] for t in tags]
            labels = [t.replace('DS1+DS2', 'Comb') for t in tags]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, vals, title, unit in zip(
                axes,
                [accs, f1s, times],
                ['Accuracy (%)', 'F1 Score (%)', 'Süre (sn)'],
                ['%', '%', 's']
            ):
                bars = ax.barh(labels, vals, color=clrs, edgecolor='white', height=0.5)
                ax.set_title(title, fontsize=11)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_width() + max(vals)*0.01,
                            bar.get_y() + bar.get_height()/2,
                            f'{v:.1f}{unit}', va='center', fontsize=9)
                ax.set_xlim(0, max(vals)*1.3)
                ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            n_cm  = len(results)
            ncols = min(3, n_cm)
            nrows = (n_cm + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes_flat = np.array(axes).flatten() if n_cm > 1 else [axes]

            for ax, (tag, r) in zip(axes_flat, results.items()):
                algo, ds = tag.split('|')
                sns.heatmap(r['cm'], annot=True, fmt='d', ax=ax, cmap='Blues',
                            xticklabels=['Human', 'AI'],
                            yticklabels=['Human', 'AI'], cbar=False)
                ax.set_title(f'{algo} / {ds}', color=ALGO_META[algo]['color'], fontsize=10)
                ax.set_ylabel('Gerçek')
                ax.set_xlabel('Tahmin')

            for ax in axes_flat[n_cm:]:
                ax.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab3:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
            for tag, r in results.items():
                algo = tag.split('|')[0]
                if r.get('y_prob') is not None:
                    fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
                    roc_auc = auc(fpr, tpr)
                    lbl = tag.replace('DS1+DS2', 'Comb')
                    ax.plot(fpr, tpr, linewidth=2,
                            color=ALGO_META[algo]['color'],
                            label=f'{lbl} (AUC={roc_auc:.3f})')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Eğrisi', fontsize=12)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:
        st.info("⬆️ Bir algoritma seçip **Çalıştır** butonuna basın.")

    st.markdown("---")

    # ── Canlı Tahmin ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("🔍 Canlı Tahmin")
        pred_col1, pred_col2 = st.columns([3, 1])

        with pred_col2:
            live_algo_options = ['MNB|DS1', 'XGB|DS1',
                                  'MNB|DS2', 'XGB|DS2',
                                  'MNB|DS1+DS2', 'XGB|DS1+DS2']
            if bert_available:
                live_algo_options += ['BERT|DS1', 'BERT|DS2', 'BERT|DS1+DS2']
            pred_algo = st.selectbox("Model", live_algo_options, key="live_pred_model")
            pred_btn  = st.button("Tahmin Et", type="primary", use_container_width=True)

        with pred_col1:
            test_text = st.text_area(
                "Test metni girin:",
                placeholder="Sınıflandırmak istediğiniz metni buraya yazın…",
                height=120,
                key="live_pred_text"
            )

        if pred_btn:
            if not test_text.strip():
                st.warning("Lütfen bir metin girin.")
            else:
                algo, ds_key = pred_algo.split('|')
                suffix = {'DS1': 'A', 'DS2': 'B', 'DS1+DS2': 'C'}[ds_key]

                with st.spinner("Tahmin yapılıyor…"):
                    if algo == 'BERT':
                        state = models.get(f'bert_{suffix}_state')
                        if state is None:
                            st.session_state['live_result'] = {'error': 'BERT modeli yüklenemedi.'}
                        else:
                            bert_model = load_bert_model(state, device='cpu')
                            pred, proba = predict_bert(bert_model, test_text)
                            if pred is None:
                                st.session_state['live_result'] = {'error': 'BERT tahmin yapamadı.'}
                            else:
                                st.session_state['live_result'] = {
                                    'pred': pred, 'proba': proba,
                                    'algo': algo, 'text_preview': test_text[:80]
                                }
                    else:
                        if algo == 'MNB':
                            model = models[f'mnb_{suffix}']
                            vec_k = 'mnb_vec_' + ('1' if suffix=='A' else '2' if suffix=='B' else 'C')
                        else:
                            model = models[f'xgb_{suffix}']
                            vec_k = 'tfidf_' + ('1' if suffix=='A' else '2' if suffix=='B' else 'C')
                        vec = models[vec_k]
                        pred, proba = predict_classic(model, vec, test_text, algo)
                        st.session_state['live_result'] = {
                            'pred': int(pred), 'proba': proba,
                            'algo': algo, 'text_preview': test_text[:80]
                        }

        if 'live_result' in st.session_state:
            res = st.session_state['live_result']
            if 'error' in res:
                st.error(f"❌ {res['error']}")
            else:
                st.caption(f"📝 Tahmin edilen metin: *{res['text_preview']}…*")
                pred  = res['pred']
                proba = res['proba']
                if pred == 1:
                    st.error(f"🤖 **AI Üretimi** — Güven: %{proba[1]*100:.1f}")
                else:
                    st.success(f"🧑 **İnsan Yazımı** — Güven: %{proba[0]*100:.1f}")
                c1, c2 = st.columns(2)
                c1.metric("Human Olasılığı", f"%{proba[0]*100:.1f}")
                c2.metric("AI Olasılığı",    f"%{proba[1]*100:.1f}")