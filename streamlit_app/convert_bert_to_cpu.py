# convert_bert_to_cpu.py
# Çalıştır: python convert_bert_to_cpu.py

import pickle
import torch
import io
from pathlib import Path

SAVE_DIR = Path(r"C:\Users\irems\Downloads\web_mining_project\streamlit_app\saved_models")

class _CPUUnpickler(pickle.Unpickler):
    """pickle içindeki torch.load çağrılarını CPU'ya zorlar."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location=torch.device('cpu'),
                weights_only=False
            )
        return super().find_class(module, name)

for key in ['bert_A_state', 'bert_B_state', 'bert_C_state']:
    p = SAVE_DIR / f"{key}.pkl"
    if not p.exists():
        print(f"⚠️  {key}.pkl bulunamadı, atlanıyor")
        continue

    print(f"🔄 {key} dönüştürülüyor...")
    try:
        with open(p, 'rb') as f:
            state_dict = _CPUUnpickler(f).load()

        # CPU'da olduğundan emin ol
        state_dict = {k: v.cpu() for k, v in state_dict.items()}

        # Yeniden kaydet
        torch.save(state_dict, p)
        print(f"✅ {key} CPU formatına dönüştürüldü ve kaydedildi")

    except Exception as e:
        print(f"❌ {key} dönüştürülemedi: {e}")