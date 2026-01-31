from pathlib import Path
import os

# =========================================================
# Repo root (kodun olduğu yer)
# common/paths.py ise: parents[2] -> repo kökü
# =========================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# Kalıcı storage root (Colab Drive varsa burası)
# Senin Drive yapın: /content/drive/MyDrive/alz-ftd-ctl-reve/
# =========================================================
DRIVE_PROJECT = Path("/content/drive/MyDrive/alz-ftd-ctl-reve")

if DRIVE_PROJECT.exists():
    STORAGE_ROOT = DRIVE_PROJECT          # Colab + Drive (kalıcı)
else:
    STORAGE_ROOT = REPO_ROOT              # Local / Drive yoksa

# =========================================================
# Dizinler
# =========================================================
DATA_ROOT = STORAGE_ROOT / "data"
WINDOW_DIR = STORAGE_ROOT / "cache" / "windows"
RESULTS_ROOT = STORAGE_ROOT / "results"

# Sık kullanılan dosyalar (örnek)
PARTICIPANTS_TXT = DATA_ROOT / "participants.txt"

# =========================================================
# Dizinleri oluştur
# =========================================================
DATA_ROOT.mkdir(parents=True, exist_ok=True)
WINDOW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
