from pathlib import Path

# =========================================================
# Repo root (kodun bulunduğu yer)
# common/paths.py -> parents[2] = repo kökü
# =========================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# Kalıcı storage (Colab + Google Drive)
# =========================================================
DRIVE_PROJECT = Path("/content/drive/MyDrive/alz-ftd-ctl-reve")

# Drive varsa orayı kullan, yoksa local repo içine yaz
STORAGE_ROOT = DRIVE_PROJECT if DRIVE_PROJECT.exists() else REPO_ROOT

# =========================================================
# Ana dizinler
# =========================================================
DATA_ROOT = STORAGE_ROOT / "data"
WINDOW_DIR = STORAGE_ROOT / "cache" / "windows"
RESULTS_ROOT = STORAGE_ROOT / "results"

# Sık kullanılan dosyalar
PARTICIPANTS_TXT = DATA_ROOT / "participants.txt"

# =========================================================
# Dizinleri oluştur (yoksa)
# =========================================================
DATA_ROOT.mkdir(parents=True, exist_ok=True)
WINDOW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
