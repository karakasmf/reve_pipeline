from pathlib import Path
import os

# Kodun olduğu repo kökü (git clone edilen yer)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Drive’daki kalıcı proje klasörün (ekrandaki klasör)
DRIVE_PROJECT = Path("/content/drive/MyDrive/alz-ftd-ctl-reve")

# Drive bağlıysa: veri/cache/results Drive'a, değilse repo içine
if DRIVE_PROJECT.exists():
    STORAGE_ROOT = DRIVE_PROJECT
else:
    STORAGE_ROOT = REPO_ROOT

# Kalıcı klasörler
WINDOW_DIR = STORAGE_ROOT / "cache" / "windows"
DATA_ROOT = STORAGE_ROOT / "data"
RESULTS_ROOT = STORAGE_ROOT / "results"

# Klasörleri oluştur
WINDOW_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
