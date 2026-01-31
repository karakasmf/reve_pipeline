from pathlib import Path
import os

# -------------------------------------------------
# Repo root (…/reve_pipeline)
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# -------------------------------------------------
# Colab / Drive algılama
# -------------------------------------------------
IN_COLAB = os.path.exists("/content")
DRIVE_ROOT = Path("/content/drive/MyDrive")

if IN_COLAB and DRIVE_ROOT.exists():
    # Colab + Drive (kalıcı)
    BASE_ROOT = DRIVE_ROOT / REPO_ROOT.name
else:
    # Local / Colab (geçici)
    BASE_ROOT = REPO_ROOT

# -------------------------------------------------
# Paths
# -------------------------------------------------
WINDOW_DIR = BASE_ROOT / "cache" / "windows"
RESULTS_ROOT = BASE_ROOT / "results"

# -------------------------------------------------
# Create dirs
# -------------------------------------------------
WINDOW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
