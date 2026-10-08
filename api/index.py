import os
import sys
from pathlib import Path

# Ensure writable cache locations in serverless / AWS Lambda / Vercel
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.main import app
