import sys
from pathlib import Path

# Allow importing the local fastmlx package without installation
ROOT = Path(__file__).resolve().parents[1] / "fastmlx"
sys.path.insert(0, str(ROOT))
