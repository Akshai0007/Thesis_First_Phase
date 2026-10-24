"""Lets the scripts run straight from a checkout (no pip install needed): puts ../src on the import path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
