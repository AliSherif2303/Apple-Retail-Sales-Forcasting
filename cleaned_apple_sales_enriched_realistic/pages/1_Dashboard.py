# -*- coding: utf-8 -*-
"""Page 1 — Dashboard (wrapper that runs the original dashboard.py)"""
import streamlit as st
import os
from pathlib import Path

# The original dashboard.py is in the parent directory
dashboard_dir = Path(__file__).resolve().parent.parent
dashboard_path = dashboard_dir / "dashboard.py"


# Change working directory so DATA_PATH resolves correctly
old_cwd = os.getcwd()
os.chdir(dashboard_dir)

try:
    code = open(dashboard_path, encoding="utf-8").read()
    exec(compile(code, str(dashboard_path), "exec"), {"__file__": str(dashboard_path), "__name__": "__main__"})
finally:
    os.chdir(old_cwd)
