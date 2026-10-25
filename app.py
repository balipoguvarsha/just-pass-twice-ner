"""
Entry point for the Streamlit web application.
Enables running either `streamlit run app.py` or `streamlit run src/app.py`.
"""

import os
import sys

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.app
