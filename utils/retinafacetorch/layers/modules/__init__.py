import os
import sys

PYTHON_PATH = os.path.abspath(os.path.join(
    os.path.abspath(__file__), "..", "..","..",".."))
os.chdir(PYTHON_PATH)
sys.path.insert(0, PYTHON_PATH)
from utils.retinafacetorch.layers.modules.multibox_loss import MultiBoxLoss

__all__ = ['MultiBoxLoss']
