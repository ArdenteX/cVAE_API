import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath('app'))
PTH_PATH = os.path.join(BASE_DIR, 'static', 'best_model.pth')
X_SCALER = os.path.join(BASE_DIR, 'static', 'Xscaler.save')
Y_SCALER = os.path.join(BASE_DIR, 'static', 'yscaler.save')
OUTPUT_DIM = 8
INPUT_DIM = 4

INPUT_PARAM = [
    'Mass',
    'Radius',
    "Fe/Mg",
    "Si/Mg"
]

OUTPUT_PARAM = [
    'WRF',
    'MRF',
    'CRF',
    'WMF',
    'CMF',
    # '$P_{\mathrm{CMB}}$(TPa)',
    # '$T_{\mathrm{CMB}}(10^{3}$K$)$',
    # '$k_{2}$'
    'P_CMB (TPa)',
    'T_CMB (10^3K)',
    'K2'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')