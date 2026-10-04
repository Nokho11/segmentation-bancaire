"""Paramètres globaux du projet (un seul endroit pour tout régler)."""
from pathlib import Path

SEED = 42                 # graine aléatoire fixée -> résultats reproductibles
N_SAMPLE = 4000           # taille de l'échantillon stratifié (consigne : 3 000 à 5 000)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
GOWER_CACHE = OUT / "gower_sample.npy"   # cache de la matrice de Gower
LAB = OUT / "labels"
for _d in (FIG, TAB, LAB):
    _d.mkdir(parents=True, exist_ok=True)
