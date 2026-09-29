"""Paramètres du projet."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Paris, Seine-et-Marne, Hauts-de-Seine, Seine-Saint-Denis, Val-de-Marne
DEPARTEMENTS = os.getenv("DVF_DEPARTEMENTS", "75,77,92,93,94").split(",")
ANNEES = os.getenv("DVF_ANNEES", "2024,2025").split(",")

DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Bornes de nettoyage (valeurs aberrantes)
SURFACE_MIN, SURFACE_MAX = 9, 400          # m²
PRIX_M2_MIN, PRIX_M2_MAX = 1_000, 25_000   # €/m²

NUMERIC_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "latitude",
    "longitude",
    "mois",
]
CATEGORICAL_FEATURES = ["type_local", "code_departement"]
TARGET = "valeur_fonciere"
