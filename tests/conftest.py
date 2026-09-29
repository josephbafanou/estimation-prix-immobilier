"""Données DVF synthétiques pour tester le pipeline sans téléchargement."""
import numpy as np
import pandas as pd
import pytest


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    deps = rng.choice(["75", "77", "92"], n)
    types = rng.choice(["Appartement", "Maison"], n, p=[0.7, 0.3])
    surface = rng.uniform(15, 150, n).round()
    prix_m2 = np.select([deps == "75", deps == "92"], [10_000, 7_000], 3_500)
    prix = surface * prix_m2 * rng.normal(1, 0.1, n)
    return pd.DataFrame(
        {
            "id_mutation": [f"2024-{i}" for i in range(n)],
            "date_mutation": pd.date_range("2024-01-01", periods=n, freq="12h").astype(str),
            "nature_mutation": "Vente",
            "valeur_fonciere": prix.round(),
            "code_departement": deps,
            "code_commune": deps + "001",
            "type_local": types,
            "surface_reelle_bati": surface,
            "nombre_pieces_principales": np.clip(surface // 20, 1, 8),
            "surface_terrain": np.where(types == "Maison", 300.0, np.nan),
            "latitude": 48.85 + rng.normal(0, 0.05, n),
            "longitude": 2.35 + rng.normal(0, 0.05, n),
        }
    )


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()
