"""Nettoyage des transactions DVF.

Une mutation (vente) peut contenir plusieurs lignes : plusieurs lots, une cave,
une dépendance… On ne garde que les ventes simples d'un seul appartement ou
d'une seule maison, pour que le prix corresponde bien au bien décrit.
"""
from pathlib import Path

import pandas as pd

from dvf import config

COLUMNS = [
    "id_mutation",
    "date_mutation",
    "nature_mutation",
    "valeur_fonciere",
    "code_departement",
    "code_commune",
    "type_local",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "latitude",
    "longitude",
]


def read_raw(paths: list[Path]) -> pd.DataFrame:
    frames = [
        pd.read_csv(p, usecols=COLUMNS, dtype={"code_departement": str, "code_commune": str},
                    low_memory=False)
        for p in paths
    ]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["nature_mutation"] == "Vente"]

    # Mutations contenant exactement un logement (et aucune autre ligne bâtie)
    bati = df[df["type_local"].notna()]
    counts = bati.groupby("id_mutation")["type_local"].agg(["count", "first"])
    simples = counts[(counts["count"] == 1) & counts["first"].isin(["Appartement", "Maison"])]

    # Le terrain est parfois sur une autre ligne de la même mutation : on le somme
    terrain = df.groupby("id_mutation")["surface_terrain"].sum(min_count=1)

    out = bati[bati["id_mutation"].isin(simples.index)].copy()
    out["surface_terrain"] = out["id_mutation"].map(terrain).fillna(0)

    out = out.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"])
    out = out[out["surface_reelle_bati"].between(config.SURFACE_MIN, config.SURFACE_MAX)]

    prix_m2 = out["valeur_fonciere"] / out["surface_reelle_bati"]
    out = out[prix_m2.between(config.PRIX_M2_MIN, config.PRIX_M2_MAX)]

    out["date_mutation"] = pd.to_datetime(out["date_mutation"])
    out["mois"] = out["date_mutation"].dt.month
    out["nombre_pieces_principales"] = out["nombre_pieces_principales"].fillna(0)

    return out.drop_duplicates("id_mutation").reset_index(drop=True)
