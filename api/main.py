"""API d'estimation de prix.

    uvicorn api.main:app --reload
    → http://localhost:8000/docs
"""
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvf import config, predict  # noqa: E402

app = FastAPI(
    title="Estimation de prix immobiliers — Île-de-France",
    description="Modèle entraîné sur les Demandes de Valeurs Foncières (DVF).",
    version="1.0.0",
)


class Bien(BaseModel):
    type_local: Literal["Appartement", "Maison"]
    surface_reelle_bati: float = Field(..., ge=9, le=400, description="Surface habitable en m²")
    nombre_pieces_principales: int = Field(..., ge=0, le=20)
    surface_terrain: float = Field(0, ge=0, description="Surface du terrain en m²")
    code_departement: str = Field(..., examples=["75", "77"])
    latitude: float = Field(..., ge=48.0, le=49.3)
    longitude: float = Field(..., ge=1.4, le=3.6)
    mois: int = Field(6, ge=1, le=12)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type_local": "Appartement",
                    "surface_reelle_bati": 45,
                    "nombre_pieces_principales": 2,
                    "surface_terrain": 0,
                    "code_departement": "77",
                    "latitude": 48.9601,
                    "longitude": 2.8788,
                    "mois": 6,
                }
            ]
        }
    }


class Estimation(BaseModel):
    prix_estime: float
    prix_m2_estime: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict", response_model=Estimation)
def estimate(bien: Bien):
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Modèle absent : lance d'abord `python -m dvf.train`")
    prix = predict.predict([bien.model_dump()])[0]
    return Estimation(prix_estime=prix, prix_m2_estime=round(prix / bien.surface_reelle_bati))
