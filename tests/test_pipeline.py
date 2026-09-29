import joblib
import pandas as pd
from fastapi.testclient import TestClient

from dvf import clean, config, predict, train


def test_clean_keeps_only_single_dwelling_sales(raw):
    extra = pd.DataFrame(
        [
            # Vente avec deux logements : exclue
            {**raw.iloc[0].to_dict(), "id_mutation": "multi"},
            {**raw.iloc[1].to_dict(), "id_mutation": "multi"},
            # Échange : exclu
            {**raw.iloc[2].to_dict(), "id_mutation": "echange", "nature_mutation": "Echange"},
            # Prix aberrant (1 €) : exclu
            {**raw.iloc[3].to_dict(), "id_mutation": "aberrant", "valeur_fonciere": 1},
        ]
    )
    df = clean.clean(pd.concat([raw, extra], ignore_index=True))

    assert not df["id_mutation"].isin(["multi", "echange", "aberrant"]).any()
    assert df["id_mutation"].is_unique
    assert {"mois"} <= set(df.columns)


def test_model_beats_naive_baseline(raw):
    df = clean.clean(raw)
    _, metrics = train.train(df)
    assert metrics["modele"]["mape"] < metrics["baseline_prix_m2_median"]["mape"]


def test_api_predict(raw, tmp_path, monkeypatch):
    model, _ = train.train(clean.clean(raw))
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict, "load_model", lambda: model)

    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "type_local": "Appartement",
            "surface_reelle_bati": 50,
            "nombre_pieces_principales": 2,
            "code_departement": "75",
            "latitude": 48.85,
            "longitude": 2.35,
        },
    )
    assert response.status_code == 200
    body = response.json()
    # Données synthétiques : ~10 000 €/m² à Paris
    assert 6_000 < body["prix_m2_estime"] < 14_000


def test_api_rejects_invalid_input():
    from api.main import app

    response = TestClient(app).post("/predict", json={"type_local": "Château"})
    assert response.status_code == 422
