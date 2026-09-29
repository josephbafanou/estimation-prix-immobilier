"""Entraînement et évaluation du modèle de prix.

    python -m dvf.train
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from dvf import clean, config, download

log = logging.getLogger(__name__)
FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def build_model() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", config.NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    n_num = len(config.NUMERIC_FEATURES)
    cat_mask = [False] * n_num + [True] * len(config.CATEGORICAL_FEATURES)
    gbr = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, categorical_features=cat_mask, random_state=42
    )
    return Pipeline([("pre", pre), ("model", gbr)])


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae_eur": round(float(mean_absolute_error(y_true, y_pred)), 0),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X, y = df[FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Le prix est très asymétrique : on apprend sur log(prix)
    model = build_model().fit(X_train, np.log1p(y_train))
    y_pred = np.expm1(model.predict(X_test))

    # Référence naïve : prix médian au m² × surface
    median_m2 = (y_train / X_train["surface_reelle_bati"]).median()
    baseline = X_test["surface_reelle_bati"] * median_m2

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "modele": evaluate(y_test, y_pred),
        "baseline_prix_m2_median": evaluate(y_test, baseline),
    }
    return model, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = clean.clean(clean.read_raw(download.download()))
    log.info("%d transactions après nettoyage", len(df))

    model, metrics = train(df)

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    log.info("Métriques : %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
