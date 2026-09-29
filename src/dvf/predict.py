"""Prédiction à partir du modèle entraîné."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from dvf import config
from dvf.train import FEATURES


@lru_cache(maxsize=1)
def load_model(path: str = str(config.MODEL_PATH)):
    return joblib.load(path)


def predict(biens: list[dict], model=None) -> list[float]:
    model = model or load_model()
    X = pd.DataFrame(biens)[FEATURES]
    return [round(float(p), -2) for p in np.expm1(model.predict(X))]
