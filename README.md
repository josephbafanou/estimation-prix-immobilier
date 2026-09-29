# 🏠 Estimation des prix immobiliers en Île-de-France

Modèle de **machine learning** qui estime le prix de vente d'un appartement ou d'une maison à partir des transactions réelles des **Demandes de Valeurs Foncières (DVF)**, publiées en open data par l'État. Le modèle est exposé via une **API FastAPI**.

![CI](https://github.com/josephbafanou/estimation-prix-immobilier/actions/workflows/ci.yml/badge.svg)

**Périmètre :** Paris (75), Seine-et-Marne (77), Hauts-de-Seine (92), Seine-Saint-Denis (93), Val-de-Marne (94) · ventes 2024–2025.

---

## Démarche

```
DVF (data.gouv.fr) ─► download.py ─► clean.py ─► train.py ─► model.joblib ─► API FastAPI
                                     nettoyage    scikit-learn               /predict
```

**1. Nettoyage** — les données DVF sont brutes : une vente peut regrouper plusieurs lots, caves ou dépendances. Je ne garde que les **ventes simples d'un seul logement**, puis j'écarte les valeurs aberrantes (surface < 9 m² ou > 400 m², prix au m² hors de 1 000 – 25 000 €).

**2. Variables** — surface habitable, nombre de pièces, surface du terrain, latitude / longitude, département, type de bien, mois de la vente.

**3. Modèle** — `HistGradientBoostingRegressor` (scikit-learn), entraîné sur **log(prix)** car la distribution des prix est très asymétrique.

**4. Évaluation** — 20 % des ventes sont mises de côté pour le test. Le modèle est comparé à une référence naïve (prix médian au m² × surface) sur trois métriques : MAE (erreur moyenne en €), MAPE (erreur moyenne en %) et R².

## Résultats

Entraînement sur **48 511 ventes** après nettoyage (38 808 pour l'apprentissage, 9 703 pour le test), lancé via le workflow GitHub Actions *Entraînement*.

| Modèle | MAE | MAPE | R² |
|---|---|---|---|
| Référence (prix médian au m² × surface) | 178 956 € | 60,9 % | 0,21 |
| **Gradient Boosting** | **67 997 €** | **21,9 %** | **0,79** |

Le modèle divise l'erreur moyenne par **2,6** par rapport à la référence, et l'erreur relative passe de 61 % à 22 %.

Les métriques sont régénérées dans `reports/metrics.json` à chaque entraînement.

## Lancer le projet

```bash
git clone https://github.com/josephbafanou/estimation-prix-immobilier.git
cd estimation-prix-immobilier
pip install -r requirements.txt
export PYTHONPATH=src          # Windows : set PYTHONPATH=src

python -m dvf.download         # télécharge les fichiers DVF (~100 Mo)
python -m dvf.train            # nettoie, entraîne, écrit models/ et reports/
uvicorn api.main:app --reload  # API sur http://localhost:8000/docs
```

Ou avec `make` : `make data train api`.

### Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"type_local": "Appartement", "surface_reelle_bati": 45, "nombre_pieces_principales": 2,
       "code_departement": "77", "latitude": 48.9601, "longitude": 2.8788}'
```

La réponse contient `prix_estime` (en €, arrondi à la centaine) et `prix_m2_estime`.

## Tests

```bash
pytest -v
```

Les tests utilisent des données synthétiques : règles de nettoyage, modèle meilleur que la référence naïve, validation des entrées de l'API. Ils tournent à chaque push via GitHub Actions. Un second workflow, **Entraînement**, se lance à la main depuis l'onglet *Actions* et entraîne le modèle sur les vraies données.

## Pistes d'évolution

- [ ] Prix moyen du quartier (IRIS) comme variable, avec un encodage sans fuite de données
- [ ] Distance aux gares et au centre de Paris
- [ ] Explicabilité des prédictions (SHAP)
- [ ] Interface Streamlit pour tester l'estimation

## Source

[Demandes de valeurs foncières géolocalisées — data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/) (Licence Ouverte 2.0)

---

Réalisé par **Joseph-Bernardin Afanou** · [LinkedIn](https://www.linkedin.com/in/joseph-bernardin-afanou)
