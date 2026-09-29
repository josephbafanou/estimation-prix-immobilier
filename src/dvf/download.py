"""Téléchargement des fichiers DVF géolocalisés (data.gouv.fr).

    python -m dvf.download
"""
import logging

import requests

from dvf import config

log = logging.getLogger(__name__)


def download(annees=config.ANNEES, departements=config.DEPARTEMENTS) -> list:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for annee in annees:
        for dep in departements:
            dest = config.DATA_DIR / f"dvf_{annee}_{dep}.csv.gz"
            paths.append(dest)
            if dest.exists():
                log.info("Déjà présent : %s", dest.name)
                continue
            url = config.DVF_URL.format(annee=annee, dep=dep)
            log.info("Téléchargement %s", url)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    download()
