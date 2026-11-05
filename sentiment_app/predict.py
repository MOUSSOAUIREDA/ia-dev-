
#!/usr/bin/env python
"""Prédiction de sentiment avec modèle ML et règles de secours."""

import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib

RACINE = Path(__file__).resolve().parents[1]


def chemin_modele() -> Path:
    """Trouve automatiquement le modèle installé ou celui du dépôt."""

    if os.environ.get("SENTIMENT_MODELE"):
        return Path(os.environ["SENTIMENT_MODELE"])

    # Modèle inclus dans le package installé (.whl)
    modele_package = (
        Path(__file__).resolve().parent / "models" / "sentiment.joblib"
    )

    if modele_package.exists():
        return modele_package

    # Modèle présent dans le dépôt source
    return RACINE / "models" / "sentiment.joblib"


MODELE = chemin_modele()

# Expressions indiquant un problème clairement négatif.
INDICES_NEGATIFS = (
    "injoignable",
    "deconseille",
    "casse",
    "cassee",
    "endommage",
    "endommagee",
    "defectueux",
    "defectueuse",
    "inutilisable",
    "ne fonctionne pas",
    "ne marche pas",
    "ne demarre pas",
    "ne repond pas",
    "ne recommande pas",
    "refuse de repondre",
    "aucune reponse",
    "service client inexistant",
    "tres tardive",
    "livraison tardive",
    "commande incomplete",
    "en panne",
)


class Resultat(TypedDict):
    """Ce que predict() renvoie : un type précis."""

    label: str
    score: float


@lru_cache(maxsize=1)
def charger_modele():
    """Charge le modèle une seule fois par processus."""

    if not MODELE.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODELE}. "
            "Vérifiez que sentiment.joblib est bien installé."
        )

    return joblib.load(MODELE)


def normaliser(texte: str) -> str:
    """Met le texte en minuscules et retire les accents."""

    texte = texte.lower()
    texte = unicodedata.normalize("NFD", texte)

    texte = "".join(
        caractere
        for caractere in texte
        if unicodedata.category(caractere) != "Mn"
    )

    return texte


def contient_indice_negatif(texte: str) -> bool:
    """Détecte des indices négatifs explicites."""

    texte_normalise = normaliser(texte)

    for indice in INDICES_NEGATIFS:
        motif = r"\b" + re.escape(indice) + r"\b"

        if re.search(motif, texte_normalise):
            return True

    return False


def predict(texte: str) -> Resultat:
    """Renvoie le label et le score associé à la classe choisie."""

    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("texte vide")

    pipeline = charger_modele()

    probabilites = pipeline.predict_proba([texte])[0]
    classes = pipeline.classes_

    scores = {
        etiquette: float(score)
        for etiquette, score in zip(classes, probabilites, strict=True)
    }

    # Règle de secours pour les avis explicitement négatifs.
    if contient_indice_negatif(texte):
        etiquette = "negative"
    else:
        etiquette = str(pipeline.predict([texte])[0])

    return {
        "label": etiquette,
        "score": round(scores[etiquette], 4),
    }


def main() -> int:
    """Point d'entrée pour la prédiction en ligne de commande."""

    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    resultat = predict(" ".join(sys.argv[1:]))

    print(
        f"{resultat['label']} "
        f"(confiance {resultat['score']:.0%})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
