
#!/usr/bin/env python
"""Prédiction de sentiment avec modèle ML et règles de secours."""

import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import joblib

MODELE = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "sentiment.joblib"
)

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


@lru_cache(maxsize=1)
def charger_modele():
    """Charge le modèle une seule fois par processus."""
    if not MODELE.exists():
        raise FileNotFoundError(
            "models/sentiment.joblib absent : "
            "lancez python src/train.py"
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


def predict(texte: str) -> dict:
    """Renvoie le label et le score associé à la classe choisie."""
    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("texte vide")

    pipeline = charger_modele()

    probabilites = pipeline.predict_proba([texte])[0]
    classes = pipeline.classes_

    scores = {
        etiquette: float(score)
        for etiquette, score in zip(classes, probabilites)
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
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    resultat = predict(" ".join(sys.argv[1:]))

    print(
        f"{resultat['label']} "
        f"(score {resultat['score']:.0%})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
