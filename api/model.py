"""Raccord entre l'API et le modèle entraîné au TP2."""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from predict import charger_modele, predict as _predict  # noqa: E402


class SentimentClassifier:
    """Enveloppe du modèle TP2 avec gestion des textes inconnus."""

    model_name = "tfidf-logreg (TP2, models/sentiment.joblib)"

    def __init__(self) -> None:
        charger_modele()

    def predict(self, text: str) -> dict:
        """Renvoie POSITIVE, NEGATIVE ou INDETERMINE."""
        modele = charger_modele()
        vecteur = modele.named_steps["tfidf"].transform([text])

        # Aucun terme du texte n'est connu du vocabulaire TF-IDF.
        if vecteur.nnz == 0:
            return {
                "label": "INDETERMINE",
                "score": 0.0,
                "text": text[:100],
            }

        resultat = _predict(text)
        return {
            "label": resultat["label"].upper(),
            "score": resultat["score"],
            "text": text[:100],
        }


def load_model() -> SentimentClassifier:
    """Construit le classifier ; le modèle est chargé dans src/predict.py."""
    return SentimentClassifier()
