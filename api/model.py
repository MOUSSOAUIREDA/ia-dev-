"""Raccord entre l'API et le modèle entraîné au TP2.

L'API n'entraîne rien et ne connaît pas scikit-learn : elle appelle la fonction
predict() du paquet (sentiment_app/predict.py), qui charge le modèle une seule fois.
"""

from sentiment_app.predict import (
    charger_modele,
    contient_indice_negatif,
)
from sentiment_app.predict import (
    predict as _predict,
)

SEUIL_ABSTENTION = 0.68


class SentimentClassifier:
    """Enveloppe du modèle TP2 avec l'interface attendue par l'API."""

    model_name = "tfidf-logreg (TP2, models/sentiment.joblib)"

    def __init__(self) -> None:
        charger_modele()

    def predict(self, text: str) -> dict:
        """Renvoie POSITIVE, NEGATIVE ou INDETERMINE."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Le texte ne peut pas être vide.")

        modele = charger_modele()

        # Règle prioritaire : reconnaître les indices négatifs explicites.
        if contient_indice_negatif(text):
            resultat = _predict(text)
            label = "NEGATIVE"

        else:
            # Vérifier si le texte contient des termes connus du modèle.
            vecteur = modele.named_steps["tfidf"].transform([text])

            if vecteur.nnz == 0:
                return {
                    "label": "INDETERMINE",
                    "score": 0.0,
                    "text": text[:100],
                }

            resultat = _predict(text)

            # Appliquer le seuil d'abstention.
            if resultat["score"] < SEUIL_ABSTENTION:
                label = "INDETERMINE"
            else:
                label = resultat["label"].upper()

        return {
            "label": label,
            "score": resultat["score"],
            "text": text[:100],
        }


def load_model() -> SentimentClassifier:
    """Construit le classifier (le cache de chargement est dans sentiment_app/predict.py)."""
    return SentimentClassifier()
