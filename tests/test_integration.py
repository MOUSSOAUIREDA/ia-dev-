
"""Tests d'intégration : démarrage et fonctionnement réel de l'API."""

from pathlib import Path

import pytest

MODELE = Path(__file__).resolve().parents[1] / "models" / "sentiment.joblib"

pytestmark = pytest.mark.skipif(
    not MODELE.exists(),
    reason="modèle non produit : python src/train.py",
)


def test_demarrage_reel_charge_le_modele():
    """D.1 : le démarrage charge le modèle."""

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        reponse = client.get("/health")

    assert reponse.status_code == 200
    assert "TP2" in reponse.json()["model"]


def test_modele_absent_empeche_demarrage(monkeypatch, tmp_path):
    """D.2 : l'API ne démarre pas si le modèle est absent."""

    from fastapi.testclient import TestClient

    import predict
    from api.main import app
    from predict import charger_modele

    modele_absent = tmp_path / "modele_inexistant.joblib"

    monkeypatch.setattr(predict, "MODELE", modele_absent)
    charger_modele.cache_clear()

    try:
        with pytest.raises(FileNotFoundError):
            with TestClient(app):
                pass
    finally:
        charger_modele.cache_clear()


def test_modele_non_charge_retourne_503(monkeypatch):
    """D.3 : les endpoints refusent les requêtes sans modèle."""

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "classifier", None)

        reponse_health = client.get("/health")
        reponse_predict = client.post(
            "/predict",
            json={"text": "Livraison rapide et produit excellent"},
        )

    assert reponse_health.status_code == 503
    assert reponse_predict.status_code == 503


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("panne du modèle"),
        ValueError("bug interne"),
    ],
    ids=["runtime-error", "value-error"],
)
def test_erreur_interne_retourne_500(monkeypatch, exception):
    """D.4 : les erreurs internes donnent 500 sans exposer leur détail."""

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        classifier = app.state.classifier

        def provoquer_erreur(text):
            raise exception

        monkeypatch.setattr(classifier, "predict", provoquer_erreur)

        reponse = client.post(
            "/predict",
            json={"text": "Livraison rapide et produit excellent"},
        )

    assert reponse.status_code == 500
    assert reponse.json()["detail"] == "Erreur interne du serveur"

    assert str(exception) not in reponse.text
