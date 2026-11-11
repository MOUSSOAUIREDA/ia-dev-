
"""Tests aux limites du contrat d'entrée (TP4, étape B)."""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

MODELE = Path(__file__).resolve().parents[1] / "models" / "sentiment.joblib"

pytestmark = pytest.mark.skipif(
    not MODELE.exists(),
    reason="modèle non produit : python src/train.py",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def en_json(valeur):
    return json.dumps(valeur).encode("utf-8")


# B.1 : huit cas de validation
CAS = [
    pytest.param(en_json({"text": ""}), 422, id="texte-vide"),
    pytest.param(en_json({"text": "a" * 2001}), 422, id="2001-caracteres"),
    pytest.param(en_json({"text": "   "}), 422, id="texte-espaces"),
    pytest.param(en_json({"text": "a"}), 200, id="un-caractere"),
    pytest.param(en_json({"text": "a" * 2000}), 200, id="2000-caracteres"),
    pytest.param(en_json({}), 422, id="champ-absent"),
    pytest.param(en_json({"text": 42}), 422, id="mauvais-type"),
    pytest.param(b"pas du json", 422, id="json-invalide"),
]


def verifier_erreur_standard(reponse):
    assert reponse.status_code == 422

    detail = reponse.json()["detail"]

    assert isinstance(detail, list)
    assert len(detail) > 0

    for erreur in detail:
        assert "loc" in erreur
        assert "msg" in erreur
        assert "type" in erreur


# B.1 : tests des frontières du contrat
@pytest.mark.parametrize("corps, code_attendu", CAS)
def test_frontieres_du_contrat(client, corps, code_attendu):
    reponse = client.post(
        "/predict",
        content=corps,
        headers={"Content-Type": "application/json"},
    )

    assert reponse.status_code == code_attendu

    if code_attendu == 422:
        verifier_erreur_standard(reponse)
    else:
        assert reponse.json()["label"] in {
            "POSITIVE",
            "NEGATIVE",
            "INDETERMINE",
        }
        assert 0 <= reponse.json()["score"] <= 1


# B.5 : tests de propriétés avec Hypothesis

@settings(max_examples=20, deadline=None)
@given(
    texte=st.text(min_size=1, max_size=2000).filter(
        lambda valeur: bool(valeur.strip())
    )
)
def test_tout_texte_non_blanc_est_accepte(texte):
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        reponse = client.post("/predict", json={"text": texte})

    assert reponse.status_code == 200

    resultat = reponse.json()

    assert resultat["label"] in {
        "POSITIVE",
        "NEGATIVE",
        "INDETERMINE",
    }
    assert 0 <= resultat["score"] <= 1


@settings(max_examples=20, deadline=None)
@given(
    texte=st.text(
        alphabet=[" ", "\t", "\n", "\r"],
        min_size=1,
        max_size=2000,
    )
)
def test_tout_texte_blanc_est_refuse(texte):
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        reponse = client.post("/predict", json={"text": texte})

    verifier_erreur_standard(reponse)
