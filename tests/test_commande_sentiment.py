"""Tests des sous-commandes de sentiment (TP6)."""

import json
import sys
from pathlib import Path

import pytest

from sentiment_app import cli

RACINE = Path(__file__).resolve().parents[1]
MODELE = RACINE / "models" / "sentiment.joblib"

avec_modele = pytest.mark.skipif(
    not MODELE.exists(),
    reason="modèle non produit : python src/train.py"
)


@avec_modele
def test_predict_reunit_les_mots(capsys):
    assert cli.main([
        "predict", "Livraison", "rapide,", "produit",
        "conforme,", "très", "satisfait"
    ]) == 0

    resultat = json.loads(capsys.readouterr().out)

    assert resultat["text"] == (
        "Livraison rapide, produit conforme, très satisfait"
    )
    assert resultat["label"] in {"POSITIVE", "NEGATIVE", "INDETERMINE"}


@avec_modele
def test_predict_s_abstient_sur_des_mots_inconnus(capsys):
    assert cli.main(["predict", "xqzvb wlkptr grombu"]) == 0

    resultat = json.loads(capsys.readouterr().out)

    assert resultat["label"] == "INDETERMINE"


def test_predict_sans_texte_est_refuse(capsys):
    with pytest.raises(SystemExit) as fin:
        cli.main(["predict"])

    assert fin.value.code == 2
    assert "texte" in capsys.readouterr().err


def test_predict_sans_modele(monkeypatch, tmp_path, capsys):
    from sentiment_app import predict

    monkeypatch.setattr(
        predict,
        "MODELE",
        tmp_path / "absent.joblib"
    )

    predict.charger_modele.cache_clear()

    try:
        assert cli.main(["predict", "Très bien"]) == 1
    finally:
        predict.charger_modele.cache_clear()

    assert "introuvable" in capsys.readouterr().err.lower()


def test_serve_lance_uvicorn_sur_l_api(monkeypatch):
    import uvicorn

    appels = []

    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, host, port: appels.append((app, host, port))
    )

    assert cli.main(["serve"]) == 0

    assert cli.main([
        "serve", "--host", "0.0.0.0", "--port", "8011"
    ]) == 0

    assert appels == [
        ("api.main:app", "127.0.0.1", 8000),
        ("api.main:app", "0.0.0.0", 8011)
    ]


def test_ui_lance_streamlit_sur_la_page(monkeypatch):
    pytest.importorskip("streamlit")

    commandes = []

    monkeypatch.setattr(
        cli.subprocess,
        "call",
        lambda commande: commandes.append(commande) or 0
    )

    assert cli.main(["ui"]) == 0

    assert commandes[0][:4] == [
        sys.executable, "-m", "streamlit", "run"
    ]

    assert Path(commandes[0][4]) == RACINE / "ui" / "app.py"


def test_ui_sans_streamlit(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "streamlit", None)

    assert cli.main(["ui"]) == 1

    assert "Streamlit absent" in capsys.readouterr().err