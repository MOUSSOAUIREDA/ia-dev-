
#!/usr/bin/env python
"""Entraîne et sauvegarde le modèle de sentiment."""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RACINE = Path(__file__).resolve().parents[1]
DEMO = RACINE / "data" / "demo_reviews.csv"
AVIS_PRODUITS = RACINE / "data" / "avis_produits.csv"

URL_ALLOCINE = (
    "https://huggingface.co/datasets/tblard/allocine/resolve/main/"
    "allocine/validation-00000-of-00001.parquet"
)

MODELE = RACINE / "models" / "sentiment.joblib"
METRIQUES = RACINE / "models" / "metrics.json"

SEUIL_F1 = 0.85
GRAINE = 42

# Poids d'apprentissage par source.
POIDS_ALLOCINE = 1.0
POIDS_PRODUITS = 40.0
POIDS_DEMO = 1.0


def charger(allocine: bool) -> tuple[pd.DataFrame, str]:
    """Charge les avis et conserve leur source."""

    if allocine:
        df = pd.read_parquet(URL_ALLOCINE)

        df["label"] = df["label"].map({
            0: "negative",
            1: "positive",
        })

        df = df[["review", "label"]].dropna()
        df["origine"] = "allocine"

        if AVIS_PRODUITS.exists():
            produits = pd.read_csv(
                AVIS_PRODUITS,
                encoding="utf-8",
            )

            produits = produits[["review", "label"]].dropna()
            produits["origine"] = "produit"

            df = pd.concat(
                [df, produits],
                ignore_index=True,
            )

        return df, "allocine+produits"

    if not DEMO.exists():
        raise SystemExit(
            "data/demo_reviews.csv absent : "
            "lancez python src/make_demo.py"
        )

    df = pd.read_csv(DEMO)
    df["origine"] = "demo"

    return df, "demo"


def construire_pipeline() -> Pipeline:
    """Construit le pipeline TF-IDF et régression logistique."""

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=GRAINE,
            ),
        ),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--allocine",
        action="store_true",
        help=(
            "utiliser Allociné avec les avis complémentaires "
            "de produits"
        ),
    )

    args = parser.parse_args()

    # 1. Chargement des données.
    df, source = charger(args.allocine)

    # 2. Nettoyage des données.
    df = df[
        df["label"].isin(["positive", "negative"])
    ].copy()

    df["review"] = df["review"].astype(str).str.strip()

    df = df[df["review"] != ""]

    df = df.drop_duplicates(
        subset=["review", "label"]
    ).reset_index(drop=True)

    if df.empty:
        raise SystemExit("Erreur : aucun avis valide à entraîner.")

    print(f"Source : {source}")
    print(f"Nombre total d'avis : {len(df)}")
    print(
        "Avis positifs : "
        f"{(df['label'] == 'positive').sum()}"
    )
    print(
        "Avis négatifs : "
        f"{(df['label'] == 'negative').sum()}"
    )

    if "origine" in df.columns:
        print("\nRépartition par source :")
        print(df["origine"].value_counts().to_string())

    # 3. Séparation entraînement / test.
    # La source est conservée pour calculer les poids.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=GRAINE,
        stratify=df["label"],
    )

    X_train = train_df["review"]
    y_train = train_df["label"]

    X_test = test_df["review"]
    y_test = test_df["label"]

    print(f"\nAvis d'entraînement : {len(X_train)}")
    print(f"Avis de test : {len(X_test)}")

    # 4. Attribution des poids d'apprentissage.
    poids_sources = {
        "allocine": POIDS_ALLOCINE,
        "produit": POIDS_PRODUITS,
        "demo": POIDS_DEMO,
    }

    poids = train_df["origine"].map(poids_sources)

    if poids.isna().any():
        raise SystemExit(
            "Erreur : une source de données n'a pas de poids défini."
        )

    nb_produits = int(
        (train_df["origine"] == "produit").sum()
    )

    print(f"Avis produits dans l'entraînement : {nb_produits}")
    print(f"Poids des avis Allociné : {POIDS_ALLOCINE}")
    print(f"Poids des avis produits : {POIDS_PRODUITS}")

    # 5. Entraînement du modèle.
    pipeline = construire_pipeline()

    pipeline.fit(
        X_train,
        y_train,
        clf__sample_weight=poids.to_numpy(),
    )

    print(
        "\nTaille du vocabulaire : "
        f"{len(pipeline.named_steps['tfidf'].vocabulary_)} termes"
    )

    # 6. Évaluation sur le jeu de test.
    predictions = pipeline.predict(X_test)

    rapport = classification_report(
        y_test,
        predictions,
        labels=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    f1 = rapport["macro avg"]["f1-score"]
    accuracy = rapport["accuracy"]
    recall_negative = rapport["negative"]["recall"]

    print("\nRapport de classification :")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["negative", "positive"],
            zero_division=0,
        )
    )

    print(f"F1 macro : {f1:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Recall négatif : {recall_negative:.4f}")

    if f1 < SEUIL_F1:
        print(
            f"Attention : le F1 macro est inférieur au seuil "
            f"de {SEUIL_F1:.2f}."
        )

    # 7. Sauvegarde du modèle.
    MODELE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(pipeline, MODELE)

    # 8. Sauvegarde des métriques.
    metriques = {
        "source": source,
        "date": date.today().isoformat(),
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_produits_train": nb_produits,
        "poids_allocine": POIDS_ALLOCINE,
        "poids_produits": POIDS_PRODUITS,
        "f1_macro": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "recall_negative": round(recall_negative, 4),
        "seuil_f1": SEUIL_F1,
    }

    METRIQUES.write_text(
        json.dumps(
            metriques,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"\nModèle enregistré : {MODELE.relative_to(RACINE)}")
    print(
        "Métriques enregistrées : "
        f"{METRIQUES.relative_to(RACINE)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
