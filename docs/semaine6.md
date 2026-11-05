# Semaine 6 : le paquet, notes du binôme

> Trois questions, quelques phrases chacune. Répondez avec ce que vous avez observé.

## 1. Pourquoi le modèle n'est-il pas dans la roue ? Comment le paquet le trouve-t-il ?

Le modèle pèse plus que le code, change à un autre rythme (on le réentraîne sans republier le code) et n'est pas versionné dans Git. La roue ne contient que le code. Le paquet trouve le modèle par la variable d'environnement `SENTIMENT_MODELE` ; sans elle, en développement, c'est `models/sentiment.joblib` à la racine du dépôt. Hors du dépôt, sans la variable, la commande répond « modèle introuvable » avec le chemin cherché et les deux remèdes possibles : c'est le comportement attendu. Le TP7 (MLflow) en fera un artefact suivi.

## 2. Qu'est-ce qui a changé dans api/model.py quand predict() est entré dans le paquet, et pourquoi est-ce mieux ?

Les lignes qui modifiaient `sys.path` pour importer depuis `src/` (avec leurs commentaires `# noqa: E402`) ont été supprimées et remplacées par des imports explicites depuis `sentiment_app.predict`. Cela permet d'importer le module partout où le paquet est installé, quel que soit le répertoire de travail courant ; les outils d'analyse statique comme Ruff, Mypy ainsi que l'EDI résolvent proprement les types et les dépendances ; et le test de non-régression `test_l_api_ne_bricole_plus_le_chemin_de_python` garantit que cette modification manuelle de `sys.path` ne sera plus réintroduite.

## 3. La roue installée hors du dépôt : qu'a-t-il fallu lui fournir, et pourquoi scikit-learn est-il épinglé dans pyproject.toml ?

Il a fallu lui fournir le modèle via la variable d'environnement `SENTIMENT_MODELE`. Toutes les dépendances logicielles (FastAPI, Uvicorn, Pydantic, Scikit-Learn, Joblib) ont été installées automatiquement par `pip` grâce aux métadonnées du paquet. `scikit-learn` est épinglé (`scikit-learn>=1.5,<1.6` dans `pyproject.toml` et `scikit-learn==1.5.*` dans la CI) car un modèle sérialisé avec Joblib/Pickle ne se recharge de manière fiable qu'avec la version mineure qui l'a produit : un modèle entraîné avec une version différente (ex. scikit-learn 1.9.1) et relu en 1.5.2 échoue avec `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`. Épingler la version garantit la compatibilité et la reproductibilité entre l'artefact et le code.
