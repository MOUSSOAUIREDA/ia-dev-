
import csv
from pathlib import Path

fichier = Path("data/jeu_or.csv")
fichier.parent.mkdir(parents=True, exist_ok=True)

avis = [
    ("Livraison rapide et produit conforme, je suis très satisfait", "POSITIVE", ""),
    ("Produit excellent, je recommande vivement ce vendeur", "POSITIVE", ""),
    ("Le service client a été très efficace et aimable", "POSITIVE", ""),
    ("Bonne qualité pour le prix, rien à redire", "POSITIVE", ""),
    ("Commande reçue avant la date prévue, parfait", "POSITIVE", ""),
    ("Installation facile et fonctionnement impeccable", "POSITIVE", ""),
    ("Je suis content de mon achat, tout fonctionne bien", "POSITIVE", ""),
    ("Emballage soigné et livraison sans problème", "POSITIVE", ""),
    ("Produit cassé à la réception, je suis très déçu", "NEGATIVE", ""),
    ("Service client injoignable, je déconseille fortement", "NEGATIVE", ""),
    ("Commande jamais reçue et aucun remboursement", "NEGATIVE", ""),
    ("Qualité médiocre, le produit ne fonctionne déjà plus", "NEGATIVE", ""),
    ("Livraison très tardive et colis endommagé", "NEGATIVE", ""),
    ("Le vendeur refuse de répondre à mes réclamations", "NEGATIVE", ""),
    ("Très mauvaise expérience, je regrette cet achat", "NEGATIVE", ""),
    ("Produit inutilisable dès le premier jour", "NEGATIVE", ""),
    ("Le produit est joli mais il ne fonctionne pas", "NEGATIVE", "difficile"),
    ("Livraison rapide, malheureusement l'article est cassé", "NEGATIVE", "difficile"),
    ("Ce n'est pas mauvais, mais je ne le recommande pas", "NEGATIVE", "difficile"),
    ("Qualité correcte, cependant le prix est beaucoup trop élevé", "NEGATIVE", "difficile"),
    ("Je pensais être déçu, mais finalement le produit est excellent", "POSITIVE", "difficile"),
    ("Le colis est arrivé en retard, mais le service client a bien réagi", "POSITIVE", "difficile"),
    ("Pas parfait, mais le rapport qualité-prix est excellent", "POSITIVE", "difficile"),
    ("Le produit fonctionne bien malgré un emballage abîmé", "POSITIVE", "difficile"),
]

with fichier.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "attendu", "difficulte", "erreur_connue"])
    for texte, attendu, difficulte in avis:
        writer.writerow([texte, attendu, difficulte, ""])

print(f"Fichier créé : {fichier.resolve()}")
print(f"Nombre d'avis : {len(avis)}")
print(f"Avis difficiles : {sum(bool(a[2]) for a in avis)}")
