"""Interface Streamlit du TP5 : communication exclusivement via l'API."""
import streamlit as st
from ui import client

st.set_page_config(
    page_title="Analyse de sentiment",
    page_icon="💬",
    layout="wide",
)

st.title("Analyse de sentiment des avis clients")

session = client.creer_session()

etat = client.sante(session)

if not etat.ok:
    st.error(etat.message)
else:
    st.caption(f"API prête — Modèle : {etat.message}")

    texte = st.text_area(
        "Avis à analyser",
        "Livraison rapide, produit conforme, très satisfait",
        height=130,
    )

    if st.button("Analyser", type="primary"):
        if not texte.strip():
            st.warning(client.REFUS)
        else:
            resultat = client.predire(session, texte)

            if not resultat.ok:
                st.error(resultat.message)
            elif resultat.label == "POSITIVE":
                st.metric("Avis positif", f"{resultat.score:.0%}")
            elif resultat.label == "NEGATIVE":
                st.metric("Avis négatif", f"{resultat.score:.0%}")
            else:
                st.metric(
                    "Je ne sais pas — vérification humaine",
                    f"{resultat.score:.0%}",
                )
                st.info(
                    "La confiance est insuffisante : le seuil d'abstention "
                    "est atteint. Faites vérifier cet avis par un humain."
                )
