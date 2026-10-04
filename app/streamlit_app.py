"""Application Streamlit : affecter un nouveau client à un segment (extension optionnelle).

Lancement depuis la racine du dépôt :
    streamlit run app/streamlit_app.py

Règle d'affectation (identique au Jour 5) : le client est encodé avec le préprocesseur
ajusté sur la population (one-hot + standardisation), puis affecté au centre de segment
le plus proche (outputs/tables/05_centroides_segments.csv).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import pairwise_distances_argmin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import preprocessing as pp          # noqa: E402
from src.config import TAB                   # noqa: E402

SEG = ["S1", "S2", "S3", "S4"]
PROFILS = {
    "S1": ("Nouveaux prospects, taux élevés",
           "Réduire la pression téléphonique : plafonner à deux appels, basculer vers le courriel ou le SMS, "
           "reporter la sollicitation à une période plus favorable."),
    "S2": ("Nouveaux prospects, taux bas",
           "Campagne d'acquisition sur mobile, concentrée sur les périodes de taux bas ; deux appels au maximum."),
    "S3": ("Clients déjà démarchés",
           "Relance personnalisée selon le résultat précédent : renouvellement pour les anciens succès, "
           "nouvel argumentaire pour les anciens échecs."),
    "S4": ("Profils réceptifs",
           "Appeler en premier, avec les conseillers les plus expérimentés ; offre d'épargne sécurisée "
           "orientée seniors et étudiants ; un ou deux appels suffisent."),
}


def load_model():
    """Préprocesseur ajusté sur la population + centres des 4 segments (espace encodé)."""
    prep = pp.prepare_all()["prep"]
    centroids = pd.read_csv(TAB / "05_centroides_segments.csv", index_col=0)
    return prep, centroids


def assign(X_mixed_new, prep, centroids):
    """Renvoie le segment ('S1'..'S4') de chaque client (données mixtes brutes, 15 variables actives)."""
    X_new, _ = pp.encode(X_mixed_new[pp.ACTIVE_NUM + pp.ACTIVE_CAT], prep=prep)
    X_new = X_new[centroids.columns]
    return pd.Series(pairwise_distances_argmin(X_new.values, centroids.values)).map(dict(enumerate(SEG))).values


def main():
    import streamlit as st

    st.set_page_config(page_title="Segmentation bancaire", layout="wide")
    st.title("Affecter un client à un segment")
    st.caption("Solution retenue : CAH de Ward, 4 segments — affectation au centre le plus proche.")

    prep, centroids = st.cache_resource(load_model)()
    sizes = pd.read_csv(TAB / "05_taille_souscription.csv", index_col=0)
    cats = dict(zip(pp.ACTIVE_CAT, prep.named_transformers_["cat"].categories_))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Profil du client")
        age = st.number_input("Âge", 17, 98, 40)
        job = st.selectbox("Profession", cats["job"], index=list(cats["job"]).index("admin."))
        marital = st.selectbox("Statut marital", cats["marital"], index=list(cats["marital"]).index("married"))
        education = st.selectbox("Niveau d'études", cats["education"],
                                 index=list(cats["education"]).index("university.degree"))
        default = st.selectbox("Crédit en défaut", cats["default"])
        housing = st.selectbox("Prêt immobilier", cats["housing"])
        loan = st.selectbox("Prêt personnel", cats["loan"])
    with col2:
        st.subheader("Relation avec la banque")
        contact = st.selectbox("Canal de contact", cats["contact"])
        campaign = st.number_input("Appels pendant la campagne", 1, 15, 2)
        previous = st.number_input("Contacts lors des campagnes précédentes", 0, 7, 0)
        poutcome = st.selectbox("Résultat de la campagne précédente", cats["poutcome"],
                                index=list(cats["poutcome"]).index("nonexistent"))
        contacted_before = st.selectbox("Recontacté après une campagne précédente", cats["contacted_before"])
    with col3:
        st.subheader("Contexte économique")
        euribor3m = st.number_input("Taux Euribor 3 mois (%)", 0.5, 5.1, 4.86, step=0.01)
        cons_price = st.number_input("Indice des prix à la consommation", 92.0, 95.0, 93.99, step=0.01)
        cons_conf = st.number_input("Indice de confiance des ménages", -51.0, -26.0, -36.4, step=0.1)

    client = pd.DataFrame([{
        "age": age, "campaign": campaign, "previous": previous, "cons.price.idx": cons_price,
        "cons.conf.idx": cons_conf, "euribor3m": euribor3m, "job": job, "marital": marital,
        "education": education, "default": default, "housing": housing, "loan": loan,
        "contact": contact, "poutcome": poutcome, "contacted_before": contacted_before}])
    s = assign(client, prep, centroids)[0]
    nom, reco = PROFILS[s]

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Segment", f"{s} — {nom}")
    b.metric("Taux de souscription observé", f"{sizes.loc[s, 'taux de souscription (%)']:.1f} %",
             f"indice {sizes.loc[s, 'indice (base 100 = moyenne)']:.0f} (moyenne = 100)")
    c.metric("Poids du segment", f"{sizes.loc[s, '% des clients']:.1f} % des clients")
    st.info(f"**Recommandation** : {reco}")
    st.caption("Rappel : S1 et S2 se distinguent surtout par la période du contact (taux Euribor). "
               "Le même client peut changer de segment si le contexte économique change.")


if __name__ == "__main__":
    main()
