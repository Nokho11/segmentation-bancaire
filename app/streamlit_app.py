"""Application Streamlit : affecter un nouveau client à un segment (extension optionnelle).

Lancement depuis la racine du dépôt :
    streamlit run app/streamlit_app.py

Règle d'affectation (identique au Jour 5) : le client est encodé avec le préprocesseur
ajusté sur la population (one-hot + standardisation), puis affecté au centre de segment
le plus proche (outputs/tables/05_centroides_segments.csv).

En complément :
- distances du client aux 4 centres (une affectation peut être franche ou limite) ;
- alerte « client atypique » par Isolation Forest, réglée comme au notebook 06 ;
- affectation par lot à partir d'un fichier CSV (format du jeu UCI accepté).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import preprocessing as pp          # noqa: E402
from src.config import SEED, TAB             # noqa: E402

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
ATYPICAL_RATE = 0.0375        # part du bruit DBSCAN dans l'échantillon (notebook 06)
ACTIVE = pp.ACTIVE_NUM + pp.ACTIVE_CAT


def load_model():
    """Préprocesseur ajusté sur la population + centres des 4 segments (espace encodé)."""
    prep = pp.prepare_all()["prep"]
    centroids = pd.read_csv(TAB / "05_centroides_segments.csv", index_col=0)
    return prep, centroids


def load_context():
    """Éléments complémentaires : détecteur d'atypiques, règles de nettoyage, profils des segments."""
    d = pp.prepare_all()
    iso = IsolationForest(n_estimators=300, contamination=ATYPICAL_RATE, random_state=SEED).fit(d["X_a"].values)
    df = d["df_clean"]
    seg = pd.read_csv(ROOT / "outputs" / "labels" / "segments_population.csv", index_col=0)["segment"]
    profile = df[pp.ACTIVE_NUM].groupby(seg).mean()
    profile.loc["Ensemble"] = df[pp.ACTIVE_NUM].mean()
    return {
        "iso": iso,
        "modes": {c: df[c].mode()[0] for c in pp.IMPUTE_MODE},     # modes appris (après imputation)
        "campaign_cap": float(df["campaign"].max()),               # plafond p99 appliqué au Jour 1
        "profile": profile,
    }


def encode_active(X_mixed_new, prep, centroids):
    X_new, _ = pp.encode(X_mixed_new[ACTIVE], prep=prep)
    return X_new[centroids.columns].values


def assign(X_mixed_new, prep, centroids):
    """Renvoie le segment ('S1'..'S4') de chaque client (données mixtes brutes, 15 variables actives)."""
    D = pairwise_distances(encode_active(X_mixed_new, prep, centroids), centroids.values)
    return np.array(SEG)[D.argmin(axis=1)]


def distances(X_mixed_new, prep, centroids):
    """Distance euclidienne (espace standardisé) de chaque client à chacun des 4 centres."""
    D = pairwise_distances(encode_active(X_mixed_new, prep, centroids), centroids.values)
    return pd.DataFrame(D, columns=SEG, index=X_mixed_new.index)


def prepare_input(raw, ctx):
    """Met un fichier client au format attendu, avec les mêmes règles que le nettoyage du Jour 1.

    Accepte soit les 15 variables actives, soit le format brut du jeu UCI (avec `pdays`).
    Renvoie (données prêtes, liste des colonnes manquantes).
    """
    X = raw.copy()
    X.columns = [c.strip() for c in X.columns]
    if "contacted_before" not in X.columns and "pdays" in X.columns:
        X["contacted_before"] = np.where(X["pdays"] == 999, "no", "yes")
    missing = [c for c in ACTIVE if c not in X.columns]
    if missing:
        return None, missing
    X = X[ACTIVE].copy()
    for c in pp.ACTIVE_CAT:
        X[c] = X[c].astype(str).str.strip().str.strip('"')
    for c, mode in ctx["modes"].items():
        X[c] = X[c].replace("unknown", mode)
    for c in pp.ACTIVE_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["campaign"] = X["campaign"].clip(upper=ctx["campaign_cap"])
    return X, []


def score(X, prep, centroids, ctx):
    """Segment, marge d'affectation et statut atypique pour chaque client."""
    D = distances(X, prep, centroids)
    srt = np.sort(D.values, axis=1)
    out = pd.DataFrame(index=X.index)
    out["segment"] = D.idxmin(axis=1)
    out["profil"] = out["segment"].map(lambda s: PROFILS[s][0])
    out["marge"] = (srt[:, 1] - srt[:, 0]).round(2)        # écart entre le 1er et le 2e centre
    out["atypique"] = np.where(ctx["iso"].predict(encode_active(X, prep, centroids)) == -1, "oui", "non")
    out["recommandation"] = out["segment"].map(lambda s: PROFILS[s][1])
    return out, D


def main():
    import streamlit as st

    st.set_page_config(page_title="Segmentation bancaire", layout="wide")
    st.title("Segmentation de clientèle bancaire")
    st.caption("Solution retenue : CAH de Ward, 4 segments — affectation au centre le plus proche. "
               "La variable de souscription n'a servi ni à construire ni à choisir les segments.")

    prep, centroids = st.cache_resource(load_model)()
    ctx = st.cache_resource(load_context)()
    sizes = pd.read_csv(TAB / "05_taille_souscription.csv", index_col=0)
    key = pd.read_csv(TAB / "05_indicateurs_cles.csv", index_col=0)
    cats = dict(zip(pp.ACTIVE_CAT, prep.named_transformers_["cat"].categories_))

    tab1, tab2, tab3 = st.tabs(["Un client", "Plusieurs clients (CSV)", "Les 4 segments"])

    # ------------------------------------------------------------------ un client
    with tab1:
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
        if poutcome != "nonexistent" and previous == 0:
            st.warning("Incohérence : un résultat de campagne précédente est indiqué, mais aucun contact antérieur.")

        res, D = score(client, prep, centroids, ctx)
        s = res.loc[0, "segment"]
        nom, reco = PROFILS[s]

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Segment", f"{s} — {nom}")
        b.metric("Taux de souscription observé", f"{sizes.loc[s, 'taux de souscription (%)']:.1f} %",
                 f"indice {sizes.loc[s, 'indice (base 100 = moyenne)']:.0f} (moyenne = 100)")
        c.metric("Poids du segment", f"{sizes.loc[s, '% des clients']:.1f} % des clients")
        st.info(f"**Recommandation** : {reco}")
        if res.loc[0, "atypique"] == "oui":
            st.warning("**Client atypique** (Isolation Forest) : son profil est rare dans la clientèle. "
                       "Dans l'étude, ces clients souscrivent à près de 50 % : à confier en priorité à un conseiller.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Distance aux centres des segments** (plus court = plus proche)")
            st.bar_chart(D.T.rename(columns={0: "distance"}), horizontal=True, height=220)
            marge = res.loc[0, "marge"]
            st.caption(f"Écart avec le 2ᵉ segment le plus proche : {marge:.2f}. "
                       + ("Affectation franche." if marge >= 0.5 else
                          "Affectation limite : le client est proche de deux segments."))
        with right:
            st.markdown(f"**Le client comparé à son segment ({s})**")
            comp = pd.DataFrame({"client": client[pp.ACTIVE_NUM].iloc[0],
                                 f"moyenne {s}": ctx["profile"].loc[s],
                                 "moyenne générale": ctx["profile"].loc["Ensemble"]}).round(2)
            st.dataframe(comp, width="stretch")
        st.caption("Rappel : S1 et S2 se distinguent surtout par la période du contact (taux Euribor). "
                   "Le même client peut changer de segment si le contexte économique change.")

    # ------------------------------------------------------------------ plusieurs clients
    with tab2:
        st.markdown("Déposez un fichier CSV de clients : soit les **15 variables actives**, soit le **format brut "
                    "du jeu UCI** (séparateur `;`, avec `pdays`). Les règles de nettoyage du Jour 1 sont appliquées "
                    "(« unknown » remplacé par le mode, `campaign` plafonné, `pdays` recodé).")
        modele = pp.load_raw().drop(columns="y").head(20)
        st.download_button("Télécharger un fichier exemple (20 clients, format UCI)",
                           modele.to_csv(sep=";", index=False).encode("utf-8"),
                           "exemple_clients.csv", "text/csv")
        up = st.file_uploader("Fichier CSV", type=["csv"])
        if up is not None:
            head = up.getvalue()[:2000].decode("utf-8", errors="ignore")
            raw = pd.read_csv(up, sep=";" if head.count(";") > head.count(",") else ",")
            X, missing = prepare_input(raw, ctx)
            if missing:
                st.error("Colonnes manquantes : " + ", ".join(missing))
            else:
                bad = X[pp.ACTIVE_NUM].isna().any(axis=1)
                if bad.any():
                    st.warning(f"{int(bad.sum())} ligne(s) ignorée(s) : valeurs numériques manquantes ou invalides.")
                X = X[~bad]
                res, _ = score(X, prep, centroids, ctx)
                out = pd.concat([raw.loc[X.index], res], axis=1)
                summary = res["segment"].value_counts().reindex(SEG, fill_value=0).rename("clients").to_frame()
                summary["%"] = (100 * summary["clients"] / len(res)).round(1)
                summary["taux de souscription attendu (%)"] = sizes["taux de souscription (%)"].round(1)
                summary["atypiques"] = res.groupby("segment")["atypique"].apply(lambda s: (s == "oui").sum()) \
                    .reindex(SEG, fill_value=0)
                st.markdown(f"**{len(res)} clients affectés**")
                st.dataframe(summary, width="stretch")
                st.dataframe(out, width="stretch", height=320)
                st.download_button("Télécharger les résultats", out.to_csv(sep=";", index=False).encode("utf-8"),
                                   "clients_segmentes.csv", "text/csv")

    # ------------------------------------------------------------------ les segments
    with tab3:
        recap = pd.DataFrame({
            "profil": [PROFILS[s][0] for s in SEG],
            "% des clients": sizes.loc[SEG, "% des clients"].round(1).values,
            "taux de souscription (%)": sizes.loc[SEG, "taux de souscription (%)"].round(1).values,
            "âge moyen": ctx["profile"].loc[SEG, "age"].round(1).values,
            "Euribor moyen": ctx["profile"].loc[SEG, "euribor3m"].round(2).values,
            "déjà démarchés (%)": key.loc[SEG, "déjà démarché (campagne passée)"].values,
            "contact sur mobile (%)": key.loc[SEG, "contact sur mobile"].values,
            "recommandation": [PROFILS[s][1] for s in SEG],
        }, index=SEG)
        st.dataframe(recap, width="stretch")
        st.bar_chart(recap["taux de souscription (%)"], height=260)
        st.caption("Profils calculés sur les 41 176 clients (notebook 05). Moyenne générale de souscription : 11,3 %.")


if __name__ == "__main__":
    main()
