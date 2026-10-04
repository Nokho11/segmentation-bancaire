"""Prétraitement réutilisable du jeu Bank Marketing (livrable du Jour 1).

Usage minimal :
    from src.preprocessing import prepare_all
    d = prepare_all()
    d["X_a"]      -> représentation (a) : one-hot + standardisation (échantillon)
    d["X_mixed"]  -> représentation (b) : données mixtes brutes (échantillon)
    d["y"]        -> vérité terrain (JAMAIS utilisée pour l'entraînement)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PATH, N_SAMPLE, SEED

# ---------------------------------------------------------------------------
# Rôle des variables
# ---------------------------------------------------------------------------
# Variables ACTIVES : elles servent à construire les clusters.
ACTIVE_NUM = ["age", "campaign", "previous",
              "cons.price.idx", "cons.conf.idx", "euribor3m"]
ACTIVE_CAT = ["job", "marital", "education", "default", "housing", "loan",
              "contact", "poutcome", "contacted_before"]
# Variables ILLUSTRATIVES : exclues du clustering, gardées pour décrire les segments.
ILLUSTRATIVE = ["duration", "month", "day_of_week", "pdays",
                "emp.var.rate", "nr.employed"]
TARGET = "y"

# Modalités "unknown" rares -> imputées par le mode. Pour `default`, "unknown"
# représente ~21 % des clients : on la garde comme modalité à part entière.
IMPUTE_MODE = ["job", "marital", "education", "housing", "loan"]


def load_raw(path=DATA_PATH):
    """Charge le fichier CSV d'origine (séparateur ';')."""
    return pd.read_csv(path, sep=";")


def clean(df, verbose=False):
    """Nettoyage : doublons, 'unknown', pdays, valeurs aberrantes.

    Retourne un nouveau DataFrame (l'original n'est pas modifié).
    """
    out = df.copy()
    report = {"n_initial": len(out)}

    # 1) Doublons stricts
    out = out.drop_duplicates().reset_index(drop=True)
    report["doublons_supprimes"] = report["n_initial"] - len(out)

    # 2) Valeurs manquantes codées 'unknown'
    for col in IMPUTE_MODE:
        mode = out.loc[out[col] != "unknown", col].mode()[0]
        n = int((out[col] == "unknown").sum())
        out[col] = out[col].replace("unknown", mode)
        report[f"unknown_{col}"] = f"{n} -> mode '{mode}'"
    report["unknown_default"] = f"{int((out['default'] == 'unknown').sum())} conservés (modalité)"

    # 3) pdays = 999 est un code ("jamais recontacté"), pas une durée :
    #    on le remplace par un indicateur binaire lisible.
    out["contacted_before"] = np.where(out["pdays"] == 999, "no", "yes")

    # 4) Valeurs aberrantes : `campaign` a une queue très longue (max = 56).
    #    On plafonne au 99e centile (winsorisation) au lieu de supprimer des clients.
    cap = out["campaign"].quantile(0.99)
    report["campaign_plafond_p99"] = float(cap)
    report["campaign_valeurs_plafonnees"] = int((out["campaign"] > cap).sum())
    out["campaign"] = out["campaign"].clip(upper=cap)

    report["n_final"] = len(out)
    if verbose:
        for k, v in report.items():
            print(f"{k:32s}: {v}")
    return out


def split_roles(df_clean):
    """Sépare variables actives (mixtes), illustratives et vérité terrain."""
    X_mixed = df_clean[ACTIVE_NUM + ACTIVE_CAT].copy()
    illus = df_clean[ILLUSTRATIVE].copy()
    y = df_clean[TARGET].copy()
    return X_mixed, illus, y


def make_preprocessor(standardize=True):
    """Représentation (a) : one-hot + standardisation z = (x - mu) / sigma.

    `drop="if_binary"` : une variable binaire donne UNE colonne 0/1 (et non deux
    colonnes redondantes qui doubleraient son poids dans la distance).
    """
    num_step = StandardScaler() if standardize else "passthrough"
    return ColumnTransformer([
        ("num", num_step, ACTIVE_NUM),
        ("cat", OneHotEncoder(drop="if_binary", sparse_output=False,
                              handle_unknown="ignore"), ACTIVE_CAT),
    ])


def encode(X_mixed, standardize=True, prep=None):
    """Applique le préprocesseur et renvoie (DataFrame encodé, préprocesseur ajusté)."""
    if prep is None:
        prep = make_preprocessor(standardize).fit(X_mixed)
    cols = [c.split("__", 1)[1] for c in prep.get_feature_names_out()]
    X = pd.DataFrame(prep.transform(X_mixed), columns=cols, index=X_mixed.index)
    return X, prep


def stratified_sample(index, y, n=N_SAMPLE, seed=SEED):
    """Échantillon aléatoire stratifié : conserve la proportion de `y`.

    `y` ne sert ici qu'à garantir la représentativité de l'échantillon ;
    aucun algorithme de clustering ne la voit.
    """
    idx, _ = train_test_split(index, train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def prepare_all(path=DATA_PATH, n_sample=N_SAMPLE, seed=SEED):
    """Pipeline complet : chargement -> nettoyage -> 2 représentations -> échantillon."""
    df = clean(load_raw(path))
    X_mixed_full, illus_full, y_full = split_roles(df)
    X_a_full, prep = encode(X_mixed_full)          # ajusté sur toute la population
    idx = stratified_sample(df.index, y_full, n_sample, seed)
    return {
        "df_clean": df, "prep": prep, "idx": idx,
        "X_mixed_full": X_mixed_full, "X_a_full": X_a_full,
        "y_full": y_full, "illus_full": illus_full,
        "X_mixed": X_mixed_full.loc[idx], "X_a": X_a_full.loc[idx],
        "y": y_full.loc[idx], "illus": illus_full.loc[idx],
    }
