"""Outils de clustering : distance de Gower, K-medoids, indice de Dunn, coude."""
from pathlib import Path

import gower
import numpy as np

from .config import SEED
from .preprocessing import ACTIVE_CAT


def gower_matrix(X_mixed, cache=None):
    """Matrice de dissimilarité de Gower (n x n) sur données mixtes.

    Le calcul prend ~40 s pour 4 000 clients : si `cache` (chemin .npy) est
    fourni, la matrice est enregistrée puis simplement relue les fois suivantes.
    """
    if cache is not None and Path(cache).exists():
        D = np.load(cache)
        if D.shape[0] == len(X_mixed):
            return D.astype(np.float64)
    X = X_mixed.copy()
    cat_flags = [c in ACTIVE_CAT for c in X.columns]
    for c, is_cat in zip(X.columns, cat_flags):
        X[c] = X[c].astype(object) if is_cat else X[c].astype(float)
    D = gower.gower_matrix(X, cat_features=cat_flags)
    if cache is not None:
        np.save(cache, D.astype(np.float32))
    return D.astype(np.float64)


# ---------------------------------------------------------------------------
# K-medoids sur matrice de distances pré-calculée
# ---------------------------------------------------------------------------
def _kmedoids_once(D, k, rng, max_iter=100):
    n = D.shape[0]
    # Initialisation type k-medoids++ : médoïdes éloignés les uns des autres
    medoids = [int(rng.integers(n))]
    for _ in range(k - 1):
        d2 = D[:, medoids].min(axis=1) ** 2
        medoids.append(int(rng.choice(n, p=d2 / d2.sum())))
    medoids = np.array(medoids)
    for _ in range(max_iter):
        labels = D[:, medoids].argmin(axis=1)          # étape d'affectation
        new = medoids.copy()
        for j in range(k):                             # étape de mise à jour
            members = np.where(labels == j)[0]
            if len(members):
                new[j] = members[D[np.ix_(members, members)].sum(axis=0).argmin()]
        if np.array_equal(np.sort(new), np.sort(medoids)):
            break
        medoids = new
    labels = D[:, medoids].argmin(axis=1)
    cost = D[np.arange(n), medoids[labels]].sum()
    return labels, medoids, cost


def kmedoids(D, k, n_init=5, seed=SEED):
    """K-medoids (algorithme alterné affectation / mise à jour des médoïdes).

    Par défaut on utilise `KMedoids` de scikit-learn-extra (demandé dans le sujet).
    Cette bibliothèque n'est plus maintenue et ne s'importe pas avec NumPy >= 2 :
    dans ce cas on bascule sur l'implémentation ci-dessus, qui suit le même principe.
    Retourne (labels, indices des médoïdes, coût total).
    """
    if int(np.__version__.split(".")[0]) < 2:
        try:
            from sklearn_extra.cluster import KMedoids
            m = KMedoids(n_clusters=k, metric="precomputed", method="alternate",
                         init="k-medoids++", random_state=seed).fit(D)
            return m.labels_, m.medoid_indices_, m.inertia_
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    runs = [_kmedoids_once(D, k, rng) for _ in range(n_init)]
    return min(runs, key=lambda r: r[2])


# ---------------------------------------------------------------------------
# Indice de Dunn (implémentation personnelle)
# ---------------------------------------------------------------------------
def dunn_index(D, labels):
    """Dunn = (plus petite distance entre deux clusters) / (plus grand diamètre).

    D : matrice de distances (n x n). Les points de bruit (label -1) sont ignorés.
    Plus l'indice est grand, plus les clusters sont compacts et bien séparés.
    """
    labels = np.asarray(labels)
    ks = [c for c in np.unique(labels) if c != -1]
    if len(ks) < 2:
        return np.nan
    groups = [np.where(labels == c)[0] for c in ks]
    diam = max(D[np.ix_(g, g)].max() for g in groups)
    sep = min(D[np.ix_(groups[i], groups[j])].min()
              for i in range(len(groups)) for j in range(i + 1, len(groups)))
    return sep / diam if diam > 0 else np.nan


def knee_point(values):
    """Position du « coude » d'une courbe : point le plus éloigné de la droite
    reliant le premier et le dernier point (méthode géométrique)."""
    v = np.asarray(values, dtype=float)
    x = np.arange(len(v))
    x_n = (x - x[0]) / (x[-1] - x[0])
    v_n = (v - v[0]) / (v[-1] - v[0])
    return int(np.abs(v_n - x_n).argmax())
