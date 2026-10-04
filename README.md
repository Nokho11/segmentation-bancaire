# Segmentation de clientèle bancaire par apprentissage non supervisé

Projet final du module « Apprentissage non supervisé » — Master BDA/IA (UN-CHK)
Auteure : Ndeye Sokhna Nokho

## Objectif

Découvrir des groupes homogènes de clients dans le jeu **Bank Marketing** (UCI,
`bank-additional-full.csv`, 41 188 contacts, 20 variables) pour mieux cibler les campagnes
de marketing téléphonique. La variable `y` (souscription) n'est jamais utilisée pour
construire les clusters : elle sert uniquement de vérité terrain pour l'évaluation.

## Résultat

Solution retenue : **CAH de Ward à 4 segments** (silhouette 0,21 ; Davies-Bouldin 1,73 ;
ARI bootstrap 0,93).

| Segment | Poids | Taux de souscription | Profil |
|---|---|---|---|
| S1 | 66 % | 5 %  | Jamais démarchés, contactés en période de taux élevés |
| S2 | 17 % | 17 % | Jamais démarchés, taux bas, joints sur mobile, plus jeunes |
| S3 | 11 % | 22 % | Déjà démarchés lors d'une campagne précédente |
| S4 | 6 %  | 41 % | Seniors, étudiants, diplômés, taux très bas |

En appelant S2, S3 et S4 (un tiers des clients), on obtient environ deux tiers des souscriptions.

La solution est choisie sur les **seuls indices internes** et la stabilité ; les indices externes
(calculés avec `y`) servent uniquement à vérifier ce choix a posteriori.

## Extensions (notebooks 06 à 08) — les quatre extensions du sujet

- **Sensibilité au contexte** : sans les trois indicateurs économiques, seul l'historique de contact
  forme un groupe net ; à période égale, l'historique et la pression d'appels discriminent plus que
  le profil socio-démographique.
- **Clients atypiques** (extension n° 1 du sujet) : bruit DBSCAN et Isolation Forest s'accordent sur
  un noyau de clients déjà démarchés, qui souscrivent à près de 50 %.
- **Sélection de variables** (extension n° 2 du sujet) : filtres variance et score laplacien, wrappers
  naïf et corrigé ; trois variables (`euribor3m`, `cons.conf.idx`, `previous`) suffisent à reproduire
  la segmentation (ARI 0,97).
- **Second jeu de données** (extension n° 4 du sujet) : Online Retail, segmentation RFM de 4 338 clients ;
  K-means à 4 segments, les Champions (16 % des clients) font 65 % du chiffre d'affaires.
- **Application Streamlit** (extension n° 3 du sujet) : affectation d'un nouveau client au segment
  le plus proche, distances aux 4 centres, alerte « client atypique » (Isolation Forest), affectation
  par lot d'un fichier CSV (format UCI accepté) avec export des résultats.

## Organisation du dépôt

```
data/bank-additional-full.csv      jeu de données (séparateur ;)
data/online_retail.csv.gz          second jeu (Online Retail, UCI), téléchargé par le notebook 08 s'il manque
src/config.py                      graine aléatoire, taille d'échantillon, chemins
src/preprocessing.py               prétraitement réutilisable -> prepare_all()
src/clustering_utils.py            Gower, K-medoids, indice de Dunn, détection du coude
notebooks/01_preparation.ipynb     Jour 1 : nettoyage, EDA, PCA / t-SNE / UMAP
notebooks/02_partitionnement.ipynb Jour 2 : distances, K-means, K-medoids, K-prototypes, GMM
notebooks/03_hierarchique_densite.ipynb  Jour 3 : CAH (4 liens), DBSCAN
notebooks/04_validation.ipynb      Jour 4 : indices internes / externes, stabilité, choix final
notebooks/05_interpretation.ipynb  Jour 5 : profilage des segments, recommandations
notebooks/06_extensions.ipynb      Extensions : sensibilité au contexte, clients atypiques
notebooks/07_selection_variables.ipynb  Extension : sélection de variables (variance, score laplacien, wrappers)
notebooks/08_online_retail_rfm.ipynb    Extension : second jeu de données, segmentation RFM
app/streamlit_app.py               application Streamlit : affecter un nouveau client à un segment
outputs/figures, tables, labels    figures (PNG), tableaux (CSV), partitions (CSV)
rapport/rapport.tex, rapport.pdf   rapport LaTeX (15 pages) ; compiler avec pdflatex depuis rapport/
soutenance/                        support de présentation (13 diapositives, notes orateur incluses)
```

## Installation et exécution

```bash
python -m venv .venv && source .venv/bin/activate     # ou un environnement conda
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Les notebooks s'exécutent **dans l'ordre** (01 → 08) : les notebooks 04 à 07 relisent les
partitions enregistrées par les notebooks précédents ; le notebook 08 est indépendant. Durée totale : 5 à 10 minutes.

Application (après l'exécution des notebooks, depuis la racine du dépôt) :

```bash
streamlit run app/streamlit_app.py
```

## Reproductibilité

- Graine aléatoire unique : `SEED = 42` dans `src/config.py`.
- Échantillon stratifié de 4 000 clients (`N_SAMPLE`), identique dans tous les notebooks.
- La matrice de Gower (40 s de calcul) et les projections t-SNE / UMAP sont mises en cache
  dans `outputs/` au premier lancement, puis relues.
- t-SNE et UMAP peuvent varier légèrement d'une machine à l'autre (visualisation
  uniquement : aucun résultat chiffré n'en dépend).

## Choix méthodologiques à connaître

- **Variables actives (15)** : `age`, `campaign`, `previous`, `cons.price.idx`, `cons.conf.idx`,
  `euribor3m`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`,
  `poutcome`, `contacted_before`.
- **Variables illustratives** (écartées du clustering, utilisées pour décrire les segments) :
  `duration` (connue seulement après l'appel), `emp.var.rate` et `nr.employed` (corrélées à
  plus de 0,90 avec `euribor3m`), `month`, `day_of_week`, `pdays` (recodée en `contacted_before`).
- **K-medoids** : `scikit-learn-extra` est utilisé s'il est installé avec `numpy<2` ; sinon,
  implémentation équivalente dans `src/clustering_utils.py`.
- **Indice de Dunn** : implémentation personnelle (`dunn_index`).
