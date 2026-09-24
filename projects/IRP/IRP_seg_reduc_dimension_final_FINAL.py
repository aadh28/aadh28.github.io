# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:40:39 2026

@author: LLA MAZ
"""

import numpy as np
import nibabel as nib
# from nilearn import plotting
from nilearn.image import resample_to_img
import pandas as pd
import seaborn as sns
import umap
from sklearn. manifold import TSNE
from scipy.stats import probplot
from scipy.stats import ks_2samp, mannwhitneyu, shapiro, ttest_ind 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
# train_test_split : séparer données en train/test
# cross_val_score : validation croisée pour estimer la performance sur le train sans biais
from sklearn.preprocessing import StandardScaler
# StandardScaler : centrage (moyenne=0) et réduction (écart-type=1) des features
# Nécessaire car le SVM est sensible à l'échelle des variables (calcul de distances)
from sklearn.decomposition import PCA
# PCA : réduction de dimensionnalité pour projeter en 2D et visualiser les frontières
from sklearn.metrics import accuracy_score, classification_report
# accuracy_score : proportion de prédictions correctes
# classification_report : précision, rappel, f1-score par classe
import matplotlib.pyplot as plt # Matplotlib pour tracer les graphiques de frontières de décision

plt.close('all')


# ======================= Extraction des régions et calcul des volumes ======================= #


# -------------- Extraction des volumes -------------- #


# Chargement atlas
atlas = nib.load(r"G:/S2 cle/IRP/atlas_image.nii")

# Chargement image cible (à remplacer par le vrai nom)
target_img = nib.load(r"G:/S2 cle/IRP/atlas_image_target.nii")

# Resample atlas -> image cible
atlas_resampled = resample_to_img(
    source_img=atlas,
    target_img=target_img,
    interpolation='nearest'
)

# Données de l'atlas resamplé
data = atlas_resampled.get_fdata()

# Lecture du fichier labels
labels = pd.read_excel(r"G:/S2 cle/IRP/atlas_labels.xlsx")

# Labels présents hors fond
labels_presents = np.unique(data)
labels_presents = labels_presents[labels_presents != 0]

# Taille voxel et volume voxel
dx, dy, dz = atlas_resampled.header.get_zooms()[:3]
voxel_volume = dx*dy*dz

print("Résolution voxel :", (dx, dy, dz))
print("Volume voxel :", voxel_volume, "mm³")

# Création des masques
masks = {}
for i in labels_presents:
    masks[i] = (data == i).astype(np.uint8)

# Création objets NIfTI pour visualisation
masks_view = {}
for i in labels_presents:
    masks_view[i] = nib.Nifti1Image(
        masks[i],
        affine=atlas_resampled.affine,
        header=atlas_resampled.header
    )
 
mat = list(masks.values())[18]   # 11e matrice
plt.figure()
plt.imshow(mat[50,:,:],cmap='gray')
plt.title("Région ID : 17  -  50e coupe sur axe x")
plt.show()

plt.figure()
plt.imshow(data[50,:,:])
plt.title("50e coupe sur axe x")
plt.show()

# Exemple visualisation
"""
region = 27
view = plotting.view_img(masks_view[region])
view.open_in_browser()
"""

# -------------- Calcul des volumes -------------- #

results = []

for i in labels_presents:                   # Parcourt chaque région
    n_voxels = np.sum(masks[i])             # Récupère le nombre de voxels : fais la somme de tous les pixels à 1 du masque
    volume_mm3 = n_voxels * voxel_volume    # Calcul le volume

    region_info = labels[labels["Region ID"] == i] # Récupère le nom de la région
    if len(region_info) > 0:
        region_name = region_info.iloc[0]["Region Name"] # On récupère le nom si c'est trouvée
    else:
        region_name = f"ROI_{int(i)}" # Sinon par défaut on met ROI_i

    # Stockage des résultat
    results.append({
        "Region ID": int(i),
        "Region Name": region_name,
        "Nb voxels": int(n_voxels),
        "Volume (mm3)": volume_mm3,
        "Volume (cm3)": volume_mm3 / 1000
    })

df_volumes = pd.DataFrame(results) # Transforme la liste results en tableau
print(df_volumes.head())

total_volume = df_volumes["Volume (mm3)"].sum() # Fais la somme de tous les volumes
df_volumes["Volume (%)"] = (df_volumes["Volume (mm3)"] / total_volume) * 100 # On convertit les volumes en pourcentages
print(df_volumes["Volume (%)"].sum()) # ça devrait faire environ 100% normalement

df_volumes.to_excel(r"G:/S2 cle/IRP/volumes_atlas_resampled.xlsx", index=False)



# ======================= ACP / t-SNE / UMAP ======================= #



# Lecture du fichier data_volumes
data_volumes = pd.read_excel(r"G:/S2 cle/IRP/data_volumes.xlsx")

"""
# 4 colonnes à analyser
zones_interets = [
    "Thalamus total volume %",
    "Hippocampus total volume %",
    "Putamen total volume %",
    "White Matter (WM) volume %"
]
X = data_volumes[zones_interets]
"""

#zones_interets = data_volumes.iloc[:, 1:88]

# Extraire les données
X = data_volumes.iloc[:,1:88]
y = data_volumes["label"]

# Standardisation
X_scaled = StandardScaler().fit_transform(X)


# -------------- ACP -------------- #


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance expliquée par PC1 :", pca.explained_variance_ratio_[0])
print("Variance expliquée par PC2 :", pca.explained_variance_ratio_[1])
print("Variance expliquée totale :", pca.explained_variance_ratio_.sum())


# -------------- Paramètres communs d'affichage -------------- #

noms_classes = {0: "Contrôles", 1: "Patients"}
couleurs = {0: "blue", 1: "red"}


# -------------- Affichage ACP -------------- #


plt.figure(figsize=(8,6))

for classe in np.unique(y):
    idx = y == classe
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        color=couleurs[classe],
        label=noms_classes[classe],
        alpha=0.8
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("ACP sur 87 variables volumétriques")
plt.legend()
plt.grid(True)
plt.show()


# -------------- t-SNE -------------- #

voisin = [5,10,20,50,75]# perplexity doit etre plus grand que 1 et moins grand que 100

for i in voisin:
    # tester sur plusieurs paramètres
    tsne = TSNE(
        n_components=2,
        perplexity=i,
        learning_rate=200,
        init='pca',
        random_state=50
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    
# -------------- UMAP -------------- #

    
    # tester sur plusieurs paramètres
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=i,
        min_dist=0.3,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_scaled)




# -------------- Affichage t-SNE -------------- #

 
    plt.figure(figsize=(8,6))
    
    for classe in np.unique(y):
        idx = y == classe
        plt.scatter(
            X_tsne[idx, 0],
            X_tsne[idx, 1],
            color=couleurs[classe],
            label=noms_classes[classe],
            alpha=0.8
        )
    
    plt.title(f"t-SNE sur 87 variables volumétriques(perplexity = {i})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------- Affichage UMAP -------------- #

    plt.figure(figsize=(8,6))
    
    for classe in np.unique(y):
        idx = y == classe
        plt.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            color=couleurs[classe],
            label=noms_classes[classe],
            alpha=0.8
        )
    
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Projection UMAP des volumes cérébraux (n_neighbors = {i})")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()


# ======================= Boxplots ======================= #


'''
regions d'interet :
    AMS: Putamen/ Cerebellum
    SAIN: Hippocampe/Thalamus 
'''
region_i = ["Putamen","Cerebellum", "Hippocampus", "Thalamus"]
reg_int_atlas = ["Both Putamen","Both Cerebellum", "Both Hippocampus", "Both Thalamus"]
reg_int_assembly = ["Putamen total volume %", "Cerebellum total volume %", "Hippocampus total volume %" ,"Thalamus total volume %"]


assembly = pd.read_excel(r"G:/S2 cle/IRP/data_volumes.xlsx")
atlas = pd.read_excel(r"G:/S2 cle/IRP/volumes_atlas_resampled_both.xlsx")

# Région putamen de assembly = "Left putamen_total dans assembly" = ligne 9 = (1 *92)
# Région putamen de atlas = "Left Putamen" ligne 11 vol = 1 data dans colonne 3

controle = assembly[assembly["label"] == 0]
patient = assembly[assembly["label"] == 1]


'''
sns.boxplot(data=controle, y='Putamen total volume %')
sns.boxplot(data=patient, y='Putamen total volume %')
ca les superposent 
'''

assembly["groupes"] = assembly["label"].map({0: "controle", 1: "patient"}) #map donne les "sous-titres" en y des groupes
    
    
for i in range(len(reg_int_atlas)):
    sns.boxplot(data=assembly, x="groupes", y=reg_int_assembly[i], palette={"controle": "skyblue", "patient": "salmon"})
    sns.stripplot(data=assembly, x="groupes", y=reg_int_assembly[i], palette={"controle": "blue", "patient": "red"})
    #sns.boxplot(data=assembly, x="label", y="Putamen total volume %") # Fonctionne aussi mais moins beau
    
    
    #ajout cst atlas
    #volume_putamen = atlas.loc[atlas.iloc[:, 1] == "Putamen", atlas.columns[5]]
    #putamen_atlas = atlas.iloc[10, 2]
    volume_putamen = atlas.loc[atlas["Region Name"] == reg_int_atlas[i], "Volume (%)"].iloc[0]
    cst = float(volume_putamen)  # sinon marche pas pq ?
    
    
    plt.axhline(y=cst, color='lime', linestyle='--', linewidth=2, label=f'Seuil = {cst:.4f}')
    
    plt.title(f"{region_i[i]}")
    plt.ylabel("Volume (%)")
    plt.legend(loc='lower right') 
    plt.show()
    
    
    # ======================= Tests statistiques ======================= #


# On prend comme région : putamen

putamen_vol = data_volumes['Putamen total volume %']
lbl = data_volumes['label']
putamen_vol_sain = []
putamen_vol_ams = []
putamen_sain = []
putamen_ams = []

for i in range(len(lbl)):
    if lbl[i] == 0:
        putamen_vol_sain.append(putamen_vol[i])
    elif lbl[i] == 1:
        putamen_vol_ams.append(putamen_vol[i])
        
putamen_sain = shapiro(putamen_vol_sain)
putamen_ams = shapiro(putamen_vol_ams)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Puramen — Test de normalité par groupe')

# Histogrammes
axes[0, 0].hist(putamen_vol_sain, bins=10, edgecolor='black', color='steelblue')
axes[0, 0].set_title('Sains — Histogramme')
axes[0, 0].set_xlabel('Volume')

axes[0, 1].hist(putamen_vol_ams, bins=10, edgecolor='black', color='salmon')
axes[0, 1].set_title('Malades — Histogramme')
axes[0, 1].set_xlabel('Volume')


# Q-Q plots
probplot(putamen_vol_sain, plot=axes[1, 0])
axes[1, 0].set_title(f'Sains — Q-Q plot  (p = {putamen_sain.pvalue:.4f})')

probplot(putamen_vol_ams, plot=axes[1, 1])
axes[1, 1].set_title(f'Malades — Q-Q plot  (p = {putamen_ams.pvalue:.4f})')

plt.tight_layout()
plt.show()

pval_sain = putamen_sain.pvalue
pval_ams = putamen_ams.pvalue



# ------------------Pour toutes les régions du cerveau 

vol_sain = data_volumes[data_volumes["label"] == 0]
vol_ams = data_volumes[data_volumes["label"] == 1]

# Colonnes à analyser (on enlève Subject et label)
colonnes = data_volumes.columns.drop(["Subject", "label"])

res = []

for col in colonnes:
    # On isole la colonne 'col' pour les tests de normalité
    # .dropna() est important pour éviter les erreurs si des cases sont vides
    data_sain = vol_sain[col].dropna()
    data_ams = vol_ams[col].dropna()  
    
    # Test de normalité
    shapiro_sain = shapiro(data_sain)
    shapiro_ams = shapiro(data_ams)
    
    # Stockage résultats
    res.append({
        "Region": col,
        "p_sain": shapiro_sain.pvalue,
        "p_ams": shapiro_ams.pvalue
    })
    '''
    # ====== PLOTS ======
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'{col} — Normalité')
    
    # Histogrammes
    axes[0, 0].hist(vol_sain, bins=10, edgecolor='black')
    axes[0, 0].set_title('Sains')
    
    axes[0, 1].hist(vol_ams, bins=10, edgecolor='black')
    axes[0, 1].set_title('Patients')
    
    # Q-Q plots
    probplot(vol_sain, plot=axes[1, 0])
    axes[1, 0].set_title(f'Sains (p={shapiro_sain.pvalue:.3f})')
    
    probplot(vol_ams, plot=axes[1, 1])
    axes[1, 1].set_title(f'Patients (p={shapiro_ams.pvalue:.3f})')
    
    plt.tight_layout()
    plt.show()
    '''
    
    
"""
p< 0.05 rejette pas l'hypothèse de normalité
p> 0.05 rejettes la normalité
"""
    
resultat = pd.DataFrame(res)


normales = resultat[
    (resultat["p_sain"] > 0.05) & 
    (resultat["p_ams"] > 0.05)
]

non_normales = resultat[
    (resultat["p_sain"] < 0.05) | 
    (resultat["p_ams"] < 0.05)
]

print(f"\nRégions où les DEUX groupes sont normaux : {len(normales)}") # données compatibles avec loi normale mais peux pas confirmer qu'elles sont parfaitement gaussienne
print(f"Régions avec au moins un groupe non normal : {len(non_normales)}")


#====================================================================================================================

#-----------------------T-test sur les regions normales

ttest_sig     = []   # différence significative
ttest_non_sig = []   # pas de différence significative

#vol_sain = data_volumes[data_volumes["label"] == 0]
#vol_ams = data_volumes[data_volumes["label"] == 1][col]

alpha =0.05

for region in normales["Region"]:
    ctrl = vol_sain[region].dropna().values
    pat  = vol_ams[region].dropna().values

    stat , p_val = ttest_ind(ctrl, pat) #, equal_var=False)   

    if p_val < alpha:
        ttest_sig.append((region, p_val))
    else:
        ttest_non_sig.append((region, p_val))
        

print("\n--- t-test ---")
print(f"  Significatives     : {len(ttest_sig)}")
print(f"  Non significatives : {len(ttest_non_sig)}")


# -------------- Mann-Whitney + KS sur les régions non normales -------------- #

# Listes résultats Mann-Whitney
mw_sig     = []
mw_non_sig = []

# Listes résultats Kolmogorov-Smirnov
ks_sig     = []
ks_non_sig = []

for region in non_normales["Region"]:
    ctrl = vol_sain[region].dropna().values
    pat  = vol_ams[region].dropna().values

    # Mann-Whitney U
    stat_mw , p_mw = mannwhitneyu(ctrl, pat)
    if p_mw < alpha:
        mw_sig.append((region, p_mw))
    else:
        mw_non_sig.append((region, p_mw))

    # Kolmogorov-Smirnov
    stat_ks , p_ks = ks_2samp(ctrl, pat)
    if p_ks < alpha:
        ks_sig.append((region, p_ks))
    else:
        ks_non_sig.append((region, p_ks))

print("\n--- Mann-Whitney U ---")
print(f"  Significatives     : {len(mw_sig)}")
print(f"  Non significatives : {len(mw_non_sig)}")

print("\n--- Kolmogorov-Smirnov ---")
print(f"  Significatives     : {len(ks_sig)}")
print(f"  Non significatives : {len(ks_non_sig)}")


# -------------- Récapitulatif final -------------- #



print("\n========== RÉGIONS SIGNIFICATIVES (p < 0.05) ==========")

print("\n[t-test — distributions normales]")
for region, p in ttest_sig:
    print(f"  {region:<45}  p = {p}")

print("\n[Mann-Whitney — distributions non normales]")
for region, p in mw_sig:
    print(f"  {region:<45}  p = {p}")

print("\n[Kolmogorov-Smirnov — distributions non normales]")
for region, p in ks_sig:
    print(f"  {region:<45}  p = {p}")


# -------------- Export Excel -------------- #

rows = []
for region, p in ttest_sig:
    rows.append({"Region": region, "Test": "t-test (Welch)", "p_value": p, "Significatif": True})
for region, p in ttest_non_sig:
    rows.append({"Region": region, "Test": "t-test (Welch)", "p_value": p, "Significatif": False})
for region, p in mw_sig:
    rows.append({"Region": region, "Test": "Mann-Whitney U", "p_value": p, "Significatif": True})
for region, p in mw_non_sig:
    rows.append({"Region": region, "Test": "Mann-Whitney U", "p_value": p, "Significatif": False})
for region, p in ks_sig:
    rows.append({"Region": region, "Test": "Kolmogorov-Smirnov", "p_value": p, "Significatif": True})
for region, p in ks_non_sig:
    rows.append({"Region": region, "Test": "Kolmogorov-Smirnov", "p_value": p, "Significatif": False})

df_resultats = pd.DataFrame(rows).sort_values(["Test", "p_value"]).reset_index(drop=True)
df_resultats.to_excel(
    r"G:/S2 cle/IRP/resultats_tests_statistiques.xlsx",
    index=False
)
print("\nRésultats exportés dans : resultats_tests_statistiques.xlsx")


# -------------- SVM -------------- #


# ligne = sujet, colonne = volume régional ou le label
dt_atlas_resample = pd.read_excel(r"G:/S2 cle/IRP/data_volumes.xlsx")

# X = matrice des features (variables explicatives)
# .values convertit le DataFrame en array NumPy, format attendu par scikit-learn
X = dt_atlas_resample[['Putamen total volume %', 'Cerebellar GM volume %', 'Pallidum total volume %']].values

y = dt_atlas_resample['label'].values
# y = vecteur des labels (variable à prédire)
# 0 = sujet contrôle, 1 = patient

kernels = ['linear', 'poly', 'rbf']
# linear : hyperplan linéaire, interprétable, adapté si les classes sont linéairement séparables
# poly : frontière polynomiale, capture des interactions non-linéaires modérées
# rbf : frontière radiale gaussienne, capture des non-linéarités complexes

# Boucle sur deux stratégies de split pour comparer leur impact sur la performance
# 70/30 = plus de données test => estimation plus fiable de la généralisation
# 80/20 = plus de données train => meilleur apprentissage si échantillon petit
for split_name, test_size in [("70/30", 0.3), ("80/20", 0.2)]:

    # random_state=50 : fixe le tirage aléatoire pour la reproductibilité
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=50, stratify=y  # stratify=y : conserve la proportion contrôle/patient
    )

    # Instanciation d'un NOUVEAU scaler à chaque split
    # Si on réutilisait le même, les statistiques (moyenne, std) du premier split
    # contamineraient le second → biais d'évaluation
    scaler = StandardScaler()

    # fit_transform sur le train : calcule moyenne et std du train, puis transforme
    # On apprend les paramètres UNIQUEMENT sur le train (pas le test)
    # Raison : en conditions réelles, on ne connaît pas les données test à l'avance
    X_train_scaled = scaler.fit_transform(X_train)

    # transform sur le test : applique la même moyenne et std du train
    # Pas de fit ici, sinon fuite d'information (data leakage)
    X_test_scaled = scaler.transform(X_test)

    # Affichage 
    print(f"\n{'='*50}")
    print(f"Split {split_name} — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"{'='*50}")

    # Boucle sur chaque type de kernel SVM
    for k in kernels:

        # Création du modèle SVM avec le kernel courant
        # random_state pour reproductibilité (certains kernels ont une composante stochastique)
        svm = SVC(kernel=k, random_state=50)

        # Entraînement du SVM sur les données train standardisées
        # Le modèle apprend la frontière de décision séparant contrôles et patients
        svm.fit(X_train_scaled, y_train)

        # Accuracy sur le train : mesure l'ajustement du modèle aux données d'entraînement
        # Si très élevée et test faible → overfitting (le modèle mémorise au lieu de généraliser)
        acc_train = accuracy_score(y_train, svm.predict(X_train_scaled))

        # Accuracy sur le test : mesure la capacité de généralisation à des données jamais vues
        # C'est la métrique qui compte réellement
        acc_test = accuracy_score(y_test, svm.predict(X_test_scaled))

        # Validation croisée 5-fold sur le train uniquement
        # Découpe le train en 5 plis, entraîne sur 4, évalue sur 1, répète 5 fois
        # Donne une estimation plus robuste que le seul acc_train
        # Détecte l'overfitting : si cv << acc_train, le modèle ne généralise pas
        cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)

        # Affichage des résultats pour ce kernel
        print(f"\n  Kernel: {k}")
        print(f"  Acc train: {acc_train:.3f}")
        print(f"  Acc test:  {acc_test:.3f}")
        # Moyenne ± écart-type du CV : l'écart-type indique la stabilité du modèle
        # Un grand écart-type signifie que la performance varie beaucoup selon le pli
        print(f"  CV 5-fold: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Classification report : détaille précision, rappel, f1 par classe
        # Précision = parmi les prédits positifs, combien sont vrais positifs
        # Rappel = parmi les vrais positifs, combien sont détectés
        # F1 = moyenne harmonique des deux, utile si classes déséquilibrées
        print(classification_report(y_test, svm.predict(X_test_scaled)))

    # ============================================================
    # PCA + VISUALISATION — dans la boucle pour chaque split
    # ============================================================

    # Réduction à 2 composantes principales pour pouvoir tracer en 2D
    # On perd de l'information mais on gagne en interprétabilité visuelle
    # La PCA est entraînée sur le train scalé uniquement (même logique que le scaler)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Affiche la part de variance expliquée par chaque composante
    # Si la somme est faible (ex: < 0.7), la projection 2D est une approximation grossière
    # Les frontières visibles ne reflètent pas exactement celles de l'espace 3D original
    print(f"\nVariance expliquée PCA ({split_name}): {pca.explained_variance_ratio_}")

    # Création de 3 sous-graphiques côte à côte, un par kernel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Titre global indiquant quel split est visualisé
    fig.suptitle(f"Frontières de décision — Split {split_name}", fontsize=14)

    # Boucle sur chaque kernel pour tracer sa frontière de décision en 2D
    for ax, k in zip(axes, kernels):

        # On ré-entraîne un SVM sur les 2 composantes PCA (pas sur les 3 features originales)
        # Ce SVM 2D sert UNIQUEMENT à la visualisation, pas à l'évaluation
        # Les performances reportées plus haut viennent du SVM entraîné sur les 3 features
        svm_2d = SVC(kernel=k, random_state=50)
        svm_2d.fit(X_train_pca, y_train)

        # Calcul des bornes de la grille avec une marge de 1 unité
        # Pour que les points ne soient pas collés aux bords du graphique
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

        # Création d'une grille de 300x300 points couvrant l'espace 2D
        # Chaque point de la grille sera classifié par le SVM pour colorier les zones
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                              np.linspace(y_min, y_max, 300))

        # Prédiction du SVM 2D sur chaque point de la grille
        # np.c_ concatène les coordonnées x et y en une matrice (n_points, 2)
        # ravel() aplatit la grille 2D en vecteur 1D
        # reshape remet le résultat en forme 2D pour le contour plot
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Colorie les zones de décision : bleu pour classe 0, rouge pour classe 1
        # alpha=0.3 : transparence pour que les points restent visibles
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

        # Affichage des points d'entraînement contrôles (classe 0) en bleu
        ax.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
                   c='blue', label='Contrôle', edgecolors='k', s=40)

        # Affichage des points d'entraînement patients (classe 1) en rouge
        ax.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
                   c='red', label='Patient', edgecolors='k', s=40)

        # Titre du sous-graphique = nom du kernel
        ax.set_title(f'Kernel: {k}')
        # Axes = composantes principales, pas les variables originales
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        # Légende pour distinguer les deux classes
        ax.legend()

    # Ajuste l'espacement entre sous-graphiques pour éviter les chevauchements
    plt.tight_layout()
    # Sauvegarde avec un nom de fichier unique par split
    # replace('/', '_') car '/' est interdit dans les noms de fichiers
    plt.savefig(f"svm_decision_boundaries_{split_name.replace('/', '_')}.png", dpi=150)
    # Affichage à l'écran
    plt.show()
