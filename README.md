# Femur Regression Project — Guide d'Entraînement

Ce guide centralise les instructions nécessaires pour se connecter au cluster, configurer, lancer et analyser les entraînements de modèles PointNet++.

### 1. Se connecter à Mesonet

```bash
ssh bapttron@juliet.mesonet.fr
```
Puis rentrer le mot de passe associé à la clé ssh. Une fois connecté, déplacez-vous dans le répertoire de travail du projet :

```bash
cd Femur-Regression-Project
```
<hr>

### 2. Lancer des entrainements

Le projet utilise une architecture unifiée par fichier de configuration. Chaque paramètre de l'entraînement (architecture des couches PointNet++, choix de l'optimiseur, hyperparamètres, etc.) est modulable de manière centralisée dans **config.yaml**.

#### 2.1 Configuration du modèle

##### 1. La Sélection des Points (`sampling_method`)

* **Aléatoire (`"random"`)** : Sélection purement au hasard à chaque époque. 
* **Farthest Point Sampling (`"fps"`)** : Sélection des points les plus éloignés les uns des autres.

##### 2. L'Augmentation de Données (`augment`)

* **`false`** : Le modèle n'apprend que sur les fémurs d'origine.
* **`true`** : Le modèle intègre également les variantes géométriques modifiées (`_aug`). Dans le dataset, il existe **3 versions augmentées** pour chaque fémur de base.

##### 3. Le Nombre de Points (`num_point`)

* **Résolution personnalisable** : Ajuste le niveau de détail (ex: 512, 1024, 2048, 4096).
* **Cas particulier** : Si `num_point = 6000`, la totalité des points disponibles du fémur est sélectionnée.

#### 2.2 Lancer un Job :

Pour soumettre un entraînement unique sur le cluster MesoNet, utilisez le script **submit.py.** Le drapeau -c spécifie la configuration de base, et le drapeau -g (ou --go) valide la soumission effective à Slurm.

```bash
python submit.py -c config.yaml -g
```

Si l'on veut modifier les paramètres du modèle sans ouvrir manuellement le fichier YAML, il faut mettre ***-p arg valeur*** pour surcharger temporairement le config.yaml. Le script se chargera de créer un sous-dossier de Job isolé, d'y injecter la configuration modifiée et de configurer dynamiquement le dossier des logs.

Par exemple, pour forcer un échantillonnage par Farthest Point Sampling (FPS) et activer l'entraînement sur les données augmentées :


```bash
python submit.py -c config.yaml -p sampling_method fps -p augment true -g
```

#### 2.3 Lancer plusieurs Jobs :

L'idée est de pouvoir modifier le *learning_rate*, le *batch_size* et le *nombre de points* (num_point) de la config automatiquement plutôt que de lancer les jobs "à la main" de manière redondante. Cela permet d'effectuer une recherche par grille (Grid Search) efficace.

*Note : Pour configurer la méthode de sampling ou l'ajout des données augmentées sur l'ensemble de la campagne, il faut le faire manuellement en modifiant le fichier racine config.yaml avant de lancer le lot.*

Modifier les listes de paramètres à croiser directement dans le script **training_batch.sh** :
```bash
# Listes des paramètres à tester
LRS=(0.001 0.0001)
BSS=(64 128 256)
NPTS=(2048 4096)
```

Avant la première utilisation, attribuez les permissions d'exécution au script Shell (à faire qu'une seule fois normalement) :
```bash 
chmod +x training_batch.sh
```

Lancer le batch d'entrainements, Le script va boucler sur toutes les combinaisons possibles, appeler submit.py à chaque itération et envoyer les différents Jobs en file d'attente Slurm :
```bash 
./training_batch.sh
```

<hr>

### 3. Résultats des entrainements

Chaque entraînement génère un dossier unique et indépendant nommé d'après son identifiant de processus dans Jobs/ID/ contenant :

**config.yaml** : la configuration exacte utilisés spécifiquement pour cet entraînement.

**best_model.pth** : Les poids du réseau de neurones ayant obtenu la plus faible erreur (MAE) sur l'ensemble de validation. C'est ce fichier qui est rechargé pour l'évaluation finale.

**all_test_errors.csv** : Un rapport d'évaluation complet et transparent sur l'ensemble de test.

Les fichiers graphiques (.png) des **courbes d'apprentissage** : Génération automatique des tracés de la perte MSE et de la précision MAE au fil des époques pour détecter visuellement un éventuel surapprentissage (overfitting).

### 4. Voir les meillerus résultats

Pour comparer instantanément les performances de toutes les expériences stockées dans le dossier Jobs/, utilisez le script de leaderboard. Il scanne les rapports textuels, extrait les hyperparamètres et affiche un tableau comparatif synthétique directement dans le terminal.

```bash 
python leaderboard.py
```