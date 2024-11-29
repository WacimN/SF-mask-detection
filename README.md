# SF-mask-detection

# SF Mask Detection

Ce projet utilise **MobileNetV2** pour une tâche de détection de masques faciaux à partir d'images. Nous appliquons le **transfert d'apprentissage** en utilisant un modèle pré-entraîné sur ImageNet pour classifier les images en deux catégories : `compliant` (avec masque) et `non-compliant` (sans masque).

## Fonctionnalités

- **Transfert d'apprentissage** : Utilisation de MobileNetV2 pour tirer parti des connaissances acquises sur ImageNet.
- **Sous-échantillonnage du dataset** : Possibilité de travailler avec un sous-ensemble des données (par exemple, 33 % du dataset de base).
- **Enregistrement des résultats** :
  - **Fichiers JSON** : Les métriques d'entraînement (perte, précision, temps, etc.) sont stockées dans `training_metrics.json`.
  - **Graphiques** : Les courbes de perte (loss) et de précision (accuracy) sont sauvegardées au format PNG/HTML dans le dossier `training`.
  - **Modèle sauvegardé** : Les poids du modèle sont enregistrés dans un fichier `.pth` avec un nom horodaté.
  

### Dataset Utilisé : `SF-MASK-padded`

Le dataset utilisé pour l'entraînement et les tests est contenu dans le dossier `SF-MASK-padded`. Ce dataset a subi des modifications spécifiques pour améliorer les performances des modèles :

- **Structure** : Le dataset est organisé en deux sous-dossiers :
  - `compliant/` : Contient les images de personnes portant correctement un masque.
  - `non-compliant/` : Contient les images de personnes ne portant pas de masque ou le portant de manière incorrecte.
  
- **Préprocessing effectué** :
  - Toutes les images ont été préalablement redimensionnées et centrées pour faciliter leur traitement par les modèles.
  - Les marges autour des objets principaux ont été ajustées pour une meilleure homogénéité dans la taille et l'alignement.

### Utilisation dans le Notebook

Dans le notebook fourni avec ce projet, **seul le dataset `SF-MASK-padded` est utilisé** pour :
- **L'entraînement** : Les images sont transformées et utilisées pour entraîner le modèle MobileNetV2.
- **Les tests** : Les images du même dataset sont utilisées pour valider les performances du modèle.

---

## Configuration de l'environnement

### Prérequis
- Python 3.10+
- Conda pour la gestion des dépendances

### Installation de l'environnement

1. Clonez le dépôt GitHub :
   ```bash
   git clone git@github.com:WacimN/SF-mask-detection.git
   cd SF-mask-detection
   ```

2. Configurez l'environnement avec Conda :
   ```bash
   conda env create -f environment.yml
   conda activate sf-mask-detection
   ```

3. Assurez-vous que PyTorch est installé correctement avec le GPU activé (si disponible).

---

## Utilisation

### Entraînement du modèle

1. Exécutez l'entraînement avec un sous-ensemble des données :
   - Par défaut, 33 % des images du dataset sont utilisées pour l'entraînement et le test.
   - Les tailles des sous-ensembles sont affichées automatiquement.

   Exemple de configuration :
   ```python
   proportion = 0.33  # Pourcentage du dataset à utiliser
   ```

2. À la fin de l'entraînement, vous obtiendrez :
   - **Fichier des poids** : `mobilenetv2_trained_<date_heure>.pth`
   - **Métriques d'entraînement** dans `training_metrics.json`
   - **Graphiques** de perte et précision dans `training/`

### Test du modèle

Pour tester une image, exécutez le script de prédiction dans la partie ## Prediction by hand  du Notebook.

---

## Résultats sauvegardés

### Fichiers générés

Pour chaque session d'entraînement, les fichiers suivants sont créés avec un horodatage unique :
- **Graphiques** : Courbes de perte et précision (PNG/HTML)
- **Poids du modèle** : Fichier `.pth` contenant les paramètres entraînés
- **Fichier JSON** : Résumé des métriques d'entraînement avec détails (nombre d'epochs, temps total, etc.)


---

## Contribution
Les contributions sont les bienvenues ! Si vous trouvez des bugs ou souhaitez proposer des améliorations, ouvrez une issue ou soumettez une pull request.
