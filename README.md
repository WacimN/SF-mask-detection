# SF-mask-detection

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

