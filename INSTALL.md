# Installation & Démarrage

## 1. Installer Python (si absent)
Télécharger Python 3.11+ depuis https://www.python.org/downloads/
Cocher "Add python.exe to PATH" lors de l'installation.

## 2. Installer les dépendances

```bash
cd C:\Users\SLM\Downloads\FA2
pip install -r requirements.txt
```

## 3. Lancer l'optimisation complète (NSGA-II + figures)

```bash
python main.py
```

Options disponibles :
```
--fast          # démo rapide (30 générations au lieu de 100)
--pop 150       # taille de population (défaut: 120)
--gen 200       # nombre de générations (défaut: 100)
--fc-min 30     # résistance minimale en MPa (défaut: 25)
--cover 50      # enrobage en mm (défaut: 40)
--carb XC4      # classe carbonatation (défaut: XC3)
--cl XS3        # classe chlorures (défaut: XS2)
--sensitivity   # courbes de sensibilité uniquement
```

## 4. Dashboard interactif

```bash
python dashboard.py
```
Ouvrir http://127.0.0.1:8050 dans le navigateur.

## 5. Résultats
Tous les graphiques sont sauvegardés dans `./results/`
