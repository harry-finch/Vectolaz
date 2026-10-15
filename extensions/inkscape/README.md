# Vectolaz pour Inkscape

Extension d'effet pour Inkscape : sélectionnez exactement une image PNG, lancez
**Extensions → Vectolaz → Vectolaz — Vectoriser les axes d'un PNG**.
Le résultat est inséré sous forme de groupe vectoriel directement au-dessus de l'image sélectionnée.
L'image source, son positionnement, son échelle et les autres éléments du document sont rigoureusement préservés.

---

## Fonctionnalités de la version 2.0

### 1. Sélecteur de préréglages (Presets)

Un menu déroulant en haut de l'onglet **Vectorisation** permet d'appliquer instantanément les réglages optimaux selon le type de dessin :

| Profil | Cas d'usage | Seuil | Tolérance géométrique | Barbes (`spur_tolerance`) | Coupures (`gap_tolerance`) | Poussières | Arcs globaux |
|---|---|---|---|---|---|---|---|
| **Personnalisé** *(défaut)* | Contrôle manuel total | Manuel GUI | Manuel GUI | Manuel GUI | Manuel GUI | Manuel GUI | Manuel GUI |
| **Dessin technique** | Plans, schémas, droites & polygones | `auto` | 2.5 px | 3.0 px | 0.0 px | 5 px | `False` |
| **Rosace** | Courbes géométriques, rosaces, guillochis | 160 (manuel) | 5.0 px | 3.0 px | 0.0 px | 10 px | `True` |
| **Tracé libre** | Lettrages, calligraphie, courbes Bézier | `auto` | 2.0 px | 2.5 px | 0.0 px | 5 px | `False` |
| **Motif fin** | Gravures, hachures denses, haute précision | 165 (manuel) | 1.0 px | 1.0 px | 0.0 px | 2 px | `False` |
| **Croquis scanné** | Dessins papier, traits crayonnés coupés | 175 (manuel) | 3.0 px | 4.5 px | 5.0 px | 25 px | `False` |

> [!NOTE]
> **Fonctionnement de l'interface Inkscape** : L'interface GTK des extensions Inkscape est générée à partir d'un fichier XML statique (`.inx`). Lorsqu'un profil est sélectionné dans le menu déroulant, les curseurs numériques en dessous ne bougent pas à l'écran : les valeurs optimales du profil sont **directement et automatiquement appliquées lors du clic sur « Appliquer »**. Pour ajuster vous-même chaque curseur manuellement, laissez le profil sur **« Personnalisé »**.

### 2. Organisation en 3 étapes logiques

Sous le menu de préréglages, les paramètres manuels sont regroupés de manière claire :
1. **1. Extraction du trait (Image)** : Seuil automatique (Otsu), Seuil manuel (0–255), Suppression des poussières / surface minimale en pixels.
2. **2. Nettoyage & Continuité du squelette** :
   - **Élagage des micro-barbes (`spur_tolerance`)** : supprime les petites épines parasites causées par les aspérités de contour lors de la squelettisation sur traits épais.
   - **Raccordement des coupures (`gap_tolerance`)** : réunit les extrémités proches sous condition de tangence et d'absence d'obstacle.
3. **3. Reconnaissance des formes & Rendu SVG** : Tolérance géométrique (px source), Regroupement des cercles et grands arcs continus (`global_arcs`), et Épaisseur vectorielle du contour SVG.

### 3. Étiquetage dynamique des groupes

Le groupe inséré dans le panneau **Calques et objets** d'Inkscape est nommé dynamiquement avec le profil et les tolérances appliqués (ex: `Vectolaz — Rosace (tol=5.0px, barbes=3.0px)`).

### 4. Onglet Aide enrichi

L'onglet **Aide** intègre :
- Le descriptif de chaque profil avec ses valeurs de référence.
- Un **Mémo de dépannage rapide** :
  - *Traits clairs manquants* → Décocher Seuil auto et monter le seuil manuel (170–200).
  - *Droites ou arcs morcelés* → Monter la tolérance géométrique (3–5 px) et l'élagage des barbes (3–4 px).
  - *Angles vifs arrondis ou petits détails perdus* → Baisser la tolérance géométrique (1.0–1.5 px) et décocher le regroupement global.
  - *Petites épines parasites sur traits épais* → Régler l'élagage des micro-barbes à la moitié de l'épaisseur du trait (ex : 3 à 5 px).

---

## Installation

Le moteur doit être installé dans son propre environnement Python 3.11+.

### 1. Préparation du moteur (venv)

Depuis la racine du projet, sur macOS ou Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m raster2vector --version
```

Sur Windows PowerShell :

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### 2. Copie de l'extension dans Inkscape

Dans Inkscape, ouvrir **Préférences → Système** et relever le dossier **Extensions utilisateur**.
Copier dans un sous-dossier `vectolaz` les fichiers `vectolaz.inx`, `vectolaz.py`, `image_source.py`, `placement.py` et `README.md`.
L'archive `dist/vectolaz-inkscape.zip` contient directement ces fichiers prêts à l'emploi.

Sur macOS / Linux :

```bash
USER_EXT="$HOME/Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions"
mkdir -p "$USER_EXT/vectolaz"
cp extensions/inkscape/vectolaz.inx extensions/inkscape/vectolaz.py \
   extensions/inkscape/image_source.py extensions/inkscape/placement.py \
   extensions/inkscape/README.md "$USER_EXT/vectolaz/"
```

Redémarrer Inkscape pour charger l'extension.

### 3. Configuration du chemin Python dans l'onglet Installation

Dans l'onglet **Installation** de Vectolaz dans Inkscape, renseigner le chemin absolu de l'exécutable Python du venv :
- macOS / Linux : `/Users/mugen/Documents/01_Projets/Vectolaz/.venv/bin/python`
- Windows : `C:\chemin\vers\Vectolaz\.venv\Scripts\python.exe`

Ce chemin est mémorisé par Inkscape.

---

## Essai rapide : La Rosace

1. Importer `samples/rosace.png` dans Inkscape (embarquée ou liée).
2. Sélectionner uniquement l'image.
3. Ouvrir **Extensions → Vectolaz → Vectolaz — Vectoriser les axes d'un PNG**.
4. Dans **Profil de motif (Préréglage)**, choisir simplement **Rosace**.
5. Cliquer sur **Appliquer**.

**Résultat attendu** : le groupe inséré `Vectolaz — Rosace (tol=5.0px, barbes=3.0px)` contient **1 cercle complet et 6 grands arcs continus** parfaitement superposés aux axes de la rosace.

---

## Formats et placement supportés

- **Sources PNG** : PNG embarqué en base64 (`data:image/png;base64,...`), chemin absolu, chemin relatif (résolu depuis le document SVG enregistré) et URI locale `file:`.
- **Placement & Transformations** : le groupe vectoriel hérite de la matrice `transform` de l'image SVG ainsi que des calculs de positionnement `x`, `y`, `width`, `height` et `preserveAspectRatio` (`meet` ou `none`).
- **Cas refusés proprement** (sans altération du document) : masques, filtres, clips, balises `<use>` (clones), viewports imbriqués ou images non PNG.

---

## Tests et validation

```bash
# Tests autonomes du moteur et des presets (sans dépendre d'inkex installé dans le venv)
.venv/bin/pytest -W error

# Tests d'intégration avec l'API inkex réelle d'Inkscape
INKSCAPE_EXTENSIONS_DIR=/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions \
  .venv/bin/pytest tests/inkscape/ -W error
```

Validation actuelle : **411 tests réussis**, 0 avertissement (`-W error`), compatibilité complète avec Inkscape 1.4+.
