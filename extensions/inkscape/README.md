# Vectolaz pour Inkscape

Première extension d'effet : sélectionnez exactement un PNG, lancez
**Extensions → Vectolaz → Vectolaz — Vectoriser les axes d'un PNG**.
Le résultat est ajouté comme groupe juste au-dessus de l'image sélectionnée.
L'image, son placement et les autres objets sont conservés.
Chaque réglage possède une infobulle : survolez son contrôle pour lire son
explication. L'onglet **Aide** regroupe les explications et les exemples
de réglages pour le motif complexe et la rosace.

## Installation

Le moteur doit être installé dans son propre environnement Python 3.11+.
Depuis la racine du projet, si l'environnement existe déjà :

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m raster2vector --version
```

Pour une première installation :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Dans Inkscape, ouvrir **Préférences → Système** et relever le dossier
**Extensions utilisateur**. Copier dans un sous-dossier `vectolaz` les quatre
fichiers `vectolaz.inx`, `vectolaz.py`, `image_source.py` et `placement.py`.
Le paquet `dist/vectolaz-inkscape.zip` contient ces fichiers et ce guide.
Redémarrer Inkscape pour charger le menu. Ne pas copier les fichiers dans
l'application Inkscape elle-même.

Commande macOS/Linux, depuis le projet, après remplacement du chemin :

```bash
mkdir -p "/chemin/Extensions utilisateur/vectolaz"
cp extensions/inkscape/vectolaz.inx extensions/inkscape/vectolaz.py \
  extensions/inkscape/image_source.py extensions/inkscape/placement.py \
  "/chemin/Extensions utilisateur/vectolaz/"
```

Dans l'onglet **Installation** de Vectolaz, sélectionner l'exécutable du moteur :

```text
/Users/mugen/Documents/01_Projets/Vectolaz/.venv/bin/python
```

Sur Windows, utiliser le chemin absolu de `.venv\Scripts\python.exe` ; pour
préparer le moteur dans PowerShell :

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

L'extension utilise inkex fourni par Inkscape. Elle appelle le moteur séparé
par un sous-processus sans shell, avec un délai maximal de **120 secondes**.
Les dépendances du moteur ne sont pas installées dans le Python d'Inkscape.
L'exécutable moteur est mémorisé avec les paramètres de l'extension par
Inkscape ; le chemin doit être actualisé si le projet ou le venv est déplacé.

## Essai de la rosace

1. Importer `samples/rosace.png` dans Inkscape, en l'embarquant ou en la liant.
2. Sélectionner uniquement cette image, puis ouvrir Vectolaz.
3. Désactiver **Seuil automatique**, entrer **160**.
4. Entrer **5** pour la tolérance géométrique et cocher le regroupement des
   cercles et grands arcs. Laisser le raccordement à 0 et l'épaisseur à 1.
5. Sélectionner le Python du moteur dans Installation, puis appliquer.

Résultat attendu : image conservée, groupe ajouté au-dessus, **1 cercle et
6 chemins d'arc**, modifiables individuellement après dégroupage. À 3 pixels,
cinq petits segments supplémentaires restent nécessaires. La tolérance
est toujours en pixels du PNG natif, pas en pixels écran ni en mm.

Déplacer temporairement l'image ou masquer son objet pour inspecter les
vecteurs. Tester ensuite une rotation et un redimensionnement de l'image,
puis relancer : la nouvelle géométrie doit rester superposée. Chaque passage
ajoute un nouveau groupe, avec des identifiants distincts.

## Formats et placement

PNG embarqué en base64, chemin local absolu/relatif et URI locale `file:`
sont pris en charge. Les liens relatifs sont résolus depuis le document
original, qui doit être enregistré ; l'extension ne relit pas ce document
sur disque et prend en compte les modifications en cours.

Le groupe inséré conserve le `transform` de l'image et le placement calculé
avec `x`, `y`, `width`, `height` et `preserveAspectRatio`. Les transformations
des parents restent héritées. Le demi-pixel du moteur est appliqué une fois.
Les alignements `meet` et `none`, rotations, miroirs, matrices affines et
échelles non uniformes sont pris en charge. Un cercle source devient
visuellement une ellipse si l'image est étirée ; le trait est étiré également.

Cette version refuse les clips, masques, filtres, mode `slice`, clones,
viewport SVG imbriqué, dimensions en pourcentage/unités relatives et
transformations ou dimensions CSS. Convertir ces cas en image PNG simple
avant de les utiliser. Les fichiers distants et autres formats sont refusés.
En cas d'échec du moteur ou de résultat vide/invalide, le document reste intact.

## Tests et contrôle visuel

Tests du moteur et des helpers autonomes :

```bash
source .venv/bin/activate
python -m pytest -q
```

Les tests de l'adaptateur utilisent le **vrai inkex** de l'installation choisie.
Ils ne sont pas remplacés par des mocks d'inkex. Pour les exécuter sur ce Mac :

```bash
python -m pip install -e ".[dev,inkscape-test]"
INKSCAPE_EXTENSIONS_DIR=/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions \
  python -m pytest -q
INKSCAPE_EXTENSIONS_DIR=/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions \
  python samples/generate_inkscape_samples.py
```

Sous Windows/Linux, adapter `INKSCAPE_EXTENSIONS_DIR` au dossier contenant
le paquet `inkex`, indiqué dans les préférences. En PowerShell :
`$env:INKSCAPE_EXTENSIONS_DIR = "C:\chemin\share\inkscape\extensions"`,
puis lancer les mêmes commandes Python sans le préfixe de variable.

La fixture génère `debug_output/inkscape/rosace_source.svg` et
`rosace_vectorized.svg` avec une image embarquée à opacité 30 %, un ratio
`meet`, une rotation de 12° et un document dimensionné en mm. Elle exécute
l'adaptateur et le moteur réels et vérifie les sept primitives.
Ouvrir le SVG résultant dans Inkscape pour comparer les axes aux traits.

Validation effectuée : **359 tests réussis**, inkex 1.4.0 issu d'Inkscape
1.4.3, exécuté avec le Python 3.14 du venv ; SVG résultant rendu et inspecté
avec Inkscape 1.4.3. Le Python 3.10 embarqué, lancé directement sur cette
machine, a quitté avec le code 137 sans diagnostic. L'installation et le
lancement depuis le menu sous l'interpréteur configuré d'Inkscape restent
donc à vérifier par l'utilisateur ; ce contrôle ne les remplace pas.

Critères de validation : menu disponible après redémarrage, moteur configuré,
PNG lié/embarqué traité, résultat correctement placé et éditable, image
conservée, messages utiles pour les cas refusés.
