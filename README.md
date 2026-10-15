# raster2vector

Outil Python de reconstruction géométrique de dessins au trait noir sur fond
blanc. L'objectif futur est de reconstruire les axes des traits, puis des
primitives géométriques, plutôt que les contours des zones noires.

## État actuel : moteur mature (411 tests) et extension Inkscape v2

Vectolaz propose une chaîne complète de vectorisation géométrique des axes médians
à partir de dessins au trait raster (PNG), avec reconstruction sous tolérance
analytique en segments de droite, cercles, arcs circulaires et courbes de Bézier cubiques.

Le projet compte désormais **411 tests unitaires et d'intégration** (validés avec
`pytest -W error`), couvrant :
- L'ensemble du pipeline raster (binarisation Otsu, débruitage, squelettisation 8-connexe, graphe topologique).
- L'**élagage topologique déterministe des micro-barbes** (`--spur-tolerance`) éliminant les aspérités de squelettisation sur les traits épais, avec dissolution automatique des fausses jonctions de degré 2.
- Le **dé-biaisage adaptatif des jonctions** ("Zhang-Suen") permettant une reconnaissance parfaite des segments et arcs arrivant sur des carrefours épais sans distorsion d'extrémité.
- La **continuité tangentielle $G^1$** aux raccords internes des chemins de Bézier multi-morceaux, garantissant des transitions fluides sans cassure visuelle.
- La **simplification des polylignes fermées** (`Polyline` en boucle fermée via Douglas-Peucker circulaire).
- Le **regroupement global** des cercles (`--global-circles`) et grands arcs continus (`--global-arcs`) à travers les intersections.
- L'**extension Inkscape v2** (`extensions/inkscape`) offrant un sélecteur de préréglages par profil de motif (`technical`, `rosace`, `organic`, `detailed`, `scanned`, `custom`), une interface en 3 étapes claires, un guide d'aide interactif et un étiquetage dynamique des calques SVG produits.

Commande courante, après installation :

```bash
raster2vector drawing.png --threshold auto --geometry-tolerance 2 \
  --spur-tolerance 3 --gap-tolerance 3 --min-component-size 10 \
  --stroke-width 1 --output drawing.svg --debug-dir debug --debug-svg
```

Le raccordement et l'élagage restent désactivés par défaut si `--gap-tolerance` ou `--spur-tolerance` ne sont pas précisés (valeur `0.0`).
Les réglages d'une image validée peuvent être conservés dans un fichier TOML.

## Installation (Python 3.11+)

Depuis la racine du projet, sur macOS ou Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Sur Windows PowerShell :

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Les dépendances d'exécution sont NumPy, OpenCV, scikit-image et SciPy. SciPy,
déjà installé via scikit-image, est déclaré directement pour l'ajustement radial.
pytest est fourni
par l'extra `dev`. Après une mise à jour du projet, relancer
`python -m pip install -e ".[dev]"` pour installer les nouvelles dépendances.

## Tests

```bash
python -m pytest -W error
```

La suite compte **411 tests unitaires et d'intégration** sans aucun avertissement.
Les tests utilisent des images synthétiques NumPy/OpenCV et vérifient :
- **Pré-traitement & Binarisation** : dimensions, chemins Unicode, gestion d'erreurs, transparence composée sur blanc, images 16 bits, seuillage automatique d'Otsu et manuel, polarités et suppression des poussières/bruit.
- **Squelettisation** : continuité 8-connexe, extrémités, intersections, conservation des trous et fidélité aux axes médians.
- **Topologie & Graphe** : nœuds, branches, boucles, coordonnées demi-pixel et les 512 motifs locaux binaires $3\times 3$.
- **Élagage des micro-barbes (`spur_tolerance`)** : suppression des branches pendantes parasites, préservation stricte d'au moins 2 branches par carrefour, dissolution topologique des nœuds de degré 2 avec raccordement sans saut spatial, conversion en boucle fermée des cycles sans nœud résiduel.
- **Primitives géométriques** : ajustement orthogonal de droites (L2/RMS/max), dé-biaisage adaptatif des jonctions de Zhang-Suen, arcs de cercle et cercles complets (quarts, demis, traversée de $\pm\pi$, flags SVG).
- **Courbes de Bézier & Lissage** : cubiques multi-morceaux, continuité tangentielle $G^1$ aux subdivisions internes sous contrôle d'erreur de Bernstein, préservation des angles vifs.
- **Polylignes & Boucles fermées** : simplification Douglas-Peucker des chemins ouverts et fermés (`Polyline(is_loop=True)`).
- **Raccordement des coupures (`gap_tolerance`)** : fusion de chaînes disjointes, fermeture de boucles, tangentes compatibles et franchissement d'obstacles.
- **Affinage des jonctions** : recalage géométrique des extrémités sur contacts communs.
- **Reconstruction globale** : cercles complets (`--global-circles`) et grands arcs continus (`--global-arcs`) à travers les intersections.
- **Extension Inkscape** : validation XML du descripteur `.inx`, application des 5 profils de préréglages, mode personnalisé, et exécution de l'adaptateur avec `inkex`.

## Essai de la CLI

Un dessin synthétique de 160 × 100 pixels est fourni dans `samples/line.png`.

```bash
python -m raster2vector samples/line.png --threshold 160 --debug-dir debug_output/line
raster2vector samples/line.png --threshold 160 --debug-dir debug_output/line
python -m raster2vector samples/line.png --output debug_output/line/output.svg
python -m raster2vector --help
```

Avec votre propre image :

```bash
python -m raster2vector "/chemin/vers/input.png" --threshold 160 --spur-tolerance 3 --debug-dir debug/
```

La sortie donne la largeur puis la hauteur en pixels et le nombre de pixels de
trait avant/après nettoyage, le nombre de pixels du squelette et les nombres
de nœuds, de branches et de boucles revenant au même nœud.
Elle indique aussi le nombre de segments reconstruits, de polylignes et de
branches conservées.
Elle indique également le nombre de cercles, d'arcs, de chemins Bézier et de cubiques.
`--debug-dir` est optionnel : sans lui, le
traitement est effectué en mémoire et aucun fichier n'est écrit.
Le code de sortie est 0 pour un traitement réussi, 1 pour une erreur d'entrée/
sortie du traitement, et 2 pour des arguments ou une configuration invalides
(y compris un fichier de configuration absent). Les erreurs usuelles sont
signalées sans traceback.

## Paramètres disponibles

| Option | Valeur par défaut | Effet |
| --- | --- | --- |
| `--threshold` | `128` | Entier entre 0 et 255, ou `auto` pour seuillage global d'Otsu ; sombre si intensité ≤ seuil. |
| `--config` | Aucun | Charge les réglages d'un fichier TOML avec table `[vectorization]`. Les options explicites prennent priorité. |
| `--spur-tolerance` | `0.0` | Supprime les micro-branches parasites (barbes de squelette) plus courtes que cette longueur en px ; 0 désactive l'élagage. |
| `--gap-tolerance` | `0.0` | Distance maximale entre deux extrémités compatibles à raccorder ; 0 désactive le raccordement. |
| `--geometry-tolerance` | `2.0` | Erreur maximale en pixels pour accepter un segment, cercle, arc ou Bézier. |
| `--polyline-tolerance` | `2.0` | Distance maximale au segment correspondant lors de la simplification d'une branche non droite. |
| `--global-circles` | Désactivé | Regroupe les branches compatibles constituant un cercle complet à travers les intersections. |
| `--global-arcs` | Désactivé | Privilégie les longs arcs continus à travers les intersections ; active aussi `--global-circles`. |
| `--min-component-size` | `0` | Supprime les objets isolés de surface strictement inférieure au minimum, en pixels ; 0 conserve tout. |
| `--stroke-width` | `1.0` | Épaisseur d'affichage du trait SVG en pixels sources, finie et strictement positive. |
| `--invert` | `auto` | `yes` : traits sombres ; `no` : traits clairs ; `auto` : polarité déduite de la bordure. |
| `--no-junction-cleanup` | Désactivé | Conserve les contacts bruts pour comparaison sans affinage des jonctions. |
| `--junction-cleanup` | Activé | Active l'affinage géométrique des jonctions (par défaut activé). |
| `--output` | Aucun | Écrit le SVG final ; le dossier parent est créé si nécessaire. |
| `--debug-dir` | Aucun | Enregistre les images intermédiaires et les contrôles de jonction. |
| `--debug-svg` | Désactivé | Ajoute `overlay.svg` dans `--debug-dir`, avec le raster embarqué sous les vecteurs. |
| `--version` | — | Affiche la version installée et quitte. |

L'inversion automatique convient aux dessins entourés de fond uniforme. Si le
dessin occupe largement le bord de l'image, forcer `--invert yes` pour un trait
sombre sur fond clair ou `--invert no` pour un trait clair sur fond sombre.
En cas d'égalité au bord, `auto` suppose un fond clair. `--threshold auto`
choisit un seuil global par Otsu ; la polarité reste indépendante du choix
de ce seuil. Un fond non uniforme peut nécessiter un réglage manuel.

Le nettoyage utilise la connexité à 8 voisins pour conserver les diagonales.
Une composante dont la surface vaut exactement le minimum est conservée. Un
petit détail volontaire peut être supprimé s'il est plus petit que ce minimum :
comparer `binary.png` à `cleaned.png` et réduire le paramètre si nécessaire.
Aucune morphologie n'est appliquée : les traits conservés ne sont ni dilatés
ni érodés. Le raccordement optionnel intervient ensuite sur le graphe ;
les images binaires et le squelette d'origine restent inchangés.

Les PNG transparents sont composés sur du blanc. Les PNG 16 bits sont ramenés
à l'échelle 8 bits avant le seuillage.

## Images de debug

```text
debug/
    grayscale.png   # niveaux de gris 8 bits
    binary.png      # masque avant nettoyage
    cleaned.png     # masque après nettoyage
    skeleton.png    # squelette d'un pixel
    graph.png       # branches colorées, nœuds N# et branches E#
    gaps.png        # avec --gap-tolerance > 0 : ponts acceptés en magenta
    graph_connected.png # avec --gap-tolerance > 0 : graphe après fusion des branches
    geometry.png    # segments rouges, arcs orange, cercles violets, Bézier cyan, polylignes vertes, brutes bleues
    geometry_before.png # géométrie avant affinage des jonctions
    geometry_local.png # avec --global-circles : géométrie avant regroupement global
    junctions.png   # jonctions acceptées vertes, refusées rouges, déplacements magenta
    overlay.svg     # seulement avec --debug-svg : raster + vecteurs
    overlay_before.svg # avec --debug-svg : comparaison avant affinage des jonctions
    overlay_local.svg # avec --debug-svg et --global-circles : comparaison avant regroupement global
```

Les masques en mémoire sont booléens (`fond=False/0`, `trait=True/1`). Pour être
lisibles à l'écran, les PNG de debug montrent les traits en noir (0) sur fond
blanc (255). Leurs dimensions correspondent à celles de la source.
Les fichiers de debug sont remplacés à chaque exécution. La CLI refuse un
fichier de sortie/debug qui écraserait le fichier source et les collisions
entre la sortie finale et les fichiers de debug.

Exemple de nettoyage du bruit avec le PNG synthétique fourni :

```bash
python -m raster2vector samples/line_noisy.png --threshold 160 --min-component-size 3 --debug-dir debug_output/noisy
```

Pour la rosace :

```bash
python -m raster2vector samples/rosace.png --threshold 160 --debug-dir debug_output/rosace
```

## Utilisation comme bibliothèque

```python
from raster2vector import (
    load_image, to_grayscale, binarize, remove_small_objects, skeletonize_mask,
    skeleton_to_graph,
    reconstruct_geometry, LineSegment,
    export_svg,
)

image = load_image("samples/line.png")
grayscale = to_grayscale(image)
binary = binarize(grayscale, threshold=160, invert="auto")
cleaned = remove_small_objects(binary, min_component_size=3)
skeleton = skeletonize_mask(cleaned)
graph = skeleton_to_graph(skeleton)
for edge in graph.edges:
    print(edge.id, edge.start_node_id, edge.end_node_id, edge.points)
geometry = reconstruct_geometry(graph, geometry_tolerance_px=2.0, polyline_tolerance_px=2.0)
for primitive in geometry.primitives:
    if isinstance(primitive, LineSegment):
        print(primitive.start, primitive.end, primitive.fit_error_max, primitive.fit_error_rms)
export_svg(geometry, "debug_output/line/output.svg", stroke_width=1.0)
```

## Architecture

```text
pyproject.toml
src/raster2vector/
    __init__.py
    __main__.py       # python -m raster2vector
    cli.py            # interface en ligne de commande
    config.py         # paramètres validés et lecture TOML (bibliothèque standard)
    image_io.py       # lecture raster indépendante des algorithmes
    coordinates.py    # Point(x, y) partagé par graphe et géométrie
    preprocessing.py  # niveaux de gris, seuil, petites composantes
    skeleton.py       # amincissement topologique 2D
    graph.py          # structures de données et extraction topologique
    geometry.py       # ajustements de droites/cercles/arcs, Douglas-Peucker et sélection des primitives
    bezier.py         # fitting cubique, découpage et borne d'erreur continue
    gaps.py           # raccords entre endpoints, fusion des chaînes et provenance
    junctions.py      # contacts communs, arcs contraints, continuités et refus atomiques
    global_circles.py # cercles complets multibranches, couverture et provenance
    global_arcs.py   # grands arcs multibranches, priorité aux courbes longues
    svg_export.py     # sérialisation XML SVG et écriture du fichier
    debug.py          # export visuel des résultats intermédiaires
tests/
samples/
debug_output/
```

Le traitement d'image, la topologie, la géométrie et les entrées/sorties restent
séparés. Les tableaux raster s'indexent en `(row, col)` ; la géométrie utilisera
des coordonnées `(x, y)` avec conversion explicite.

## Squelettisation (étape 2)

Le masque nettoyé est aminci par `skimage.morphology.skeletonize`, avec la méthode
2D de Zhang-Suen explicitement choisie. Elle est déterministe et préserve la
topologie des traits. L'entrée n'est pas modifiée ; la sortie est un masque
booléen de mêmes dimensions, avec les mêmes conventions `(row, col)`.

L'amincissement approche l'axe central mais ne calcule pas un axe médian exact :
un décalage local d'environ un pixel peut apparaître, notamment pour des traits
d'épaisseur paire. Les extrémités épaisses peuvent être raccourcies. Aucun
élagage, raccordement ou correction d'intersection n'est appliqué. Si un trait
est déjà interrompu dans `cleaned.png`, cette étape ne répare pas la coupure.

Six PNG synthétiques sont fournis dans `samples/skeleton_cases/` : ligne
horizontale, diagonale, L, T, croix et cercle épais. Pour les régénérer :

```bash
python samples/generate_skeleton_samples.py
```

Pour traiter les six cas (macOS/Linux) :

```bash
for name in horizontal diagonal L T cross circle; do
  python -m raster2vector "samples/skeleton_cases/$name.png" \
    --debug-dir "debug_output/skeleton_cases/$name"
done
```

Pour traiter un seul cas, sur toutes les plateformes :

```bash
python -m raster2vector samples/skeleton_cases/T.png --debug-dir debug_output/skeleton_cases/T
```

## Conversion en graphe (étape 3)

Le moteur utilise des dataclasses typées sans NetworkX :

- `Point(x, y)` : coordonnées dans le repère de l'image, x=col et y=row.
- `GraphNode(id, kind, position, source_pixels)` : `endpoint`, `junction`,
  `loop` (ancre d'une boucle sans intersection) ou `isolated`. Les pixels
  sources restent en `(row, col)` ; la position est leur barycentre.
- `GraphEdge(id, start_node_id, end_node_id, points)` : chemin ordonné des
  pixels d'une branche, convertis explicitement en `Point(x, y)`.
- `SkeletonGraph(width, height, nodes, edges)` : les identifiants indexent les
  tuples de nœuds et de branches. Leur ordre est déterministe.

Les voisins sont recherchés dans les huit directions. Une diagonale est
ignorée lorsqu'un passage orthogonal existe déjà : cela évite les triangles
artificiels aux angles. Dans un bloc plein 2x2, le côté vertical droit est
également redondant et supprimé du comptage des liaisons ; ce bloc n'est pas
un trou raster et ne doit pas devenir une boucle. Ces règles ne changent
aucun pixel du squelette et conservent sa connexité.

Les pixels de degré supérieur ou égal à trois et ceux des blocs pleins 2x2
forment les zones d'intersection. Les pixels de ces zones directement voisins
sont regroupés. Une branche entre deux zones reste conservée, même si elle
est courte. Les véritables cycles internes à une zone sont conservés comme
boucles. Un pixel isolé reste représenté par un nœud sans branche.

Une boucle sans intersection possède un nœud artificiel `loop` et une arête
revenant sur ce nœud, avec répétition du premier point à la fin du chemin.
`edge.is_loop` indique que les deux identifiants de nœud sont identiques.
Pour une boucle attachée à une zone d'intersection, les deux contacts peuvent
être des pixels différents de cette même zone : ils ne sont pas déplacés.
Les autres cycles peuvent parcourir plusieurs nœuds : le compteur de boucles
de la CLI compte uniquement les arêtes revenant sur un même nœud.

`graph.png` conserve les dimensions de l'image source. Les branches sont
colorées et identifiées `E#`. Les nœuds sont identifiés `N#` : extrémités
vertes, intersections rouges, ancres de boucle violettes et pixels isolés
noirs. Dans les zones denses, des traits gris relient les étiquettes décalées
à leur position. Les couleurs et marqueurs servent uniquement à l'inspection.

Le graphe conserve les points du raster, sans reconstruire de droite ou
d'arc. Les petites branches et intersections parasites de la rosace restent
donc visibles. Avec le seuil 160, elle donne actuellement 28 nœuds et 44
branches. Le barycentre d'un nœud reste une donnée raster ; l'étape 11
conserve séparément les contacts géométriques affinés dans `JunctionResult`.

## Reconstruction de segments (étape 4)

Chaque branche ouverte est ajustée par PCA/SVD sur **tous** ses points centrés.
La direction retenue minimise la somme des carrés des distances
perpendiculaires à la droite : le traitement est identique pour une droite
horizontale, verticale ou oblique. Aucun point aberrant n'est rejeté.

`fit_line(points)` fournit un `LineFit` : barycentre, direction unitaire,
bornes projetées et erreurs maximale/RMS en pixels. Les bornes sont les
projections extrêmes de tous les points afin que le segment couvre la branche
entière. Pour un trait parcouru sans retour en arrière, elles correspondent
généralement aux projections des extrémités raster. Le sens de parcours est
conservé lorsque la projection du premier au dernier point est non nulle.

L'ajustement à une seule droite, premier traitement de
`reconstruct_geometry(graph, geometry_tolerance_px=2.0)`, produit un
`GeometryResult(width, height, primitives)`, avec une primitive par branche :

- `LineSegment(start, end, fit_error_max, fit_error_rms, source_edge_id,
  source_points)` si l'erreur maximale est **inférieure ou égale** à la tolérance.
- `RawPolyline(points, source_edge_id, is_loop, fit_error_max, fit_error_rms)`
  sinon, en conservant les points ordonnés inchangés et les erreurs mesurées,
  si la simplification de l'étape 5 est désactivée ou n'apporte aucun gain.

Les boucles ne deviennent jamais des segments : à l'étape 7, elles sont
ajustées comme cercles ou conservées comme `RawPolyline`. Les branches sans
deux points distincts restent brutes.
Les erreurs sont `None` lorsqu'aucun ajustement n'a été effectué. Le graphe,
les zones d'intersection et les points sources ne sont pas modifiés. Une
primitive référence sa branche par `source_edge_id` ; les données raster
sont conservées pour contrôler son erreur.

Une tolérance trop faible peut refuser une ligne raster qui paraît droite,
à cause du crénelage ou d'un petit crochet de squelettisation aux extrémités.
Une tolérance élevée peut accepter une portion de courbe peu incurvée. Le
critère mesure un écart en pixels, pas l'intention du dessin. L'affinage
des raccords entre branches est décrit à l'étape 11.

`geometry.png` montre les arcs en **orange**, les cercles en **violet**, les segments en **rouge**, les polylignes en **vert**
avec leurs sommets marqués, et les branches brutes en **bleu**, sur le raster
original éclairci en niveaux de gris. Le repère et les
dimensions restent inchangés. Les pixels isolés n'ont pas de branche à ajuster
et restent visibles dans le raster et dans le graphe.

Cinq PNG synthétiques sont fournis dans `samples/geometry_cases/` : ligne
horizontale, verticale, diagonale, ligne légèrement bruitée et courbe.
Pour les régénérer et comparer les deux derniers :

```bash
python samples/generate_geometry_samples.py
python -m raster2vector samples/geometry_cases/noisy_line.png \
  --geometry-tolerance 2 --debug-dir debug_output/geometry_cases/noisy_line
python -m raster2vector samples/geometry_cases/curve.png \
  --geometry-tolerance 2 --debug-dir debug_output/geometry_cases/curve
```

Pour les traiter tous (macOS/Linux) :

```bash
for name in horizontal vertical diagonal noisy_line curve; do
  python -m raster2vector "samples/geometry_cases/$name.png" \
    --geometry-tolerance 2 --debug-dir "debug_output/geometry_cases/$name"
done
```

Pour la rosace :

```bash
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 2 --debug-dir debug_output/rosace
```

Avec ces paramètres à l'étape 5, la rosace produit 26 segments et 18 polylignes.
Les longues courbes sont approximées par des polylignes ; les petites branches
suffisamment proches d'une droite deviennent des segments. Le cercle extérieur
de la rosace est réparti entre plusieurs branches ouvertes : l'étape 7 ne le
recompose pas, même si elle sait maintenant ajuster une branche fermée isolée.

## Polylignes et découpage (étape 5)

Si l'ajustement à une seule droite dépasse `geometry_tolerance_px`, la branche
ouverte est simplifiée par Douglas-Peucker, en utilisant
`polyline_tolerance_px`. Ces deux tolérances sont indépendantes : la première
accepte une droite ajustée, la seconde borne l'écart des sous-chaînes à leurs
segments. L'ajustement global reste prioritaire pour éviter de découper une
branche déjà compatible avec une seule droite.

La simplification utilise les distances aux **segments finis**. Elle conserve
les extrémités, les coins nécessaires et l'ordre du tracé, y compris les retours
en arrière. Une passe retire ensuite les sommets redondants, uniquement si la
sous-chaîne fusionnée respecte encore la même tolérance. Aucun ajustement
séparé des segments n'est appliqué, afin de conserver leurs sommets partagés.

Une branche simplifiée produit :

```python
Polyline(
    points, fit_error_max, fit_error_rms,
    source_edge_id, source_points, source_indices,
)
```

`points` contient les sommets retenus, et `segment_count` le nombre de segments.
`source_indices` indique leur position dans les données d'origine, permettant
de vérifier chaque sous-chaîne. Les erreurs maximale et RMS mesurent la
distance de chaque point à son segment correspondant, plutôt qu'au segment
le plus proche parmi toute la polyligne.

La fonction publique `simplify_polyline(points, tolerance_px=2.0)` renvoie les
sommets d'une polyligne ouverte. Elle rejette les boucles fermées ; celles du
graphe sont traitées par l'ajustement de cercle, puis restent des `RawPolyline`
si elles sont incompatibles. Une branche sans gain de
simplification reste également brute. Aucun point source n'est perdu : les
primitives conservent les données d'origine.

Une tolérance de zéro autorise seulement la suppression des points
strictement colinéaires, sans écart. Pour désactiver la simplification des
branches ouvertes dans la bibliothèque, utiliser :

```python
geometry = reconstruct_geometry(graph, 2.0, polyline_tolerance_px=None, detect_arcs=False)
```

Une courbe douce est encore représentée par des segments droits. Une petite
tolérance produit davantage de segments ; une grande tolérance peut effacer
des détails. Aucun nombre maximal arbitraire de segments n'est imposé, et
Douglas-Peucker ne garantit pas le minimum mathématique de sommets.
Les angles arrondis du squelette peuvent demander des segments supplémentaires.
Les cercles fermés sont ajustés à l'étape 7 et les arcs à l'étape 8 ;
les courbes libres viendront ensuite.

Quatre PNG synthétiques sont fournis dans `samples/polyline_cases/`.
Pour les régénérer et les traiter (macOS/Linux) :

```bash
python samples/generate_polyline_samples.py
for name in L zigzag almost_straight smooth_curve; do
  python -m raster2vector "samples/polyline_cases/$name.png" \
    --geometry-tolerance 2 --polyline-tolerance 2 \
    --debug-dir "debug_output/polyline_cases/$name"
done
```

Sur toutes les plateformes, pour un seul exemple :

```bash
python -m raster2vector samples/polyline_cases/L.png \
  --geometry-tolerance 2 --polyline-tolerance 2 --debug-dir debug_output/polyline_cases/L
```

## Export SVG minimal (étape 6)

Le fichier final utilise des chemins et, depuis l'étape 7, des éléments
`<circle>`, avec `fill="none"` et `stroke="black"`. Il supporte `LineSegment`,
`Polyline`, `RawPolyline`, `Circle`, `CircularArc` et `BezierPath`,
avec une épaisseur configurable et des extrémités/jointures arrondies. Les
identifiants `edge-#` correspondent aux branches sources. L'export repose sur
`xml.etree.ElementTree` de la bibliothèque standard, sans nouvelle dépendance.

Commande complète :

```bash
python -m raster2vector samples/rosace.png \
  --threshold 160 --geometry-tolerance 2 --polyline-tolerance 2 \
  --stroke-width 1 --output debug_output/rosace/output.svg \
  --debug-dir debug_output/rosace --debug-svg
```

Dans un environnement activé, `raster2vector` peut remplacer
`python -m raster2vector`. Sans `--output`, le programme reste utilisable
pour inspecter les étapes intermédiaires. `--debug-svg` nécessite
`--debug-dir` et fonctionne aussi sans sortie finale.

`output.svg` contient uniquement les vecteurs sur fond transparent.
`overlay.svg` contient le PNG original embarqué en base64, avec une opacité
de 35 %, puis les mêmes vecteurs au-dessus. Il est autonome : déplacer le SVG
ne casse pas le lien vers l'image. Les fichiers existants de même nom sont
remplacés lors d'un nouvel export.

La `viewBox` est `0 0 largeur hauteur`, et la taille du document est celle du
PNG, en pixels. Les primitives conservent leurs coordonnées `(x, y)`.
Le groupe SVG des vecteurs applique une translation de `(0.5, 0.5)` : un pixel
OpenCV indexé `(row, col)` a son centre SVG à `(col+0.5, row+0.5)` lorsque
l'image couvre `[0, largeur] × [0, hauteur]`. Cette translation aligne les
strokes avec le raster embarqué, sans modifier les données géométriques.

Les boucles raster explicitement fermées sont exportées avec `Z`. Si une
branche est une boucle topologique dont les contacts raster sont distincts
dans une zone d'intersection, son tracé est conservé tel quel : l'export
n'invente pas de raccord. L'étape 11 peut auparavant ajuster ces contacts
à un point commun si la tolérance le permet. Les pixels isolés restent
dans le graphe et le debug raster, sans primitive SVG à cette étape.

La rosace est désormais reconstruite en segments, arcs et quelques polylignes.
Pour gagner en finesse sur les branches restant sous forme de polylignes,
réduire `--polyline-tolerance`, au prix d'un plus grand nombre de sommets.

Les exemples de la rosace et du cercle ont été ouverts et rendus sans erreur
avec l'installation locale d'Inkscape. Leur rendu de contrôle est disponible
dans `debug_output/rosace/svg_render.png`,
`debug_output/rosace/overlay_render.png` et
`debug_output/skeleton_cases/circle/svg_render.png`.

Pour produire le SVG du L :

```bash
python -m raster2vector samples/polyline_cases/L.png \
  --output debug_output/polyline_cases/L/output.svg \
  --debug-dir debug_output/polyline_cases/L --debug-svg
```

L'API Python propose `svg_document(geometry, stroke_width=1.0)` pour obtenir
le XML et `export_svg(geometry, path, stroke_width=1.0)` pour l'écrire.
Le paramètre nommé `background_png` accepte les octets d'un PNG de mêmes
dimensions pour créer un document de comparaison. Un appel direct à l'API
doit choisir une destination distincte de ses fichiers sources.

## Détection des cercles (étape 7)

Le moteur ajuste chaque branche **explicitement fermée** à un cercle à partir
de ses points sources, avant tout autre traitement de cette boucle. Il ne
cherche pas des cercles par Hough et ne regroupe pas plusieurs branches.

L'estimation initiale résout un système algébrique avec `numpy.linalg.lstsq`
sur des coordonnées centrées et normalisées. `scipy.optimize.least_squares`
affine ensuite le centre en minimisant les écarts radiaux ; le rayon optimal
est la moyenne des distances au centre. Aucun point aberrant n'est rejeté.
Une ancre répétée en fin de boucle n'est comptée qu'une fois.

`fit_circle(points)` fournit un `CircleFit(center, radius, fit_error_max,
fit_error_rms)`. L'erreur d'un point est `abs(distance(point, center) - radius)`.
Les données insuffisantes, colinéaires ou dégénérées et les ajustements qui ne
convergent pas renvoient `None`. Les coordonnées non finies sont rejetées.

Une branche devient :

```python
Circle(center, radius, fit_error_max, fit_error_rms, source_edge_id, source_points)
```

si son erreur radiale maximale est inférieure ou égale à
`geometry_tolerance_px` **et** son parcours effectue un tour autour du centre.
Cette vérification évite de transformer un arc refermé par un retour en arrière
en cercle complet. Le graphe et les pixels sources restent conservés.

Les boucles incompatibles, dont les ellipses dépassant la tolérance, restent
brutes avec les erreurs mesurées. Les boucles topologiques ayant des contacts
raster distincts dans une zone d'intersection ne sont pas ajustées en cercle.
Une tolérance élevée peut accepter une ellipse proche d'un cercle : elle
exprime un écart maximal, pas une classification indépendante de cet écart.

`Circle` est exporté comme un vrai `<circle cx="…" cy="…" r="…">`, avec les
mêmes styles et translation de 0.5 pixel que les chemins. Il apparaît en
violet dans `geometry.png`. Son centre et son rayon restent fractionnaires.

Trois PNG synthétiques sont fournis dans `samples/circle_cases/` : cercle
raster, cercle légèrement bruité et ellipse. Pour les régénérer :

```bash
python samples/generate_circle_samples.py
```

Pour traiter les trois cas (macOS/Linux) :

```bash
for name in circle noisy_circle ellipse; do
  python -m raster2vector "samples/circle_cases/$name.png" \
    --geometry-tolerance 2 \
    --output "debug_output/circle_cases/$name/output.svg" \
    --debug-dir "debug_output/circle_cases/$name" --debug-svg
done
```

Pour un seul exemple sur toutes les plateformes :

```bash
python -m raster2vector samples/circle_cases/circle.png \
  --geometry-tolerance 2 --output debug_output/circle_cases/circle/output.svg \
  --debug-dir debug_output/circle_cases/circle --debug-svg
```

Avec une tolérance de 2 pixels, le cercle fourni a un rayon ajusté d'environ
54.94 pixels et une erreur maximale de 0.76 pixel ; le cercle bruité a une
erreur maximale de 1.26 pixel. L'ellipse a une erreur maximale de 19.56 pixels
et reste brute. Les SVG du cercle et de la comparaison du cercle bruité ont
été rendus avec Inkscape sans erreur.

À l'étape 7, la rosace conservait 26 segments et 18 polylignes : aucune branche
n'y parcourt seule un cercle complet. Depuis l'étape 8, les branches ouvertes
compatibles sont ajustées comme arcs. L'étape 11 affine désormais les
raccordements aux intersections compatibles avec la tolérance.

## Détection des arcs (étape 8)

Une branche ouverte qui échoue à l'ajustement d'une droite est testée comme
arc **avant** Douglas-Peucker. `fit_arc(points, tolerance_px=2.0)` reprend
l'ajustement de cercle sur tous les points, projette les extrémités sur ce
cercle et déroule leurs angles pour gérer les passages par +/-pi.

La représentation expose :

```python
CircularArc(
    center, radius, start_angle_rad, end_angle_rad,
    fit_error_max, fit_error_rms, source_edge_id, source_points,
)
```

`end_angle_rad` est déroulé, et `sweep_angle_rad` est sa différence avec
`start_angle_rad`. Les propriétés `start` et `end` donnent les extrémités
projetées. `clockwise`, `large_arc_flag` et `sweep_flag` décrivent le parcours.
Dans le repère image, Y descend : un balayage positif est horaire et donne
`sweep_flag=1`, un balayage négatif donne `sweep_flag=0`.
`large_arc_flag=1` lorsque l'arc dépasse un demi-tour. Une très petite marge
numérique stabilise le flag pour un demi-tour exact.

La tolérance borne la distance à l'**arc fini**, pas seulement à son cercle
support. Un point hors du balayage est comparé aux extrémités. Les retours
angulaires dépassant la tolérance en longueur d'arc sont refusés, ainsi que les
boucles, balayages nuls ou couvrant un tour complet. Les branches refusées
continuent vers la simplification en polyligne ou restent brutes.

Les arcs très courts qui respectent déjà la tolérance d'une droite restent
des segments ; ce choix évite des cercles de rayon mal déterminé sur des
traits presque droits. Sur les autres arcs courts, le centre et le rayon
peuvent être moins stables que sur un arc long : l'écart au tracé reste le
critère de validation. Tous les points sources sont conservés.

Le SVG utilise `M … A r r 0 large-arc-flag sweep-flag …` pour chaque arc,
avec les styles et la translation de 0.5 pixel déjà employés. Le rendu
`geometry.png` montre les arcs en orange. Le debug PNG les échantillonne pour
affichage ; le SVG contient une vraie commande d'arc.

L'API expose aussi `ArcFit` et `fit_arc`. Pour inspecter uniquement les
ajustements antérieurs à cette étape, utiliser
`reconstruct_geometry(graph, 2.0, 2.0, detect_arcs=False, detect_beziers=False)`.

Cinq exemples sont fournis dans `samples/arc_cases/` : quart de cercle,
demi-cercle, arc court, arc de plus de 180 degrés et demi-ellipse non circulaire.
Pour les régénérer et les traiter (macOS/Linux) :

```bash
python samples/generate_arc_samples.py
for name in quarter half short large non_arc; do
  python -m raster2vector "samples/arc_cases/$name.png" \
    --geometry-tolerance 2 --output "debug_output/arc_cases/$name/output.svg" \
    --debug-dir "debug_output/arc_cases/$name" --debug-svg
done
```

Pour la rosace, sur toutes les plateformes :

```bash
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 2 --polyline-tolerance 2 \
  --output debug_output/rosace/output.svg --debug-dir debug_output/rosace --debug-svg
```

À l'étape 8, avec ces paramètres, elle produisait **15 arcs, 26 segments et 3 polylignes**.
Les arcs remplacent les longues courbes compatibles ; les petits détours et
les raccords aux zones d'intersection restent à traiter. Les rendus du SVG
final, de la comparaison de la rosace et du grand arc ont été vérifiés dans
Inkscape sans erreur.

## Validation de l'étape 8

- Les tests passent et les deux points d'entrée fonctionnent.
- Le SVG final s'ouvre dans Inkscape, avec une page de mêmes dimensions que le PNG.
- Les objets sont des chemins ou cercles à stroke noir sans remplissage, et l'épaisseur
  respecte `--stroke-width`.
- Dans `overlay.svg`, les traits passent près des axes du raster, dans la
  limite des tolérances de reconstruction choisies.
- Les quatre exemples circulaires ouverts produisent chacun un arc SVG `A`.
- Le contre-exemple elliptique reste une polyligne à 2 pixels de tolérance.
- Les flags donnent le bon sens et la bonne portion du cercle, notamment pour
  un arc dépassant 180 degrés ou traversant +/-pi.
- Les arcs orange se superposent aux axes des traits dans la tolérance choisie.
- Les centres, rayons, angles, erreurs et points sources sont accessibles en Python.
- Le SVG final est indépendant du fichier raster ; le SVG de comparaison
  reste utilisable après déplacement grâce à son image embarquée.

Étape 8 validée ; les courbes libres sont maintenant traitées à l'étape 9.

## Courbes libres (étape 9)

Les droites, cercles et arcs gardent la priorité. Pour une branche ouverte
restante, le moteur examine la polyligne simplifiée. Un angle de changement
de direction de **45° ou plus** conserve cette polyligne ; ce seuil explicite
protège les coins, mais peut aussi conserver certaines courbes très serrées.
Une courbe plus douce peut devenir un `BezierPath` si le nombre de points
diminue : `3 × nombre de cubiques + 1`, en comptant les poignées, doit être
strictement inférieur au nombre de sommets de la polyligne.

Chaque cubique est ajustée par moindres carrés avec extrémités imposées.
Le paramétrage initial suit les longueurs de corde, puis jusqu'à 12 itérations
de Newton le rapprochent de la courbe, en conservant l'ordre des sources.
Le moteur mesure les erreurs maximale et RMS aux points sources, et une
**borne continue** `error_bound` : sur chaque intervalle source, il compare
les quatre contrôles de la restriction cubique aux contrôles du segment
raster correspondant. L'enveloppe convexe de Bernstein borne toute la
différence, y compris entre les points. Cette borne doit respecter
`--geometry-tolerance` ; elle peut être plus conservatrice que la distance
minimale réelle à la courbe. Elle concerne la chaîne du squelette, pas les
bords du trait épais du PNG.

Si la borne dépasse la tolérance, la chaîne est découpée ; les morceaux
adjacents sont ensuite fusionnés lorsque leur nouvel ajustement est accepté.
La recherche est déterministe et gloutonne, sans garantie du minimum global
de cubiques. Les extrémités internes sont exactement partagées (continuité C0),
mais les tangentes ne sont pas contraintes à être identiques. Les points
sources, indices et paramètres restent disponibles pour vérifier l'ajustement.
À tolérance nulle, les arrondis numériques peuvent empêcher l'acceptation
d'une cubique ; le moteur conserve alors la représentation antérieure.

Le SVG contient des commandes `M … C …`, avec strokes noirs sans remplissage.
Dans `geometry.png`, les Bézier sont en **cyan**, avec leurs points de jonction
marqués ; les poignées ne sont pas dessinées sur le raster. Les arcs restent
orange, les segments rouges et les polylignes vertes.

L'API publique expose `CubicBezier`, `BezierPath`, `fit_bezier` et
`evaluate_cubic`. `reconstruct_geometry(..., detect_beziers=False)` permet
de comparer avec l'étape 8. `polyline_tolerance_px=None` désactive la
simplification et donc la sélection des Bézier dans le moteur ; le fitting
direct reste disponible via `fit_bezier(points, tolerance_px=2)`.

Exemples reproductibles (depuis la racine du projet, macOS/Linux) :

```bash
source .venv/bin/activate
python -m pytest
python samples/generate_bezier_samples.py
for name in s_curve wave corner; do
  python -m raster2vector "samples/bezier_cases/$name.png" \
    --geometry-tolerance 2 --polyline-tolerance 2 \
    --output "debug_output/bezier_cases/$name/output.svg" \
    --debug-dir "debug_output/bezier_cases/$name" --debug-svg
done
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 2 --polyline-tolerance 2 \
  --output debug_output/rosace/output.svg --debug-dir debug_output/rosace --debug-svg
```

Les exemples produisent une cubique pour `s_curve`, deux pour `wave`, et une
polyligne à deux segments pour `corner`. La rosace produit **26 segments,
15 arcs, 2 chemins Bézier (3 cubiques) et 1 polyligne**. Les défauts locaux
des raccords aux intersections restent visibles : le raccordement des petits
trous est prévu à l'étape 10 et le nettoyage des intersections à l'étape 11.

## Validation de l'étape 9

- Les tests passent et les SVG s'ouvrent dans Inkscape.
- `s_curve` et `wave` restent proches des axes raster dans `overlay.svg`,
  avec respectivement une et deux commandes cubiques `C`.
- Le L conserve ses deux segments et son angle marqué.
- La borne d'erreur des Bézier respecte la tolérance demandée ; les points
  sources et les extrémités partagées sont conservés.
- Les chemins contiennent moins de points que la polyligne de comparaison,
  et les cercles/arcs/segments reconnus gardent leur priorité.

Étape 9 validée ; le raccordement optionnel est maintenant disponible à l'étape 10.

## Petits trous et raccordement (étape 10)

`--gap-tolerance` mesure la distance euclidienne entre **centres des endpoints
du squelette**, pas la largeur blanche entre les bords du trait épais. Il vaut
0 par défaut : aucun raccord n'est alors recherché et le graphe est inchangé.
Le raccordement intervient après l'extraction du graphe et avant l'ajustement
des primitives.

Seuls deux nœuds `endpoint` sont candidats. Une tangente sortante est ajustée
par PCA sur au plus **12 pixels de longueur de chaîne** à chaque extrémité.
Une portée inférieure à **3 pixels**, ou une erreur locale supérieure à
**1.25 pixel**, refuse la tangente. Les deux tangentes doivent viser l'autre
extrémité avec un angle d'au plus **20°** chacune. Ces constantes internes
sont nommées dans `gaps.py` ; elles permettent un comportement conservateur
sans ajouter des options CLI supplémentaires.

Le pont rectiligne doit rester à plus de **0.75 pixel** des autres traits du
squelette et des pixels isolés. Les voisins locaux du contact situés derrière
sa tangente sont exclus de ce contrôle ; un trait incident qui revient dans
la coupure reste un obstacle. Si une extrémité a plusieurs candidats encore
compatibles, aucun n'est choisi pour elle. Deux ponts proposés qui se croisent
ou passent à moins de 0.75 pixel l'un de l'autre sont tous deux refusés.
Les tests d'obstacle portent sur le squelette, pas sur les bords des strokes
épais ; ils ne garantissent pas de retrouver l'intention du dessin. Les courbes
serrées, branches très courtes et coupures ambiguës peuvent rester ouvertes.

Les ponts acceptés réunissent les branches dans un nouveau graphe. Les
coordonnées sources restent conservées, avec deux contacts consécutifs qui
définissent le segment ajouté. Les anciens endpoints intermédiaires sont
retirés ; une chaîne refermée conserve une ancre de boucle. Les intersections
existantes ne sont ni déplacées ni affinées à cette étape. Les identifiants
du graphe raccordé sont réindexés ; les identifiants SVG `edge-#` désignent
désormais ces branches raccordées.

L'API `connect_gaps(graph, gap_tolerance_px=3)` retourne un `GapResult` :
`graph` contient le résultat, `connections` les ponts acceptés avec leurs
endpoints originaux et leur distance, et `source_edge_ids[i]` les identifiants
des anciennes branches réunies dans la nouvelle branche i. Le graphe fourni
n'est jamais modifié. Exemple :

```python
from raster2vector import connect_gaps, reconstruct_geometry

raccords = connect_gaps(graph, gap_tolerance_px=3)
geometry = reconstruct_geometry(raccords.graph, geometry_tolerance_px=2)
```

`graph.png` et `skeleton.png` montrent toujours la topologie d'origine.
Avec `--gap-tolerance > 0`, `gaps.png` montre les seuls ponts acceptés en
magenta, et `graph_connected.png` les branches fusionnées avec leurs nouveaux
identifiants. `geometry.png`, `output.svg` et `overlay.svg` utilisent le graphe
raccordé. La console indique le nombre de raccords acceptés et les nouveaux
comptages du graphe.

Commandes de contrôle depuis la racine du projet (macOS/Linux) :

```bash
source .venv/bin/activate
python -m pytest
python samples/generate_gap_samples.py
for name in aligned diagonal perpendicular parallel obstacle ambiguous; do
  python -m raster2vector "samples/gap_cases/$name.png" \
    --gap-tolerance 12 --geometry-tolerance 2 \
    --output "debug_output/gap_cases/$name/output.svg" \
    --debug-dir "debug_output/gap_cases/$name" --debug-svg
done
python -m raster2vector samples/rosace.png --threshold 160 \
  --gap-tolerance 3 --geometry-tolerance 2 \
  --output debug_output/rosace_gaps/output.svg \
  --debug-dir debug_output/rosace_gaps --debug-svg
```

À 12 pixels, `aligned` et `diagonal` produisent chacun **un raccord et un seul
segment SVG**. Les quatre contre-exemples ne produisent aucun raccord.
Les rendus des cas aligné, diagonal, perpendiculaire et ambigu ont été vérifiés
dans Inkscape. À 3 pixels, la rosace produit **zéro raccord** et garde les
mêmes primitives que l'étape 9. Ses irrégularités aux jonctions attendent le
nettoyage topologique prévu à l'étape 11.

## Validation de l'étape 10

- Les tests passent et le SVG aligné contient un seul chemin continu.
- Les ponts magenta apparaissent dans les deux coupures compatibles.
- Les traits perpendiculaires, parallèles voisins, ambigus ou bloqués restent séparés.
- Réduire la tolérance sous la distance entre endpoints empêche le raccord.
- Sans `--gap-tolerance`, la géométrie de l'étape 9 est conservée.
- Le raster et le squelette d'origine restent disponibles pour comparer.

Étape 10 validée ; l'affinage géométrique des intersections est maintenant disponible.

## Nettoyage des intersections (étape 11)

`refine_junctions(graph, geometry, tolerance_px=2)` retourne un `JunctionResult`
avec la géométrie affinée, les `adjustments` acceptés, les `continuations`
entre segments presque colinéaires et les `rejected_node_ids` conservés.
La CLI l'applique après la reconstruction des primitives, en utilisant
`--geometry-tolerance`. `--no-junction-cleanup` permet de retrouver le résultat
de l'étape 10.

Chaque nœud `junction` reçoit une proposition de point commun estimée à partir
des supports des branches : droites ajustées, tangentes d'arc ou de Bézier,
et estimation locale pour les polylignes. Les segments et portées de moins
de **6 pixels** ne guident pas cette estimation ; les supports locaux sont
mesurés sur au plus **12 pixels de chaîne**. Si les supports sont presque
parallèles (rapport des valeurs singulières inférieur à **0.05**), le moteur
utilise le barycentre de la zone plutôt qu'une intersection instable lointaine.
La proposition doit rester à moins de la tolérance de chaque contact source.
Ces constantes sont nommées dans `junctions.py`.

Deux segments situés de part et d'autre de la jonction peuvent former une
continuation si l'écart de direction est au plus **5°**, si chaque branche
a un seul partenaire, et si l'ajustement commun de leurs sources respecte
la tolérance. Le support commun participe à l'estimation du nœud. Les
continuations sont également vérifiées après ajustement ; leurs branches
gardent des objets SVG distincts partageant le contact, pour conserver les
identifiants et la structure du graphe. Cette étape ne transforme pas les
croisements en passages indépendants.

Les branches incidentes sont ensuite ajustées ensemble :

- Les segments partagent exactement le point commun, avec contrôle des
  distances aux segments **finis** ; un segment ne peut pas être réduit à zéro.
- Les arcs sont réajustés avec leurs deux contacts imposés : leur centre est
  recherché sur la médiatrice de la corde. L'erreur à l'arc fini, l'ordre des
  sources et le sens de parcours sont contrôlés. `JoinedArc` reste un
  `CircularArc`, avec des contacts communs exacts dans l'export SVG `A`.
- Les polylignes changent uniquement leurs contacts extrêmes ; leurs indices
  sources et distances aux segments correspondants sont conservés/recalculés.
- Les Bézier déplacent les contacts et leurs poignées voisines, puis leur
  borne continue est recalculée, y compris entre les points sources.
- Les chaînes brutes conservent leurs sources dans `source_points`, avec
  une borne de déplacement des contacts valable aussi entre les points.

Pour les segments et arcs, la nouvelle erreur doit respecter la tolérance
géométrique. Pour les polylignes et Bézier, elle ne doit pas dépasser le plus
grand de cette tolérance et de la mesure précédente, afin de conserver les
tolérances distinctes déjà employées. La borne de déplacement des contacts
reste toujours celle de la jonction. La continuité des Bézier reste C0,
sans imposer une tangence entre branches.

Si une branche ne peut pas respecter ces critères, le moteur tente le
barycentre source une fois, puis conserve les contacts d'origine de cette
jonction. La validation est atomique au niveau du nœud : toutes ses branches
partagent le nouveau point, ou aucun de ses contacts n'est remplacé. Un
cercle fermé incident conserve sa primitive et empêche cet ajustement.
Les zones de contacts trop étendues sont refusées.

Le **graphe reste inchangé** : aucune branche courte, composante ou cycle
n'est supprimé ou fusionné. Les points et zones sources restent disponibles.
Des petits détours réellement présents dans le squelette peuvent donc
subsister. Ce choix protège les détails du dessin plutôt que de supposer
qu'ils sont parasites. La géométrie affinée ne constitue pas une nouvelle
extraction du graphe ; les positions de `GraphNode` restent celles du raster.

Les fichiers `geometry_before.png` et `overlay_before.svg` montrent le
résultat antérieur à cet affinage. `geometry.png`, `output.svg` et
`overlay.svg` montrent le résultat final. `junctions.png` marque les points
acceptés en vert, les refus en rouge et les déplacements en magenta.
Avec `--no-junction-cleanup`, ces nouveaux fichiers de contrôle ne sont pas
écrits. La console indique les jonctions acceptées/conservées et les
continuations reconnues.

Commandes de contrôle depuis la racine du projet (macOS/Linux) :

```bash
source .venv/bin/activate
python -m pytest
python samples/generate_junction_samples.py
for name in T cross double_T oblique arc_contacts loop_contacts; do
  python -m raster2vector "samples/junction_cases/$name.png" \
    --geometry-tolerance 2 \
    --output "debug_output/junction_cases/$name/output.svg" \
    --debug-dir "debug_output/junction_cases/$name" --debug-svg
done
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 2 --output debug_output/rosace_junctions/output.svg \
  --debug-dir debug_output/rosace_junctions --debug-svg
```

Le T produit un contact commun et une continuation, la croix deux
continuations et le double T deux contacts communs. `arc_contacts` conserve
un vrai arc et deux segments partageant un contact ; `loop_contacts`
conserve sa boucle. L'exemple oblique comporte deux jonctions raster et reste
inchangé à 2 pixels : il constitue un cas de refus conservateur.

À 2 pixels, la rosace conserve ses **28 nœuds et 44 branches**, avec
**17 jonctions acceptées, 11 conservées et 3 continuités presque colinéaires**
au centre. Les types restent 26 segments, 15 arcs, 2 chemins Bézier et
1 polyligne. Des irrégularités locales restent visibles près des refus.
Les SVG de la rosace, du T et de l'arc incident ont été rendus dans Inkscape.

Pour examiner une tolérance plus souple, sans remplacer le résultat à 2 pixels :

```bash
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 3 --output debug_output/rosace_junctions_3px/output.svg \
  --debug-dir debug_output/rosace_junctions_3px --debug-svg
```

À 3 pixels, la rosace produit 20 jonctions acceptées et 8 conservées,
26 segments et 18 arcs. Ce réglage modifie aussi la reconnaissance des
primitives ; comparer avec son propre `overlay_before.svg`.

## Validation de l'étape 11

- Les tests passent et les SVG s'ouvrent dans Inkscape.
- Les T/croix partagent leurs contacts, sans supprimer leurs branches.
- Les arcs incidents restent de vraies commandes SVG `A` avec le bon sens.
- Les erreurs recalculées restent dans les limites documentées.
- Les cycles et sources sont conservés ; les jonctions refusées restent inchangées.
- Sur la rosace, comparer le centre et les pointes dans les fichiers avant/après,
  en acceptant les petits détours conservés sous la tolérance choisie.

Étape 11 validée ; la CLI et la configuration sont maintenant consolidées.

## CLI et configuration (étape 12)

Les deux points d'entrée `raster2vector` et `python -m raster2vector` utilisent
le même moteur. `--help` décrit les paramètres, les unités, les priorités et
les codes de sortie, avec des exemples. `--version` lit la version du package
installé. Ces deux commandes ne chargent ni image ni configuration.

```bash
raster2vector --help
python -m raster2vector --help
raster2vector --version
```

Les valeurs par défaut des étapes précédentes sont conservées : seuil 128,
tolérances géométrique et polyligne de 2 pixels, raccordement désactivé,
nettoyage des jonctions activé. `--output` et `--debug-dir` restent optionnels.
Seuls les fichiers explicitement demandés sont écrits.

### Seuillage automatique

`--threshold auto` applique Otsu sur les niveaux de gris 8 bits après gestion
de l'alpha et des images 16 bits. Il n'ajoute aucun lissage, aucune morphologie
et aucun apprentissage. Le masque emploie ensuite le même seuil inclusif que
le mode manuel, avec la polarité définie par `--invert`.
La console affiche le seuil numérique réellement appliqué.

Otsu sépare les classes d'intensité globales ; il ne corrige pas un éclairage
irrégulier et n'optimise pas la topologie du squelette. Une petite variation
du seuil peut modifier les pixels retenus et les branches. Sur la rosace,
Otsu choisit **164**, avec 29 nœuds et 45 branches ; le seuil manuel validé
**160** conserve les 28 nœuds et 44 branches de l'étape 11.
Un fond uniformément coloré donne un masque vide avec la polarité automatique.
Un trait très pâle sur fond clair est également couvert par les tests.

### Configuration TOML

`samples/settings.toml` fournit un exemple complet, avec seuil automatique :

```toml
[vectorization]
threshold = "auto"        # ou un entier de 0 à 255
invert = "auto"           # auto, yes ou no
geometry_tolerance = 2.0
polyline_tolerance = 2.0
gap_tolerance = 0.0
spur_tolerance = 0.0      # élagage des micro-barbes
min_component_size = 0
stroke_width = 1.0
junction_cleanup = true
global_circles = false
global_arcs = false
```

Le fichier doit contenir uniquement la table `[vectorization]`. Les clés
peuvent être omises : elles prennent leur valeur par défaut. Les paramètres
inconnus, fautes de syntaxe, mauvais types, nombres non finis et valeurs
hors limites sont refusés. Le fichier est validé avant application des options
CLI ; une configuration invalide n'est pas rendue valide par une surcharge.
Les noms TOML emploient `_`, les options CLI `-`.

Priorité : **option explicitement fournie > fichier TOML > valeur par défaut**.
Une valeur CLI égale au défaut reste une surcharge explicite ; par exemple
`--gap-tolerance 0` désactive un raccordement demandé par le fichier.
`--junction-cleanup` et `--no-junction-cleanup` sont mutuellement exclusifs.

Les chemins d'entrée, de sortie et de debug restent dans la commande ; ils
ne sont pas stockés dans le fichier. Les chemins relatifs se résolvent depuis
le dossier courant, y compris celui passé à `--config`. Un fichier de sortie
ne peut pas écraser l'image ni la configuration source. TOML est lu par
`tomllib`, inclus dans Python 3.11+ : aucune dépendance n'a été ajoutée.

L'API expose `VectorizationConfig`, `load_config` et `automatic_threshold` :

```python
from pathlib import Path
from raster2vector import load_config

settings = load_config(Path("samples/settings.toml"))
settings = settings.with_overrides(threshold=160, geometry_tolerance=2.0)
```

### Contrôle de l'étape

Depuis la racine du projet, sur macOS/Linux :

```bash
source .venv/bin/activate
python -m pytest
raster2vector --help
raster2vector --version
raster2vector samples/rosace.png --config samples/settings.toml \
  --output debug_output/rosace_cli/output.svg \
  --debug-dir debug_output/rosace_cli --debug-svg
python -m raster2vector samples/rosace.png --config samples/settings.toml \
  --threshold 160 --output debug_output/rosace_cli_manual/output.svg \
  --debug-dir debug_output/rosace_cli_manual --debug-svg
```

Le premier résultat emploie Otsu et affiche `Seuil : 164 (auto, Otsu)`.
Le second surcharge le fichier avec 160 et reproduit le SVG validé de
l'étape 11. Les debug habituels restent disponibles pour comparer.
Le SVG de comparaison automatique a été rendu et inspecté dans Inkscape.

Sur Windows PowerShell, activer `.venv\\Scripts\\Activate.ps1`, puis utiliser
les mêmes commandes sur une seule ligne.

Critères de validation : aide utilisable sans image, configuration strictement
validée, priorité des options explicites respectée, seuil appliqué affiché,
SVG et fichiers de contrôle inspectables, et tests au vert.

Attendre la validation avant l'étape 13 : analyser l'architecture d'intégration
Inkscape, sans implémenter l'extension avant une autorisation distincte.

## Reconstruction globale — première étape : cercle complet

Le découpage aux intersections est conservé dans le graphe, mais plusieurs
branches peuvent désormais former une primitive SVG commune. Cette étape
intervient **après** la reconstruction locale et l'affinage des jonctions.
L'option `--global-circles` est volontairement explicite ; elle n'est pas une
clé supplémentaire du fichier TOML. Sans cette option, le résultat de
l'étape 12 est conservé.

Les branches ouvertes sont ajustées à un cercle depuis leurs points sources,
indépendamment de leur classification locale en arc, polyligne ou Bézier.
Une branche ne guide la recherche que si son balayage net atteint **30°** et
si son erreur respecte la tolérance. Les centres et rayons sont comparés avec
une limite de recherche égale à **4 fois la tolérance** ; chaque ajout exige
un nouvel ajustement commun accepté sur toutes les sources. Cette limite
sert à proposer des regroupements, elle n'autorise pas une erreur plus grande.

Les petits connecteurs de longueur au plus 4 fois la tolérance peuvent être
inclus uniquement s'ils existent déjà dans le graphe et sont compatibles
avec le même cercle. Les branches pendantes sont écartées du candidat ; un
cycle doit avoir exactement deux branches candidates à chacun de ses nœuds.
Une ramification ambiguë est refusée plutôt que de choisir arbitrairement
un parcours.

Le cycle ordonné est réajusté et doit faire **un tour complet** autour de son
centre. Ses retours angulaires ne peuvent pas dépasser la tolérance exprimée
en longueur d'arc. L'erreur radiale est contrôlée sur toute la chaîne, y
compris entre les points : la distance minimale du centre à chaque segment
source contrôle notamment la flèche de sa corde. Les zones raster des nœuds
traversés doivent également respecter la tolérance. Un arc ouvert ou des
portions déconnectées ne deviennent donc pas un cercle, même si leurs angles
couvrent ensemble 360°. Les raccords explicitement autorisés à l'étape 10
peuvent en revanche faire partie du graphe fourni.

Les propositions sont déterministes ; les sources d'une branche ne sont
consommées qu'une fois. En cas de regroupements concurrents, le candidat
couvrant le plus de points est considéré en premier. Cette recherche est
conservatrice et ne garantit pas de reconnaître tous les cercles, notamment
si chaque fragment est inférieur à 30° ou si les intersections sont ambiguës.
Les cercles déjà reconnus sur une seule boucle restent inchangés.

L'API `reconstruct_global_circles(graph, geometry, tolerance_px=3)` retourne
un `GlobalCircleResult` avec sa géométrie et ses `circles` acceptés. Chaque
`GlobalCircle` est aussi un `Circle`, avec les points de son cycle source et
`source_edge_ids` listant toutes les branches regroupées. Le graphe reste
inchangé. La nouvelle géométrie peut avoir moins de primitives que de
branches ; elle ne doit pas être repassée à `refine_junctions`, qui attend
la correspondance locale d'une primitive par branche.

Le SVG contient un vrai `<circle id="global-circle-…">`, avec l'attribut
`data-source-edges` pour sa provenance. Les branches remplacées n'y sont
pas exportées une seconde fois. Les autres primitives restent inchangées.
Leur raccord exact au nouveau cercle n'est pas imposé à cette étape ; une
petite différence locale peut subsister aux contacts. Le debug montre le
cercle global en violet dans `geometry.png`, et les branches précédentes
dans `geometry_local.png`. Avec `--debug-svg`, comparer `overlay_local.svg`
à `overlay.svg`.

Commande de contrôle sur la rosace :

```bash
source .venv/bin/activate
python -m pytest
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 3 --global-circles \
  --output debug_output/rosace_global/output.svg \
  --debug-dir debug_output/rosace_global --debug-svg
```

Le cercle extérieur remplace **11 branches : 6 arcs et 5 petits connecteurs**.
Son rayon est d'environ **267.94 pixels**, avec une erreur maximale de
**2.75 pixels**. Le SVG contient alors 1 cercle, 12 arcs internes et
21 segments. Ces derniers incluent les petits raccords internes, conservés
pour la prochaine étape. Une tolérance de 2 pixels refuse ce cercle commun ;
3 pixels permet de l'accepter sans dépasser l'erreur demandée.

Les tests couvrent les cercles traversés par plusieurs traits, les branches
non consommées, les arcs séparés, les ellipses, les zones de jonction
incompatibles, l'erreur entre points, deux cercles distincts, la provenance
SVG et le pipeline CLI sur une image raster synthétique. Les SVG de la
rosace et des comparaisons ont été rendus et inspectés dans Inkscape.

Critères de validation : le contour extérieur est un seul élément `<circle>`,
les fragments remplacés ne sont plus doublonnés, les traits internes sont
conservés et la superposition reste dans la tolérance annoncée.

Cette étape est validée ; les grands arcs internes sont traités ci-dessous.

## Reconstruction globale — grands arcs continus

L'option CLI `--global-arcs` recherche les arcs circulaires répartis entre
plusieurs branches du graphe. Elle active d'abord les cercles globaux, puis
regroupe les arcs restants après l'affinage local des jonctions. Elle ne
constitue pas une nouvelle clé TOML. Sans cette option, les étapes précédentes
gardent leur comportement.

Les fragments ouverts couvrant au moins 30° proposent un support circulaire.
Des centres et rayons proches sont comparés avec une limite de recherche de
8 fois la tolérance : les ajustements de fragments courts peuvent avoir des
centres assez différents. Un ajustement commun doit toutefois respecter la
tolérance réelle sur chaque branche. Les chemins reliant les fragments sont
recherchés uniquement parmi les branches existantes ; leur présélection peut
atteindre 2 fois la tolérance, mais leur ajustement final doit respecter la
tolérance demandée. Aucun intervalle déconnecté n'est comblé par cette étape.

Les composantes connectées compatibles fournissent le balayage de l'arc.
Les sources et les zones raster des jonctions doivent rester sous tolérance.
La distance à l'arc fini est contrôlée sur chaque segment source, y compris
entre les points et près des extrémités. Le résultat est un véritable chemin
SVG `M … A …`, pouvant traverser plusieurs intersections sans être découpé.
Le sens d'export est normalisé ; les deux orientations des branches sources
produisent la même courbe.

En cas de concurrence, la longueur géométrique `rayon × balayage` détermine
la priorité, puis l'erreur et les identifiants assurent un résultat déterministe.
Les petites irrégularités et boucles compatibles peuvent être absorbées dans
la courbe idéale. Le graphe original, ses branches et ses cycles sont conservés
pour l'analyse et la provenance ; la géométrie exportée peut donc avoir une
topologie simplifiée à l'intérieur de la tolérance.

Chaque branche remplacée appartient à une seule primitive exportée. Un petit
connecteur d'intersection peut néanmoins étayer plusieurs arcs compatibles.
`data-source-edges` indique les branches consommées exclusivement, tandis que
`data-support-edges` conserve toutes les branches qui ont servi à ajuster l'arc.
L'API `reconstruct_global_arcs(graph, geometry, tolerance_px=3)` retourne un
`GlobalArcResult` contenant la géométrie et ses `arcs`. `GlobalArc` dérive de
`CircularArc` ; ses `source_points` réunissent les sources de plusieurs branches
et ne constituent pas un chemin global ordonné. L'ordre de chaque branche
reste disponible dans le graphe.

La recherche nécessite au moins deux fragments guides compatibles et accepte
des balayages de 30° à 330°. Elle reste heuristique et ne garantit pas le
regroupement maximal de tout dessin. Elle n'impose ni symétrie parfaite, ni
égalité des rayons, ni coïncidence exacte des contacts entre primitives globales.
Les branches dépassant la tolérance restent exportées séparément. Les grandes
droites et les Bézier multibranches ne sont pas regroupés par cette option.

Commandes de contrôle sur la rosace, depuis la racine du projet :

```bash
source .venv/bin/activate
python -m pytest -q
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 3 --global-arcs \
  --output debug_output/rosace_global_arcs_3px/output.svg \
  --debug-dir debug_output/rosace_global_arcs_3px --debug-svg
python -m raster2vector samples/rosace.png --threshold 160 \
  --geometry-tolerance 5 --global-arcs \
  --output debug_output/rosace_global_arcs_5px/output.svg \
  --debug-dir debug_output/rosace_global_arcs_5px --debug-svg
```

À **3 pixels**, le résultat contient **1 cercle, 6 grands arcs et 5 petits
segments** : ces derniers ne peuvent pas encore être absorbés sous cette
tolérance. À **5 pixels**, les 44 branches sources deviennent **7 objets SVG :
1 cercle et 6 grands arcs**, sans fragments résiduels. Les arcs internes
couvrent environ **119 à 120°** et leur erreur maximale atteint **4.31 pixels**,
sous la limite de 5 explicitement choisie. Le cercle extérieur garde son
erreur d'environ 2.75 pixels. Les écarts concernent les axes reconstruits et
les zones de jonction du squelette, pas les contours du trait épais original.

Comparer `geometry_local.png` à `geometry.png` et `overlay_local.svg` à
`overlay.svg` dans chaque dossier.

---

## Élagage topologique des micro-barbes (`spur_tolerance`)

Lors de la squelettisation d'images comportant des traits épais ou des aspérités
de bordure, des petites branches parasites pendantes ("barbes" ou *spurs*)
apparaissent souvent autour des intersections et le long des contours. Ces épines
créent artificiellement des nœuds de type `junction` et fragmentent des traits
qui devraient être continus.

L'option `--spur-tolerance PIXELS` (fonction publique `prune_spurs` dans
`raster2vector.graph`) permet d'éliminer automatiquement ces micro-branches :

1. **Identification des barbes candidates** : arêtes non fermées reliant un nœud
   `endpoint` à une `junction`, dont la longueur curviligne $\sum \|\Delta p\| \le \text{spur\_tolerance}$.
2. **Règle de préservation topologique** : pour chaque jonction, le nombre de barbes
   élaguées est plafonné pour que son degré résiduel reste au moins égal à 2. Les barbes
   les plus courtes sont éliminées en priorité (départagées de manière déterministe).
   Une étoile de micro-branches ne disparaît donc jamais dans le vide.
3. **Dissolution topologique des fausses jonctions** : toute jonction dont le degré
   résiduel devient exactement 2 est dissoute. Ses deux branches restantes sont fusionnées
   en une branche unique continue. Si les points de contact diffèrent, un pont continu
   8-connexe sans doublon est tracé à travers les `source_pixels` de la jonction dissoute.
4. **Conservation des cycles fermés** : si une composante cyclique entière de jonctions
   est dissoute, un nœud d'ancre `"loop"` est conservé pour fermer proprement la boucle.
5. **Ré-indexation déterministe** : nœuds ré-indexés de `0..N-1` et arêtes de `0..E-1`.

---

## Dé-biaisage des jonctions et continuité $G^1$ des courbes Bézier

### Dé-biaisage adaptatif des jonctions ("Zhang-Suen")

À l'approche d'une jonction épaisse (croix, T, bifurcation), l'algorithme de
squelettisation dévie systématiquement les derniers pixels de chaque branche
vers le centre de gravité de la composante (effet "patte d'oie"). Cette déviation
locale artificielle augmentait l'erreur maximale et empêchait la détection de
segments de droite ou d'arcs de cercle par ailleurs parfaits.

Le moteur applique désormais un rognage adaptatif temporaire des derniers pixels
adjacents à une jonction (`_trimmed_junction_slice`) lors du calcul d'ajustement
géométrique. Une fois le segment ou l'arc reconnu, ses extrémités sont prolongées
proprement jusqu'aux contacts de la jonction sans subir la distorsion de squelette.

### Continuité tangentielle $G^1$ aux raccords internes de Bézier

Pour les tracés organiques et courbes fluides subdivisés en plusieurs cubiques
consécutives (`fit_piecewise_bezier`), les tangentes aux points de raccord
internes sont harmonisées via `_smooth_internal_joints` :
- Les poignées de contrôle encadrant le raccord sont projetées pour devenir
  colinéaires et de sens opposés.
- L'alignement n'est validé que si la déformation résultante respecte strictement
  la borne d'erreur continue de Bernstein sous la tolérance géométrique choisie.
- Les angles vifs détectés (`corner_indices`) sont systématiquement préservés
  sans lissage forcé.

---

## Reconnaissance et simplification des polygones fermés

Les boucles fermées non circulaires (rectangles, triangles, polygones, contours
mixtes) sont désormais simplifiées analytiquement sous forme de `Polyline(is_loop=True)`
plutôt que de retomber en `RawPolyline` pixelisée brute :
- Un sommet de courbure maximale est sélectionné de façon déterministe comme point
  d'ancrage de départ.
- L'algorithme de Douglas-Peucker circulaire est appliqué le long du périmètre fermé.
- L'export SVG génère un tracé compact `M … L … Z` parfaitement fermé.

---

## Extension Inkscape v2

L'extension officielle située dans `extensions/inkscape` dispose d'une ergonomie
complètement repensée pour Inkscape 1.4+ :

- **Menu de préréglages (Presets)** : profils optimisés prêts à l'emploi (`Dessin technique`,
  `Rosace`, `Tracé libre`, `Motif fin`, `Croquis scanné`, et mode `Personnalisé`).
- **Fonctionnement dans Inkscape** : les préréglages appliquent directement leurs valeurs
  optimales lors du clic sur **Appliquer** (l'interface GTK d'Inkscape étant statique,
  les curseurs à l'écran restent disponibles pour le mode Personnalisé).
- **Organisation en 3 étapes** : 1. Extraction du trait (Image), 2. Nettoyage &
  Continuité du squelette (avec `spur_tolerance` et `gap_tolerance`), 3. Reconnaissance
  des formes & Rendu SVG.
- **Étiquetage dynamique** : le groupe inséré dans Inkscape indique le profil et les
  tolérances réels (ex : `Vectolaz — Rosace (tol=5.0px, barbes=3.0px)`).
- **Aide et dépannage intégrés** : guide par type de motif et mémo des réglages fréquents.

Voir [le guide d'installation et d'utilisation de l'extension](extensions/inkscape/README.md)
et [l'architecture d'intégration](docs/inkscape-integration.md).

---

## Bilan de validation

La suite de tests automatisée compte aujourd'hui **411 tests réussis**, 0 avertissement
(`pytest -W error`), testée sous Python 3.11 à Python 3.14 sur macOS, Linux et Windows.
