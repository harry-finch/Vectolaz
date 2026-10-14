# Courbes et boucles du motif complexe

Correction du 9 octobre 2026, sur `samples/motif_complexe.png`, PNG RGBA
fourni par l'utilisateur. L'alpha est composé sur blanc par le moteur
existant ; aucun changement de seuil, de transparence ou de tolérance
n'est nécessaire pour cette correction.

## Cause et changement

Le choix précédent refusait tout Bézier si une branche avait un angle de
45° ou plus dans sa polyligne simplifiée. Il exigeait aussi moins de points,
poignées comprises, que cette polyligne, simplifiée à 2 pixels indépendamment
de la tolérance géométrique. Toutes les boucles non circulaires restaient
brutes. Ces trois règles empêchaient la reconstruction courbe de ce motif.

Les angles de 45° ou plus servent désormais de contacts imposés : les
portions de part et d'autre sont ajustées séparément à des cubiques, sans
fusionner à travers le sommet. Les portions essentiellement droites entre
contacts restent des polylignes. Les courbes doivent réduire le nombre de
points par rapport à la source raster dense, poignées comprises ; elles
ne doivent plus battre une polyligne utilisant une autre tolérance.
Les segments droits et arcs circulaires restent prioritaires lorsqu'ils
décrivent toute la branche dans la tolérance.

Une boucle explicitement fermée, sans points intérieurs répétés, peut devenir
un `BezierPath(is_loop=True)`. Des coupures initiales en quatre portions
évitent un ajustement à extrémités confondues ; les angles détectés ajoutent
leurs propres coupures. Le SVG contient des commandes `C` et une fermeture
`Z`, avec des contacts identiques au début et à la fin. Les boucles dont les
contacts raster diffèrent, et les parcours retracés, gardent leur forme brute :
aucune fermeture manquante n'est inventée.

`fit_piecewise_bezier` conserve les indices des points relativement à la
branche entière. Le contrôle continu de Bernstein est appliqué à chaque
morceau, et l'affinage des jonctions remesure la même borne après déplacement.
Le graphe et les identifiants des branches restent inchangés. La continuité
est C0 ; la continuité des tangentes entre toutes les cubiques n'est pas
imposée. La classification des angles est une heuristique et n'assure pas
la reconnaissance de chaque intention géométrique.

## Résultats et commandes

Les trois résultats emploient le seuil manuel 160, la tolérance de polyligne
par défaut 2 pixels, l'affinage des jonctions et `--global-arcs`.

| Résultat | Segments | Polylignes | Boucles brutes | Arcs | Chemins Bézier | Cubiques |
|---|---:|---:|---:|---:|---:|---:|
| Avant, 1 px | 322 | 614 | 22 | 194 | 0 | 0 |
| Après, 1 px | 322 | 0 | 0 | 194 | 636 | 2242 |
| Après, 2 px | 349 | 15 | 0 | 190 | 598 | 1768 |

Les 820 nœuds et 1154 branches du graphe sont conservés. Les arcs globaux
regroupent 4 branches en 2 arcs ; aucune hypothèse nouvelle de symétrie ou
de regroupement Bézier multibranches n'est ajoutée.

```bash
source .venv/bin/activate
python -m pytest -q
python -m raster2vector samples/motif_complexe.png --threshold 160 \
  --geometry-tolerance 1 --global-arcs \
  --output debug_output/motif_curves_1px/output.svg \
  --debug-dir debug_output/motif_curves_1px --debug-svg
python -m raster2vector samples/motif_complexe.png --threshold 160 \
  --geometry-tolerance 2 --global-arcs \
  --output debug_output/motif_curves_2px/output.svg \
  --debug-dir debug_output/motif_curves_2px --debug-svg
```

Le résultat historique se trouve dans `debug_output/motif_before`. La
comparaison détaillée `debug_output/motif_curves_1px/comparaison.svg` et son
PNG montrent une boucle centrale avant/après, sans le fond raster ;
`overlay.svg` sert à vérifier la superposition avec l'image.

## Extension Inkscape

L'adaptateur existant appelle déjà le moteur depuis le venv du projet.
Comme ce moteur est installé en mode éditable avec `pip install -e`, la
correction s'applique au prochain lancement de l'extension : **pas besoin
de recopier l'extension ni de modifier le Python d'Inkscape**.

Sélectionner le PNG original, saisir 160 en seuil manuel et 1 ou 2 en
tolérance géométrique, puis appliquer. Masquer l'ancien résultat pour
comparer les courbes nouvelles. Si le moteur est installé ailleurs en mode
non éditable, le réinstaller depuis cette version du projet.

La suite complète avec l'inkex réel comporte **365 tests réussis**. Les
nouveaux contrôles couvrent la conservation d'un angle entre portions
courbes, les deux orientations, les indices sources, les bornes entre
points, les ellipses fermées, `C`/`Z`, les contacts distincts et la fermeture
non inventée. Le test historique du L raster reste vert. La rosace passée
par l'adaptateur conserve **1 cercle et 6 arcs**. Le détail du motif a été
rendu et inspecté avec Inkscape.

Validation attendue : absence des escaliers sur les boucles centrales,
portions arrondies éditables en Bézier, sommets du motif conservés et
superposition fidèle à la précision choisie.
