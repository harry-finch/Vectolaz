# Étape 13 — proposition d'intégration Inkscape

Statut : architecture validée le 9 octobre 2026, première implémentation
disponible dans `extensions/inkscape`. Voir [le guide de test](../extensions/inkscape/README.md).
Le texte ci-dessous décrit la proposition validée ; les fichiers envisagés
ont été créés. L'installation utilisateur et la validation du menu restent
à effectuer. La reconstruction globale des cercles et grands arcs est validée.

## Architecture proposée

Une extension d'effet reçoit le document courant, traite exactement une image
PNG sélectionnée et ajoute un groupe de vecteurs juste au-dessus de cette
image, dans le même parent. L'image est conservée pour comparaison. Le moteur
continue à travailler dans les pixels natifs du PNG, indépendamment de sa
taille d'affichage dans Inkscape.

```text
Document courant et sélection
       ↓
Adaptateur inkex : sélection, PNG, placement
       ↓ PNG temporaire + paramètres explicites
Python du moteur → python -m raster2vector → SVG temporaire
       ↓
Adaptateur inkex : groupe vectoriel, identifiants, transformation
       ↓
Document modifié renvoyé à Inkscape
```

Fichiers envisagés, à créer seulement après validation :

```text
extensions/inkscape/
    vectolaz.inx           # menu et paramètres
    vectolaz.py            # EffectExtension, orchestration
    image_source.py        # PNG embarqué ou lié
    placement.py           # pixels → unités locales du document
    README.md              # installation et Python du moteur
tests/inkscape/             # tests de l'adaptateur dans son environnement
```

Ces fichiers restent séparés de `src/raster2vector`. Aucun import d'inkex
dans le moteur, aucun import direct NumPy/OpenCV dans l'adaptateur (inkex
a ses propres dépendances). La CLI et
son SVG existants suffisent au premier échange ; pas de nouveau protocole
ni de refonte du pipeline nécessaires.

## Réception et retour du document

Le descripteur `.inx` déclare un effet, ses paramètres et son script Python.
`inkex.EffectExtension` charge le document transmis par Inkscape et expose
`self.svg` ainsi que `self.svg.selection`. La sélection est contrôlée dans
le script : exactement un élément `inkex.Image`, sans traitement implicite
des autres images. Le document transmis inclut les modifications non encore
enregistrées : ne pas relire le SVG sauvegardé sur disque.

Le SVG modifié est renvoyé par inkex ; stdout est réservé à cette sortie.
Les logs du moteur sont capturés, les erreurs utiles passent par
`inkex.AbortExtension`/stderr. Le descripteur désactive l'aperçu automatique
pour éviter de relancer le traitement à chaque changement de paramètre.
Sources : [effets et descripteur INX](https://inkscape.gitlab.io/extensions/documentation/authors/inx-overview.html),
[EffectExtension](https://inkscape.gitlab.io/extensions/documentation/source/inkex.extensions.html).

## PNG sélectionné

Lire l'attribut `href`, puis `xlink:href` pour les documents anciens.
Ne pas utiliser la propriété générique `element.href` pour obtenir ces octets :
la version locale d'inkex l'utilise pour résoudre des références d'éléments.

Pour `data:image/png;base64,…`, décoder les octets et vérifier la signature
PNG avant de les écrire dans un dossier temporaire. Une erreur de base64
ou de format provoque un message clair avant toute insertion.

Pour un lien externe, prendre en charge un chemin absolu, un chemin relatif
et une URI locale `file:` décodée correctement, espaces et accents compris.
Résoudre les liens relatifs depuis le dossier du document original avec
`document_path()`/`absolute_href(..., default=None)`, après conversion de
l'URI en chemin local. Le fichier temporaire fourni à l'extension ne sert
pas de base de résolution. Un document jamais enregistré doit être enregistré
ou son image embarquée si le lien est relatif. Les URI distantes et formats
autres que PNG sont hors du premier périmètre.

Le même dossier temporaire contient `input.png` et `output.svg`, puis est
supprimé à la fin. Le debug persistant n'est écrit que dans un dossier
explicitement choisi. Le moteur garde son traitement actuel de la transparence.
Sources : [API des chemins](https://inkscape.gitlab.io/extensions/documentation/source/inkex.base.html),
[extraction d'image Inkscape](https://inkscape.gitlab.io/inkscape/doxygen-extensions/image__extract_8py_source.html).

## Interpréteurs et dépendances

Constat local : `Info.plist` indique **Inkscape 1.4.3** et le paquet contient
`Contents/Frameworks/Python.framework/Versions/3.10/bin/python3.10`.
Le projet exige **Python ≥ 3.11** et son environnement `.venv` utilise
Python 3.14.3. La présence du Python embarqué ne prouve pas l'interpréteur
effectivement configuré par l'utilisateur ; il faudra vérifier celui-ci lors
de l'installation. Un import direct du moteur dans le Python 3.10 serait
incompatible avec son contrat actuel.

Décision proposée : laisser inkex dans le Python de l'extension et appeler
un exécutable Python du moteur configuré par chemin absolu. Sur macOS/Linux,
ce sera `.venv/bin/python`, sur Windows `.venv/Scripts/python.exe`.
Utiliser `subprocess.run` avec une liste d'arguments, sans shell, un timeout
documenté et capture de stdout/stderr. Contrôler le code de retour, l'existence
du SVG et son contenu avant d'insérer le groupe. Aucun changement global du
Python d'Inkscape ni installation automatique de dépendances n'est proposé.

Paramètres de l'interface : seuil automatique ou manuel, tolérance géométrique
en pixels sources, tolérance de raccordement, suppression des petites
composantes, épaisseur du trait en pixels sources et reconstruction globale
des cercles/arcs. Les autres réglages conservent leurs valeurs actuelles.
Le chemin Python appartient à un onglet d'installation. La reconstruction
globale est une option visible, sans seuil augmenté implicitement.

## Placement des vecteurs

Soient `W,H` les dimensions natives du PNG et `x,y,w,h` le rectangle de
l'image en unités locales de son parent, avant son attribut `transform`.
Les longueurs doivent être résolues dans ce repère, pas converties au moyen
de la boîte englobante tournée. Le `viewBox` du document détermine les unités ;
aucun facteur DPI arbitraire ne doit être ajouté.

Pour `preserveAspectRatio="none"` : `sx=w/W`, `sy=h/H`, `dx=x`, `dy=y`.
Pour `xMidYMid meet`, valeur par défaut : `s=min(w/W,h/H)`,
`sx=sy=s`, `dx=x+(w-sW)/2`, `dy=y+(h-sH)/2`.
Les autres alignements min/mid/max règlent ces décalages avec les facteurs
0, 1/2 ou 1. Le mode `slice` demanderait aussi de reproduire le recadrage ;
il est refusé explicitement dans la première version.

Notons `A` la transformation propre de l'image et
`B=translate(dx,dy) × scale(sx,sy)` le placement des pixels. Le groupe ajouté
comme frère de l'image porte **`A × B`**, puis contient le groupe `vectors`
exporté par le moteur, avec son **`translate(0.5,0.5)` conservé une seule fois**.
Un point du squelette `(u,v)` est donc affiché en
`A × B × (u+0.5,v+0.5,1)`. Les transformations des parents s'appliquent déjà
aux deux frères ; les recopier sur le groupe produirait un double déplacement.

Exemple : PNG 200×100 dans un rectangle `(10,20,300,300)`, `xMidYMid meet`.
L'échelle vaut 1.5 et le placement est `(10,95)`, avec 75 unités de marge
verticale de chaque côté. Le pixel source `(0,0)` apparaît en `(10.75,95.75)`
avant rotation éventuelle de l'image et transformations de ses parents.

Sous une échelle non uniforme, un cercle source apparaît comme une ellipse,
conformément à l'image étirée. Garder les primitives et les matrices préserve
les arcs sans recalcul délicat de leurs flags. L'épaisseur du trait est aussi
transformée ; la première version n'utilise pas de stroke non redimensionnable.
Source : [coordonnées et preserveAspectRatio SVG](https://www.w3.org/TR/SVG2/coords.html#PreserveAspectRatioAttribute).

## Insertion et limites du premier périmètre

Importer uniquement le groupe `vectors` du SVG temporaire, ses styles,
primitives et attributs de provenance. Ne pas importer un nouveau document
racine ni son fond raster. Réattribuer tous les identifiants avec les outils
inkex afin de permettre plusieurs vectorisations dans le même document.
Insérer le groupe après l'image sélectionnée dans son parent et lui donner
un nom lisible. La modification du document intervient seulement après
réussite du moteur, validation du SVG et calcul du placement.

Prendre en charge translations, rotations, matrices affines, échelles
uniformes/non uniformes, inversions et groupes parents transformés. Refuser
avec explication les transformations singulières, dimensions nulles,
géométries CSS/longueurs en pourcentage non résolues, `slice`, clips, masques
ou filtres affectant l'image ou ses ancêtres. Les `<use>`/clones ne sont pas
des images sélectionnables dans cette version. Ces refus évitent de vectoriser
un contenu différent de celui visible sans avoir prévu sa gestion.

## Vérifications prévues après autorisation

Tests de l'adaptateur : PNG embarqué/lié, chemins relatifs et `file:` Unicode,
sélection invalide, source absente, PNG invalide, échec/timeout du moteur,
absence de mutation en cas d'erreur et identifiants distincts au second passage.
Placement : translation, rotation, groupe parent, miroir, échelle non uniforme,
document en mm avec `viewBox`, marges `meet`, alignements et demi-pixel.
Les cas refusés doivent produire des messages utiles.

Comparer ensuite raster et vecteurs sur des fixtures SVG dans Inkscape,
notamment la rosace validée : un cercle et six arcs au réglage de 5 pixels.
Ces tests d'intégration sont à réaliser dans l'environnement inkex et avec le
moteur externe ; la suite actuelle reste indépendante d'Inkscape.

À cette étape documentaire, contrôle du moteur existant :

```bash
source .venv/bin/activate
python -m pytest -q
python -m raster2vector --help
python -m raster2vector --version
```

Critères de validation de la proposition : appel du moteur dans un Python
séparé, traitement d'un PNG sélectionné, image conservée, insertion dans le
même parent, placement tenant compte du ratio et des transformations, limites
initiales explicites. Valider cette architecture avant de créer l'extension.

## Références locales inspectées

Dans `/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions/` :
`inkex/base.py`, `inkex/extensions.py`, `inkex/elements/_base.py`,
`inkex/elements/_image.py`, `image_extract.py` et `image_embed_selected.inx`.
La version locale de `Image` ne possède pas `embed_image`, contrairement à
la documentation actuelle qui l'annonce depuis 1.5 : l'adaptateur prévu ne
dépendra donc pas de cette méthode plus récente.
