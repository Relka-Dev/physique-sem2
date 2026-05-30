### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ df96d71a-8c94-4268-b4b5-f899db9a42d1
using Plots

# ╔═╡ 72ce9fa6-f5a7-4861-971b-b868db127313
using LinearAlgebra

# ╔═╡ b460e0df-5aa3-4a1e-b2b3-a92ed3007fa3
using Test

# ╔═╡ e575895b-63ef-44b2-9265-aac1af08cb60
using Statistics

# ╔═╡ cec34ee7-b494-4607-820c-4be5f76539e9
using StatsBase

# ╔═╡ e69a6728-e8ee-4a23-afc4-08dc8aafa669
md"""

# Simulation multi-agents : de la théorie cinétique des gaz à la modélisation de la condensation

- **Auteur** 	: Karel Vilém Svoboda  
- **Cours**  	: Computational physics 2   
- **Date** 	: 22.05.2026  

"""

# ╔═╡ d774739f-4fec-4de3-a467-158ce2f57efd
md"""
## 1 La base du code : la classe Molecule
"""

# ╔═╡ 4d43a8ca-1ec8-44c0-8a6b-d16eaa95a3bd
md"""
### 1.1 Développez une classe nommée `Molecule`
"""

# ╔═╡ 6312f49c-0e57-11f1-004c-852d60e5287d
mutable struct Molecule
    position::Vector{Float64}    # position spatiale du centre [x, y, z] en m
    velocity::Vector{Float64}    # vitesse [vx, vy, vz] en m/s
    mass::Float64                # masse de la molécule en kg
    radius::Float64              # rayon en m
    formula::String              # formaule chimique   
end

# ╔═╡ 1479e2d1-55e5-4aba-b3d9-bd755e7b19eb
md"""
## 2 Différents types de molécules

### 2.1 Trouvez les valeurs des attributs de la classe Molecule pour les molécules suivantes : He, N e, N2, O2.
"""

# ╔═╡ 98816279-a0f4-48f4-9c11-a2d99934d799
begin
	He = Molecule([0.0, 0.0, 0.0], [100.0,  50.0, 0.0], 6.6465e-27, 1.40e-10, "He")
    Ne = Molecule([0.0, 0.0, 0.0], [ 80.0, 120.0, 0.0], 3.3509e-26, 1.54e-10, "Ne")
    N2 = Molecule([0.0, 0.0, 0.0], [ 60.0, -80.0, 0.0], 4.6518e-26, 1.85e-10, "N2")
    O2 = Molecule([0.0, 0.0, 0.0], [ 40.0,  90.0, 0.0], 5.3135e-26, 1.73e-10, "O2")
    molecules = [He, Ne, N2, O2]
end

# ╔═╡ 3c864515-564e-4611-bdc5-2f9f770f610d
md"""
### 2.2 Quelles sont les différences entre ces molécules ? Est-ce que la théorie cinétique des gaz fonctionne pour toutes ces molécules ?

#### Les différences

He est la plus légère et la plus petite O2 est la plus lourde et N2 est la plus grande. Au niveau de la structure atomique, He et Ne sont monoatomiques et N2 et 02 sont diatomiques. Le a différence c'est le niveau de liberté, les atomes monoatomique en ont 3 et les diatomiques 5.

#### Est-ce que la théorie cinétique fonctionne pour toutes ?

Pour les atomes monoatomiques, He et Ne, oui, dans interactions, le modèle s'applique parfaitement. Pour les diatomiques, partiellement, on effectue une simplification en ignorant la forme allongée, la retation et la vibration des molécules et les interrations électostatiques entre les deux atomes de la molécule.
"""

# ╔═╡ 0206106c-81ab-4f35-ac9e-93d9504d4e61
md"""
## 3 Le déplacement d'une molécule

Maintenant que vous avez réussi à créer la classe `Molecule`, vous allez mettre en mouvement celle-ci. Il est possible
de simpliffier sa dynamique sous la forme d'un mouvement de projectile sans interaction avec son environnement.

### 3.1 En utilisant la seconde loi de Newton, écrivez les équations gouvernant le mouvement de ces molécules

En mouvement réctiligne uniforme, sans gravité et d'interration entre les molécules, la seconde loi de Newton donne :

$$\vec{F} = m\cdot\vec{a} = \vec{0}$$

$$\frac{d\vec{v}}{dt} = \vec{0} \implies \vec{v}(t) = \vec{v}(0) = \text{constante}$$

$$\frac{d\vec{r}}{dt} = \vec{v} \implies \vec{r}(t) = \vec{r}(0) + \vec{v}\cdot t$$

### 3.2 Utilisez les différences fnies sur le système d'équation obtenu à la question précédente.

La discrétisation dans le temps avec la méthode d'Euler explicite donne :

$$\vec{v}(t + \Delta t) = \vec{v}(t)$$

$$\vec{r}(t + \Delta t) = \vec{r}(t) + \vec{v}(t) \cdot \Delta t$$
"""

# ╔═╡ db24edb4-8b4b-4d83-a875-ecfee8f89e9a
md"""
### 3.3 Implémentation `computeNextPosition`
Implémentez ce calcul dans la méthode computeNextPosition calculant la position de la molécule au pas de temps d'après. Pour cette méthode, il vous est fortement conseillé de mettre en argument l'instance de la molécule puis de changer sa position directement dans la fonction.
"""

# ╔═╡ 05fd32d1-97e9-4ab5-a1d6-3fcbbd9f925b
function computeNextPosition(molecule::Molecule, dt::Float64)
	molecule.position .+= molecule.velocity .* dt
end

# ╔═╡ f7164257-150d-430a-9fd4-84115dcad863
md"""
### 3.3 Visuation de trajectoire
Afin de vérifier et valider la nouvelle fonction implémentée, vous allez instancier une des molécules de l'exercice 2.1 et visualisez sa trajectoire au cours du temps. Quels position, vitesse, pas de temps et temps final avez-vous choisi et pourquoi ?

#### Choix de position et vitesse initale
J'ai placé la molécule au centre de la simulation [0, 0, 0] et donné une vitesse réalise pour une molécule d'hydrogène à température ambiante [100.0, 50.0, 0.0].

Si on imagine une boite de taille 10e-8m et que la molécule bouge à vitesse constante à 100ms, alors la boite est traversée en 10e-10s. Done un delta de temps de 10e-12 est adéquat. Pour le temps final, on va commencer par 100 iérations dont 1e-9. 

"""

# ╔═╡ 185757a3-3bc2-43f6-a843-e1da59146e79
function displayOneMoleculeMouvement(molecule::Molecule, deltaTime::Float64, finalTime::Float64)
    positions_x = []
    positions_y = []
    
    t = 0.0
    while t <= finalTime
        push!(positions_x, molecule.position[1])
        push!(positions_y, molecule.position[2])
        computeNextPosition(molecule, deltaTime)
        t += deltaTime
        # print(molecule.velocity)
    end
    plot(positions_x, positions_y)
end

# ╔═╡ 1d452f32-3c01-4269-afa6-fa151bd990b2
begin
displayOneMoleculeMouvement(He, 10e-12, 1e-9)
end

# ╔═╡ a9662ab2-221e-40bc-801f-884aa443a1fd
md"""
#### Analyse

Comme on peut le voir, on observe un mouvement uniforme au cous du temps avec une vitesse constante.
"""

# ╔═╡ 1b0775ba-e51a-462e-8aa3-ef3bc0ec9fdf
md"""
## 4 Intéraction entre deux molécules

Lorsqu'un système contient plusieurs molécules en mouvement, deux molécules peuvent s'approcher suffisamment l'une de l'autre et intéragir entre elles.

### 4.1 
Décrivez le type d'interaction qui peut se produire lorsque deux molécules s'approchent suffisamment l'une de l'autre. Expliquez également dans quelles conditions ce modèle d'interaction est pertinent et quelles hypothèses permettent de l'utiliser dans ce contexte

C'est une collision élastique entre deux sphères rigides. Cela se produit quand se prduit quand leurs centres sont inférieurs ou égales à la somme de leur rayons :

$$d \leq \rho_1 + \rho_2$$

#### Hypthèses du modèle 
	1. Les molécules sont des sphères rigides  
	2. La collision est élastique (l'énergie cinétioque totale et la quatité de mouvement totale sont conservés) 
	3. On ignore les degrés de liberté internes (pas de frictions ni de rotation)

#### Pertinence du modèle
	1. Pression basse pour que les interractions soient rares et brès
	2. haute température, l'énergie cinétique doit dominer les forces d'attractions
	3. Avoir des gaz nobles 

"""

# ╔═╡ b534f811-34d5-473f-bb44-b7c1b2841f98
md"""
### 4.2 Tests

Avant d'implémenter le modèle de collision, proposez au minimum 4 scénarios de test permettant de vérifier que votre code fonctionne correctement. Vos cas de test doivent notamment permettre de contrôler : la conservation de la quantité de mouvement, la conservation de l'énergie cinétique et l'influence de l'angle de contact entre les deux molécules. Décrivez clairement chaque scénario (conditions initiales).

L'objectif est de voir dans différents scénarios si :

$$\vec{p_{tot}} = m_1\vec{v}_1 + m_2\vec{v}_2 = contanste$$
$$E_{cin} = \frac{1}{2}m_1v^2_1 + \frac{1}{2}m_2v^2_2 = constante$$

#### Scenario 1, Choc frontal avec 2 molécules `He`

L'objectif est que les vitesses des molécules s'échangangent leurs vitesses.

```
État initial

Molécule 1 (He): Vitesse [100, 0, 0]
Molécule 2 (He): Vitesse [-100, 0, 0]

Résultat attendu
Molécule 1 (He): Vitesse [-100, 0, 0]
Molécule 2 (He): Vitesse [100, 0, 0]
```

#### Scenario 2, Choc frontal avec 2 molécules `He` et `O2`

La molécule `He` est en mouvement pendant que la molécule `O2` est immobile.

```
État initial

Molécule 1 (He): Vitesse [100, 0, 0]
Molécule 2 (O2): Vitesse [0, 0, 0]

Résultat attendu
La molécule `He` repart en arrière lentement pendant que 02 avance lentement
```

#### Scenario 3, Collision à 90° 

Deux molécules `He` à la même vitesse font un choc à 90°.

```
État initial

Molécule 1 (He): Vitesse [100, 0, 0]
Molécule 2 (He): Vitesse [0, 100, 0]

Résultat attendu
Les vitesses sont redistribuées selon l'angle de contact.
```


#### Scenario 4, Pas de collision

Les deux molécules ne devraient pas se toucher.

```
État initial

Molécule 1 (He): Vitesse [100, 0, 0] Position [100, 100, 100]
Molécule 2 (O2): Vitesse [0, 100, 0] Position [200, 200, 200]

Résultat attendu
Aucun changement
```

"""

# ╔═╡ 5abe6eee-76ec-4d7d-b27e-7b72a784b1c7
md"""
### 4.3 `detectCollision`

Rédigez une fonction qui détecte si deux molécules sont en interaction. Cette fonction doit renvoyer true lorsque la
distance entre leurs centres est inférieure ou égale à la somme de leurs rayons et false dans le cas contraire.

Implémenter : $d = ||\vec{r_1}-\vec{r_2}|| \leq \rho_1 + \rho_2$
"""

# ╔═╡ ed7f37a2-4a5a-4c35-842f-747353e48abd
function detectCollision(molecule1:: Molecule, molecule2:: Molecule)
	return norm(molecule1.position - molecule2.position) <= molecule1.radius + molecule2.radius
end

# ╔═╡ 8d92dc16-ef6b-48d6-94f1-3b1287b64c2c
md"""
### 4.4 Màj des collision et tests

Implémentez l'algorithme étudié en cours avec Jessen permettant de mettre à jour l'état des molécules après une collision. À l'aide des cas de test défnis à la question 4.2, vérifiez que votre implémentation respecte les propriétés physiques attendues.

La maj des vitesses après la collision se fait selon la composante normale au plan de contact.

$$\vec{p}_{tot} = m_1\vec{v}_1 + m_2\vec{v}_2$$

$$E_{cin} = \frac{1}{2}m_1||\vec{v}_1||^2 + \frac{1}{2}m_2||\vec{v}_2||^2$$

---

$$\hat{n} = \frac{\vec{r_1}-\vec{r_2}}{||\vec{r_1}-\vec{r_2}||}$$

$$\vec{v}_1' = \vec{v}_1 - \frac{2m_2}{m_1+m_2}\left[(\vec{v}_1 - \vec{v}_2)\cdot\hat{n}\right]\hat{n}$$

$$\vec{v}_2' = \vec{v}_2 + \frac{2m_1}{m_1+m_2}\left[(\vec{v}_1 - \vec{v}_2)\cdot\hat{n}\right]\hat{n}$$

"""

# ╔═╡ 11b51dba-e52e-4153-a1a3-151b4e3f850f
function computeCollision(molecule1::Molecule, molecule2::Molecule)
	v1 = copy(molecule1.velocity)
	v2 = copy(molecule2.velocity)

	# compostante normale au plan de contact
	n = (molecule1.position - molecule2.position) / (norm(molecule1.position-molecule2.position))
	
	v1_prime = v1 - (2*molecule2.mass)/(molecule1.mass+molecule2.mass) * dot(v1-v2, n) .* n

	v2_prime = v2 + (2*molecule1.mass)/(molecule1.mass+molecule2.mass) * dot(v1-v2, n) .* n

	molecule1.velocity .= v1_prime
	molecule2.velocity .= v2_prime
end

# ╔═╡ 78a927f5-5c01-4e44-a73b-35d952dba07a
function movementQuantity(molecule1::Molecule, molecule2::Molecule)
	return molecule1.mass*molecule1.velocity + molecule2.mass*molecule2.velocity
end

# ╔═╡ 7c60fb05-85f0-4fe8-a9a0-8c59f9706f38
function cineticEnergy(molecule1::Molecule, molecule2::Molecule)
    return 0.5*molecule1.mass*norm(molecule1.velocity)^2 + 0.5*molecule2.mass*norm(molecule2.velocity)^2
end

# ╔═╡ f53cb4ae-5b48-4f15-bd9b-f707dc8a3fc6
@testset "Collision Tests" begin
    
    @testset "Scenario 1 - Choc frontal He-He" begin
        m1 = Molecule([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        m2 = Molecule([3.0e-10, 0.0, 0.0], [-100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        computeCollision(m1, m2)
        @test m1.velocity ≈ [-100.0, 0.0, 0.0]
        @test m2.velocity ≈ [100.0, 0.0, 0.0]
    end

    @testset "Scenario 2 - Choc frontal He-O2" begin
        m1 = Molecule([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        m2 = Molecule([3.0e-10, 0.0, 0.0], [0.0, 0.0, 0.0], 5.3135e-26, 1.73e-10, "O2")
        p_avant = movementQuantity(m1, m2)
        e_avant = cineticEnergy(m1, m2)
        computeCollision(m1, m2)
        @test movementQuantity(m1, m2) ≈ p_avant
        @test cineticEnergy(m1, m2) ≈ e_avant
        @test m1.velocity[1] < 0
        @test m2.velocity[1] > 0
    end

    @testset "Scenario 3 - Collision 90°" begin
        m1 = Molecule([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        m2 = Molecule([3.0e-10, 0.0, 0.0], [0.0, 100.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        p_avant = movementQuantity(m1, m2)
        e_avant = cineticEnergy(m1, m2)
        computeCollision(m1, m2)
        @test movementQuantity(m1, m2) ≈ p_avant
        @test cineticEnergy(m1, m2) ≈ e_avant
    end

    @testset "Scenario 4 - Pas de collision" begin
        m1 = Molecule([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        m2 = Molecule([100.0, 100.0, 100.0], [0.0, 100.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        v1_avant = copy(m1.velocity)
        v2_avant = copy(m2.velocity)
        if detectCollision(m1, m2)
            computeCollision(m1, m2)
        end
        @test m1.velocity ≈ v1_avant
        @test m2.velocity ≈ v2_avant
    end

end

# ╔═╡ eaa968af-d013-4d16-b62f-713500a68b18
md"""
## 5 Le domaine de simulation

Le domaine représente l'espace dans lequel vos molécules évoluent et interagissent. Dans ce laboratoire, cet espace
est modélisé comme un pavé droit (un cuboïd), de dimensions Lx × Ly × Lz centré autour de la position (0, 0, 0).
"""

# ╔═╡ b2ca41ad-c8b8-423e-a75f-56bd2b737890
md"""
### 5.1 class `Domain`

Développez une nouvelle classe nommée Domain qui représente le volume dans lequel se trouvent les molécules.
Cette classe doit contenir les attributs suivants :
- la longueur de l'axe x : Lx (m)
- la longueur de l'axe y : Ly (m)
- la longueur de l'axe z : Lz (m)
"""

# ╔═╡ 375c8598-0034-445e-b3e8-bcd58a471c5f
struct Domain
	L::Vector{Float64} # Taille en 3D du domaine en m
end

# ╔═╡ 22b1998c-045d-4d0b-9ca8-40ab8c208d95
md"""
### 5.2
Ajoutez une fonction permettant de calculer automatiquement son volume
"""

# ╔═╡ e018bcc6-5f90-429f-a749-6b4011dfbad4
function domainVolume(domain::Domain)
	return prod(domain.L)
end

# ╔═╡ 6b2dc248-8208-47fd-987c-5c1726127353
md"""
### 5.3
Après avoir proposé une vérification de votre classe et de votre méthode de calcul du volume, implémentez cette vérification pour vous assurer de leur bon fonctionnement.

Un volume qui fait 10m de large avec 5m de largeur et 3 m de profondeur devrait faire :

$$10*5*3 = 150m^3$$ 
"""

# ╔═╡ c6e70817-cebf-4ac8-bf11-79df05a39e6c
@testset "Domain Tests" begin
    d = Domain([10.0, 5.0, 3.0])
    @test domainVolume(d) ≈ 150.0
end

# ╔═╡ ff8a5c41-8ec5-4208-a1d8-165dfb6e9d99
md"""
## 6 Condition de bord/limite : réflexion spéculaire

Si une molécule rebondit sur les parois par réflécion spéculaire, si une molécule sort du domaine sur un axe, il faut corrifer la position en la rléfléchissant par rapport au mur. en inversant la composante de vitesse correspondante.

Nouvelle position :

$$x_{new} = Lx - d$$

Vitesse corrigée : 

$$v_x \rightarrow -vx$$
"""

# ╔═╡ 8c2b6820-8a02-4879-afe5-3b35cebe1ea8
md"""
### 6.1
Implémentez une fonction qui corrige la position et la vitesse d'une molécule lorsqu'elle se retrouve en dehors du
domaine. Cette fonction doit vérifier chaque axe séparément et appliquer la réflexion si nécessaire.
"""

# ╔═╡ 40d672a8-4dee-48f7-a241-5e14df293bc2
function applyBoundaries(molecule::Molecule, domain::Domain)
    for i in 1:3
        if molecule.position[i] < 0
            molecule.position[i] = -molecule.position[i]
            molecule.velocity[i] = -molecule.velocity[i]
        elseif molecule.position[i] > domain.L[i]
            molecule.position[i] = 2*domain.L[i] - molecule.position[i]
            molecule.velocity[i] = -molecule.velocity[i]
        end
    end
end

# ╔═╡ ab3d206c-35d9-40ef-8154-3c1cc6b6fc4e
md"""
### 6.2

Proposez puis implémentez au minimum 8 scénarios de test permettant de vérifer que votre code fonctionne correctement. Vos cas de test doivent notamment permettre de contrôler : le changement de vitesse et de position
pour chaque paroi (une ou plusieurs)
"""

# ╔═╡ aff6073d-d727-4b0e-b3b2-166b2175e72e
@testset "Boundary Tests" begin
    domain = Domain([10.0, 10.0, 10.0])

    @testset "Bord gauche x < 0" begin
        m = Molecule([-1.0, 5.0, 5.0], [-100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[1] ≈ 1.0
        @test m.velocity[1] ≈ 100.0
    end

    @testset "Bord droit x > Lx" begin
        m = Molecule([11.0, 5.0, 5.0], [100.0, 0.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[1] ≈ 9.0
        @test m.velocity[1] ≈ -100.0
    end

    @testset "Bord bas y < 0" begin
        m = Molecule([5.0, -2.0, 5.0], [0.0, -100.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[2] ≈ 2.0
        @test m.velocity[2] ≈ 100.0
    end

    @testset "Bord haut y > Ly" begin
        m = Molecule([5.0, 12.0, 5.0], [0.0, 100.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[2] ≈ 8.0
        @test m.velocity[2] ≈ -100.0
    end

    @testset "Bord avant z < 0" begin
        m = Molecule([5.0, 5.0, -3.0], [0.0, 0.0, -100.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[3] ≈ 3.0
        @test m.velocity[3] ≈ 100.0
    end

    @testset "Bord arrière z > Lz" begin
        m = Molecule([5.0, 5.0, 13.0], [0.0, 0.0, 100.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[3] ≈ 7.0
        @test m.velocity[3] ≈ -100.0
    end

    @testset "Coin x et y simultanément" begin
        m = Molecule([-1.0, -2.0, 5.0], [-100.0, -100.0, 0.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position[1] ≈ 1.0
        @test m.position[2] ≈ 2.0
        @test m.velocity[1] ≈ 100.0
        @test m.velocity[2] ≈ 100.0
    end

    @testset "Pas de sortie" begin
        m = Molecule([5.0, 5.0, 5.0], [100.0, 100.0, 100.0], 6.6465e-27, 1.40e-10, "He")
        applyBoundaries(m, domain)
        @test m.position ≈ [5.0, 5.0, 5.0]
        @test m.velocity ≈ [100.0, 100.0, 100.0]
    end

end

# ╔═╡ 06bcf237-ccc7-411a-b9e9-f8cff71b8fd4
md"""
## 7 Première simulation d'un gaz d'Helium

Première simulation multi agents. Les atômes sont initialisés de façon random dans le domaine. Chaque atome sera initialisé aà la même vitesse mais avec des positions aléatoires.

Conditions initiales de la simulation :
- Domaine de simulation (m) : [−5 · 10−9, 5 · 10−9] × [−5 · 10−9, 5 · 10−9] × [−5 · 10−9, 5 · 10−9]
- Masse de l'hélium (kg) : 6.646 · 10−27
- Rayon de l'hélium - modèle hard sphere (m) : 1.1 · 10−10
- Nombre d'atomes : 400
- Vitesse initiale de chaque atome (m/s) : 1400
- Pas de temps (s) : 1 · 10−14
- Temps nal (s) : 2 · 10−1
"""

# ╔═╡ f04d6d55-c7e8-4437-8d9e-1c1bc8acf8a3
md"""
### 7.1
Faites une animation illustrant le comportement des atomes de cette simulation.

### 7.2 :
Calculez puis affichez l'évolution de la vitesse moyenne des atomes au cours du temps. Comment cette vitesse évolue-t-elle au sain de la simulation ?
"""

# ╔═╡ f00cee09-d424-4b57-986e-0215dde37cc0
function initaliseGasAtRandom(N::Int, domaine::Domain, moleculesPrototypes::Vector{Molecule}, initialSpeed::Float64)
	final_molecules = Molecule[]
	v = randn(3)

	
	for i in 1:N
		# Vecteur gaussien aléatoire
		random_speed = randn(3)
		random_speed = random_speed / norm(random_speed) * initialSpeed

		random_position = rand(3) .* domaine.L
		
		prototype = rand(moleculesPrototypes)
		m = Molecule(random_position, random_speed, prototype.mass, prototype.radius, prototype.formula)
		push!(final_molecules, m)
	end

	return final_molecules
end

# ╔═╡ 2eae66aa-4497-4b74-b629-76f40d269728
md"""
#### Analyse

Elle commence par descendre légérement puis à se stabiliser. Cette stabilité c'est l'équilibre thermodynamique. Les grandeurs macroscopiques comme la vitesse moyennes ne varient plus.
"""

# ╔═╡ c7efe043-a7dc-4217-8fb9-d20bcd3ed36f
md"""
### 7.3
Affichez la distribution de la magnitude de la vitesse des atomes au temps final. Qu'observez-vous ?

Selon l'histograme, on observe une distribution de Maxwell-Botzmann. Elle suit une courbe en cloche mais n'est pas symétrique dans sa moyenne. Elle possède une borne à gauche car elle peut pas être négative et la queue est plus longue à droite.
"""

# ╔═╡ 9de0000d-0399-4c9a-9ae4-74a386e06932
md"""
### 7.4

Calculez puis affichez α au cours du temps. Comment évolue α au cours du temps ? Qu'est-ce que α peut représenter
physiquement et quelle est son unité ?

$$α = \frac{m <v2>}{3kb}$$
avec m la masse des atomes, < v2 > la moyenne des vitesses au carrées et kb = 1.380649·10−23(J/K) la constante
de Boltzmann.

C'est la température en Kelvins [K]
"""

# ╔═╡ a2bf77de-0cbc-480c-899e-2a2892dcca56
md"""
### 7.5

Calculez puis achez β au cours du temps. Comment évolue β au cours du temps ? Qu'est-ce que β peut représenter
physiquement et quelle est son unité ?
β = N m < v2 >
3V

$$\beta = \frac{Nm <v2>}{3V}$$

avec N le nombre d'atome dans le domaine, m la masse des atomes, < v2 > la moyenne des vitesses au carrées
et V le volume de ce domaine.

C'est la pression en Pascal [Pa]

"""

# ╔═╡ 54811911-1303-4acb-a666-7bb72dcf3ebb
md"""
## 8 Ajout de la gravité

La gravité est une force volumique agissant sur l'ensemble des atomes, même si son effet et est extrêmement faible à l'échelle microscopique. L'objectif est de comprendre quand et comment la gravité peut modifier la répartition spatiale d'un gaz.

### 8.1 :
Dans votre méthode computeNextPosition, ajoutez l'influence de la gravité. Cette force suivra l'axe z.
"""

# ╔═╡ b7b7b4e9-a125-47e3-8054-5ecf5b21d704
function computeNextPositionGravity(molecule::Molecule, dt::Float64, gravity::Vector{Float64})
	molecule.position .+= molecule.velocity .* dt .+ 0.5 .* gravity .* dt^2
	molecule.velocity .+= gravity .* dt
end

# ╔═╡ 1ed02f4a-6ba4-416c-8554-7b7d71e04368
md"""
### 8.4 
Vous simulez la même configuration que précédemment (partie 7) mais avec ces changements.

Conditions initiales de la simulation
- Domaine de simulation (m) : [−1 · 10−8, 1 · 10−8] × [−1 · 10−8, 1 · 10−8] × [−1 · 10−8, 1 · 10−8]
- Nombre d'atomes : 500
- Vitesse initiale de chaque atome (m/s) : 1367
- Temps nal (s) : 5 · 10−11
- g (m/s2) : 9.81 · 1013

Afin de pouvoir visualiser dans un temps respectable l'influuence de la gravité sur la répartition, celle-ci sera très fortement exagérée. Comment évolue cette probabilité de présence en fonction de l'altitude ?

-> point 8.2
"""

# ╔═╡ 431388d3-71a6-4187-8476-a4a939fbfa37
# ╠═╡ disabled = true
#=╠═╡
function simulateWithGravity(molecules::Vector{Molecule}, domain::Domain, dt::Float64, finalTime::Float64, gravity::Vector{Float64})
    mean_speeds = []
    times = []
    alphas = []
    betas = []
    
    @gif for t in 0:dt:finalTime 
        current_speed_sum = 0.0
        for molecule in molecules
            computeNextPositionGravity(molecule, dt, gravity)
            applyBoundaries(molecule, domain)
        end
        for i in 1:length(molecules)
            for j in i+1:length(molecules)
                if detectCollision(molecules[i], molecules[j])
                    computeCollision(molecules[i], molecules[j])
                end
            end
            current_speed_sum += norm(molecules[i].velocity)
        end
        push!(mean_speeds, current_speed_sum / length(molecules))
        push!(times, t)
        alpha = He.mass * mean(norm.(getfield.(molecules, :velocity)).^2) / (3 * 1.380649e-23)
push!(alphas, alpha)
        beta = length(molecules) * He.mass * mean([norm(m.velocity)^2 for m in molecules]) / (3 * domainVolume(domain))
push!(betas, beta)
        p1 = scatter([m.position[1] for m in molecules], [m.position[2] for m in molecules], xlims=(0, domain.L[1]), ylims=(0, domain.L[2]), legend=false, title="Positions")
        plot(p1, layout=(1,1))
    end every 100
end
  ╠═╡ =#

# ╔═╡ 7bb387ab-f469-4179-8733-912203ba94b0
md"""

### 8.2

Afin d'observer l'effet de la gravité dans votre simulation, implémentez l'affichage d'un graphique représentant

l'évolution de la probabilité de présence des particules en fonction de z.

On observe une décroissance exponnentielle à cause de la gravité, plus l'axe z est élevé, moins c'est probable da'voir des particules présentes.
"""

# ╔═╡ f3d01dff-1774-4cad-8829-652bbf4cba9b
md"""
### 8.3
La vitesse de gravité est modélisée par le coeficient suivant

$$v_{grav} = \frac{mgD}{kbT}$$

avec m la masse de la molécule, g la constante de gravité, D un coeficient de diffusion (6.5 · 10−5 m2/s), kb la constant de Boltzmann et T la température.
Combien de temps faudrait-il à une molécule d'Hélium pour parcourir une mètre de distance due uniquement à la force de gravité pour une température de 26.85 ◦C ?

Temps pour parcourir 1 mètre :

$$t = \frac{d}{v_{grav}} = \frac{1}{v_{grav}}$$
"""

# ╔═╡ 4e987429-a0b3-45ae-b973-396afb05f951
begin
m = 6.646e-27
g = 9.81
D = 6.5e-5
kb = 1.380649e-23
T = 273.15 + 26.85

v_grav = (m * g * D) / (kb * T)
t = 1 / v_grav
println("v_grav = $v_grav m/s")
println("t = $t s")
end

# ╔═╡ c47a9bf4-6b63-4551-8db2-e5e692294908
md"""
### 8.5
Qu'est-ce que la turbopause ? Quel mécanisme physique permet d'expliquer cette limite ?

La turbopause c'est la limite dans l'athmosphère terrestre qui sépare deux régions, elle se situe à 100km d'altitude :

En dessous (homosphère): Grâce au méange athmosphérique, la composition des gaz reste homogène, les différents types sont mélangés indépendamment de leur masse.

Au dessus (hétérosphère): le mélange disparait et chaque type se distribue selon sa propre masse suivant la loi barométique :

$$n(z) = n_0 \cdot e^{-\frac{mgz}{k_BT}}$$

La complession entre les deux effets suivants :
- L'agitation thermique qui tend à mélanger uniformément les gaz
- La gravité qui tend à séparer les espèces selon leur masse (les plus lourdes en bas, les plus légères en haut)

En dessous de la turbopause, la turbulence domine et homogénéise. Au dessus, la gravité domine et les espèces se séparent.
"""

# ╔═╡ 3f5235ca-9adf-4e9d-b6f1-22ea36606712
md"""
### 8.6
Dans le même esprit que la question 8.3, affichez l'évolution de la pression en fonction z. Que se passe-t-il ?

La pression décroit exponnentiellement avec l'altitude
"""

# ╔═╡ f71e8ecb-e04a-4df3-9568-f3d946b0ec60
md"""
## 9 Simulation multi-espèces (He-Ar)
Dans cette partie, vous allez simuler un système composé de plusieurs espèces atomiques. Cette simulation consid-
érera un mélange d'atomes d'hélium (He) et d'argon (Ar), initialisés aléatoirement dans le domaine de simulation.

L'objectif est d'étudier l'inuence de la masse/taille des particules sur leur dynamique, leur répartition spatiale et les grandeurs thermodynamiques du système.

Conditions initiales de la simulation :
	- Domaine de simulation (m) : [−1 · 10−8, 1 · 10−8] × [−1 · 10−8, 1 · 10−8] × [−1 · 10−8, 1 · 10−8]
	- Pas de temps (s) : 1 · 10−14
	- Temps nal (s) : 5 · 10−11
	- g (m/s2) : 9.81 · 1013
	- Hélium :
	- Masse de l'hélium (kg) : 6.646 · 10−27
	- Rayon de l'hélium - modèle hard sphere (m) : 1.1 · 10−10
	- Nombre d'atomes d'hélium : 400
	- Vitesse initiale des atomes d'héliun (m/s) : 789.45
- Argon :
	- Masse de l'Argon (kg) : 6.634 · 10−26
	- Rayon de l'Argon - modèle hard sphere (m) : 1.88 · 10−10
	- Nombre d'atomes d'Argon : 200
	- Vitesse initiale des atomes d'Argon (m/s) : 249.88

## 9.1
Réalisez la simulation correspondant à ces conditions initiales et produisez une animation illustrant l'évolution du système. Afin de distinguer clairement les deux espèces, utilisez une couleur différente pour les atomes d'hélium et d'argon.
"""

# ╔═╡ 6835f289-57cd-44cd-a5f3-32cf0f8232ba
# ╠═╡ disabled = true
#=╠═╡
 function simulateWithMultiSpicies(molecules::Vector{Molecule}, domain::Domain, dt::Float64, finalTime::Float64, gravity::Vector{Float64})
    mean_speeds = []
    times = []
    
    @gif for t in 0:dt:finalTime 
        current_speed_sum = 0.0
        for molecule in molecules
            computeNextPositionGravity(molecule, dt, gravity)
            applyBoundaries(molecule, domain)
        end
        for i in 1:length(molecules)
            for j in i+1:length(molecules)
                if detectCollision(molecules[i], molecules[j])
                    computeCollision(molecules[i], molecules[j])
                end
            end
            current_speed_sum += norm(molecules[i].velocity)
        end
        push!(mean_speeds, current_speed_sum / length(molecules))
        push!(times, t)
        p1 = scatter([m.position[1] for m in molecules if m.formula == "He"],
             [m.position[2] for m in molecules if m.formula == "He"],
             color=:blue, label="He", xlims=(0, domain.L[1]), ylims=(0, domain.L[2]))
scatter!(p1, [m.position[1] for m in molecules if m.formula == "Ar"],
             [m.position[2] for m in molecules if m.formula == "Ar"],
             color=:red, label="Ar")
plot(p1)
    end every 100
end
  ╠═╡ =#

# ╔═╡ 855ace40-74b1-4aa3-b345-a9b4c23d8173
md"""
### 9.2
Afin d'analyser l'effet de la gravité sur chaque espèce, calculez et affichez l'évolution de la probabilité de présence des particules en fonction de l'altitude z. Cette analyse devra être exectuée séparément pour l'hélium et pour l'argon.
"""

# ╔═╡ 7078bc83-93c7-4736-b407-fa665aa78c02
md"""
### 9.3
Expliquez la différence physique entre les grandeurs moyennes < v2 > et < mv2 > ?

La moyenne des carrées des vitesses -> grandeur cinématique

$$<v^2>$$ 

La moyenne des corrées de la vitesse pondérée par la masse -> grandeur énergétique

$$<mv^2>$$

Différence : L'équilibre fondamentale : à l0équilibre thermodynmaique, deux expèces différentes ont la même valeur de $$<mv^2>$$ mais la $$<v^2>$$ sont différentes car les molécules plus lourdes se déplacent plus lentement. -> Principe d0équiparatition de l'énergie.
"""

# ╔═╡ b9eb2927-b1fd-41d4-be8c-9d633eb7bbff
md"""
### 9.4
Calculez et représentez l'évolution en fonction de z des grandeurs suivantes : : < mv2 >, p et T en fonction de z.
"""

# ╔═╡ 77a04d8c-c3c2-4fc5-8673-ea206a6b4d73
md"""
## 10 Critère de stabilité d'une simulation physique

Dans une simulation multi-agents, les trajectoires individuelles des atomes évoluent en permanence et ne se sta-
bilisent complètement. Cependant, certaines grandeurs statistiques globales du système, telles que < mv2 >, p
et T , peuvent atteindre un régime stationnaire au-delà duquel elles ne présentent plus que de faibles fuctuations
autour d'une valeur moyenne.
Cette stabilisation permet de définir un critère de stabilité (ou de convergence) et de considérer que le système
a atteint un état d'équilibre statistique. Un tel critère est essentiel afin de déterminer si une simulation a été menée sur un temps suffisamment long pour produire des résultats physiquement pertinents.
"""

# ╔═╡ d7d821a0-e809-4d09-b0fc-f2cab5a970c7
md"""
### 10.1 
Proposez un ou plusieurs critères permettant d'évaluer la stabilité d'une simulation physique.

Écart-type relatif : on calcule l'écart-type de la grandeur (T, p, < mv² >) sur une fenêtre glissante des N derniers pas de temps. Si σ/μ < seuil (ex. 5%), le système est stable.

$$\frac{\sigma}{\micro} < \epsilon$$
"""

# ╔═╡ e617fe99-22c3-4f21-843c-7c3c3a086d97
md"""
### 10.2
Implémentez et testez au moins un des critères proposés sur l'une des simulations réalisées précédemment. Analysez
son évolution au cours du temps et discutez de sa pertinence pour juger de la convergence du système
"""

# ╔═╡ fda906ad-394f-4b72-9f2f-cef7a5821b22
function simulateCollectStats(molecules::Vector{Molecule}, domain::Domain, dt::Float64, finalTime::Float64, gravity::Vector{Float64})
    alphas = []
    times = []
    
    for t in 0:dt:finalTime 
        for molecule in molecules
            computeNextPositionGravity(molecule, dt, gravity)
            applyBoundaries(molecule, domain)
        end
        for i in 1:length(molecules)
            for j in i+1:length(molecules)
                if detectCollision(molecules[i], molecules[j])
                    computeCollision(molecules[i], molecules[j])
                end
            end
        end
        alpha = mean([m.mass * norm(m.velocity)^2 for m in molecules]) / (3 * 1.380649e-23)
        push!(alphas, alpha)
        push!(times, t)
    end
    
    return times, alphas
end

# ╔═╡ 8fb8ce4e-ea89-4008-a5fa-99be6205fafc
md"""
## 11 Distribution de ???

### 11.1 :
La probabilité de présence en fonction de la coordonnée z présente une forme caractéristique. Quelle est cette distribution ? 

La distribution de Boltzmann.

La loi de Boltzmann s'écrit :
$$P(\text{état}) \propto e^{-\frac{E}{k_BT}}$$

Quels paramètres physiques influencent la forme de cette distribution et de quelle manière ?

La temperature `T`, plut `T` est elevée, plus loa distrubution est étalée.

L'énergie `E`, selon le contexte

Pour la distribution spatiale : $E = mgz$ (énergie potentielle)  

Pour la distribution des vitesses $E = \frac{mv^2}{2}$

Plus E croît vite, plus la distribution décroit vite.

$$\frac{E}{k_BT}$$

C'est la compétition entre l'énergie du système et l'agitation thermique

### 11.2 :
Quelles grandeurs thermodynamiques obéissent à cette même loi de distribution ? Expliquez pourquoi.

Les grandeurs qui obéissent à cette loi :

**Densité de particules :**
$$n(z) = n_0 \cdot e^{-\frac{mgz}{k_BT}}$$

**Pression :**
$$P(z) = P_0 \cdot e^{-\frac{mgz}{k_BT}}$$

**Distribution des vitesses (Maxwell) :**
$$p(v) = C \cdot e^{-\frac{mv^2}{2k_BT}}$$

"""

# ╔═╡ ead9a443-4aa3-4a45-9a14-e028e9862dc7
md"""
## 12 Entropie de Shannon

Probabilité d'apparition en un point x.

$$H(x) = -\sum{P(x)log_2(P(x))}$$

Dans le cas à quatre variables indépendantes, cette fonction peut s'écrire sous la forme suivante

$$H(X, Y, Z, <v^2>) = H(X) + H(Y) + H(Z) + H(<v^2>)$$
"""

# ╔═╡ 2be52041-7d82-40e5-b62b-301ac0bf59b5
md"""
## 12.1 :
Implémentez le calcul de cette information en fonction des distributions de probabilité pour les variables aléatoires : x, y, z et < v2 >. Les distributions spatiales doivent être divisées en 10. La distribution de < v2 > doit être
divisée en 200 sur le domaine [0; 200000] (m2/s2).
"""

# ╔═╡ 1c2bd9bc-c679-4b41-8a8d-fbb79556c1a9
function entropy(hist)
        p = hist.weights / sum(hist.weights)
        return -sum(p[p .> 0] .* log2.(p[p .> 0]))
    end

# ╔═╡ d28f93f2-db90-4658-95c4-39297f1cb7ac
function shannonEntropy(molecules::Vector{Molecule}, domain::Domain, nx_bins::Int=10)
    x_hist = fit(Histogram, [m.position[1] for m in molecules], range(0, domain.L[1], length=nx_bins+1))
    y_hist = fit(Histogram, [m.position[2] for m in molecules], range(0, domain.L[2], length=11))
    z_hist = fit(Histogram, [m.position[3] for m in molecules], range(0, domain.L[3], length=11))
    
    # Distribution de <v²> (200 bins sur [0, 200000])
    v2_hist = fit(Histogram, [norm(m.velocity)^2 for m in molecules], range(0, 200000, length=201))
    
    return entropy(x_hist) + entropy(y_hist) + entropy(z_hist) + entropy(v2_hist)
end

# ╔═╡ 158780a9-a947-47c3-9205-0aaae8d4978e
md"""
### 12.2 :
Réalisez la simulation suivante sans l'action de la gravité. Affichez l'évolution de l'entropie au cours du temps.
Qu'observez-vous ?

Conditions initiales de la simulation
- Domaine de simulation (m) : [−5 · 10−9, 5 · 10−9] × [−5 · 10−9, 5 · 10−9] × [−5 · 10−9, 5 · 10−9]
- Masse de l'hélium (kg) : 6.646 · 10−27
- Rayon de l'hélium - modèle hard sphere (m) : 1.1 · 10−10
- Nombre d'atomes : 400
- Position initiale des atomes : [−5 · 10−9, 0] × [−5 · 10−9, 0] × [−5 · 10−9, 0] ( 1
8 ème de domaine)
- Vitesse initiale de chaque atome (m/s) : 1400
- Pas de temps (s) : 1 · 10−14
- Temps nal (s) : 5 · 10−11
"""

# ╔═╡ 25290cf7-6296-41fa-80f7-1bf2f47f2450
function initaliseGasInRegion(N::Int, domain::Domain, prototype::Molecule, initialSpeed::Float64, x_range, y_range, z_range)
    final_molecules = Molecule[]
    for i in 1:N
        random_speed = randn(3)
        random_speed = random_speed / norm(random_speed) * initialSpeed
        random_position = [
            x_range[1] + rand() * (x_range[2] - x_range[1]),
            y_range[1] + rand() * (y_range[2] - y_range[1]),
            z_range[1] + rand() * (z_range[2] - z_range[1])
        ]
        m = Molecule(random_position, random_speed, prototype.mass, prototype.radius, prototype.formula)
        push!(final_molecules, m)
    end
    return final_molecules
end

# ╔═╡ c28d8f25-a33f-47df-b728-0e9a899608f6
molecules12 = initaliseGasInRegion(400, Domain([10e-9, 10e-9, 10e-9]), He, 1400.0, 
    (0, 5e-9), (0, 5e-9), (0, 5e-9))

# ╔═╡ eb7cfb2b-dd24-4fb8-ac76-aebe082f9c8d
function simulate12(molecules::Vector{Molecule}, domain::Domain, dt::Float64, finalTime::Float64)
    mean_speeds = []
    times = []
    alphas = []
    betas = []
    entropies = []
    
    @gif for t in 0:dt:finalTime 
        current_speed_sum = 0.0
        for molecule in molecules
            computeNextPosition(molecule, dt)
            applyBoundaries(molecule, domain)
        end
        for i in 1:length(molecules)
            for j in i+1:length(molecules)
                if detectCollision(molecules[i], molecules[j])
                    computeCollision(molecules[i], molecules[j])
                end
            end
            current_speed_sum += norm(molecules[i].velocity)
        end
        push!(mean_speeds, current_speed_sum / length(molecules))
        push!(times, t)
        entropy = shannonEntropy(molecules, domain)
        push!(entropies, entropy)
        alpha = He.mass * mean(norm.(getfield.(molecules, :velocity)).^2) / (3 * 1.380649e-23)
push!(alphas, alpha)
        beta = length(molecules) * He.mass * mean([norm(m.velocity)^2 for m in molecules]) / (3 * domainVolume(domain))
push!(betas, beta)
        p1 = scatter([m.position[1] for m in molecules], [m.position[2] for m in molecules], xlims=(0, domain.L[1]), ylims=(0, domain.L[2]), legend=false, title="Positions")

        plot(p1)
    end every 100
    return times, entropies
end

# ╔═╡ 2945adb0-86db-4265-93bf-262de9271a05
function simulateOpening(molecules::Vector{Molecule}, domain::Domain, dt::Float64, nsteps::Int;
                         nx_bins::Int = 10, animate::Bool = true, save_every::Int = 100)
    times     = Float64[]
    entropies = Float64[]

    function record!(k)
        for m in molecules
            computeNextPosition(m, dt)            # mouvement libre, sans gravité
            applyBoundaries(m, domain)
        end
        for i in 1:length(molecules), j in i+1:length(molecules)
            if detectCollision(molecules[i], molecules[j])
                computeCollision(molecules[i], molecules[j])
            end
        end
        push!(times, k*dt)
        push!(entropies, shannonEntropy(molecules, domain, nx_bins))
    end

    if animate
        anim = @animate for k in 0:nsteps
            record!(k)
            scatter([m.position[1] for m in molecules],
                    [m.position[2] for m in molecules],
                    [m.position[3] for m in molecules],
                    xlims=(0, domain.L[1]), ylims=(0, domain.L[2]), zlims=(0, domain.L[3]),
                    legend=false, markersize=2, markerstrokewidth=0,
                    camera=(30, 25), title="Ouverture de la paroi (3D)")
        end every save_every
    else
        anim = nothing
        for k in 0:nsteps
            record!(k)
        end
    end

    return (anim=anim, times=times, entropies=entropies)
end

# ╔═╡ 15b32504-f8fe-4bdb-adc0-687c7c848bc6
begin
    He_hs = Molecule([0.0,0.0,0.0], [0.0,0.0,0.0], 6.646e-27, 1.1e-10, "He")  # rayon hard sphere (énoncé)

    # Phase 1 — paroi fermée : 400 atomes confinés dans 1/8 du domaine
    domain_small = Domain([10e-9, 10e-9, 10e-9])
    gasOpen = initaliseGasInRegion(400, domain_small, He_hs, 1400.0,
                                   (0, 5e-9), (0, 5e-9), (0, 5e-9))
    phase1 = simulateOpening(gasOpen, domain_small, 1e-14, 5000; nx_bins=10, animate=false)

    # Phase 2 — on ouvre : le domaine double sur x ([-5,15]·1e-9), +5000 pas
    domain_large = Domain([20e-9, 10e-9, 10e-9])
    phase2 = simulateOpening(gasOpen, domain_large, 1e-14, 5000; nx_bins=20, animate=true)
end

# ╔═╡ 188655a1-3880-4340-a9bd-e20410d5461d
gif(phase2.anim, fps=20)

# ╔═╡ 171250e6-688e-4510-a1d2-61e82cec4b82
begin
    t_open = vcat(phase1.times, phase2.times .+ phase1.times[end])
    H_open = vcat(phase1.entropies, phase2.entropies)
    plot(t_open, H_open, xlabel="t (s)", ylabel="H (bits)",
         title="Entropie — paroi fermée puis ouverte", legend=false, lw=2)
    vline!([phase1.times[end]], ls=:dash, color=:red, label="ouverture")
end

# ╔═╡ ecbb55fd-f273-4238-be8a-6c5fb4ec4929
md"""
On observe une chute ou niveau de l'entropie, puis elle remontre progressivement se dispersant dans le nouveau domaine.
"""

# ╔═╡ 04eb480c-95cf-4bd6-b26d-865d95d2c2b1
md"""
## 13.1 :
Pourquoi les vitesses suivent des distributions diérentes ?
"""

# ╔═╡ a3a5e147-819d-414c-9162-6821e97c7cdc
md"""
### 13.2 :
Implémentez la nouvelle condition de bord en prenant en ajoutant en paramètres: la température du bord haut (z+) et la température du bord bas (z−).
"""

# ╔═╡ 9e2170bf-752c-48bf-bce5-4b8ce0a32ded
function applyBoundariesWithTemp(molecule::Molecule, domain::Domain, T_low::Float64, T_high::Float64)
    # axes x et y — réflexion spéculaire normale
    for i in 1:2
        if molecule.position[i] < 0
            molecule.position[i] = -molecule.position[i]
            molecule.velocity[i] = -molecule.velocity[i]
        elseif molecule.position[i] > domain.L[i]
            molecule.position[i] = 2*domain.L[i] - molecule.position[i]
            molecule.velocity[i] = -molecule.velocity[i]
        end
    end
    
    # axe z — réémission thermique
    if molecule.position[3] < 0
        σ = sqrt(1.380649e-23 * T_low / molecule.mass)
        molecule.position[3] = -molecule.position[3]
        molecule.velocity[1] = randn() * σ
        molecule.velocity[2] = randn() * σ
        molecule.velocity[3] = σ * sqrt(-2*log(rand()))  # Rayleigh, positif vers intérieur
    elseif molecule.position[3] > domain.L[3]
        σ = sqrt(1.380649e-23 * T_high / molecule.mass)
        molecule.position[3] = 2*domain.L[3] - molecule.position[3]
        molecule.velocity[1] = randn() * σ
        molecule.velocity[2] = randn() * σ
        molecule.velocity[3] = -σ * sqrt(-2*log(rand()))  # Rayleigh, négatif vers intérieur
    end
end

# ╔═╡ 4ffc6f10-cc9c-4572-84e7-fa63deac5ca5
function simulate(molecules::Vector{Molecule}, domain::Domain, dt::Float64, finalTime::Float64;
        gravity::Vector{Float64} = [0.0, 0.0, 0.0],  
        T_low  = nothing,                              
        T_high = nothing,
        view::Symbol     = :xyz,                      
        color_species::Bool   = false,                 
        compute_entropy::Bool = false,                 
        animate::Bool         = true,                  
        save_every::Int       = 100)

    kb = 1.380649e-23
    V  = domainVolume(domain)

    times = Float64[]; mean_speeds = Float64[]
    Ts = Float64[]; Ps = Float64[]; entropies = Float64[]

    # --- un pas de simulation + collecte des grandeurs ---
    function record!(t)
        for m in molecules
            computeNextPositionGravity(m, dt, gravity)
            isnothing(T_low) ? applyBoundaries(m, domain) :
                               applyBoundariesWithTemp(m, domain, T_low, T_high)
        end
        for i in 1:length(molecules), j in i+1:length(molecules)
            if detectCollision(molecules[i], molecules[j])
                computeCollision(molecules[i], molecules[j])
            end
        end
        sum_mv2 = sum(m.mass * norm(m.velocity)^2 for m in molecules)
        push!(times, t)
        push!(mean_speeds, mean(norm(m.velocity) for m in molecules))
        push!(Ts, sum_mv2 / (length(molecules) * 3 * kb))   # α : <mv²>/(3kb) → T
        push!(Ps, sum_mv2 / (3 * V))                         # β : Σmv²/(3V) → P
        compute_entropy && push!(entropies, shannonEntropy(molecules, domain))
    end

    # --- construction d'une image selon la vue choisie ---
    function frameplot()
        getf = view === :xy ? (m->m.position[1], m->m.position[2]) :
               view === :xz ? (m->m.position[1], m->m.position[3]) :
                              (m->m.position[1], m->m.position[2], m->m.position[3])
        lims = view === :xy ? (xlims=(0,domain.L[1]), ylims=(0,domain.L[2])) :
               view === :xz ? (xlims=(0,domain.L[1]), ylims=(0,domain.L[3])) :
                              (xlims=(0,domain.L[1]), ylims=(0,domain.L[2]), zlims=(0,domain.L[3]))
        common = view === :xyz ? (markersize=2, markerstrokewidth=0, camera=(30,25)) :
                                 (markersize=2, markerstrokewidth=0)
        coords(sub) = [[f(m) for m in sub] for f in getf]

        if color_species
            sps = unique(m.formula for m in molecules)
            plt = scatter(coords(filter(m->m.formula==sps[1], molecules))...;
                          label=sps[1], legend=true, title="Simulation", lims..., common...)
            for sp in sps[2:end]
                scatter!(plt, coords(filter(m->m.formula==sp, molecules))...; label=sp, common...)
            end
            return plt
        else
            return scatter(coords(molecules)...;
                           legend=false, title="Simulation", lims..., common...)
        end
    end

    if animate
        anim = @animate for t in 0:dt:finalTime
            record!(t)
            frameplot()
        end every save_every
    else
        anim = nothing
        for t in 0:dt:finalTime
            record!(t)
        end
    end

    return (anim=anim, times=times, mean_speeds=mean_speeds, T=Ts, P=Ps, entropy=entropies)
end

# ╔═╡ 17f7570b-318a-4582-bcff-d2230c49013a
begin
	domain = Domain([10e-9, 10e-9, 10e-9])
	moleculesRandom = initaliseGasAtRandom(400, domain, [He], 1400.0)
	res7 = simulate(moleculesRandom, domain, 1e-14, 1e-10)
	gif(res7.anim, fps=20)
end

# ╔═╡ 205df3c6-ff87-458c-8b65-232bd38a0f6b
histogram([norm(m.velocity) for m in moleculesRandom], 
    xlabel="v (m/s)", ylabel="nombre d'atomes", 
    legend=false, title="Distribution des vitesses")

# ╔═╡ 402cdb02-9388-4efe-b25b-c05fdbf494fc
begin
    domain8 = Domain([20e-9, 20e-9, 20e-9])
    molecules8 = initaliseGasAtRandom(500, domain8, [He], 1367.0)
    res8 = simulate(molecules8, domain8, 1e-14, 5e-11; gravity=[0.0, 0.0, -9.81e13], view=:xyz)
end

# ╔═╡ a25b46f6-b91d-4ce3-b2c7-4be439c4bc47
gif(res8.anim, fps=20)

# ╔═╡ 8047a34e-aec3-47da-8f82-6924d6d6895d
histogram([m.position[3] for m in molecules8], 
    normalize=true,
    xlabel="z (m)", ylabel="probabilité", 
    legend=false, title="Probabilité de présence en z")

# ╔═╡ d056e4e7-29cb-4db5-864c-e255415064a0
histogram([m.position[3] for m in molecules8],
    weights=[norm(m.velocity)^2 for m in molecules8] .* He.mass / (3 * domainVolume(domain8)),
    normalize=false,
    xlabel="z (m)", ylabel="pression (Pa)",
    legend=false, title="Pression en fonction de z")

# ╔═╡ 1347075f-4dd5-4b72-9ad7-ab49d6f99621
begin
    Ar = Molecule([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 6.634e-26, 1.88e-10, "Ar")
    
    domain9 = Domain([20e-9, 20e-9, 20e-9])
    molecules9 = vcat(
        initaliseGasAtRandom(400, domain9, [He], 789.45),
        initaliseGasAtRandom(200, domain9, [Ar], 249.88)
    )
    res9 = simulate(molecules9, domain9, 1e-14, 5e-11;
    gravity=[0.0,0.0,-9.81e13], color_species=true)
    gif(res9.anim, fps=20)
end

# ╔═╡ e28de34a-99c6-4c34-94b9-dcdaa11d2eef
begin
p_he = histogram([m.position[3] for m in molecules9 if m.formula == "He"],
    normalize=true, color=:blue, label="He",
    xlabel="z (m)", ylabel="probabilité")

p_ar = histogram([m.position[3] for m in molecules9 if m.formula == "Ar"],
    normalize=true, color=:red, label="Ar",
    xlabel="z (m)", ylabel="probabilité")

plot(p_he, p_ar, layout=(1,2))
end

# ╔═╡ eb953969-6e32-47cd-afa3-22e324ed32ea
begin
weights_he = [m.mass * norm(m.velocity)^2 for m in molecules9 if m.formula == "He"]
weights_ar = [m.mass * norm(m.velocity)^2 for m in molecules9 if m.formula == "Ar"]
z_he = [m.position[3] for m in molecules9 if m.formula == "He"]
z_ar = [m.position[3] for m in molecules9 if m.formula == "Ar"]
end

# ╔═╡ bf9992bc-5637-4fc4-9bc0-4332b64bfdbc
begin
    p1 = histogram(z_he, weights=weights_he, color=:blue, label="He", title="<mv²>", alpha=0.5)
    histogram!(p1, z_ar, weights=weights_ar, color=:red, label="Ar", alpha=0.5)
    plot(p1)
end

# ╔═╡ 3150f9cb-d3c6-4a47-872b-ecf879dc3032
begin
p3 = histogram(z_he, weights=weights_he ./ (3*1.380649e-23), color=:blue, label="He", title="Température", alpha=0.5)
histogram!(p3, z_ar, weights=weights_ar ./ (3*1.380649e-23), color=:red, label="Ar", alpha=0.5)
end

# ╔═╡ d68b317d-fcd5-45bf-abef-e487b8a5d591
begin
p2 = histogram(z_he, weights=weights_he ./ (3*domainVolume(domain9)), color=:blue, label="He", title="Pression", alpha=0.5)
histogram!(p2, z_ar, weights=weights_ar ./ (3*domainVolume(domain9)), color=:red, label="Ar", alpha=0.5)

plot(p2)
end

# ╔═╡ 69e417a1-1822-4976-a26f-3186430d68e2
begin
times, alphas = simulateCollectStats(molecules9, domain9, 1e-14, 5e-11, [0.0, 0.0, -9.81e13])

window = 1000
stability = []

for i in window:length(alphas)
    fenetre = alphas[i-window+1:i]
    push!(stability, std(fenetre)/mean(fenetre))
end

plot(stability,
    xlabel="pas de temps", ylabel="σ/μ",
    title="Critère de stabilité", legend=false)
hline!([0.05], color=:red, label="seuil 5%")
end

# ╔═╡ 8f2fbc00-f600-435e-b3c3-c190be49b158
"""md
Question 13.3 :
Réalisez la simulation suivante.
Conditions initiales de la simulation
 Domaine de simulation (m) : [−2 · 10−9, 2 · 10−9] × [−2 · 10−9, 2 · 10−9] × [−5 · 10−9, 5 · 10−9]
 Température du bord haut (K) : 700
 Température du bord bas (K): 300
 Masse de l'hélium (kg) : 6.646 · 10−27
 Rayon de l'hélium - modèle hard sphere (m) : 1.1 · 10−10
 Nombre d'atomes : 500
 Vitesse initiale de chaque atome (m/s) : 1400
 Pas de temps (s) : 1 · 10−14
 Temps nal (s) : 1 · 10−10
Question 13.4 :
Achez l'évolution de l'entropie au cours du temps.
Question 13.5 :
A la n de la simulation tracez l'évolution de la température en fonction de z. Qu'observez-vous ?
"""

# ╔═╡ f11f0c92-44bc-4690-afde-d9be21b4f536
# ╠═╡ disabled = true
#=╠═╡
function simulate13(molecules::Vector{Molecule}, domain::Domain, dt::Float64, finalTime::Float64, T_low::Float64, T_high::Float64)
    times = []
    entropies = []
    
    @gif for t in 0:dt:finalTime
        for molecule in molecules
            computeNextPosition(molecule, dt)
            applyBoundariesWithTemp(molecule, domain, T_low, T_high)
        end
        for i in 1:length(molecules)
            for j in i+1:length(molecules)
                if detectCollision(molecules[i], molecules[j])
                    computeCollision(molecules[i], molecules[j])
                end
            end
        end
        push!(times, t)
        push!(entropies, shannonEntropy(molecules, domain))
        
        scatter([m.position[1] for m in molecules], [m.position[3] for m in molecules],
            xlims=(0, domain.L[1]), ylims=(0, domain.L[3]),
            legend=false, title="Positions (x,z)")
    end every 100
    
    return g, times, entropies
end
  ╠═╡ =#

# ╔═╡ 06be4d04-b604-44ad-b21c-41f3f0d154fe
begin
	domain13 = Domain([4e-9, 4e-9, 10e-9])
molecules13 = initaliseGasAtRandom(500, domain13, [He], 1400.0)
res13 = simulate(molecules13, domain13, 1e-14, 1e-10;
                 T_low=300.0, T_high=700.0, view=:xyz, compute_entropy=true)
gif(res13.anim, fps=20)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
Plots = "~1.41.6"
StatsBase = "~0.34.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "f7cf84feed17865ab16e11fc767925aa177caf91"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a21c5464519504e41e0cbc91f0188e8ca23d7440"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "01ba9d15e9eae375dc1eb9589df76b3572acd3f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "b7bfd56fa66616138dfe5237da4dc13bbd83c67f"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee0585b62671ce88e48d3409733230b401c9775c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.22"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7dd7173f7129a1b6f84e0f03e0890cd1189b0659"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.22+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "926c6af3a037c68d02596a44c22ec3595f5f760b"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╠═e69a6728-e8ee-4a23-afc4-08dc8aafa669
# ╠═d774739f-4fec-4de3-a467-158ce2f57efd
# ╠═df96d71a-8c94-4268-b4b5-f899db9a42d1
# ╠═4d43a8ca-1ec8-44c0-8a6b-d16eaa95a3bd
# ╠═6312f49c-0e57-11f1-004c-852d60e5287d
# ╠═1479e2d1-55e5-4aba-b3d9-bd755e7b19eb
# ╠═98816279-a0f4-48f4-9c11-a2d99934d799
# ╠═3c864515-564e-4611-bdc5-2f9f770f610d
# ╠═0206106c-81ab-4f35-ac9e-93d9504d4e61
# ╠═db24edb4-8b4b-4d83-a875-ecfee8f89e9a
# ╠═05fd32d1-97e9-4ab5-a1d6-3fcbbd9f925b
# ╠═f7164257-150d-430a-9fd4-84115dcad863
# ╠═185757a3-3bc2-43f6-a843-e1da59146e79
# ╠═1d452f32-3c01-4269-afa6-fa151bd990b2
# ╠═a9662ab2-221e-40bc-801f-884aa443a1fd
# ╠═1b0775ba-e51a-462e-8aa3-ef3bc0ec9fdf
# ╠═b534f811-34d5-473f-bb44-b7c1b2841f98
# ╠═5abe6eee-76ec-4d7d-b27e-7b72a784b1c7
# ╠═ed7f37a2-4a5a-4c35-842f-747353e48abd
# ╠═8d92dc16-ef6b-48d6-94f1-3b1287b64c2c
# ╠═72ce9fa6-f5a7-4861-971b-b868db127313
# ╠═11b51dba-e52e-4153-a1a3-151b4e3f850f
# ╠═b460e0df-5aa3-4a1e-b2b3-a92ed3007fa3
# ╠═78a927f5-5c01-4e44-a73b-35d952dba07a
# ╠═7c60fb05-85f0-4fe8-a9a0-8c59f9706f38
# ╠═f53cb4ae-5b48-4f15-bd9b-f707dc8a3fc6
# ╠═eaa968af-d013-4d16-b62f-713500a68b18
# ╠═b2ca41ad-c8b8-423e-a75f-56bd2b737890
# ╠═375c8598-0034-445e-b3e8-bcd58a471c5f
# ╠═22b1998c-045d-4d0b-9ca8-40ab8c208d95
# ╠═e018bcc6-5f90-429f-a749-6b4011dfbad4
# ╠═6b2dc248-8208-47fd-987c-5c1726127353
# ╠═c6e70817-cebf-4ac8-bf11-79df05a39e6c
# ╠═ff8a5c41-8ec5-4208-a1d8-165dfb6e9d99
# ╠═8c2b6820-8a02-4879-afe5-3b35cebe1ea8
# ╠═40d672a8-4dee-48f7-a241-5e14df293bc2
# ╠═ab3d206c-35d9-40ef-8154-3c1cc6b6fc4e
# ╠═aff6073d-d727-4b0e-b3b2-166b2175e72e
# ╠═06bcf237-ccc7-411a-b9e9-f8cff71b8fd4
# ╠═f04d6d55-c7e8-4437-8d9e-1c1bc8acf8a3
# ╠═f00cee09-d424-4b57-986e-0215dde37cc0
# ╠═e575895b-63ef-44b2-9265-aac1af08cb60
# ╠═4ffc6f10-cc9c-4572-84e7-fa63deac5ca5
# ╠═17f7570b-318a-4582-bcff-d2230c49013a
# ╠═2eae66aa-4497-4b74-b629-76f40d269728
# ╠═c7efe043-a7dc-4217-8fb9-d20bcd3ed36f
# ╠═205df3c6-ff87-458c-8b65-232bd38a0f6b
# ╠═9de0000d-0399-4c9a-9ae4-74a386e06932
# ╠═a2bf77de-0cbc-480c-899e-2a2892dcca56
# ╠═54811911-1303-4acb-a666-7bb72dcf3ebb
# ╠═b7b7b4e9-a125-47e3-8054-5ecf5b21d704
# ╠═1ed02f4a-6ba4-416c-8554-7b7d71e04368
# ╟─431388d3-71a6-4187-8476-a4a939fbfa37
# ╠═402cdb02-9388-4efe-b25b-c05fdbf494fc
# ╠═a25b46f6-b91d-4ce3-b2c7-4be439c4bc47
# ╠═7bb387ab-f469-4179-8733-912203ba94b0
# ╠═8047a34e-aec3-47da-8f82-6924d6d6895d
# ╠═f3d01dff-1774-4cad-8829-652bbf4cba9b
# ╠═4e987429-a0b3-45ae-b973-396afb05f951
# ╠═c47a9bf4-6b63-4551-8db2-e5e692294908
# ╠═3f5235ca-9adf-4e9d-b6f1-22ea36606712
# ╠═d056e4e7-29cb-4db5-864c-e255415064a0
# ╠═f71e8ecb-e04a-4df3-9568-f3d946b0ec60
# ╟─6835f289-57cd-44cd-a5f3-32cf0f8232ba
# ╠═1347075f-4dd5-4b72-9ad7-ab49d6f99621
# ╠═855ace40-74b1-4aa3-b345-a9b4c23d8173
# ╠═e28de34a-99c6-4c34-94b9-dcdaa11d2eef
# ╠═7078bc83-93c7-4736-b407-fa665aa78c02
# ╠═b9eb2927-b1fd-41d4-be8c-9d633eb7bbff
# ╠═eb953969-6e32-47cd-afa3-22e324ed32ea
# ╠═bf9992bc-5637-4fc4-9bc0-4332b64bfdbc
# ╠═d68b317d-fcd5-45bf-abef-e487b8a5d591
# ╠═3150f9cb-d3c6-4a47-872b-ecf879dc3032
# ╠═77a04d8c-c3c2-4fc5-8673-ea206a6b4d73
# ╠═d7d821a0-e809-4d09-b0fc-f2cab5a970c7
# ╠═e617fe99-22c3-4f21-843c-7c3c3a086d97
# ╟─fda906ad-394f-4b72-9f2f-cef7a5821b22
# ╠═69e417a1-1822-4976-a26f-3186430d68e2
# ╠═8fb8ce4e-ea89-4008-a5fa-99be6205fafc
# ╠═ead9a443-4aa3-4a45-9a14-e028e9862dc7
# ╠═2be52041-7d82-40e5-b62b-301ac0bf59b5
# ╠═cec34ee7-b494-4607-820c-4be5f76539e9
# ╠═1c2bd9bc-c679-4b41-8a8d-fbb79556c1a9
# ╠═d28f93f2-db90-4658-95c4-39297f1cb7ac
# ╠═158780a9-a947-47c3-9205-0aaae8d4978e
# ╠═25290cf7-6296-41fa-80f7-1bf2f47f2450
# ╠═c28d8f25-a33f-47df-b728-0e9a899608f6
# ╟─eb7cfb2b-dd24-4fb8-ac76-aebe082f9c8d
# ╠═2945adb0-86db-4265-93bf-262de9271a05
# ╠═15b32504-f8fe-4bdb-adc0-687c7c848bc6
# ╠═188655a1-3880-4340-a9bd-e20410d5461d
# ╠═171250e6-688e-4510-a1d2-61e82cec4b82
# ╠═ecbb55fd-f273-4238-be8a-6c5fb4ec4929
# ╠═04eb480c-95cf-4bd6-b26d-865d95d2c2b1
# ╠═a3a5e147-819d-414c-9162-6821e97c7cdc
# ╠═9e2170bf-752c-48bf-bce5-4b8ce0a32ded
# ╠═8f2fbc00-f600-435e-b3c3-c190be49b158
# ╟─f11f0c92-44bc-4690-afde-d9be21b4f536
# ╠═06be4d04-b604-44ad-b21c-41f3f0d154fe
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
