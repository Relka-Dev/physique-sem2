### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ df96d71a-8c94-4268-b4b5-f899db9a42d1
using Plots, Statistics

# ╔═╡ 6312f49c-0e57-11f1-004c-852d60e5287d
mutable struct Molecule
    r::Vector{Float64}   # position [x, y, z] en m
    v::Vector{Float64}   # vitesse [vx, vy, vz] en m/s
    m::Float64           # masse en kg
    ρ::Float64           # rayon en m
    formule::String
end

# ╔═╡ section4-collision-detection
md"""
## Section 4 : Interaction entre deux molécules

### Q4.1
Les deux molécules peuvent subir une **collision élastique** (choc élastique) lorsqu'elles se rapprochent suffisamment. Ce modèle est pertinent sous les hypothèses de la théorie cinétique des gaz :
- Les molécules sont des sphères dures (hard spheres)
- Les collisions conservent l'énergie cinétique et la quantité de mouvement
- On néglige les forces à longue portée (van der Waals, etc.)

### Q4.2 — Scénarios de test
| # | Description | Conditions initiales |
|---|-------------|----------------------|
| 1 | Choc frontal égal | m1=m2, v1=(1,0,0), v2=(-1,0,0) → échange de vitesses |
| 2 | Choc frontal masses inégales | m1=2m2, v1=(1,0,0), v2=0 → conservation p et Ec |
| 3 | Choc tangentiel | contact à 90°, vérifier composantes |
| 4 | Pas de collision | centres distants, fonction doit retourner false |
"""

# ╔═╡ q4-3-detection
"""
Détecte si deux molécules sont en collision.
Retourne true si la distance entre centres ≤ somme des rayons.
"""
function areColliding(m1::Molecule, m2::Molecule)::Bool
    dist = norm(m1.r - m2.r)
    return dist <= (m1.ρ + m2.ρ)
end

# ╔═╡ q4-4-collision
"""
Met à jour les vitesses de deux molécules après une collision élastique.
Algorithme : collision de sphères dures (composante normale échangée selon les masses).
"""
function resolveCollision!(m1::Molecule, m2::Molecule)
    # Vecteur normal unitaire de m1 vers m2
    d = m2.r - m1.r
    dist = norm(d)
    if dist == 0.0
        return  # positions identiques, on ignore
    end
    n = d / dist

    # Vitesses relatives dans la direction normale
    v_rel = dot(m1.v - m2.v, n)

    # Si les molécules s'éloignent déjà, pas de collision à résoudre
    if v_rel <= 0
        return
    end

    # Impulsion scalaire (collision élastique)
    j = 2 * v_rel / (1/m1.m + 1/m2.m)

    m1.v -= (j / m1.m) * n
    m2.v += (j / m2.m) * n
end

# ╔═╡ q4-4-tests
md"""
### Tests Q4.4
"""

# ╔═╡ q4-test-run
begin
    # Test 1 : choc frontal, masses égales → échange de vitesses
    t1_a = Molecule([-1.5e-10, 0.0, 0.0], [100.0, 0.0, 0.0], 1.0, 1.0e-10, "A")
    t1_b = Molecule([ 1.5e-10, 0.0, 0.0], [-100.0, 0.0, 0.0], 1.0, 1.0e-10, "B")
    p_avant = t1_a.m * t1_a.v + t1_b.m * t1_b.v
    Ec_avant = 0.5 * t1_a.m * norm(t1_a.v)^2 + 0.5 * t1_b.m * norm(t1_b.v)^2
    resolveCollision!(t1_a, t1_b)
    p_apres = t1_a.m * t1_a.v + t1_b.m * t1_b.v
    Ec_apres = 0.5 * t1_a.m * norm(t1_a.v)^2 + 0.5 * t1_b.m * norm(t1_b.v)^2
    println("Test 1 (choc frontal égal) :")
    println("  Conservation quantité de mouvement : ", isapprox(p_avant, p_apres, atol=1e-20))
    println("  Conservation énergie cinétique     : ", isapprox(Ec_avant, Ec_apres, atol=1e-20))
    println("  v1 après = ", t1_a.v, " (attendu: [-100, 0, 0])")
    println("  v2 après = ", t1_b.v, " (attendu: [100, 0, 0])")

    # Test 2 : masse au repos, m1 = 2*m2
    t2_a = Molecule([-1.5e-10, 0.0, 0.0], [100.0, 0.0, 0.0], 2.0, 1.0e-10, "A")
    t2_b = Molecule([ 1.5e-10, 0.0, 0.0], [0.0, 0.0, 0.0],   1.0, 1.0e-10, "B")
    p_avant2 = t2_a.m * t2_a.v + t2_b.m * t2_b.v
    Ec_avant2 = 0.5 * t2_a.m * norm(t2_a.v)^2 + 0.5 * t2_b.m * norm(t2_b.v)^2
    resolveCollision!(t2_a, t2_b)
    p_apres2 = t2_a.m * t2_a.v + t2_b.m * t2_b.v
    Ec_apres2 = 0.5 * t2_a.m * norm(t2_a.v)^2 + 0.5 * t2_b.m * norm(t2_b.v)^2
    println("\nTest 2 (choc frontal, m1=2*m2) :")
    println("  Conservation quantité de mouvement : ", isapprox(p_avant2, p_apres2, atol=1e-20))
    println("  Conservation énergie cinétique     : ", isapprox(Ec_avant2, Ec_apres2, atol=1e-20))

    # Test 3 : pas en collision → areColliding = false
    t3_a = Molecule([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 1.0, 1.0e-10, "A")
    t3_b = Molecule([1.0, 0.0, 0.0], [0.0, 0.0, 0.0],  1.0, 1.0e-10, "B")
    println("\nTest 3 (pas de collision, distance >> rayons) :")
    println("  areColliding = ", areColliding(t3_a, t3_b), " (attendu: false)")

    # Test 4 : en collision → areColliding = true
    t4_a = Molecule([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1.0e-10, "A")
    t4_b = Molecule([1.5e-10, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1.0e-10, "B")
    println("\nTest 4 (en collision, distance < 2*rayon) :")
    println("  areColliding = ", areColliding(t4_a, t4_b), " (attendu: true)")
end

# ╔═╡ section5-domain
md"""
## Section 5 : Le domaine de simulation
"""

# ╔═╡ q5-1-domain-struct
mutable struct Domain
    Lx::Float64   # longueur axe x (m)
    Ly::Float64   # longueur axe y (m)
    Lz::Float64   # longueur axe z (m)
end

# ╔═╡ q5-2-volume
"""Calcule le volume du domaine cubique."""
volume(d::Domain) = d.Lx * d.Ly * d.Lz

# ╔═╡ q5-3-test-volume
begin
    d_test = Domain(2.0, 3.0, 4.0)
    println("Test volume Domain(2,3,4) = ", volume(d_test), " (attendu: 24.0)")
    d_cube = Domain(1.0, 1.0, 1.0)
    println("Test volume Domain(1,1,1) = ", volume(d_cube), " (attendu: 1.0)")
end

# ╔═╡ section6-boundary
md"""
## Section 6 : Condition de bord — réflexion spéculaire
"""

# ╔═╡ q6-1-apply-boundary
"""
Corrige la position et la vitesse d'une molécule si elle sort du domaine.
Le domaine est centré en 0 : axe x ∈ [-Lx/2, Lx/2], idem y et z.
"""
function applyBoundary!(mol::Molecule, dom::Domain)
    half = [dom.Lx/2, dom.Ly/2, dom.Lz/2]
    for i in 1:3
        if mol.r[i] > half[i]
            mol.r[i] = 2*half[i] - mol.r[i]   # réflexion
            mol.v[i] = -mol.v[i]
        elseif mol.r[i] < -half[i]
            mol.r[i] = -2*half[i] - mol.r[i]   # réflexion
            mol.v[i] = -mol.v[i]
        end
    end
end

# ╔═╡ q6-2-boundary-tests
begin
    dom_test = Domain(2.0, 2.0, 2.0)  # x,y,z ∈ [-1, 1]

    # Test 1 : sortie par le bord +x
    b1 = Molecule([1.2, 0.0, 0.0], [10.0, 5.0, 0.0], 1.0, 0.1, "T")
    applyBoundary!(b1, dom_test)
    println("Test 1 (+x) : r=", round.(b1.r, digits=3), " v=", b1.v)
    println("  attendu r≈[0.8,0,0], vx<0 : ", b1.r[1] ≈ 0.8 && b1.v[1] < 0)

    # Test 2 : sortie par le bord -x
    b2 = Molecule([-1.3, 0.0, 0.0], [-10.0, 0.0, 0.0], 1.0, 0.1, "T")
    applyBoundary!(b2, dom_test)
    println("Test 2 (-x) : r=", round.(b2.r, digits=3), " v=", b2.v)
    println("  attendu r≈[-0.7,0,0], vx>0 : ", b2.r[1] ≈ -0.7 && b2.v[1] > 0)

    # Test 3 : sortie par +y
    b3 = Molecule([0.0, 1.1, 0.0], [0.0, 5.0, 0.0], 1.0, 0.1, "T")
    applyBoundary!(b3, dom_test)
    println("Test 3 (+y) : vy<0 : ", b3.v[2] < 0)

    # Test 4 : sortie par -y
    b4 = Molecule([0.0, -1.4, 0.0], [0.0, -5.0, 0.0], 1.0, 0.1, "T")
    applyBoundary!(b4, dom_test)
    println("Test 4 (-y) : vy>0 : ", b4.v[2] > 0)

    # Test 5 : sortie par +z
    b5 = Molecule([0.0, 0.0, 1.05], [0.0, 0.0, 3.0], 1.0, 0.1, "T")
    applyBoundary!(b5, dom_test)
    println("Test 5 (+z) : vz<0 : ", b5.v[3] < 0)

    # Test 6 : sortie par -z
    b6 = Molecule([0.0, 0.0, -1.2], [0.0, 0.0, -3.0], 1.0, 0.1, "T")
    applyBoundary!(b6, dom_test)
    println("Test 6 (-z) : vz>0 : ", b6.v[3] > 0)

    # Test 7 : sortie par deux bords en même temps (+x et +y)
    b7 = Molecule([1.3, 1.2, 0.0], [5.0, 3.0, 0.0], 1.0, 0.1, "T")
    applyBoundary!(b7, dom_test)
    println("Test 7 (coin +x+y) : vx<0 && vy<0 : ", b7.v[1] < 0 && b7.v[2] < 0)

    # Test 8 : molécule dans le domaine → rien ne change
    b8 = Molecule([0.5, 0.3, -0.2], [1.0, 2.0, 3.0], 1.0, 0.1, "T")
    v_before = copy(b8.v); r_before = copy(b8.r)
    applyBoundary!(b8, dom_test)
    println("Test 8 (dans domaine) : pas de changement : ", b8.r == r_before && b8.v == v_before)
end

# ╔═╡ section7-helium-simulation
md"""
## Section 7 : Première simulation d'un gaz d'Hélium
"""

# ╔═╡ q7-setup
begin
    # Paramètres
    kb = 1.380649e-23   # constante de Boltzmann

    He_m    = 6.646e-27   # masse He (kg)
    He_ρ    = 1.1e-10     # rayon He (m)
    N_atoms = 400
    v0      = 1400.0      # vitesse initiale (m/s)
    dt      = 1e-14       # pas de temps (s)
    t_final = 2e-11       # temps final (s)
    n_steps = Int(t_final / dt)

    L = 5e-9  # demi-longueur du domaine (m)
    dom7 = Domain(2L, 2L, 2L)

    # Initialisation aléatoire des positions et directions de vitesse
    function init_helium(N, L, v0, m, ρ)
        mols = Vector{Molecule}(undef, N)
        for i in 1:N
            r = [rand()*2L - L, rand()*2L - L, rand()*2L - L]
            # direction aléatoire sur la sphère unité
            θ = acos(1 - 2*rand())
            φ = 2π * rand()
            v = v0 * [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
            mols[i] = Molecule(r, v, m, ρ, "He")
        end
        return mols
    end

    atoms = init_helium(N_atoms, L, v0, He_m, He_ρ)
    println("Simulation initialisée : ", length(atoms), " atomes He")
end

# ╔═╡ q7-run-simulation
begin
    # Historique pour les grandeurs statistiques (1 point toutes les 100 steps)
    record_every = 100
    n_records = div(n_steps, record_every)

    mean_v_hist  = zeros(n_records)
    alpha_hist   = zeros(n_records)
    beta_hist    = zeros(n_records)
    time_hist    = zeros(n_records)

    rec_idx = 1

    for step in 1:n_steps
        # 1. Déplacer toutes les molécules
        for mol in atoms
            mol.r .+= mol.v .* dt
        end

        # 2. Appliquer les conditions de bord
        for mol in atoms
            applyBoundary!(mol, dom7)
        end

        # 3. Détecter et résoudre les collisions (O(N²) simple)
        for i in 1:N_atoms-1
            for j in i+1:N_atoms
                if areColliding(atoms[i], atoms[j])
                    resolveCollision!(atoms[i], atoms[j])
                end
            end
        end

        # 4. Enregistrer les statistiques périodiquement
        if step % record_every == 0
            speeds = [norm(mol.v) for mol in atoms]
            v2_mean = mean([norm(mol.v)^2 for mol in atoms])
            mean_v_hist[rec_idx]  = mean(speeds)
            alpha_hist[rec_idx]   = He_m * v2_mean / (3 * kb)
            beta_hist[rec_idx]    = N_atoms * He_m * v2_mean / (3 * volume(dom7))
            time_hist[rec_idx]    = step * dt
            rec_idx += 1
        end
    end
    println("Simulation terminée.")
end

# ╔═╡ q7-2-mean-speed
begin
    p1 = plot(time_hist .* 1e12, mean_v_hist,
        xlabel="Temps (ps)", ylabel="Vitesse moyenne (m/s)",
        title="Q7.2 — Évolution de la vitesse moyenne", legend=false)
    display(p1)
end

# ╔═╡ q7-3-speed-distribution
begin
    final_speeds = [norm(mol.v) for mol in atoms]
    p2 = histogram(final_speeds, bins=40,
        xlabel="||v|| (m/s)", ylabel="Nombre d'atomes",
        title="Q7.3 — Distribution des vitesses (temps final)",
        legend=false)
    display(p2)
end

# ╔═╡ q7-4-alpha
begin
    p3 = plot(time_hist .* 1e12, alpha_hist,
        xlabel="Temps (ps)", ylabel="α (K)",
        title="Q7.4 — α = m<v²>/(3kb) ≡ Température cinétique",
        legend=false)
    display(p3)
    println("α final ≈ ", round(alpha_hist[end], digits=2), " K")
    println("→ α représente la température cinétique du gaz (en Kelvin).")
end

# ╔═╡ q7-5-beta
begin
    p4 = plot(time_hist .* 1e12, beta_hist,
        xlabel="Temps (ps)", ylabel="β (Pa)",
        title="Q7.5 — β = Nm<v²>/(3V) ≡ Pression",
        legend=false)
    display(p4)
    println("β final ≈ ", round(beta_hist[end], digits=2), " Pa")
    println("→ β représente la pression du gaz (en Pascal), via l'équation cinétique des gaz.")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Plots = "~1"
"""
