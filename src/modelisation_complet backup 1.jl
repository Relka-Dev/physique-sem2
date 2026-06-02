### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ deps
using Plots, Statistics, Random

# ════════════════════════════════════════════════════════════════
# SECTION 1 & 2 — Classe Molecule + types de molécules
# ════════════════════════════════════════════════════════════════

# ╔═╡ molecule-struct
mutable struct Molecule
    r::Vector{Float64}   # position [x, y, z] (m)
    v::Vector{Float64}   # vitesse  [vx,vy,vz] (m/s)
    m::Float64           # masse (kg)
    ρ::Float64           # rayon (m)
    formule::String
end

# ╔═╡ molecule-instances
md"""
## Ex 2.1 — Valeurs pour He, Ne, N₂, O₂
| Molécule | Masse (kg)    | Rayon (m)    | Remarques        |
|----------|---------------|--------------|------------------|
| He       | 6.646e-27     | 1.40e-10     | gaz noble monoatomique |
| Ne       | 3.351e-26     | 1.54e-10     | gaz noble monoatomique |
| N₂       | 4.652e-26     | 1.85e-10     | diatomique       |
| O₂       | 5.314e-26     | 1.73e-10     | diatomique       |

## Ex 2.2 — Différences et applicabilité de la TKG
He et Ne sont **monoatomiques** : pas de degrés de liberté de rotation/vibration.
N₂ et O₂ sont **diatomiques** : ont des degrés de liberté rotationnels.
La théorie cinétique des gaz s'applique à toutes ces molécules pour le mouvement de translation,
mais le théorème d'équipartition de l'énergie doit tenir compte des degrés de liberté supplémentaires
pour N₂ et O₂ (capacité thermique différente).
"""

# ════════════════════════════════════════════════════════════════
# SECTION 3 — Déplacement (Euler explicite)
# ════════════════════════════════════════════════════════════════

# ╔═╡ move-theory
md"""
## Q3.1 — Équations de Newton
Sans force extérieure : m·(d²r/dt²) = 0  ⟹  dv/dt = 0,  dr/dt = v

## Q3.2 — Différences finies (Euler explicite)
v(t+dt) = v(t)
r(t+dt) = r(t) + v(t)·dt

Avec gravité selon -z :
v_z(t+dt) = v_z(t) - g·dt
r(t+dt)   = r(t) + v(t+dt)·dt  (ou v(t)·dt pour Euler standard)
"""

# ╔═╡ compute-next-position
"""
Calcule la position de la molécule au pas de temps suivant (Euler explicite).
Gravité optionnelle selon -z (défaut g=0).
"""
function computeNextPosition!(mol::Molecule, dt::Float64; g::Float64=0.0)
    mol.v[3] -= g * dt          # effet de la gravité sur vz
    mol.r   .+= mol.v .* dt     # mise à jour de la position
end

# ╔═╡ q3-4-validation
begin
    mol_test = Molecule([0.0, 0.0, 0.0], [100.0, 50.0, 0.0], 6.646e-27, 1.4e-10, "He")
    dt_test  = 1e-14
    t_max_test = 1e-11
    traj = Vector{Vector{Float64}}()
    push!(traj, copy(mol_test.r))
    for _ in 1:Int(t_max_test/dt_test)
        computeNextPosition!(mol_test, dt_test)
        push!(traj, copy(mol_test.r))
    end
    xs = [p[1] for p in traj]
    ys = [p[2] for p in traj]
    plot(xs, ys, xlabel="x (m)", ylabel="y (m)",
         title="Q3.4 — Trajectoire He (MRU)", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 4 — Collision entre deux molécules
# ════════════════════════════════════════════════════════════════

# ╔═╡ collision-theory
md"""
## Q4.1 — Type d'interaction
Quand deux molécules sont suffisamment proches, elles subissent une **collision élastique**
(modèle hard sphere). Hypothèses :
- Les molécules sont des sphères rigides impénétrables
- La collision est instantanée et conserve l'énergie cinétique et la quantité de mouvement
- On ignore les forces à longue portée (van der Waals, électrostatiques)

Ce modèle est pertinent pour des gaz nobles (He, Ar) à densité modérée et haute température.
"""

# ╔═╡ q4-3-detection
"""
Retourne true si les deux molécules sont en contact (collision possible).
"""
function areColliding(m1::Molecule, m2::Molecule)::Bool
    return norm(m1.r - m2.r) <= (m1.ρ + m2.ρ)
end

# ╔═╡ q4-4-resolve
"""
Résout la collision élastique entre deux sphères dures.
Seule la composante normale à la surface de contact est modifiée.
"""
function resolveCollision!(m1::Molecule, m2::Molecule)
    d    = m2.r - m1.r
    dist = norm(d)
    dist == 0.0 && return
    n     = d / dist
    v_rel = dot(m1.v - m2.v, n)
    v_rel <= 0.0 && return          # déjà en train de s'éloigner
    j = 2.0 * v_rel / (1/m1.m + 1/m2.m)
    m1.v -= (j / m1.m) .* n
    m2.v += (j / m2.m) .* n
end

# ╔═╡ q4-2-4-tests
begin
    # Test 1 : choc frontal masses égales → échange de vitesses
    t1a = Molecule([-1.5e-10,0.,0.], [100.,0.,0.], 1., 1e-10, "A")
    t1b = Molecule([ 1.5e-10,0.,0.], [-100.,0.,0.], 1., 1e-10, "B")
    p0  = t1a.m.*t1a.v .+ t1b.m.*t1b.v
    E0  = 0.5*t1a.m*norm(t1a.v)^2 + 0.5*t1b.m*norm(t1b.v)^2
    resolveCollision!(t1a, t1b)
    p1  = t1a.m.*t1a.v .+ t1b.m.*t1b.v
    E1  = 0.5*t1a.m*norm(t1a.v)^2 + 0.5*t1b.m*norm(t1b.v)^2
    println("Test 1 — choc frontal égal :")
    println("  conservation p  : ", isapprox(p0, p1, atol=1e-18))
    println("  conservation Ec : ", isapprox(E0, E1, rtol=1e-10))
    println("  v1x après ≈ -100 : ", isapprox(t1a.v[1], -100., atol=1e-8))

    # Test 2 : m1=2m2, m2 au repos → formule analytique
    t2a = Molecule([-1.5e-10,0.,0.], [100.,0.,0.], 2., 1e-10, "A")
    t2b = Molecule([ 1.5e-10,0.,0.], [0.,0.,0.],   1., 1e-10, "B")
    p0b = t2a.m.*t2a.v .+ t2b.m.*t2b.v
    E0b = 0.5*t2a.m*norm(t2a.v)^2
    resolveCollision!(t2a, t2b)
    E1b = 0.5*t2a.m*norm(t2a.v)^2 + 0.5*t2b.m*norm(t2b.v)^2
    p1b = t2a.m.*t2a.v .+ t2b.m.*t2b.v
    println("\nTest 2 — m1=2*m2 :")
    println("  conservation p  : ", isapprox(p0b, p1b, atol=1e-18))
    println("  conservation Ec : ", isapprox(E0b, E1b, rtol=1e-10))
    println("  v1x ≈ 33.3 : ", isapprox(t2a.v[1], 100/3, rtol=1e-8))
    println("  v2x ≈ 133.3 : ", isapprox(t2b.v[1], 400/3, rtol=1e-8))

    # Test 3 : pas en collision
    t3a = Molecule([0.,0.,0.], [10.,0.,0.], 1., 1e-10, "A")
    t3b = Molecule([1.,0.,0.], [0.,0.,0.],  1., 1e-10, "B")
    println("\nTest 3 — pas de collision : areColliding = ",
            areColliding(t3a, t3b), " (attendu: false)")

    # Test 4 : en collision
    t4a = Molecule([0.,0.,0.],      [0.,0.,0.], 1., 1e-10, "A")
    t4b = Molecule([1.5e-10,0.,0.], [0.,0.,0.], 1., 1e-10, "B")
    println("Test 4 — en collision    : areColliding = ",
            areColliding(t4a, t4b), " (attendu: true)")

    # Test 5 : molécules qui s'éloignent → vitesses non modifiées
    t5a = Molecule([-1e-10,0.,0.], [-100.,0.,0.], 1., 1e-10, "A")
    t5b = Molecule([ 1e-10,0.,0.], [ 100.,0.,0.], 1., 1e-10, "B")
    v_before = copy(t5a.v)
    resolveCollision!(t5a, t5b)
    println("Test 5 — s'éloignent     : vitesses inchangées = ",
            t5a.v == v_before)
end

# ════════════════════════════════════════════════════════════════
# SECTION 5 — Domaine de simulation
# ════════════════════════════════════════════════════════════════

# ╔═╡ domain-struct
mutable struct Domain
    Lx::Float64
    Ly::Float64
    Lz::Float64
end

# ╔═╡ domain-volume
"""Calcule le volume du domaine (m³)."""
volume(d::Domain) = d.Lx * d.Ly * d.Lz

# ╔═╡ domain-tests
begin
    d1 = Domain(2., 3., 4.)
    println("Test volume(2,3,4) = ", volume(d1), " (attendu 24.0) : ", volume(d1) == 24.0)
    d2 = Domain(1e-8*2, 1e-8*2, 1e-8*2)
    println("Test volume domaine He ≈ 8e-24 m³ : ", isapprox(volume(d2), 8e-24, rtol=1e-10))
end

# ════════════════════════════════════════════════════════════════
# SECTION 6 — Réflexion spéculaire
# ════════════════════════════════════════════════════════════════

# ╔═╡ apply-boundary
"""
Corrige la position et la vitesse d'une molécule si elle sort du domaine.
Le domaine est centré en 0 : axe i ∈ [-Li/2, +Li/2].
"""
function applyBoundary!(mol::Molecule, dom::Domain)
    half = [dom.Lx/2, dom.Ly/2, dom.Lz/2]
    for i in 1:3
        if mol.r[i] > half[i]
            mol.r[i] = 2*half[i] - mol.r[i]
            mol.v[i] = -mol.v[i]
        elseif mol.r[i] < -half[i]
            mol.r[i] = -2*half[i] - mol.r[i]
            mol.v[i] = -mol.v[i]
        end
    end
end

# ╔═╡ boundary-tests
begin
    dom_t = Domain(2., 2., 2.)   # ∈ [-1, 1]³

    # Test 1 : sortie +x
    b = Molecule([1.2,0.,0.], [10.,5.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 1 (+x)  r=", round.(b.r, digits=3), " vx<0:", b.v[1]<0,
            " correct:", isapprox(b.r[1], 0.8, atol=1e-12))

    # Test 2 : sortie -x
    b = Molecule([-1.3,0.,0.], [-10.,0.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 2 (-x)  r=", round.(b.r, digits=3), " vx>0:", b.v[1]>0,
            " correct:", isapprox(b.r[1], -0.7, atol=1e-12))

    # Test 3 : sortie +y
    b = Molecule([0.,1.1,0.], [0.,5.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 3 (+y)  vy<0:", b.v[2]<0,
            " correct:", isapprox(b.r[2], 0.9, atol=1e-12))

    # Test 4 : sortie -y
    b = Molecule([0.,-1.4,0.], [0.,-5.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 4 (-y)  vy>0:", b.v[2]>0,
            " correct:", isapprox(b.r[2], -0.6, atol=1e-12))

    # Test 5 : sortie +z
    b = Molecule([0.,0.,1.05], [0.,0.,3.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 5 (+z)  vz<0:", b.v[3]<0,
            " correct:", isapprox(b.r[3], 0.95, atol=1e-12))

    # Test 6 : sortie -z
    b = Molecule([0.,0.,-1.2], [0.,0.,-3.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 6 (-z)  vz>0:", b.v[3]>0,
            " correct:", isapprox(b.r[3], -0.8, atol=1e-12))

    # Test 7 : coin +x+y
    b = Molecule([1.3,1.2,0.], [5.,3.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 7 (coin +x+y)  vx<0:", b.v[1]<0, " vy<0:", b.v[2]<0)

    # Test 8 : dans le domaine → rien ne change
    b = Molecule([.5,.3,-.2], [1.,2.,3.], 1., .1, "T")
    r_bef, v_bef = copy(b.r), copy(b.v)
    applyBoundary!(b, dom_t)
    println("Test 8 (inchangé)  : ", b.r == r_bef && b.v == v_bef)
end

# ════════════════════════════════════════════════════════════════
# SECTION 7 — Simulation He
# ════════════════════════════════════════════════════════════════

# ╔═╡ sim7-params
begin
    const kb  = 1.380649e-23
    const He_m   = 6.646e-27
    const He_ρ   = 1.1e-10
    const N7     = 400
    const v0_7   = 1400.0
    const dt7    = 1e-14
    const t_fin7 = 2e-11
    const L7     = 5e-9
    dom7 = Domain(2L7, 2L7, 2L7)
end

# ╔═╡ sim7-init
function init_atoms(N, L, v0, m, ρ, formule; seed=42)
    Random.seed!(seed)
    mols = Vector{Molecule}(undef, N)
    for i in 1:N
        r = [rand()*2L - L for _ in 1:3]
        θ = acos(1 - 2*rand()); φ = 2π*rand()
        v = v0 .* [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
        mols[i] = Molecule(r, v, m, ρ, formule)
    end
    return mols
end

# ╔═╡ sim7-run
begin
    atoms7 = init_atoms(N7, L7, v0_7, He_m, He_ρ, "He")
    n_steps7   = Int(t_fin7 / dt7)
    rec_every  = 100
    n_rec      = div(n_steps7, rec_every)
    t_hist     = zeros(n_rec)
    vmean_hist = zeros(n_rec)
    alpha_hist = zeros(n_rec)
    beta_hist  = zeros(n_rec)
    idx = 1

    for step in 1:n_steps7
        # 1. déplacement
        for mol in atoms7
            computeNextPosition!(mol, dt7)
        end
        # 2. conditions de bord
        for mol in atoms7
            applyBoundary!(mol, dom7)
        end
        # 3. collisions O(N²)
        for i in 1:N7-1, j in i+1:N7
            if areColliding(atoms7[i], atoms7[j])
                resolveCollision!(atoms7[i], atoms7[j])
            end
        end
        # 4. stats
        if step % rec_every == 0
            speeds = [norm(mol.v) for mol in atoms7]
            v2m    = mean(norm(mol.v)^2 for mol in atoms7)
            t_hist[idx]     = step * dt7
            vmean_hist[idx] = mean(speeds)
            alpha_hist[idx] = He_m * v2m / (3kb)
            beta_hist[idx]  = N7 * He_m * v2m / (3 * volume(dom7))
            idx += 1
        end
    end
    println("Simulation 7 terminée.")
end

# ╔═╡ q7-2-plot
plot(t_hist.*1e12, vmean_hist,
     xlabel="Temps (ps)", ylabel="Vitesse moyenne (m/s)",
     title="Q7.2 — Vitesse moyenne au cours du temps", legend=false)

# ╔═╡ q7-3-plot
histogram([norm(mol.v) for mol in atoms7], bins=40,
          xlabel="|v| (m/s)", ylabel="Nombre d'atomes",
          title="Q7.3 — Distribution des vitesses (Maxwell-Boltzmann)", legend=false)

# ╔═╡ q7-4-plot
begin
    p_α = plot(t_hist.*1e12, alpha_hist,
               xlabel="Temps (ps)", ylabel="α (K)",
               title="Q7.4 — α = m<v²>/(3kb) = Température cinétique", legend=false)
    println("α final ≈ ", round(alpha_hist[end], digits=1), " K")
    println("→ α est la température cinétique du gaz (unité : Kelvin)")
    p_α
end

# ╔═╡ q7-5-plot
begin
    p_β = plot(t_hist.*1e12, beta_hist,
               xlabel="Temps (ps)", ylabel="β (Pa)",
               title="Q7.5 — β = Nm<v²>/(3V) = Pression", legend=false)
    println("β final ≈ ", round(beta_hist[end], digits=2), " Pa")
    println("→ β est la pression cinétique du gaz (unité : Pascal)")
    p_β
end

# ════════════════════════════════════════════════════════════════
# SECTION 8 — Gravité
# ════════════════════════════════════════════════════════════════

# ╔═╡ gravity-theory
md"""
## Q8.1 — Gravité dans computeNextPosition!
Déjà intégré via le paramètre `g` :
`mol.v[3] -= g * dt`

## Q8.3 — Temps pour parcourir 1m (He, T = 300 K)
vgrav = m·g·D / (kb·T)
t = 1 / vgrav
"""

# ╔═╡ q8-3-calc
begin
    T_300 = 26.85 + 273.15   # K
    D_diff = 6.5e-5           # m²/s
    g_real = 9.81
    v_grav = He_m * g_real * D_diff / (kb * T_300)
    t_1m   = 1.0 / v_grav
    println("vgrav = ", v_grav, " m/s")
    println("Temps pour 1m = ", t_1m, " s ≈ ", round(t_1m/(3600*24*365), digits=1), " ans")
end

# ╔═╡ sim8-params
begin
    const N8      = 500
    const v0_8    = 1367.0
    const t_fin8  = 5e-11
    const L8      = 1e-8
    const g8      = 9.81e13   # gravité exagérée
    dom8 = Domain(2L8, 2L8, 2L8)
    n_steps8   = Int(t_fin8 / dt7)
    n_zbins    = 20
end

# ╔═╡ sim8-run
begin
    atoms8 = init_atoms(N8, L8, v0_8, He_m, He_ρ, "He")
    z_snapshots = Vector{Vector{Float64}}()

    for step in 1:n_steps8
        for mol in atoms8
            computeNextPosition!(mol, dt7; g=g8)
        end
        for mol in atoms8
            applyBoundary!(mol, dom8)
        end
        for i in 1:N8-1, j in i+1:N8
            if areColliding(atoms8[i], atoms8[j])
                resolveCollision!(atoms8[i], atoms8[j])
            end
        end
        if step == n_steps8
            push!(z_snapshots, [mol.r[3] for mol in atoms8])
        end
    end
    println("Simulation 8 terminée.")
end

# ╔═╡ q8-2-plot
begin
    z_final = z_snapshots[end]
    histogram(z_final, bins=n_zbins,
              xlabel="z (m)", ylabel="Nombre d'atomes",
              title="Q8.2/8.4 — Probabilité de présence en z", legend=false,
              normalize=:probability)
end

# ╔═╡ gravity-comments
md"""
## Q8.4 — Comment évolue la probabilité ?
Avec la gravité exagérée, les atomes s'accumulent progressivement vers les z négatifs
(bas du domaine). La distribution devient asymétrique : densité plus élevée en bas.

## Q8.5 — Turbopause
La **turbopause** (≈ 100 km d'altitude) est la frontière au-dessus de laquelle la
turbulence atmosphérique cesse de mélanger les gaz. En-dessous, le brassage turbulent
maintient une composition quasi-uniforme (air bien mélangé). Au-dessus, la diffusion
moléculaire domine : chaque espèce se distribue selon sa propre échelle de hauteur
barométrique H = kbT/(mg), proportionnelle à 1/m. Les molécules légères (He, H₂)
ont une grande échelle de hauteur et s'étendent loin dans l'atmosphère.
"""

# ╔═╡ q8-6-pressure-vs-z
begin
    z_vals  = [mol.r[3] for mol in atoms8]
    # Couper en tranches z et calculer la pression locale β_local = n·m·<v²>/3 dans chaque tranche
    z_edges = range(-L8, L8, length=n_zbins+1)
    z_centers = [(z_edges[i]+z_edges[i+1])/2 for i in 1:n_zbins]
    p_z = zeros(n_zbins)
    for (idx_z, mol) in enumerate(atoms8)
        bin = searchsortedfirst(z_edges, mol.r[3]) - 1
        bin = clamp(bin, 1, n_zbins)
        p_z[bin] += He_m * norm(mol.v)^2 / 3
    end
    dz = z_edges[2] - z_edges[1]
    V_slice = 2L8 * 2L8 * dz
    p_z ./= V_slice
    plot(z_centers.*1e9, p_z,
         xlabel="z (nm)", ylabel="Pression (Pa)",
         title="Q8.6 — Pression en fonction de z", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 9 — Simulation multi-espèces He-Ar
# ════════════════════════════════════════════════════════════════

# ╔═╡ sim9-params
begin
    const Ar_m   = 6.634e-26
    const Ar_ρ   = 1.88e-10
    const N_He9  = 400
    const N_Ar9  = 200
    const v0_He9 = 789.45
    const v0_Ar9 = 249.88
    const t_fin9 = 5e-11
    const L9     = 1e-8
    dom9 = Domain(2L9, 2L9, 2L9)
    n_steps9 = Int(t_fin9 / dt7)
end

# ╔═╡ sim9-run
begin
    he_atoms = init_atoms(N_He9, L9, v0_He9, He_m, He_ρ, "He"; seed=1)
    ar_atoms = init_atoms(N_Ar9, L9, v0_Ar9, Ar_m, Ar_ρ, "Ar"; seed=2)
    all9 = vcat(he_atoms, ar_atoms)
    N9_tot = length(all9)

    for step in 1:n_steps9
        for mol in all9
            computeNextPosition!(mol, dt7; g=9.81e13)
        end
        for mol in all9
            applyBoundary!(mol, dom9)
        end
        for i in 1:N9_tot-1, j in i+1:N9_tot
            if areColliding(all9[i], all9[j])
                resolveCollision!(all9[i], all9[j])
            end
        end
    end
    println("Simulation 9 terminée.")
end

# ╔═╡ q9-2-prob-z
begin
    z_he = [mol.r[3] for mol in he_atoms]
    z_ar = [mol.r[3] for mol in ar_atoms]
    p_he = histogram(z_he, bins=20, normalize=:probability,
                     xlabel="z (m)", ylabel="P(z)",
                     title="Q9.2 — P(z) He vs Ar", label="He", alpha=0.6)
    histogram!(p_he, z_ar, bins=20, normalize=:probability,
               label="Ar", alpha=0.6, color=:red)
    p_he
end

# ╔═╡ q9-3-comment
md"""
## Q9.3 — Différence entre <v²> et <mv²>
- **<v²>** est la moyenne des carrés des vitesses : grandeur cinématique pure, ne tient pas
  compte de la masse. Deux espèces à la même température ont des <v²> différents
  (les particules légères vont plus vite).
- **<mv²>** est proportionnel à l'énergie cinétique moyenne (×½) : c'est une grandeur
  dynamique. À l'équilibre thermique, **<mv²> est identique pour toutes les espèces**
  (équipartition), ce qui implique que les particules légères ont un <v²> plus grand.
"""

# ╔═╡ q9-4-z-profiles
begin
    n_zb9  = 20
    z_ed9  = range(-L9, L9, length=n_zb9+1)
    z_c9   = [(z_ed9[i]+z_ed9[i+1])/2 for i in 1:n_zb9]
    dz9    = z_ed9[2]-z_ed9[1]
    Vs9    = 2L9 * 2L9 * dz9

    mv2_z  = zeros(n_zb9)
    p9_z   = zeros(n_zb9)
    T9_z   = zeros(n_zb9)
    count9 = zeros(Int, n_zb9)

    for mol in all9
        b = clamp(searchsortedfirst(z_ed9, mol.r[3])-1, 1, n_zb9)
        v2 = norm(mol.v)^2
        mv2_z[b]  += mol.m * v2
        p9_z[b]   += mol.m * v2 / 3
        T9_z[b]   += mol.m * v2 / (3kb)
        count9[b] += 1
    end
    p9_z  ./= Vs9
    # moyennes par atome
    for b in 1:n_zb9
        if count9[b] > 0
            mv2_z[b] /= count9[b]
            T9_z[b]  /= count9[b]
        end
    end

    pa = plot(z_c9.*1e9, mv2_z, xlabel="z (nm)", ylabel="<mv²> (J)",
              title="Q9.4a — <mv²> vs z", legend=false)
    pb = plot(z_c9.*1e9, p9_z,  xlabel="z (nm)", ylabel="p (Pa)",
              title="Q9.4b — Pression vs z", legend=false)
    pc = plot(z_c9.*1e9, T9_z,  xlabel="z (nm)", ylabel="T (K)",
              title="Q9.4c — Température vs z", legend=false)
    plot(pa, pb, pc, layout=(1,3), size=(900,300))
end

# ════════════════════════════════════════════════════════════════
# SECTION 10 — Critère de stabilité
# ════════════════════════════════════════════════════════════════

# ╔═╡ stability-comment
md"""
## Q10.1 — Critères de stabilité proposés
1. **Variance glissante** : calculer l'écart-type de α (ou β, ou T) sur une fenêtre
   temporelle glissante. La simulation est stable quand cet écart-type descend en dessous
   d'un seuil relatif (ex. < 1% de la moyenne).
2. **Variation relative** : |α(t) - α(t-Δt)| / α(t) < ε
3. **Test du plateau** : régression linéaire sur une fenêtre ; stable si la pente ≈ 0.
"""

# ╔═╡ q10-2-stability
begin
    # Critère : écart-type relatif sur les 20 derniers points de alpha
    window = 20
    if length(alpha_hist) >= window
        last_alpha = alpha_hist[end-window+1:end]
        σ_rel = std(last_alpha) / mean(last_alpha)
        println("Écart-type relatif de α sur les 20 derniers points : ",
                round(σ_rel * 100, digits=3), " %")
        println("Système stable (σ_rel < 1%) : ", σ_rel < 0.01)
    end
    plot(t_hist.*1e12, alpha_hist,
         xlabel="Temps (ps)", ylabel="α (K)",
         title="Q10.2 — Convergence de α (température cinétique)", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 11 — Distribution de ???
# ════════════════════════════════════════════════════════════════

# ╔═╡ section11-comment
md"""
## Q11.1 — Distribution barométrique (de Boltzmann)
La probabilité de présence en z suit une loi **exponentielle décroissante** :
P(z) ∝ exp(-mgz / kbT)
C'est la **distribution barométrique** (ou de Boltzmann). Les paramètres physiques qui
influencent sa forme :
- La **masse** m : plus la particule est lourde, plus la distribution est concentrée en bas
- La **température** T : plus T est grande, plus la distribution est étalée (agitation thermique
  compense la gravité)
- La **constante de gravité** g

L'échelle de hauteur caractéristique est H = kbT/(mg).

## Q11.2 — Autres grandeurs obéissant à la loi de Boltzmann
- La **pression** : P(z) = P₀·exp(-mgz/kbT)  (loi barométrique)
- La **densité** : ρ(z) = ρ₀·exp(-mgz/kbT)
- La **distribution de Maxwell-Boltzmann** des vitesses : P(v) ∝ v²·exp(-mv²/2kbT)
Toutes ces distributions proviennent du facteur de Boltzmann exp(-E/kbT) qui décrit la
probabilité d'occupation d'un état d'énergie E à la température T.
"""

# ════════════════════════════════════════════════════════════════
# SECTION 12 — Entropie de Shannon
# ════════════════════════════════════════════════════════════════

# ╔═╡ shannon-function
"""
Entropie de Shannon H = -Σ p·log2(p) pour une liste de valeurs
discrétisées en `nbins` intervalles sur [vmin, vmax].
"""
function shannon_entropy(values, nbins::Int, vmin::Float64, vmax::Float64)::Float64
    counts = zeros(Int, nbins)
    width  = (vmax - vmin) / nbins
    for x in values
        idx = floor(Int, (x - vmin) / width) + 1
        idx = clamp(idx, 1, nbins)
        counts[idx] += 1
    end
    n = length(values)
    H = 0.0
    for c in counts
        c == 0 && continue
        p = c / n
        H -= p * log2(p)
    end
    return H
end

"""
Entropie totale H(X,Y,Z,<v²>) = H(X) + H(Y) + H(Z) + H(<v²>)
"""
function total_entropy(mols, dom::Domain)::Float64
    Lx, Ly, Lz = dom.Lx, dom.Ly, dom.Lz
    xs  = [mol.r[1] for mol in mols]
    ys  = [mol.r[2] for mol in mols]
    zs  = [mol.r[3] for mol in mols]
    v2s = [norm(mol.v)^2 for mol in mols]
    Hx = shannon_entropy(xs,  10, -Lx/2, Lx/2)
    Hy = shannon_entropy(ys,  10, -Ly/2, Ly/2)
    Hz = shannon_entropy(zs,  10, -Lz/2, Lz/2)
    Hv = shannon_entropy(v2s, 200, 0.0, 200000.0)
    return Hx + Hy + Hz + Hv
end

# ╔═╡ sim12-params
begin
    const L12    = 5e-9
    const N12    = 400
    const v0_12  = 1400.0
    const dt12   = 1e-14
    const t_fin12 = 5e-11
    n_steps12 = Int(t_fin12 / dt12)
    dom12 = Domain(2L12, 2L12, 2L12)
end

# ╔═╡ sim12-run
begin
    # Atomes initialisés dans le 1/8ème : x,y,z ∈ [-5e-9, 0]
    Random.seed!(42)
    atoms12 = Vector{Molecule}(undef, N12)
    for i in 1:N12
        r = [rand()*L12 - L12, rand()*L12 - L12, rand()*L12 - L12]
        θ = acos(1 - 2*rand()); φ = 2π*rand()
        v = v0_12 .* [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
        atoms12[i] = Molecule(r, v, He_m, He_ρ, "He")
    end

    rec12_every = 100
    n_rec12     = div(n_steps12, rec12_every)
    H12_hist    = zeros(n_rec12)
    t12_hist    = zeros(n_rec12)
    idx12 = 1

    for step in 1:n_steps12
        for mol in atoms12
            computeNextPosition!(mol, dt12)
        end
        for mol in atoms12
            applyBoundary!(mol, dom12)
        end
        for i in 1:N12-1, j in i+1:N12
            if areColliding(atoms12[i], atoms12[j])
                resolveCollision!(atoms12[i], atoms12[j])
            end
        end
        if step % rec12_every == 0
            H12_hist[idx12] = total_entropy(atoms12, dom12)
            t12_hist[idx12] = step * dt12
            idx12 += 1
        end
    end
    println("Simulation 12 terminée.")
end

# ╔═╡ q12-2-plot
begin
    p12 = plot(t12_hist.*1e12, H12_hist,
               xlabel="Temps (ps)", ylabel="H (bits)",
               title="Q12.2 — Entropie de Shannon au cours du temps",
               legend=false)
    println("H initial ≈ ", round(H12_hist[1], digits=3),
            "  H final ≈ ", round(H12_hist[end], digits=3))
    p12
end

# ╔═╡ q12-3-expanded-domain
begin
    # Reprise depuis l'état final de sim12, domaine élargi sur x : [-5e-9, 15e-9]
    dom12b = Domain(20e-9, 2L12, 2L12)   # Lx = 20 nm (de -5 à +15)
    n_steps12b = 5000
    rec12b_every = 50
    n_rec12b   = div(n_steps12b, rec12b_every)
    H12b_hist  = zeros(n_rec12b)
    t12b_hist  = zeros(n_rec12b)
    # On cumule après le temps final de la simulation précédente
    t_offset = t_fin12
    idx12b = 1

    for step in 1:n_steps12b
        for mol in atoms12
            computeNextPosition!(mol, dt12)
        end
        for mol in atoms12
            applyBoundary!(mol, dom12b)
        end
        for i in 1:N12-1, j in i+1:N12
            if areColliding(atoms12[i], atoms12[j])
                resolveCollision!(atoms12[i], atoms12[j])
            end
        end
        if step % rec12b_every == 0
            H12b_hist[idx12b] = total_entropy(atoms12, dom12b)
            t12b_hist[idx12b] = t_offset + step * dt12
            idx12b += 1
        end
    end

    # Graphe combiné
    all_t = vcat(t12_hist, t12b_hist) .* 1e12
    all_H = vcat(H12_hist, H12b_hist)
    vline_t = t_fin12 * 1e12

    plot(all_t, all_H,
         xlabel="Temps (ps)", ylabel="H (bits)",
         title="Q12.3 — Entropie : domaine initial puis élargi",
         legend=false)
    vline!([vline_t], linestyle=:dash, color=:red, label="ouverture paroi")
end

# ╔═╡ q12-observations
md"""
## Q12.2 — Observations
L'entropie **augmente** au cours du temps pour atteindre un plateau : les atomes, initialement
confinés dans 1/8 du domaine, se répartissent progressivement dans tout le volume (diffusion).
L'entropie spatiale converge vers log₂(10) ≈ 3.32 bits par dimension quand la distribution
est uniforme.

## Q12.3 — Observations avec domaine élargi
Quand la paroi s'ouvre, l'entropie **augmente à nouveau** car les atomes colonisent le nouveau
volume disponible. Ceci illustre le **second principe de la thermodynamique** : l'entropie
d'un système isolé ne peut qu'augmenter. Elle atteint un nouveau plateau plus élevé
correspondant à la distribution uniforme dans le domaine élargi.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Plots = "~1"
"""
