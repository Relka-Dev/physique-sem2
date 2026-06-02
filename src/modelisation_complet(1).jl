### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 6310bcd4-55b2-11f1-86a7-f1f13ec531a7
begin
    using Plots, Statistics, Random, LinearAlgebra
end

# ════════════════════════════════════════════════════════════════
# SECTION 1 — Classe Molecule
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310c044-55b2-11f1-84de-9fea747f4d7f
mutable struct Molecule
    r::Vector{Float64}   # position [x, y, z] (m)
    v::Vector{Float64}   # vitesse  [vx,vy,vz] (m/s)
    m::Float64           # masse (kg)
    ρ::Float64           # rayon (m)
    formule::String
end

# ╔═╡ 6310c13e-55b2-11f1-9e2c-ff7d27813657
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
mais le théorème d'équipartition doit tenir compte des degrés de liberté supplémentaires
pour N₂ et O₂ (capacité thermique différente).
"""

# ════════════════════════════════════════════════════════════════
# SECTION 3 — Déplacement (Euler explicite)
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310c5aa-55b2-11f1-b3f8-af71452a6674
md"""
## Q3.1 — Équations de Newton
Sans force extérieure : m·(d²r/dt²) = 0  ⟹  dv/dt = 0,  dr/dt = v

## Q3.2 — Différences finies (Euler explicite)
v(t+dt) = v(t)
r(t+dt) = r(t) + v(t)·dt

Avec gravité selon -z :
v_z(t+dt) = v_z(t) - g·dt
r(t+dt)   = r(t) + v(t)·dt
"""

# ╔═╡ 6310c6d4-55b2-11f1-a5d5-33d28add3bff
function computeNextPosition!(mol::Molecule, dt::Float64; g::Float64=0.0)
    mol.v[3] -= g * dt
    mol.r   .+= mol.v .* dt
end

# ╔═╡ 6310c74c-55b2-11f1-a5a4-29b79f7a6f07
begin
    mol_test = Molecule([0.0, 0.0, 0.0], [100.0, 50.0, 0.0], 6.646e-27, 1.4e-10, "He")
    dt_test  = 1e-14
    t_max_test = 1e-11
    traj = Vector{Vector{Float64}}()
    push!(traj, copy(mol_test.r))
    for _ in 1:round(Int, t_max_test/dt_test)
        computeNextPosition!(mol_test, dt_test)
        push!(traj, copy(mol_test.r))
    end
    xs_traj = [p[1] for p in traj]
    ys_traj = [p[2] for p in traj]
    plot(xs_traj, ys_traj, xlabel="x (m)", ylabel="y (m)",
         title="Q3.4 — Trajectoire He (MRU)", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 4 — Collision entre deux molécules
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310cad0-55b2-11f1-b77a-0f95a2917373
md"""
## Q4.1 — Type d'interaction
Quand deux molécules sont suffisamment proches, elles subissent une **collision élastique**
(modèle hard sphere). Hypothèses :
- Les molécules sont des sphères rigides impénétrables
- La collision est instantanée et conserve l'énergie cinétique et la quantité de mouvement
- On ignore les forces à longue portée (van der Waals, électrostatiques)

Ce modèle est pertinent pour des gaz nobles (He, Ar) à densité modérée et haute température.

## Q4.2 — Scénarios de test
| # | Description | Conditions initiales | Propriété vérifiée |
|---|-------------|----------------------|-------------------|
| 1 | Choc frontal, masses égales | v1=(100,0,0), v2=(-100,0,0) | échange de vitesses, conservation p et Ec |
| 2 | Choc frontal, m1=2·m2 | v1=(100,0,0), v2=0 | formules analytiques, conservation p et Ec |
| 3 | Pas de collision | centres distants >> 2ρ | areColliding = false |
| 4 | En collision | distance < 2ρ | areColliding = true |
| 5 | Molécules qui s'éloignent | v_rel < 0 | vitesses inchangées |
"""

# ╔═╡ 6310ce2c-55b2-11f1-9997-a1c3e3ce4dab
function areColliding(m1::Molecule, m2::Molecule)::Bool
    return norm(m1.r - m2.r) <= (m1.ρ + m2.ρ)
end

# ╔═╡ 6310cea4-55b2-11f1-8462-2b0a971147bc
function resolveCollision!(m1::Molecule, m2::Molecule)
    d    = m2.r - m1.r
    dist = norm(d)
    dist == 0.0 && return
    n     = d / dist
    v_rel = dot(m1.v - m2.v, n)
    v_rel <= 0.0 && return
    j = 2.0 * v_rel / (1/m1.m + 1/m2.m)
    m1.v -= (j / m1.m) .* n
    m2.v += (j / m2.m) .* n
end

# ╔═╡ 6310d796-55b2-11f1-83c8-6f6bffddfd0b
mutable struct Domain
    Lx::Float64
    Ly::Float64
    Lz::Float64
end

# ╔═╡ 6310d80e-55b2-11f1-9454-2b5f24106b82
volume(d::Domain) = d.Lx * d.Ly * d.Lz

# ╔═╡ 6310d852-55b2-11f1-b8ce-eb8b0d931e60
begin
    d1 = Domain(2., 3., 4.)
    println("Test volume(2,3,4) = ", volume(d1), " correct: ", volume(d1) == 24.0)
    d2 = Domain(2e-8, 2e-8, 2e-8)
    println("Test volume He ≈ 8e-24 m³ : ", isapprox(volume(d2), 8e-24, rtol=1e-10))
end

# ════════════════════════════════════════════════════════════════
# SECTION 6 — Réflexion spéculaire
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310daa2-55b2-11f1-9571-2998aef2fcaa
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

# ╔═╡ 6310e1b4-55b2-11f1-bf13-b5351da53e6d
begin
    const kb   = 1.380649e-23
    const He_m = 6.646e-27
    const He_ρ = 1.1e-10
end

# ╔═╡ 6310e240-55b2-11f1-a30e-d7eaa045a5e3
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

# ╔═╡ 6310e362-55b2-11f1-8b6e-4380846a6cae
begin
    const N7     = 400
    const v0_7   = 1400.0
    const dt7    = 1e-14
    const t_fin7 = 2e-11
    const L7     = 5e-9
    dom7 = Domain(2L7, 2L7, 2L7)

    atoms7     = init_atoms(N7, L7, v0_7, He_m, He_ρ, "He")
    n_steps7   = round(Int, t_fin7 / dt7)
    rec_every  = 100
    n_rec7     = div(n_steps7, rec_every)
    t_hist7    = zeros(n_rec7)
    vmean_hist = zeros(n_rec7)
    alpha_hist = zeros(n_rec7)
    beta_hist  = zeros(n_rec7)
    idx7 = 1

    for step in 1:n_steps7
        for mol in atoms7; computeNextPosition!(mol, dt7); end
        for mol in atoms7; applyBoundary!(mol, dom7); end
        for i in 1:N7-1, j in i+1:N7
            if areColliding(atoms7[i], atoms7[j])
                resolveCollision!(atoms7[i], atoms7[j])
            end
        end
        if step % rec_every == 0
            v2m = mean(norm(mol.v)^2 for mol in atoms7)
            t_hist7[idx7]    = step * dt7
            vmean_hist[idx7] = mean(norm(mol.v) for mol in atoms7)
            alpha_hist[idx7] = He_m * v2m / (3kb)
            beta_hist[idx7]  = N7 * He_m * v2m / (3 * volume(dom7))
            idx7 += 1
        end
    end
    println("Simulation 7 terminée.")
end

# ╔═╡ 6310e75e-55b2-11f1-b095-ed83a3b47b5c
plot(t_hist7.*1e12, vmean_hist,
     xlabel="Temps (ps)", ylabel="Vitesse moyenne (m/s)",
     title="Q7.2 — Vitesse moyenne", legend=false)

# ╔═╡ 6310e7f4-55b2-11f1-b29d-137cdafc2ca7
histogram([norm(mol.v) for mol in atoms7], bins=40,
          xlabel="|v| (m/s)", ylabel="Nombre d'atomes",
          title="Q7.3 — Distribution Maxwell-Boltzmann", legend=false)

# ╔═╡ 6310e89c-55b2-11f1-87ce-c9695185f7b1
begin
    println("α final ≈ ", round(alpha_hist[end], digits=1),
            " K  (température cinétique)")
    plot(t_hist7.*1e12, alpha_hist,
         xlabel="Temps (ps)", ylabel="α (K)",
         title="Q7.4 — α = m<v²>/(3kb) = Température cinétique", legend=false)
end

# ╔═╡ 6310e9ac-55b2-11f1-8a51-fd0e92fb089e
begin
    println("β final ≈ ", round(beta_hist[end], digits=2),
            " Pa  (pression cinétique)")
    plot(t_hist7.*1e12, beta_hist,
         xlabel="Temps (ps)", ylabel="β (Pa)",
         title="Q7.5 — β = Nm<v²>/(3V) = Pression", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 8 — Gravité
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310ec10-55b2-11f1-8379-c95c614fa57b
md"""
## Q8.1 — Gravité dans computeNextPosition!
Déjà intégré via le paramètre `g` :
`mol.v[3] -= g * dt`

## Q8.3 — Temps pour parcourir 1m (He, T = 300 K)
vgrav = m·g·D / (kb·T)  →  t = 1 / vgrav
"""

# ╔═╡ 6310ecf4-55b2-11f1-aa1f-7940dd8fd1f4
begin
    T_300  = 26.85 + 273.15
    D_diff = 6.5e-5
    g_real = 9.81
    v_grav = He_m * g_real * D_diff / (kb * T_300)
    t_1m   = 1.0 / v_grav
    println("vgrav = ", v_grav, " m/s")
    println("Temps pour 1m = ", round(t_1m / (3600*24*365), digits=1), " ans")
end

# ╔═╡ 6310edf8-55b2-11f1-9f42-1b291aa58ac8
begin
    const N8     = 500
    const v0_8   = 1367.0
    const t_fin8 = 5e-11
    const L8     = 1e-8
    const g8     = 9.81e13
    dom8      = Domain(2L8, 2L8, 2L8)
    n_steps8  = round(Int, t_fin8 / dt7)
    n_zbins8  = 20

    atoms8 = init_atoms(N8, L8, v0_8, He_m, He_ρ, "He"; seed=10)

    for step in 1:n_steps8
        for mol in atoms8; computeNextPosition!(mol, dt7; g=g8); end
        for mol in atoms8; applyBoundary!(mol, dom8); end
        for i in 1:N8-1, j in i+1:N8
            if areColliding(atoms8[i], atoms8[j])
                resolveCollision!(atoms8[i], atoms8[j])
            end
        end
    end
    println("Simulation 8 terminée.")
end

# ╔═╡ 6310f032-55b2-11f1-b660-6f33dcd8f60f
histogram([mol.r[3] for mol in atoms8], bins=n_zbins8,
          xlabel="z (m)", ylabel="Probabilité",
          title="Q8.2/8.4 — P(z) avec gravité exagérée",
          legend=false, normalize=:probability)

# ╔═╡ 6310f0f0-55b2-11f1-861a-29c2ea0791e4
md"""
## Q8.4 — Évolution de P(z)
Avec la gravité exagérée, les atomes s'accumulent vers les z négatifs (bas du domaine).
La distribution devient asymétrique : densité plus élevée en bas.

## Q8.5 — Turbopause
La **turbopause** (≈ 100 km) est la limite au-dessus de laquelle la turbulence
atmosphérique cesse de mélanger les gaz. En dessous, le brassage turbulent maintient une
composition quasi-uniforme. Au-dessus, la diffusion moléculaire domine : chaque espèce
se distribue selon son échelle de hauteur H = kbT/(mg). Les molécules légères (He, H₂)
ont une grande H et s'étendent loin.
"""

# ╔═╡ 6310f2ee-55b2-11f1-acc5-59866099fa26
begin
    z_edges8  = range(-L8, L8, length=n_zbins8+1)
    z_centers8 = [(z_edges8[i]+z_edges8[i+1])/2 for i in 1:n_zbins8]
    dz8  = z_edges8[2]-z_edges8[1]
    Vs8  = 2L8 * 2L8 * dz8
    p_z8 = zeros(n_zbins8)
    for mol in atoms8
        b = clamp(searchsortedfirst(collect(z_edges8), mol.r[3])-1, 1, n_zbins8)
        p_z8[b] += He_m * norm(mol.v)^2 / 3
    end
    p_z8 ./= Vs8
    plot(z_centers8.*1e9, p_z8,
         xlabel="z (nm)", ylabel="Pression (Pa)",
         title="Q8.6 — Pression en fonction de z", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 9 — Simulation multi-espèces He-Ar
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310f622-55b2-11f1-8f07-ad2bb1c922f1
begin
    const Ar_m   = 6.634e-26
    const Ar_ρ   = 1.88e-10
    const N_He9  = 400
    const N_Ar9  = 200
    const v0_He9 = 789.45
    const v0_Ar9 = 249.88
    const t_fin9 = 5e-11
    const L9     = 1e-8
    dom9     = Domain(2L9, 2L9, 2L9)
    n_steps9 = round(Int, t_fin9 / dt7)

    he_atoms = init_atoms(N_He9, L9, v0_He9, He_m, He_ρ, "He"; seed=1)
    ar_atoms = init_atoms(N_Ar9, L9, v0_Ar9, Ar_m, Ar_ρ, "Ar"; seed=2)
    all9     = vcat(he_atoms, ar_atoms)
    N9_tot   = length(all9)

    for step in 1:n_steps9
        for mol in all9; computeNextPosition!(mol, dt7; g=g8); end
        for mol in all9; applyBoundary!(mol, dom9); end
        for i in 1:N9_tot-1, j in i+1:N9_tot
            if areColliding(all9[i], all9[j])
                resolveCollision!(all9[i], all9[j])
            end
        end
    end
    println("Simulation 9 terminée.")
end

# ╔═╡ 6310f8e6-55b2-11f1-be0b-f17e734b5a94
begin
    ph = histogram([mol.r[3] for mol in he_atoms], bins=20,
                   normalize=:probability, label="He", alpha=0.6,
                   xlabel="z (m)", ylabel="P(z)",
                   title="Q9.2 — P(z) He vs Ar")
    histogram!(ph, [mol.r[3] for mol in ar_atoms], bins=20,
               normalize=:probability, label="Ar", alpha=0.6, color=:red)
    ph
end

# ╔═╡ 6310fa46-55b2-11f1-88e9-4b7561821b4a
md"""
## Q9.3 — <v²> vs <mv²>
- **<v²>** : moyenne des carrés des vitesses. Grandeur cinématique pure. À l'équilibre,
  les particules légères ont un <v²> plus grand que les lourdes.
- **<mv²>** : proportionnel à l'énergie cinétique (= 2×Ec). À l'équilibre thermique,
  **<mv²> est le même pour toutes les espèces** (équipartition), ce qui implique
  que les particules légères vont plus vite.
"""

# ╔═╡ 6310fbc2-55b2-11f1-85d8-dda2d011335b
begin
    n_zb9  = 20
    z_ed9  = range(-L9, L9, length=n_zb9+1)
    z_c9   = [(z_ed9[i]+z_ed9[i+1])/2 for i in 1:n_zb9]
    dz9    = z_ed9[2]-z_ed9[1]
    Vs9    = 2L9 * 2L9 * dz9
    mv2_z  = zeros(n_zb9)
    p9_z   = zeros(n_zb9)
    T9_z   = zeros(n_zb9)
    cnt9   = zeros(Int, n_zb9)

    for mol in all9
        b = clamp(searchsortedfirst(collect(z_ed9), mol.r[3])-1, 1, n_zb9)
        v2 = norm(mol.v)^2
        mv2_z[b] += mol.m * v2
        p9_z[b]  += mol.m * v2 / 3
        T9_z[b]  += mol.m * v2 / (3kb)
        cnt9[b]  += 1
    end
    p9_z ./= Vs9
    for b in 1:n_zb9
        if cnt9[b] > 0
            mv2_z[b] /= cnt9[b]
            T9_z[b]  /= cnt9[b]
        end
    end

    pa = plot(z_c9.*1e9, mv2_z, xlabel="z (nm)", ylabel="<mv²> (J)",  title="<mv²> vs z", legend=false)
    pb = plot(z_c9.*1e9, p9_z,  xlabel="z (nm)", ylabel="p (Pa)",     title="Pression vs z", legend=false)
    pc = plot(z_c9.*1e9, T9_z,  xlabel="z (nm)", ylabel="T (K)",      title="Température vs z", legend=false)
    plot(pa, pb, pc, layout=(1,3), size=(900,300))
end

# ════════════════════════════════════════════════════════════════
# SECTION 10 — Critère de stabilité
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6311007c-55b2-11f1-bcf8-c9aebec926a1
md"""
## Q10.1 — Critères proposés
1. **Écart-type relatif** sur une fenêtre glissante : σ(α)/mean(α) < ε  (ex. 1%)
2. **Variation relative** entre deux instants : |α(t) - α(t-Δt)| / α(t) < ε
3. **Pente de régression** sur une fenêtre : stable si pente ≈ 0
"""

# ╔═╡ 6311018a-55b2-11f1-8e71-4fe346d8861a
begin
    window = 20
    last_alpha = alpha_hist[end-window+1:end]
    σ_rel = std(last_alpha) / mean(last_alpha)
    println("Écart-type relatif de α (20 derniers points) : ",
            round(σ_rel * 100, digits=3), " %")
    println("Stable (σ_rel < 1%) : ", σ_rel < 0.01)
    plot(t_hist7.*1e12, alpha_hist,
         xlabel="Temps (ps)", ylabel="α (K)",
         title="Q10.2 — Convergence de α", legend=false)
end

# ════════════════════════════════════════════════════════════════
# SECTION 11 — Distribution de Boltzmann
# ════════════════════════════════════════════════════════════════

# ╔═╡ 63110482-55b2-11f1-9685-350ab23719c9
md"""
## Q11.1 — Distribution barométrique
La probabilité de présence en z suit une loi **exponentielle** :
P(z) ∝ exp(-mgz / kbT)

C'est la **distribution de Boltzmann** (ou barométrique). Paramètres :
- **m** : plus la masse est grande, plus la décroissance est rapide (H = kbT/(mg) petit)
- **T** : plus T est élevée, plus la distribution est étalée (H grand)
- **g** : plus g est grand, plus la distribution est concentrée en bas

## Q11.2 — Autres grandeurs obéissant à Boltzmann
- **Pression** : P(z) = P₀·exp(-mgz/kbT)
- **Densité** : ρ(z) = ρ₀·exp(-mgz/kbT)
- **Distribution de Maxwell-Boltzmann** : P(v) ∝ v²·exp(-mv²/2kbT)

Toutes proviennent du facteur de Boltzmann exp(-E/kbT).
"""

# ════════════════════════════════════════════════════════════════
# SECTION 12 — Entropie de Shannon
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6311084c-55b2-11f1-8573-3b82c01d4568
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

# ╔═╡ 631109a0-55b2-11f1-923d-2f88a361a366
function total_entropy(mols, dom::Domain)::Float64
    Lx, Ly, Lz = dom.Lx, dom.Ly, dom.Lz
    Hx = shannon_entropy([mol.r[1] for mol in mols], 10, -Lx/2, Lx/2)
    Hy = shannon_entropy([mol.r[2] for mol in mols], 10, -Ly/2, Ly/2)
    Hz = shannon_entropy([mol.r[3] for mol in mols], 10, -Lz/2, Lz/2)
    Hv = shannon_entropy([norm(mol.v)^2 for mol in mols], 200, 0.0, 200000.0)
    return Hx + Hy + Hz + Hv
end

# ╔═╡ 63110af4-55b2-11f1-947c-19b51357a89f
begin
    const L12     = 5e-9
    const N12     = 400
    const v0_12   = 1400.0
    const dt12    = 1e-14
    const t_fin12 = 5e-11
    dom12     = Domain(2L12, 2L12, 2L12)
    n_steps12 = round(Int, t_fin12 / dt12)

    Random.seed!(42)
    atoms12 = Vector{Molecule}(undef, N12)
    for i in 1:N12
        # position initiale dans le 1/8 de domaine : x,y,z ∈ [-L12, 0]
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
        for mol in atoms12; computeNextPosition!(mol, dt12); end
        for mol in atoms12; applyBoundary!(mol, dom12); end
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

# ╔═╡ 63110f4a-55b2-11f1-829a-bb6eff8e136a
begin
    println("H initial ≈ ", round(H12_hist[1], digits=3),
            "  H final ≈ ", round(H12_hist[end], digits=3))
    plot(t12_hist.*1e12, H12_hist,
         xlabel="Temps (ps)", ylabel="H (bits)",
         title="Q12.2 — Entropie de Shannon", legend=false)
end

# ╔═╡ 63111058-55b2-11f1-b70f-21873504d8cd
begin
    # Domaine élargi sur x : [-5e-9, 15e-9] = longueur 20e-9
    dom12b     = Domain(20e-9, 2L12, 2L12)
    n_steps12b = 5000
    rec12b_every = 50
    n_rec12b   = div(n_steps12b, rec12b_every)
    H12b_hist  = zeros(n_rec12b)
    t12b_hist  = zeros(n_rec12b)
    t_offset   = t_fin12
    idx12b     = 1

    for step in 1:n_steps12b
        for mol in atoms12; computeNextPosition!(mol, dt12); end
        for mol in atoms12; applyBoundary!(mol, dom12b); end
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
    println("Simulation 12b terminée.")
end

# ╔═╡ 63111332-55b2-11f1-b77c-a90f74757667
begin
    all_t12 = vcat(t12_hist, t12b_hist) .* 1e12
    all_H12 = vcat(H12_hist, H12b_hist)
    p12 = plot(all_t12, all_H12,
               xlabel="Temps (ps)", ylabel="H (bits)",
               title="Q12.3 — Entropie : domaine initial puis élargi",
               legend=false)
    vline!(p12, [t_fin12*1e12], linestyle=:dash, color=:red)
    p12
end

# ╔═╡ 6311147c-55b2-11f1-bd7b-69e1df1f36f1
md"""
## Q12.2 — Observations
L'entropie **augmente** puis atteint un plateau : les atomes, initialement
confinés dans 1/8 du domaine, se répartissent dans tout le volume (diffusion).
L'entropie maximale vaut log₂(10) ≈ 3.32 bits par dimension spatiale (distribution uniforme).

## Q12.3 — Observations avec domaine élargi
Quand la paroi s'ouvre, l'entropie **augmente à nouveau** : les atomes colonisent le
nouveau volume. Ceci illustre le **second principe** : l'entropie d'un système isolé
ne peut qu'augmenter. Elle atteint un nouveau plateau correspondant à la distribution
uniforme dans le domaine élargi.
"""

# ╔═╡ 6310cfa8-55b2-11f1-b0f5-9dc2d8940a4b
begin
    # Test 1 : choc frontal masses égales → échange de vitesses
    t1a = Molecule([-1.5e-10,0.,0.], [100.,0.,0.], 1., 1e-10, "A")
    t1b = Molecule([ 1.5e-10,0.,0.], [-100.,0.,0.], 1., 1e-10, "B")
    p0  = t1a.m .* t1a.v .+ t1b.m .* t1b.v
    E0  = 0.5*t1a.m*norm(t1a.v)^2 + 0.5*t1b.m*norm(t1b.v)^2
    resolveCollision!(t1a, t1b)
    p1  = t1a.m .* t1a.v .+ t1b.m .* t1b.v
    E1  = 0.5*t1a.m*norm(t1a.v)^2 + 0.5*t1b.m*norm(t1b.v)^2
    println("Test 1 — choc frontal égal :")
    println("  conservation p  : ", isapprox(p0, p1, atol=1e-18))
    println("  conservation Ec : ", isapprox(E0, E1, rtol=1e-10))
    println("  échange vitesses : ", isapprox(t1a.v[1], -100., atol=1e-8))

    # Test 2 : m1=2m2, m2 au repos
    t2a = Molecule([-1.5e-10,0.,0.], [100.,0.,0.], 2., 1e-10, "A")
    t2b = Molecule([ 1.5e-10,0.,0.], [0.,0.,0.],   1., 1e-10, "B")
    p0b = t2a.m .* t2a.v .+ t2b.m .* t2b.v
    E0b = 0.5*t2a.m*norm(t2a.v)^2
    resolveCollision!(t2a, t2b)
    E1b = 0.5*t2a.m*norm(t2a.v)^2 + 0.5*t2b.m*norm(t2b.v)^2
    p1b = t2a.m .* t2a.v .+ t2b.m .* t2b.v
    println("\nTest 2 — m1=2·m2 :")
    println("  conservation p  : ", isapprox(p0b, p1b, atol=1e-18))
    println("  conservation Ec : ", isapprox(E0b, E1b, rtol=1e-10))
    println("  v1x ≈ 33.3 : ", isapprox(t2a.v[1], 100/3, rtol=1e-8))
    println("  v2x ≈ 133.3 : ", isapprox(t2b.v[1], 400/3, rtol=1e-8))

    # Test 3 : pas de collision
    t3a = Molecule([0.,0.,0.], [10.,0.,0.], 1., 1e-10, "A")
    t3b = Molecule([1.,0.,0.], [0.,0.,0.],  1., 1e-10, "B")
    println("\nTest 3 — pas de collision : ", !areColliding(t3a, t3b))

    # Test 4 : en collision
    t4a = Molecule([0.,0.,0.],      [0.,0.,0.], 1., 1e-10, "A")
    t4b = Molecule([1.5e-10,0.,0.], [0.,0.,0.], 1., 1e-10, "B")
    println("Test 4 — en collision    : ", areColliding(t4a, t4b))

    # Test 5 : s'éloignent → inchangé
    t5a = Molecule([-1e-10,0.,0.], [-100.,0.,0.], 1., 1e-10, "A")
    t5b = Molecule([ 1e-10,0.,0.], [ 100.,0.,0.], 1., 1e-10, "B")
    v_bef = copy(t5a.v)
    resolveCollision!(t5a, t5b)
    println("Test 5 — s'éloignent     : ", t5a.v == v_bef)
end

# ════════════════════════════════════════════════════════════════
# SECTION 5 — Domaine
# ════════════════════════════════════════════════════════════════

# ╔═╡ 6310dbce-55b2-11f1-8af6-91e3bbc7cbb4
begin
    dom_t = Domain(2., 2., 2.)   # ∈ [-1, 1]³

    b = Molecule([1.2,0.,0.], [10.,5.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 1 (+x)  correct: ", isapprox(b.r[1], 0.8, atol=1e-12) && b.v[1] < 0)

    b = Molecule([-1.3,0.,0.], [-10.,0.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 2 (-x)  correct: ", isapprox(b.r[1], -0.7, atol=1e-12) && b.v[1] > 0)

    b = Molecule([0.,1.1,0.], [0.,5.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 3 (+y)  correct: ", isapprox(b.r[2], 0.9, atol=1e-12) && b.v[2] < 0)

    b = Molecule([0.,-1.4,0.], [0.,-5.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 4 (-y)  correct: ", isapprox(b.r[2], -0.6, atol=1e-12) && b.v[2] > 0)

    b = Molecule([0.,0.,1.05], [0.,0.,3.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 5 (+z)  correct: ", isapprox(b.r[3], 0.95, atol=1e-12) && b.v[3] < 0)

    b = Molecule([0.,0.,-1.2], [0.,0.,-3.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 6 (-z)  correct: ", isapprox(b.r[3], -0.8, atol=1e-12) && b.v[3] > 0)

    b = Molecule([1.3,1.2,0.], [5.,3.,0.], 1., .1, "T")
    applyBoundary!(b, dom_t)
    println("Test 7 (coin +x+y)  correct: ", b.v[1] < 0 && b.v[2] < 0)

    b = Molecule([.5,.3,-.2], [1.,2.,3.], 1., .1, "T")
    r_bef, v_bef = copy(b.r), copy(b.v)
    applyBoundary!(b, dom_t)
    println("Test 8 (inchangé)  correct: ", b.r == r_bef && b.v == v_bef)
end

# ════════════════════════════════════════════════════════════════
# SECTION 7 — Simulation He
# ════════════════════════════════════════════════════════════════

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Plots = "~1.41.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "c475f066e07223d20cc5a14828586a844c1c1a1e"

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
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

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
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

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
git-tree-sha1 = "8f05e9a2e7c2e3eb524102bb2926c5743c07fbe1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

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
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "44716a1a667cb867ee0e9ec8edc31c3e4aa5afdc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.24"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "be8a1b8065959e24fdc1b51402f39f3b6f0f6653"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.24+0"

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
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

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
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "f76f7560267b840e492180f9899b472f30b88450"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

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
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

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
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

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
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

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
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

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
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "d5b7dd0e226774cbd87e2790e34def09245c7eab"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

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
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
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
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

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
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

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
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

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
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

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
# ╠═6310bcd4-55b2-11f1-86a7-f1f13ec531a7
# ╠═6310c044-55b2-11f1-84de-9fea747f4d7f
# ╠═6310c13e-55b2-11f1-9e2c-ff7d27813657
# ╠═6310c5aa-55b2-11f1-b3f8-af71452a6674
# ╠═6310c6d4-55b2-11f1-a5d5-33d28add3bff
# ╠═6310c74c-55b2-11f1-a5a4-29b79f7a6f07
# ╠═6310cad0-55b2-11f1-b77a-0f95a2917373
# ╠═6310ce2c-55b2-11f1-9997-a1c3e3ce4dab
# ╠═6310cea4-55b2-11f1-8462-2b0a971147bc
# ╠═6310cfa8-55b2-11f1-b0f5-9dc2d8940a4b
# ╠═6310d796-55b2-11f1-83c8-6f6bffddfd0b
# ╠═6310d80e-55b2-11f1-9454-2b5f24106b82
# ╠═6310d852-55b2-11f1-b8ce-eb8b0d931e60
# ╠═6310daa2-55b2-11f1-9571-2998aef2fcaa
# ╠═6310dbce-55b2-11f1-8af6-91e3bbc7cbb4
# ╠═6310e1b4-55b2-11f1-bf13-b5351da53e6d
# ╠═6310e240-55b2-11f1-a30e-d7eaa045a5e3
# ╠═6310e362-55b2-11f1-8b6e-4380846a6cae
# ╠═6310e75e-55b2-11f1-b095-ed83a3b47b5c
# ╠═6310e7f4-55b2-11f1-b29d-137cdafc2ca7
# ╠═6310e89c-55b2-11f1-87ce-c9695185f7b1
# ╠═6310e9ac-55b2-11f1-8a51-fd0e92fb089e
# ╠═6310ec10-55b2-11f1-8379-c95c614fa57b
# ╠═6310ecf4-55b2-11f1-aa1f-7940dd8fd1f4
# ╠═6310edf8-55b2-11f1-9f42-1b291aa58ac8
# ╠═6310f032-55b2-11f1-b660-6f33dcd8f60f
# ╠═6310f0f0-55b2-11f1-861a-29c2ea0791e4
# ╠═6310f2ee-55b2-11f1-acc5-59866099fa26
# ╠═6310f622-55b2-11f1-8f07-ad2bb1c922f1
# ╠═6310f8e6-55b2-11f1-be0b-f17e734b5a94
# ╠═6310fa46-55b2-11f1-88e9-4b7561821b4a
# ╠═6310fbc2-55b2-11f1-85d8-dda2d011335b
# ╠═6311007c-55b2-11f1-bcf8-c9aebec926a1
# ╠═6311018a-55b2-11f1-8e71-4fe346d8861a
# ╠═63110482-55b2-11f1-9685-350ab23719c9
# ╠═6311084c-55b2-11f1-8573-3b82c01d4568
# ╠═631109a0-55b2-11f1-923d-2f88a361a366
# ╠═63110af4-55b2-11f1-947c-19b51357a89f
# ╠═63110f4a-55b2-11f1-829a-bb6eff8e136a
# ╠═63111058-55b2-11f1-b70f-21873504d8cd
# ╠═63111332-55b2-11f1-b77c-a90f74757667
# ╠═6311147c-55b2-11f1-bd7b-69e1df1f36f1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
