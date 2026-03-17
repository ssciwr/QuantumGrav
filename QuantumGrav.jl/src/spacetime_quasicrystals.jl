"""
    quasicrystal(ρ::Real) -> Tuple{Vector{Float64},Vector{Float64}}
Generate a 2D spacetime quasicrystal via a 4D cut-and-project construction
with internal acceptance radius `ρ`.
The construction:
- enumerates integer lattice points in 4D satisfying analytic bounds,
- projects them to physical spacetime using `PhysicalProj`,
- computes lightcone coordinates `(α_in, α_out)` via Minkowski inner products,
- returns the points restricted to the unit causal diamond,
  sorted by `α_in` for efficient range queries.

Returns
-------
A tuple `(α_in, α_out)` where:
- `α_in  :: Vector{Float64}`
- `α_out :: Vector{Float64}`

Both vectors have equal length and are sorted by increasing `α_in`.
"""
function quasicrystal(ρ::Real)::Tuple{Vector{Float64},Vector{Float64}}

    √ = sqrt

    vout = [
        -√(4 + √17) + (5 + √17)/2,
        (5 + √17 - 2*√(53/2 + (13*√17)/2)) / 4,
        (5 + √17 - 2*√(13/2 + (5*√17)/2)) / 4,
        1.0
    ]

    vin = [
        √(4 + √17) + (5 + √17)/2,
        (5 + √17 + 2*√(53/2 + (13*√17)/2)) / 4,
        (5 + √17 + 2*√(13/2 + (5*√17)/2)) / 4,
        1.0
    ]
   
    # Minkowski metric η = diag(-1, 1, 1, 1)
    η = LinearAlgebra.Diagonal([-1.0, 1.0, 1.0, 1.0])

    # Minkowski inner product ⟨u, v⟩ = u† η v
    minkowski(u, v) = LinearAlgebra.dot(u, η * v)

    # compute products
    vin_vout  = minkowski(vin, vout)

    PhysicalProj = [
        1/2 + 7/(2*√17)   0.0               -2/√17           -2/√17;
        0.0               1/2 + 3/(2*√17)    1/√17            -1/√17;
        2/√17             1/√17              1/2 - 5/(2*√17)  -1/√17;
        2/√17            -1/√17             -1/√17            1/2 - 5/(2*√17)
    ]
    
    points = Vector{NTuple{4,Int}}()

    # helper: generate points for a fixed x0
    function points_for_x0(x0::Int)
        local_pts = NTuple{4,Int}[]

        # x1 bounds
        x1_lo = ceil(Int,  -2.22529571428407 + 0.5 * x0 - 0.898756954935066 * ρ)
        x1_hi = floor(Int,  2.22529571428407 + 0.5 * x0 + 0.898756954935066 * ρ)
        x1_lo > x1_hi && return local_pts

        for x1 in x1_lo:x1_hi
            # x2 bounds
            lb1 = 0.001621130114261006 * (-2829.855912710005 + 857.6660139559601 * x0 - 1098.478395249714 * x1)
            lb2 = 0.359611796797792 * x0 + 0.280776406404415 * x1 - 1.85283492847788 * ρ
            ub1 = 0.001621130114261006 * ( 2829.855912710005 + 857.6660139559601 * x0 - 1098.478395249714 * x1)
            ub2 = 0.359611796797792 * x0 + 0.280776406404415 * x1 + 1.85283492847788 * ρ

            x2_lo = ceil(Int, max(lb1, lb2))
            x2_hi = floor(Int, min(ub1, ub2))
            x2_lo > x2_hi && continue # reject point if no x2 possible

            for x2 in x2_lo:x2_hi
                # radicand
                D = -1.969690009882569 * x0^2 - 3.075774975293578 * x0 * x1 - 
                    1.200746266059174 * x1^2 + 10.95453501482385 * x0 * x2 + 
                    8.553042482705505 * x1 * x2 - 15.23105625617661 * x2^2 + 
                    52.28817457999083 * ρ^2
                D < 0 && continue # reject point if no x3 possible

                sqrtD = sqrt(D)
                center = 0.4384471871911697 * x0 - 0.2192235935955849 * x1 - 0.2192235935955849 * x2

                # x3 lower bounds
                lb_lin1 = 0.25*(-75.23105625617661 + 29.64663624374690 * x0 - 23.72450097900972 * x1 - 
                            17.32256025724875 *x2)
                lb_lin2 = 0.25*(-75.23105625617661 + 6.845786258723745 * x0 + 5.478289727774397 * x1 - 
                            0.9236509939865662 * x2)
                lb_quad = center - 0.25 * sqrtD

                # x3 upper bounds
                ub_lin1 = 0.006646191411927027 * (1115.173879529864 * x0 - 892.4096339007985 * x1 - 
                            651.5972526070451 * x2)
                ub_lin2 = 0.006646191411927027 * (257.5078655739034 * x0 + 206.0687613489150 * x1 - 
                            34.74361994483840 * x2)
                ub_quad = center + 0.25 * sqrtD

                x3_lo = ceil(Int, max(lb_lin1, lb_lin2, lb_quad))
                x3_hi = floor(Int, min(ub_lin1, ub_lin2, ub_quad))
                x3_lo > x3_hi && continue # reject point if no x3 possible

                for x3 in x3_lo:x3_hi
                    push!(local_pts, (x0,x1,x2,x3))
                end
            end
        end

        return local_pts
    end

    # scan x0 symmetrically using convexity
    x0 = 0
    empty_pos = false
    empty_neg = false

    while true
        if !empty_pos
            pts = points_for_x0(x0)
            if isempty(pts)
                empty_pos = true
            else
                append!(points, pts)
            end
        end

        if x0 != 0 && !empty_neg
            pts = points_for_x0(-x0)
            if isempty(pts)
                empty_neg = true
            else
                append!(points, pts)
            end
        end

        empty_pos && empty_neg && break
        x0 += 1
    end

    projected = Vector{Vector{Float64}}(undef, length(points))
    for (i,(x0,x1,x2,x3)) in enumerate(points)
        projected[i] = PhysicalProj * Float64[x0,x1,x2,x3]
    end

    αin = [minkowski(vout, point) ./ vin_vout for point in projected]
    αout = [minkowski(vin, point) ./ vin_vout for point in projected]

    perm = sortperm(αin)
    αin_sorted = αin[perm]
    αout_sorted = αout[perm]

    return (αin_sorted, αout_sorted)
end

"""
    translate_sub_spacetime_crystal(
        N::Int,
        center::NTuple{2,Float64};
        ρ::Union{Float64,Nothing} = nothing,
        crystal::Union{Tuple{Vector{Float64},Vector{Float64}},Nothing} = nothing
    ) -> CausalSets.Coordinates{2}

Extract a local causal diamond of expected size `N` from a 2D spacetime
quasicrystal in lightcone coordinates `(α_in, α_out)`.

The function:
- takes a quasicrystal filling unit causal diamond,
- assumes the crystal is sorted by increasing `α_in`,
- selects a rectangular causal diamond around `center`
  with side length `ℓ = sqrt(N / N_unit)`,
- returns all points inside that diamond.

Arguments
---------
- `N` : target causal set size (expected, up to discreteness fluctuations)
- `center` : `(α_in, α_out)` with values inside `(ℓ,1-ℓ)×(ℓ,1-ℓ)`
- `ρ` : internal acceptance radius (used only if `crystal` is not provided)
- `crystal` : optional precomputed quasicrystal `(α_in, α_out)`

Returns
-------
`CausalSets.Coordinates{2}` (alias for `Vector{NTuple{2,Float64}}`)
containing the selected points."""
function translate_sub_spacetime_crystal(
    N::Int64,
    center::NTuple{2,Float64};
    ρ::Union{Float64,Nothing} = nothing,
    crystal::Union{Tuple{Vector{Float64},Vector{Float64}},Nothing} = nothing,
    exact_size::Bool = true,
    deviation_from_mean_size::Float64=0.1,
    max_iter::Int64=100,
)::Vector{CausalSets.Coordinates{2}}

    αin₀, αout₀ = center

    # --- 1. get quasicrystal in unit diamond
    if ρ === nothing && crystal === nothing
        error("Either ρ or crystal must be provided")
    elseif crystal === nothing
        αin, αout = quasicrystal(ρ)
    else
        αin, αout = crystal
    end

    Nunit = length(αin)
    Nunit == 0 && error("Empty quasicrystal")

    # --- 2. local diamond side length
    ℓ = sqrt(N / Nunit)
    halfℓ = ℓ / 2

    # --- ensure local diamond is fully contained in unit diamond

    if αin₀  - halfℓ < 0 || αin₀  + halfℓ > 1 ||
       αout₀ - halfℓ < 0 || αout₀ + halfℓ > 1
        error("Local causal diamond not fully contained in unit diamond. Choose center further inside or reduce N.")
    end


    if exact_size

        hℓ_lo = (1 - deviation_from_mean_size) * halfℓ
        hℓ_hi = (1 + deviation_from_mean_size) * halfℓ

        # Precompute maximal candidate slice for bisection (monotonicity region)
        αin_lo_max  = αin₀  - hℓ_hi
        αin_hi_max  = αin₀  + hℓ_hi
        αout_lo_max = αout₀ - hℓ_hi
        αout_hi_max = αout₀ + hℓ_hi

        i_lo_max = searchsortedfirst(αin, αin_lo_max)
        i_hi_max = searchsortedlast(αin, αin_hi_max)

        cand_αin  = Float64[]
        cand_αout = Float64[]

        for i in i_lo_max:i_hi_max
            if αout_lo_max ≤ αout[i] ≤ αout_hi_max
                push!(cand_αin,  αin[i])
                push!(cand_αout, αout[i])
            end
        end

        function count_for_halfℓ(hℓ)
            αin_lo  = αin₀  - hℓ
            αin_hi  = αin₀  + hℓ
            αout_lo = αout₀ - hℓ
            αout_hi = αout₀ + hℓ

            count = 0
            for (a_in, a_out) in zip(cand_αin, cand_αout)
                if αin_lo ≤ a_in ≤ αin_hi && αout_lo ≤ a_out ≤ αout_hi
                    count += 1
                end
            end
            return count
        end

        count_mid = count_for_halfℓ(halfℓ)
        hℓ_mid = (hℓ_lo + hℓ_hi) / 2

        for i in (1:max_iter)
            count_mid = count_for_halfℓ(hℓ_mid)
            if count_mid > N
                hℓ_hi = hℓ_mid
            elseif count_mid < N
                hℓ_lo = hℓ_mid
            end
            hℓ_mid = (hℓ_lo + hℓ_hi) / 2

            if count_mid == N
                break
            end

            if i == max_iter && count_mid != N
                @warn "Max iterations reached in size adjustment; got $count_mid points (target $N). Increase `deviation_from_mean_size` or `max_iter`."
            end
        end

        halfℓ = hℓ_mid
    end


    # --- 3. bounds of local diamond
    αin_lo  = αin₀  - halfℓ
    αin_hi  = αin₀  + halfℓ
    αout_lo = αout₀ - halfℓ
    αout_hi = αout₀ + halfℓ

    # --- 4. extract points
    coords = NTuple{2,Float64}[]
    
    i_lo = searchsortedfirst(αin, αin_lo)
    i_hi = searchsortedlast(αin, αin_hi)

    if i_lo <= i_hi
        for i in i_lo:i_hi
            if αout_lo ≤ αout[i] ≤ αout_hi
                push!(coords, (αin[i], αout[i]))
            end
        end
    end

    return coords
end

function create_Minkowski_quasicrystal_cset(
    N::Int64,
    center::NTuple{2,Float64};
    ρ::Union{Float64,Nothing} = nothing,
    crystal::Union{Tuple{Vector{Float64},Vector{Float64}},Nothing} = nothing,
    exact_size::Bool = true,
    deviation_from_mean_size::Float64=0.1,
    max_iter::Int64=100,
)::CausalSets.BitArrayCauset

    if ρ === nothing && crystal === nothing
        error("Either ρ or crystal must be provided")
    end

    point_set = translate_sub_spacetime_crystal(
        N,
        center;
        ρ = ρ,
        crystal = crystal,
        exact_size = exact_size,
        deviation_from_mean_size = deviation_from_mean_size,
        max_iter = max_iter,
    )

    # convert from lightcone (α_in, α_out) to Cartesian Minkowski (t, x)
    # conventions: t = (α_in + α_out)/2, x = (α_out - α_in)/2
    cartesian_points = Vector{CausalSets.Coordinates{2}}([
        ((αin + αout)/2, (αout - αin)/2) for (αin, αout) in point_set
    ])

    # order by time coordinate
    sort!(cartesian_points, by = p -> p[1])

    return CausalSets.BitArrayCauset(CausalSets.MinkowskiManifold{2}(), cartesian_points)
end