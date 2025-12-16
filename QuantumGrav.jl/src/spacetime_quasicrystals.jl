using Plots
using LinearAlgebra
using CausalSets
using Base: searchsortedfirst, searchsortedlast


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
    η = Diagonal([-1.0, 1.0, 1.0, 1.0])

    # Minkowski inner product ⟨u, v⟩ = u† η v
    minkowski(u, v) = dot(u, η * v)

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
        x1_lo = ceil(Int, -2.2253 + 0.5*x0 - 0.898757*ρ)
        x1_hi = floor(Int,  2.2253 + 0.5*x0 + 0.898757*ρ)
        x1_lo > x1_hi && return local_pts

        for x1 in x1_lo:x1_hi
            # x2 bounds
            lb1 = 0.00162113*(-2829.86 + 857.666*x0 - 1098.48*x1)
            lb2 = 0.359612*x0 + 0.280776*x1 - 1.85283*ρ
            ub1 = 0.00162113*( 2829.86 + 857.666*x0 - 1098.48*x1)
            ub2 = 0.359612*x0 + 0.280776*x1 + 1.85283*ρ

            x2_lo = ceil(Int, max(lb1, lb2))
            x2_hi = floor(Int, min(ub1, ub2))
            x2_lo > x2_hi && continue # reject point if no x2 possible

            for x2 in x2_lo:x2_hi
                # radicand
                D =
                    -1.96969*x0^2 - 3.07577*x0*x1 - 1.20075*x1^2 +
                     10.9545*x0*x2 + 8.55304*x1*x2 - 15.2311*x2^2 +
                     52.2882*ρ^2
                D < 0 && continue # reject point if no x3 possible

                sqrtD = sqrt(D)
                center = 0.438447*x0 - 0.219224*x1 - 0.219224*x2

                # x3 lower bounds
                lb_lin1 = 0.25*(-75.2311 + 29.6466*x0 - 23.7245*x1 - 17.3226*x2)
                lb_lin2 = 0.25*(-75.2311 + 6.84579*x0 + 5.47829*x1 - 0.923651*x2)
                lb_quad = center - 0.25*sqrtD

                # x3 upper bounds
                ub_lin1 = 0.00664619*(1115.17*x0 - 892.41*x1 - 651.597*x2)
                ub_lin2 = 0.00664619*( 257.508*x0 + 206.069*x1 - 34.7436*x2)
                ub_quad = center + 0.25*sqrtD

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

big_set =quasicrystal(700.)

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
    exact_size::Bool = false,
    deviation_from_mean_size::Float64=0.03,
    max_iter::Int64=20,
)

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

    if exact_size

        function count_for_halfℓ(hℓ)
            αin_lo  = αin₀  - hℓ
            αin_hi  = αin₀  + hℓ
            αout_lo = αout₀ - hℓ
            αout_hi = αout₀ + hℓ

            i_lo = searchsortedfirst(αin, αin_lo)
            i_hi = searchsortedlast(αin, αin_hi)

            count = 0
            if i_lo <= i_hi
                for i in i_lo:i_hi
                    if αout_lo ≤ αout[i] ≤ αout_hi
                        count += 1
                    end
                end
            end
            return count
        end

        hℓ_lo = (1 - deviation_from_mean_size) * halfℓ
        hℓ_hi = (1 + deviation_from_mean_size) * halfℓ
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
                #@warn "Max iterations reached in size adjustment; got $count_mid points (target $N). Increase `deviation_from_mean_size` or `max_iter`."
                return "not converged"
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

count = 0
for i in 1:1000
    center = (rand(), rand())
    if translate_sub_spacetime_crystal(
        256, center; 
        crystal=big_set, 
        exact_size=true, 
        deviation_from_mean_size=0.0015,
        max_iter=30) == "not converged"
        count += 1
        println("Failed at center: $center")
    end
end

count