using Plots
using LinearAlgebra

function quasicrystal_non_translated(num_points::Int)
    
    ρ = sqrt((num_points - a) / b)

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


    return (αin, αout)
end

quasicrystal_non_translated(1024)[1]

pts = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, 30)
for i in 1:30
    pts[i] = quasicrystal_non_translated(i)
end

x = collect(1:3)
y = length.(first.(pts[1:3]))

# design matrix
X = hcat(ones(length(x)), x.^2)

# least-squares fit
coeffs = X \ y
a, b = coeffs

xfit = range(1, 3; length=300)
yfit = a .+ b .* xfit.^2

ρ = sqrt((num_points - a) / b)

scatter(
    x,
    y;
    label = "data",
    xlabel = "ρ",
    ylabel = "number of points"
)

plot!(
    xfit,
    yfit;
    label = "fit: a + b ρ²",
    linewidth = 2
)

scatter(
    pts[30][1],
    pts[30][2],
    markersize = 3,
    xlabel = "α_in",
    ylabel = "α_out",
    aspect_ratio = :equal,
    xlims = (0, 1.),
    ylims = (0, 1.),
    legend = false,
)

test=quasicrystal_non_translated(500)

length(test[1])

length.(pts[:,1])

using Plots

scatter(
    pts[30][1],
    pts[30][2],
    markersize = 3,
    xlabel = "α_in",
    ylabel = "α_out",
    aspect_ratio = :equal,
    xlims = (0, 1.),
    ylims = (0, 1.),
    legend = false,
)

quasicrystal_non_translated(100.0)

# convenience
using LinearAlgebra

chop(z::ComplexF64; tol::Float64=1e-12) =
    complex(
        abs(real(z)) < tol ? 0.0 : real(z),
        abs(imag(z)) < tol ? 0.0 : imag(z),
    )

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

v1 = Complex[
    (5 - √17)/2 + im*√(-4 + √17),
    (5 - √17)/4 - im*√(2*(-53 + 13*√17))/4,
    (5 - √17)/4 + im*√(2*(-13 + 5*√17))/4,
    1.0
]

v1star = Complex[
    (5 - √17)/2 - im*√(-4 + √17),
    (5 - √17)/4 + im*√(2*(-53 + 13*√17))/4,
    (5 - √17)/4 - im*√(2*(-13 + 5*√17))/4,
    1.0
]

PhysicalProj = [
    1/2 + 7/(2*√17)   0.0               -2/√17           -2/√17;
    0.0               1/2 + 3/(2*√17)    1/√17            -1/√17;
    2/√17             1/√17              1/2 - 5/(2*√17)  -1/√17;
    2/√17            -1/√17             -1/√17            1/2 - 5/(2*√17)
]

InternalProj = [
    1/2 - 7/(2*√17)   0.0                2/√17            2/√17;
    0.0               1/2 - 3/(2*√17)   -1/√17            1/√17;
   -2/√17            -1/√17              1/2 + 5/(2*√17)  1/√17;
   -2/√17             1/√17              1/√17            1/2 + 5/(2*√17)
]

# Minkowski metric η = diag(-1, 1, 1, 1)
η = Diagonal([-1.0, 1.0, 1.0, 1.0])

# Minkowski inner product ⟨u, v⟩ = uᵀ η v
minkowski(u, v) = dot(u, η * v)

# compute products
vin_vin   = minkowski(vin, vin)
vout_vout = minkowski(vout, vout)
vin_vout  = minkowski(vin, vout)

minkowski(v1, v1star)

# 4D integer grid as vectors
grid = [Float64[i, j, k, l]
        for i in -1000:1000, j in -1000:1000, k in -1000:1000, l in -1000:1000]

# flatten if you want a 1D collection
grid = vec(grid)

# now linear algebra works
proj_internal = [InternalProj * g for g in grid]
proj_physical = [PhysicalProj * g for g in grid]

norm = minkowski(v1, v1)
norm = chop(minkowski(v1star, v1star); tol=1e-15)

int_coord = [minkowski(v1,point) ./ norm for point in proj_internal]

int_coord = int_coord .+ (1 + im * sqrt(17)) 

int_coord

int_coord_norms = real([dot(point,point) for point in int_coord])

accepted_points = []
for (i, norm) in enumerate(int_coord_norms)
    if norm <= 1
        push!(accepted_points, i)
    end
end

proj_physical_accepted = proj_physical[accepted_points]

αin = [minkowski(vout, point) ./ vin_vout for point in proj_physical_accepted]
αout = [minkowski(vin, point) ./ vin_vout for point in proj_physical_accepted]

αpairs = collect(zip(αin, αout))

αmat = hcat(αin, αout)  # size (N, 2)
# element type: Tuple{Float64,Float64} (assuming real)

using Plots

scatter(
    real.(αin),
    real.(αout),
    markersize = 3,
    xlabel = "α_in",
    ylabel = "α_out",
    aspect_ratio = :equal,
    xlims = (-.5, .5),
    ylims = (-.5, .5),
    legend = false,
)