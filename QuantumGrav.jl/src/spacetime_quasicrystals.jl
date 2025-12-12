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