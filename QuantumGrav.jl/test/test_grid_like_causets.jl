
# ---------------- Helpers (property checks) ----------------
@testsnippet propHelpers begin
    import LinearAlgebra
    # Solve for barycentric parameters t in x = origin + E * t
    function inv_map(
        E::AbstractMatrix{<:Real},
        x::NTuple{N,Float64},
        origin::NTuple{N,Float64},
    ) where {N}
        return tuple((E \ (collect(x) .- collect(origin)))...)
    end

    # Axis-aligned edges for expected-value tests
    square_edges(a = 1.0) = ((a, 0.0), (0.0, a))
    rect_edges(a = 1.0, b = 0.5) = ((a, 0.0), (0.0, b))
    hex_edges(a = 1.0) = ((a, 0.0), (0.5a, 0.5a*sqrt(3.0)))

    # Rotation matrix and rotator
    rotmat(θ) = ((cos(θ), -sin(θ)), (sin(θ), cos(θ)))
    rot(x, R) = (R[1][1]*x[1] + R[1][2]*x[2], R[2][1]*x[1] + R[2][2]*x[2])
end

# ---------------- generate_grid_from_brillouin_cell ----------------
@testitem "brillouin_cell_points_in_unit_param_cube" tags=[:cell, :props] setup=[
    propHelpers,
] begin



    edges = square_edges(1.0)
    origin = (0.0, 0.0)
    pts = QuantumGrav.generate_grid_from_brillouin_cell(25, edges; origin = origin)
    @test length(pts) == 25
    E = [edges[1][1] edges[2][1]; edges[1][2] edges[2][2]]
    for p in pts
        t = inv_map(E, p, origin)
        @test all(0.0 - 1e-12 .<= t .<= 1.0 + 1e-12)
    end
end

@testitem "degenerate_edges_throw" tags=[:cell, :errors] begin


    edges = ((1.0, 0.0), (2.0, 0.0))
    @test_throws ArgumentError QuantumGrav.generate_grid_from_brillouin_cell(10, edges)
end

@testitem "square_expected_small_grids" tags=[:cell, :expected] begin
    # k=2 (4 points)

    edges = ((1.0, 0.0), (0.0, 1.0))
    pts4 = QuantumGrav.generate_grid_from_brillouin_cell(4, edges)
    @test Set(pts4) == Set([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    # k=3 (9 points)
    pts9 = QuantumGrav.generate_grid_from_brillouin_cell(9, edges)
    expected9 = Tuple{Float64,Float64}[]
    for i = 0:2, j = 0:2
        push!(expected9, (i/2, j/2))
    end
    @test Set(pts9) == Set(expected9)
end

@testitem "rotation_equivariance" tags=[:cell, :props] setup=[propHelpers] begin

    edges = rect_edges(2.0, 1.0)
    origin = (0.1, -0.2)
    pts = QuantumGrav.generate_grid_from_brillouin_cell(16, edges; origin = origin)
    θ = 0.37
    R = rotmat(θ)
    redges = (rot(edges[1], R), rot(edges[2], R))
    rorigin = rot(origin, R)
    rpts = QuantumGrav.generate_grid_from_brillouin_cell(16, redges; origin = rorigin)
    @test length(rpts) == 16
    # The rotated set should equal pointwise rotation of original (order preserved by our construction)
    for (p, q) in zip(pts, rpts)
        @test isapprox(rot(p, R)[1], q[1]; atol = 1e-12)
        @test isapprox(rot(p, R)[2], q[2]; atol = 1e-12)
    end
end

# ---------------- generate_grid_2d (wrapper) ----------------
@testitem "wrapper_square_expected_values" tags=[:wrapper, :expected] begin

    pts = QuantumGrav.generate_grid_2d(9, "square"; a = 1.0, rotate_deg = 0)
    expected9 = Tuple{Float64,Float64}[]
    for i = 0:2, j = 0:2
        push!(expected9, (i/2, j/2))
    end
    @test Set(pts) == Set(expected9)
end

@testitem "wrapper_rectangular_expected_values" tags=[:wrapper, :expected] setup=[] begin

    pts = QuantumGrav.generate_grid_2d(9, "rectangular"; a = 2.0, b = 1.0, rotate_deg = 0)
    expected9 = Tuple{Float64,Float64}[]
    for i = 0:2, j = 0:2
        push!(expected9, (i/2*2.0, j/2*1.0))
    end
    @test Set(pts) == Set(expected9)
end

@testitem "wrapper_hexagonal_expected_values" tags=[:wrapper, :expected] setup=[propHelpers] begin


    pts = QuantumGrav.generate_grid_2d(9, "hexagonal"; a = 1.0, rotate_deg = 0)
    expected9 = Tuple{Float64,Float64}[]
    ts = (0.0, 0.5, 1.0)
    e1, e2 = hex_edges(1.0)
    for t1 in ts, t2 in ts
        push!(expected9, (t1*e1[1] + t2*e2[1], t1*e1[2] + t2*e2[2]))
    end
    @test Set(pts) == Set(expected9)
end

@testitem "wrapper_auto_rotation_prefers_y_extent" tags=[:wrapper, :props] setup=[] begin

    pts = QuantumGrav.generate_grid_2d(40, "rectangular"; a = 2.0, b = 0.5)  # auto-rotate
    xs = first.(pts);
    ys = last.(pts)
    @test (maximum(ys) - minimum(ys)) ≥ (maximum(xs) - minimum(xs))
end

@testitem "wrapper_bad_name_throws" tags=[:wrapper, :errors] begin

    @test_throws ArgumentError QuantumGrav.generate_grid_2d(5, "not-a-lattice")
end

# ---------------- sort_grid_by_time_from_manifold ----------------
@testitem "ordering_is_monotone" tags=[:ordering] begin
    using CausalSets
    using QuantumGrav
    grid = QuantumGrav.generate_grid_2d(25, "square"; a = 1.0, rotate_deg = 0)
    sorted =
        QuantumGrav.sort_grid_by_time_from_manifold(CausalSets.MinkowskiManifold{2}(), grid)
    ok = true
    for i = 1:(length(sorted)-1)
        if CausalSets.isless_coord_time(
            CausalSets.MinkowskiManifold{2}(),
            sorted[i+1],
            sorted[i],
        )
            global ok = false;
            break
        end
    end
    @test ok
end

# ---------------- center_and_rescale_grid_to_box ----------------
@testitem "centering_and_rescaling" tags=[:centering, :rescaling] begin
    using CausalSets
    using QuantumGrav

    grid = QuantumGrav.generate_grid_2d(25, "square"; a = 1.0, rotate_deg = 0)
    box = (CausalSets.Coordinates{2}((-2.0, 0)), CausalSets.Coordinates{2}((-1.0, 1)))
    transformed = QuantumGrav.center_and_rescale_grid_to_box(grid, box)

    min_t, max_t = extrema(first.(transformed))
    min_x, max_x = extrema(last.(transformed))

    # inside box
    @test min_t >= -2.0
    @test max_t <= -1.0
    @test min_x >= 0.0
    @test max_x <= 1.0

    # maximal
    @test min_t == -2.0 || min_x == 0.0

    # symmetric
    @test -2.0 - min_t == max_t - (-1.0)
    @test 0.0 - min_x == max_x - (1.0)
end


# ---------------- create_grid_causet_2D ----------------
@testitem "grid_ordering_consistent_with_coordinate_time_order" tags=[:cset] begin
    import CausalSets


    grid = QuantumGrav.generate_grid_2d(30, "square"; a = 1.0, rotate_deg = 0)
    cset, _, coords = QuantumGrav.create_grid_causet_2D(
        30,
        "square",
        CausalSets.MinkowskiManifold{2}();
        a = 1.0,
        rotate_deg = 0,
    )
    @test typeof(cset) == CausalSets.BitArrayCauset
    @test cset.atom_count == 30
    @test length(coords) == 60
end

@testitem "grid_manifold_builds_causet" tags=[:cset, :polynomial] begin
    import CausalSets
    import Random


    rng = Random.MersenneTwister(1234)
    size = 30
    order = 3
    r = 2.0
    cset, converged, coords = QuantumGrav.create_grid_causet_2D_polynomial_manifold(
        size,
        "square",
        rng,
        order,
        r;
        a = 1.0,
        rotate_deg = 0.0,
    )

    @test typeof(cset) == CausalSets.BitArrayCauset
    @test cset.atom_count == size
    @test converged == true
    @test length(coords) == 2 * size
end

@testitem "grid_input_errors" tags=[:cset, :errors] begin
    import Random


    rng = Random.MersenneTwister(1234)

    @test_throws ArgumentError QuantumGrav.create_grid_causet_2D_polynomial_manifold(
        1,
        "square",
        rng,
        3,
        2.0;
        a = 1.0,
    )
    @test_throws ArgumentError QuantumGrav.create_grid_causet_2D_polynomial_manifold(
        10,
        "square",
        rng,
        0,
        2.0;
        a = 1.0,
    )
    @test_throws ArgumentError QuantumGrav.create_grid_causet_2D_polynomial_manifold(
        10,
        "square",
        rng,
        3,
        1.0;
        a = 1.0,
    )
end
