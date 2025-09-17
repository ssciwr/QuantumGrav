"""
    generate_random_branch_points(nPoints::Int, nTuples::Int; d::Int64 = 2, rng::AbstractRNG = Random.default_rng())
        -> Tuple{Vector{Coordinates{d}}, Vector{Tuple{Coordinates{d}, Coordinates{d}}}}
Randomly generates `nPoints` distinct coordinate points and `nTuples` pairs of coordinate points in the hypercube `[-1,1]^d`. The returned branch points are sorted by coordinate time.

# Arguments
- `nPoints::Int`: Number of single random points to generate (must be ≥ 0).
- `nTuples::Int`: Number of random pairs of points to generate (must be ≥ 0).
- `d::Int64`: Dimension of the coordinate space (default 2). Currently, only d=2 is supported.
- `rng::AbstractRNG`: Optional RNG (default: `Random.default_rng()`).

# Returns
- `Vector{Coordinates{d}}`: `nPoints` randomly selected points, sorted by coordinate time (first coordinate).
- `Vector{Tuple{Coordinates{d}, Coordinates{d}}}`: `nTuples` pairs of random points, each tuple is sorted by the first element's coordinate time, and the list of tuples is sorted by the first element’s coordinate time.

# Throws
- `ArgumentError` if `nPoints < 0` or `nTuples < 0`.
"""
function generate_random_branch_points(nPoints::Int, nTuples::Int; d::Int64 = 2, rng::Random.AbstractRNG = Random.default_rng())::Tuple{Vector{CausalSets.Coordinates{d}}, Vector{Tuple{CausalSets.Coordinates{d}, CausalSets.Coordinates{d}}}}

    if nPoints < 0
        throw(ArgumentError("nPoints must be at least 0, got $(nPoints)."))
    end

    if nTuples < 0
        throw(ArgumentError("nTuples must be at least 0, got $(nTuples)."))
    end

    # Generate random points in [-1,1]^N
    single_points = [ntuple(_ -> rand(rng, -1.0:0.0001:1.0), d) for _ in 1:nPoints]

    # Generate tuples of 2 random points each, order each tuple so a[1] ≤ b[1], then sort by a[1]
    raw_tuples = [ (ntuple(_ -> rand(rng, -1.0:0.0001:1.0), d), ntuple(_ -> rand(rng, -1.0:0.0001:1.0), d)) for _ in 1:nTuples ]
    tuple_points = sort([(a[1] <= b[1] ? (a, b) : (b, a)) for (a, b) in raw_tuples], by = x -> x[1][1])

    return sort(single_points, by = x -> x[1]), tuple_points # sort by coordinate time
end

"""
    point_segment_distance(p, a, b; tolerance=1e-12)

Compute the minimal Euclidean distance between a point `p` and a line segment from `a` to `b` in 2D Minkowski-like coordinates (interpreted as Euclidean here).

# Arguments
- `p::CausalSets.Coordinates{2}`: The point as a 2D coordinate (t, x).
- `a::CausalSets.Coordinates{2}`: The first endpoint of the segment as a 2D coordinate.
- `b::CausalSets.Coordinates{2}`: The second endpoint of the segment as a 2D coordinate.
- `tolerance::Float64`: (Keyword, default `1e-12`) Tolerance for detecting degenerate (zero-length) segments.

# Returns
- `Float64`: The minimal Euclidean distance from `p` to the segment [a,b]. If the segment is degenerate (i.e., `a ≈ b` within `tolerance`), returns the distance from `p` to `a`.

# Notes
- The function treats the coordinates as points in 2D Euclidean space, regardless of their interpretation as Minkowski coordinates.
- When the segment is degenerate (zero length), the distance from `p` to `a` is returned.
- Used for filtering sprinkled points near branch cuts and for argument validation (e.g., ensuring no point is too close to a cut).
"""
function point_segment_distance(
    p::CausalSets.Coordinates{2},
    a::CausalSets.Coordinates{2},
    b::CausalSets.Coordinates{2};
    tolerance::Float64=1e-12,
)::Float64
    pt, px = p
    at, ax = a
    bt, bx = b
    abt, abx = bt - at, bx - ax
    denom = abt^2 + abx^2
    if denom < tolerance
        return sqrt((pt - at)^2 + (px - ax)^2)
    end
    s = clamp(((pt - at)*abt + (px - ax)*abx)/denom, 0.0, 1.0)
    projt = at + s*abt
    projx = ax + s*abx
    return sqrt((pt - projt)^2 + (px - projx)^2)
end

"""
    filter_sprinkling_near_cuts(sprinkling, branch_point_info; tmax=1.0, ε=1e-12)

Remove points from `sprinkling` that lie within distance `ε` of any cut.
Cuts are either:
- Tuples `(p, q)` representing finite cut segments, or
- Single branch points `b` representing vertical cuts from `b` up to `(tmax, b[2])`.

# Arguments
- `sprinkling::Vector{Coordinates{2}}`: List of points.
- `branch_point_info::Tuple{Vector{Coordinates{2}}, Vector{Tuple{Coordinates{2}, Coordinates{2}}}}`: A tuple containing the single branch points and pair cuts.
- `tolerance::Float64`: Minimum allowed Euclidean distance to a cut (keyword, default 1e-12).

# Returns
- Filtered `Vector{Coordinates{2}}`.
"""
function filter_sprinkling_near_cuts(
    sprinkling::Vector{CausalSets.Coordinates{2}},
    branch_point_info::Tuple{Vector{CausalSets.Coordinates{2}},Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}}};
    tolerance::Float64 = 1e-12
)

    if tolerance <= 0
        throw(ArgumentError("tolerance must be positive, got $tolerance"))
    end

    if any(sprinkling[i][1] > sprinkling[i+1][1] for i in 1:length(sprinkling)-1)
        throw(ArgumentError("sprinkling must be sorted by coordinate time."))
    end

    single_branch_points = branch_point_info[1] # cuts until boundary in coordinate-time direction -> trouser-geometry like
    branch_point_tuples = branch_point_info[2] # cuts between branch points given as tuples

    tmax = sprinkling[end][1] # used to create vertical cuts from single_branch_points

    filtered = CausalSets.Coordinates{2}[]
    for p in sprinkling
        too_close = false
        # check tuples
        for (a,b) in branch_point_tuples
            if point_segment_distance(p,a,b; tolerance=tolerance) < tolerance
                too_close = true
                break
            end
        end
        # check vertical cuts
        if !too_close
            for b in single_branch_points
                v_end = CausalSets.Coordinates{2}((tmax,b[2]))
                if point_segment_distance(p,b,v_end; tolerance=tolerance) < tolerance
                    too_close = true
                    break
                end
            end
        end
        if !too_close
            push!(filtered,p)
        end
    end
    return filtered
end

"""
    intersections_with_cuts(manifold, ray_origin, slope, branch_point_tuples, y) -> Union{Tuple{Coordinates{2}, Int}, Nothing}

Compute the earliest intersection between a null ray (with slope ±1) starting from `ray_origin` and any branch cut segment in `branch_point_tuples`, restricted to occur before the event `y`.

# Arguments
- `manifold`: The background manifold (used for causal ordering).
- `ray_origin::Coordinates{2}`: The starting point of the null ray.
- `slope::Float64`: Slope of the null ray; should be ±1.
- `branch_point_tuples`: Vector of cut segments, each as a tuple `(p, q)` of `Coordinates{2}`.
- `y::Coordinates{2}`: The event restricting the search; only intersections in the causal past of `y` are considered.

# Returns
- Either `nothing` if no valid intersection is found, or a tuple `(intersection::Coordinates{2}, index::Int)` where `intersection` is the earliest intersection point and `index` is the 1-based index of the cut segment intersected.

# Notes
- The null ray is parameterized as `t = t₀ + a`, `x = x₀ + slope*a`, with `a ≥ 0`, starting from `ray_origin = (t₀, x₀)`.
- Each branch cut is a segment from `p` to `q`, parameterized by `s ∈ [0, 1]`.
- The function solves for intersections between the ray and each segment, only accepting those in the future of `ray_origin` (i.e., `a ≥ 0`) and in the causal past of `y`.
- Only intersections occurring before the coordinate time of `y` are considered, and the earliest such intersection (smallest `t`) is returned.
"""
function intersections_with_cuts(
    manifold::CausalSets.AbstractManifold,
    ray_origin::CausalSets.Coordinates{2},
    slope::Float64,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
    y::CausalSets.Coordinates{2};
    tolerance::Float64 = 1e-12
)::Union{Tuple{CausalSets.Coordinates{2},Int}, Nothing}
    best_intersection = nothing
    best_index = nothing
    best_time = Inf
    t0, x0 = ray_origin
    for (idx, (p, q)) in enumerate(branch_point_tuples)
        t1, x1 = p
        t2, x2 = q
        # Parametrize the segment: (x(t), t(t)) = p + s*(q - p), s in [0,1]
        dx = x2 - x1
        dt = t2 - t1
        # Parametrize the ray: t = t0 + a, x = x0 + slope*a, a >= 0
        # Solve for intersection: x0 + slope*a = x1 + s*dx and t0 + a = t1 + s*dt
        # From the two equations:
        # a = t1 + s*dt - t0
        # x0 + slope*a = x1 + s*dx
        # Substitute a:
        # x0 + slope*(t1 + s*dt - t0) = x1 + s*dx
        # Rearranged:
        # slope*t1 + slope*s*dt - slope*t0 + x0 = x1 + s*dx
        # Collect terms with s:
        # s*(slope*dt - dx) = x1 - x0 - slope*(t1 - t0)
        denom = slope*dt - dx
        # If denom is zero, lines are parallel or colinear
        if abs(denom) < tolerance
            continue
        end
        s = (x1 - x0 - slope*(t1 - t0)) / denom
        a = t1 + s*dt - t0
        # Check if s in [0,1] and a >= 0 for valid intersection
        if -tolerance <= s <= 1.0 + tolerance && a >= -tolerance
            intersection_point = CausalSets.Coordinates{2}((t0 + a, x0 + slope*a))
            if CausalSets.in_past_of(manifold, intersection_point, y)
                if (t0 + a) < best_time
                    best_time = t0 + a
                    best_intersection = intersection_point
                    best_index = idx
                end
            end
        end
    end
    if best_intersection !== nothing
        return (best_intersection, best_index)
    else
        return nothing
    end
end

# Compute the two intersection points of the future lightcone of x and the past lightcone of y in 2D Minkowski spacetime
"""
    diamond_corners(x::CausalSets.Coordinates{2}, y::CausalSets.Coordinates{2}) -> Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}

Given two 2D coordinates `x` and `y`, returns the two intersection points of the future lightcone of `x` and the past lightcone of `y`.
These are the corners of the causal diamond defined by `x` and `y`.

# Arguments
- `x::CausalSets.Coordinates{2}`: The starting event (past).
- `y::CausalSets.Coordinates{2}`: The ending event (future).

# Returns
- `Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}`: The two intersection points as (left, right) corners.
"""
function diamond_corners(x::CausalSets.Coordinates{2}, y::CausalSets.Coordinates{2})
    tx, xx = x
    ty, xy = y
    # Right-going edge intersection
    tR = 0.5 * (tx - xx + ty + xy)
    xR = 0.5 * (-tx + xx + ty + xy)
    # Left-going edge intersection
    tL = 0.5 * (tx + xx + ty - xy)
    xL = 0.5 * (tx + xx - ty + xy)
    return (CausalSets.Coordinates{2}((tL, xL)), CausalSets.Coordinates{2}((tR, xR)))
end

function point_in_diamond(manifold::CausalSets.AbstractManifold, p::CausalSets.Coordinates{2}, x::CausalSets.Coordinates{2}, y::CausalSets.Coordinates{2})
    return in_past_of(manifold, x, p) && in_past_of(manifold, p, y)
end

"""
    cut_crosses_diamond(x::CausalSets.Coordinates{2}, y::CausalSets.Coordinates{2}, cut::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}) -> Bool

Check whether the cut segment crosses the causal diamond between `x` and `y`.
A cut crosses the diamond if:
- The endpoints lie strictly to the left and right of the diamond spatially, and
- The cut passes through the vertical line at `y[2]` at a time less than `y[1]`.

Returns true if the cut crosses the diamond, false otherwise.
"""
function cut_crosses_diamond(manifold::CausalSets.AbstractManifold, x::CausalSets.Coordinates{2}, y::CausalSets.Coordinates{2}, cut::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}})
    (b1, b2) = cut

    # Check if endpoints are on different sides of y[2]
    if (b1[2] - y[2]) * (b2[2] - y[2]) > 0
        return false
    end

    # Check whether cut is vertical
    Δx = b2[2] - b1[2]
    if abs(Δx) < 1e-12
        return false
    end

    # Check whether at least one endpoint is inside the diamond
    if point_in_diamond(manifold, b1, x, y) || point_in_diamond(manifold, b2, x, y)
        return false
    end

    # Interpolate time at which cut crosses spatial position of upper corner y[2]
    s = (y[2] - b1[2]) / Δx
    t_cross = b1[1] + s * (b2[1] - b1[1])

    # Cut crosses diamond if crossing time is between x[1] (lower corner) and y[2] (upper corner)
    return x[1] < t_cross && t_cross < x[2]
end

"""
    propagate_ray(manifold, x, slope, branch_point_tuples, tmax) -> Vector{Coordinates{2}}

Propagate a null ray starting at event `x` with slope `±1` through
branch cuts up to time `tmax`.

At each cut intersection, the ray continues from the *opposite endpoint*
of the cut (opposite in spatial direction to the approach).
If no intersection occurs before `tmax`, the ray propagates straight.

# Arguments
- `manifold`: The background manifold.
- `x::Coordinates{2}`: Starting event.
- `slope::Float64`: ±1, slope of the null ray.
- `branch_point_tuples`: Vector of cut segments.
- `tmax::Float64`: Final time to propagate to (usually `y[1]`).

# Returns
- `Vector{Coordinates{2}}`: The full piecewise path of the ray as a vector of coordinates.
"""
function propagate_ray(
    manifold::CausalSets.AbstractManifold,
    x::CausalSets.Coordinates{2},
    slope::Float64,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
    y::CausalSets.Coordinates{2},
)::Vector{CausalSets.Coordinates{2}}
    pos = x
    path = [x]
    tmax = y[1]
    while pos[1] < tmax
        hit = intersections_with_cuts(manifold, pos, slope, branch_point_tuples, y)
        if hit === nothing
            # No more intersections: propagate straight to tmax
            Δt = tmax - pos[1]
            newpos = CausalSets.Coordinates{2}((tmax, pos[2] + slope*Δt))
            push!(path, newpos)
            continue
        end

        intersection, idx = hit
        (p, q) = branch_point_tuples[idx]
        # Move to intersection point
        push!(path, intersection)

        # Choose the endpoint opposite in spatial direction,
        # but if that endpoint is beyond tmax, interpolate to tmax along the cut
        candidate = if slope > 0
            (p[2] > q[2]) ? q : p
        else
            (p[2] < q[2]) ? q : p
        end
        if candidate[1] <= tmax
            pos = candidate
            push!(path, pos)
        else
            # Interpolate along (p,q) to t = tmax
            t1, x1 = p
            t2, x2 = q
            s = (tmax - t1) / (t2 - t1)
            x_at_tmax = x1 + s * (x2 - x1)
            pos = CausalSets.Coordinates{2}((tmax, x_at_tmax))
            push!(path, pos)
        end
    end
    return path
end

"""
    propagate_wedge(manifold, x, branch_point_tuples, tmax) -> Bool

Propagate the causal diamond of event `x` forward in time to `tmax`, taking into
account branch cuts. Both the left- and right-going null rays are followed, with
deflections at cuts handled as in `propagate_ray`. If the left ray overtakes
the right ray (wedge collapse), the function returns `false`.

# Arguments
- `manifold`: The background manifold.
- `x::Coordinates{2}`: Starting event.
- `branch_point_tuples`: Vector of cut segments.
- `tmax::Float64`: Final time (usually `y[1]`).

# Returns
- `Bool`: `true` if the wedge remains open until `tmax`, else `false`.
"""
function in_wedge_of(
    manifold::CausalSets.AbstractManifold,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
)::Bool
    # Use propagate_ray for both left and right rays
    left_path = propagate_ray(manifold, x, -1.0, branch_point_tuples, y)
    right_path = propagate_ray(manifold, x, 1.0, branch_point_tuples, y)

    if left_path[end][2] > y[2] || right_path[end][2] < y[2]
        return false
    end

    # Helper: linear interpolation of x at time t along a path segment
    function interp_x(path, idx, t)
        t0, x0 = path[idx]
        t1, x1 = path[idx+1]
        dt = t1 - t0
        if abs(dt) < 1e-12
            return x0
        end
        α = (t - t0) / dt
        return x0 + α * (x1 - x0)
    end

    # Collect set of times for duplicate-time skipping
    left_times = Set{Float64}(pt[1] for pt in left_path)

    # Optimized single-pass pointer approach leveraging that left_path is time sorted
    # Check for each point in left_path (excluding endpoints) that left_x <= right_x (interpolated) at the same time
    j = 1
    for i in 2:(length(left_path)-1)
        t_left = left_path[i][1]
        # Advance j so that right_path[j+1][1] >= t_left (or j+1 > length)
        while right_path[j+1][1] < t_left - 1e-12
            j += 1
        end
        # Directly interpolate and compare at t_left
        x_left = left_path[i][2]
        x_right = interp_x(right_path, j, t_left)
        if x_left > x_right + 1e-12
            return false
        end
    end
    # Check for each point in right_path (excluding endpoints) that left_x (interpolated) <= right_x at the same time
    j = 1
    for i in 2:(length(right_path)-1)
        t_right = right_path[i][1]
        # Skip if this time is also present in left_path (duplicate time)
        if t_right in left_times
            continue
        end
        # Advance j so that left_path[j+1][1] >= t_right (or j+1 > length)
        while left_path[j+1][1] < t_right - 1e-12
            j += 1
        end
        # Directly interpolate and compare at t_right
        x_right = right_path[i][2]
        x_left = interp_x(left_path, j, t_right)
        if x_left > x_right + 1e-12
            return false
        end
    end
    return true
end

"""
    in_past_of(manifold::ConformallyTimesliceableManifold{N}, x::Coordinates{N}, y::Coordinates{N}, branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}) -> Bool

Determines whether the point `x` is in the causal past of `y` in a N-dimensional (currently only N=2 is implemented) conformally timesliceable manifold with possible branching singularities, including both vertical branch cuts and finite branch cut segments.

This function extends the standard causal relation by enforcing topological constraints from branch cuts:

- The function takes `branch_point_info` as a tuple `(single_branch_points, branch_point_tuples)`, where:
    - `single_branch_points` is a vector of single branch points (each induces a vertical cut extending upward in time).
    - `branch_point_tuples` is a vector of tuples, each representing a finite branch cut segment between two points.

- Throws `ArgumentError` if `x` or `y` lies directly on a branch cut (either a vertical cut or a finite cut segment, as determined by `point_segment_distance`).

- For efficiency, irrelevant cut segments are pruned by checking for both temporal and spatial overlap with the causal diamond between `x` and `y`.

- The function proceeds as follows:
    1. Checks whether `x` is causally related to `y` in the standard spacetime sense (`x ≺ y`).
    2. Considers all vertical cuts (from `single_branch_points`): if any cut lies strictly between the spatial positions of `x` and `y` and is not in the causal future of `x`, the path is obstructed.
    3. For finite branch cuts, tests whether any cut segment crosses the causal diamond between `x` and `y` using intersection and wedge propagation checks.
    4. If no obstruction is found, returns `true`; otherwise, returns `false`.

This models scenarios such as topology change (e.g., trousers geometry) or arbitrary branch cuts where causal curves cannot cross unconnected regions.

# Arguments
- `manifold::ConformallyTimesliceableManifold{N}`: The spacetime background.
- `x::Coordinates{N}`: Potential past event.
- `y::Coordinates{N}`: Potential future event.
- `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`: Tuple containing single branch points (vertical cuts) and branch cut segments (finite cuts).

# Returns
- `Bool`: `true` if `x` is causally in the past of `y` and no branch cut obstructs the path; `false` otherwise.

# Throws
- `ArgumentError` if `x` or `y` lies directly on a branch cut (vertical or finite segment).
"""
function CausalSets.in_past_of(
    manifold::CausalSets.ConformallyTimesliceableManifold{N},
    x::CausalSets.Coordinates{N},
    y::CausalSets.Coordinates{N},
    branch_point_info::Tuple{
        Vector{CausalSets.Coordinates{N}},
        Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}
    };
    tolerance::Float64=1e-12
)::Bool where N

    if N != 2
        throw(ArgumentError("Currently, nontrivial topologies are only implemented for dimensionality N=2, is $(N)."))
    end
    
    # Check standard causal relation
    CausalSets.in_past_of(manifold, x, y) || return false

    single_branch_points, branch_point_tuples = branch_point_info

    # Prune irrelevant cuts
    corners = diamond_corners(x,y)
    branch_point_tuples = [
    (p,q) for (p,q) in branch_point_tuples
    if y[1] > p[1] && q[1] > x[1] && min(p[2],q[2]) < corners[2][2] && max(p[2],q[2]) > corners[1][2]
    ]

    # Check for branch cuts that obstruct causality
    for b in single_branch_points
        if min(x[2], y[2]) < b[2] < max(x[2], y[2])
            CausalSets.in_past_of(manifold, x, b) || return false
        end
    end

    # Check whether causal diamond edges are inhibited by cuts
    past_intersections   = Vector{Union{Nothing, Tuple{CausalSets.Coordinates{2}, Int}}}(undef, 2)
    future_intersections = Vector{Union{Nothing, Tuple{CausalSets.Coordinates{2}, Int}}}(undef, 2)
    for i in 1:2 
        past_intersections[i] = intersections_with_cuts(manifold, x, (-1.)^i, branch_point_tuples, corners[i])
        future_intersections[i] = intersections_with_cuts(manifold, corners[i], (-1.)^(i+1), branch_point_tuples, y)
        if isnothing(past_intersections[i]) && isnothing(future_intersections[i])
            return true
        end
    end
    
    # Efficient obstruction test for single cuts intersecting whole diamond
    for cut in branch_point_tuples
        if cut_crosses_diamond(manifold, x, y, cut)
            return false
        end
    end

    return in_wedge_of(manifold, branch_point_tuples, x, y)
end

struct BranchedManifoldCauset{N, M} <: CausalSets.AbstractCauset where {N, M<:CausalSets.AbstractManifold}
    atom_count::Int64
    manifold::M
    sprinkling::Vector{CausalSets.Coordinates{N}}
    branch_point_info::Tuple{
        Vector{CausalSets.Coordinates{N}},
        Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}
    }
end

"""
    BranchedManifoldCauset(manifold, sprinkling, branch_point_info)

Construct a `BranchedManifoldCauset{N, M}` representing a causal set sprinkled into a background manifold with possible branch points or cuts.

# Arguments
- `manifold::M`: The background manifold, of type subtype of `AbstractManifold{N}`. Determines the spacetime geometry and causal structure.
- `sprinkling::Vector{Coordinates{N}}`: List of sprinkled spacetime points (atoms), typically sorted by coordinate time.
- `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`: Tuple containing information about branch points and branch cuts.
    - The first element, `single_branch_points`, is a vector of single branch points, each inducing a vertical cut to the boundary (extending in coordinate time).
    - The second element, `branch_point_tuples`, is a vector of finite branch cut segments, each as a tuple `(p, q)` of coordinates.

# Returns
- A `BranchedManifoldCauset{N, M}` instance with the following fields:
    - `atom_count::Int64`: Number of atoms (points) in the causet.
    - `manifold::M`: The background manifold.
    - `sprinkling::Vector{Coordinates{N}}`: The list of sprinkled points.
    - `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`: Tuple containing single branch points and branch cut segments.

# Throws
- `ArgumentError` if the dimensions of `manifold`, `sprinkling`, or `branch_point_info` do not match (`N`).
"""
function BranchedManifoldCauset(
    manifold::M,
    sprinkling::Vector{CausalSets.Coordinates{N}},
        branch_point_info::Tuple{
        Vector{CausalSets.Coordinates{N}},
        Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}
    }
)::BranchedManifoldCauset{N, M} where {N, M<:CausalSets.AbstractManifold{N}}
    return BranchedManifoldCauset{N, M}(length(sprinkling), manifold, sprinkling, branch_point_info)
end

"""
    in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int) -> Bool

Returns `true` if element `i` is in the past of element `j`, based on both spacetime and branch relations.
"""
function CausalSets.in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int)::Bool
    x = causet.sprinkling[i]
    y = causet.sprinkling[j]
    return CausalSets.in_past_of(causet.manifold, x, y, causet.branch_point_info)
end

"""
    CausalSets.BitArrayCauset(causet::BranchedManifoldCauset)

Constructs a `BitArrayCauset` by computing causal relations from a `BranchedManifoldCauset`.
"""
function CausalSets.BitArrayCauset(causet::BranchedManifoldCauset{N})::CausalSets.BitArrayCauset where {N}
    return convert(CausalSets.BitArrayCauset, causet)
end

"""
    convert(BitArrayCauset, causet::BranchedManifoldCauset)

Computes the causal matrix for a `BranchedManifoldCauset` and returns a `BitArrayCauset`. Note: this function uses all available threads.
"""
function Base.convert(::Type{CausalSets.BitArrayCauset}, causet::BranchedManifoldCauset{N}; tolerance::Float64=1e-12)::CausalSets.BitArrayCauset where {N}
    atom_count = causet.atom_count

    future_relations = Vector{BitVector}(undef, atom_count)
    past_relations = Vector{BitVector}(undef, atom_count)

    for i in 1:atom_count
        future_relations[i] = falses(atom_count)
        past_relations[i] = falses(atom_count)
    end

    Threads.@threads for i in 1:atom_count
        for j in i+1:atom_count
            if CausalSets.in_past_of(causet.manifold, causet.sprinkling[i], causet.sprinkling[j], causet.branch_point_info; tolerance=tolerance)
                future_relations[i][j] = true
                past_relations[j][i] = true
            end
        end
    end

    return CausalSets.BitArrayCauset(atom_count, future_relations, past_relations)
end

"""
    make_branched_manifold_cset(
        npoints::Int64,
        n_vertical_cuts::Int64,
        n_finite_cuts::Int64,
        rng::Random.AbstractRNG,
        order::Int64,
        r::Float64;
        d::Int64 = 2,
        type::Type{T} = Float32,
        tolerance::Float64=1e-12,
    ) -> Tuple{CausalSets.BitArrayCauset, Vector{Tuple{T,Vararg{T}}}, Tuple{Vector{Tuple{T,Vararg{T}}}, Vector{Tuple{Tuple{T,Vararg{T}}, Tuple{T,Vararg{T}}}}}, Matrix{T}}

Generate a causal set embedded into a branched 2D polynomial manifold with both vertical cuts (from single branch points) and finite cuts (between pairs of points). The spacetime is divided into causally separated sectors by these branch cuts.

# Arguments
- `npoints::Int64`: Number of sprinkled points in the spacetime. Must be > 0.
- `n_vertical_cuts::Int64`: Number of vertical cuts (single branch points from which a vertical cut extends up in time). Must be ≥ 0.
- `n_finite_cuts::Int64`: Number of finite cuts (segments between pairs of points). Must be ≥ 0.
- `rng::AbstractRNG`: Random number generator for reproducibility.
- `order::Int64`: Truncation order for the Chebyshev expansion. Must be > 0.
- `r::Float64`: Decay base for the Chebyshev coefficients. Must be > 1.
- `d::Int64`: Dimension of the spacetime. Only `d = 2` is currently supported.
- `tolerance::Float64`: Minimum allowed Euclidean distance to a cut for filtering sprinkled points (default: `1e-12`).
- `type::Type{T}`: Type for output coordinates and coefficient matrix (default: `Float32`).

# Description
- The function generates a random polynomial manifold (via Chebyshev expansion), sprinkles `npoints` into it, and inserts both vertical and finite branch cuts:
    - *Vertical cuts* originate from single branch points and extend upward in coordinate time, modeling e.g. trousers topology.
    - *Finite cuts* are segments between random pairs of points and act as branch cuts between regions.
- The function filters out any sprinkled points lying too close to any cut (within `tolerance`).

# Returns
- A 4-tuple `(cset, sprinkling, branch_point_info, chebyshev_coefs)` where:
    - `cset`: The `CausalSets.BitArrayCauset` representing the causal set with branch cuts.
    - `sprinkling`: The filtered list of coordinates (after removing points near cuts).
    - `branch_point_info`: A tuple `(vertical_branch_points, finite_cut_segments)` where
        - `vertical_branch_points` is a vector of single branch points (for vertical cuts),
        - `finite_cut_segments` is a vector of tuples, each representing a finite cut segment.
    - `chebyshev_coefs`: The coefficient matrix for the Chebyshev expansion of the background polynomial manifold.

# Throws
- `ArgumentError` if any of the following hold:
    - `npoints <= 0`
    - `n_vertical_cuts < 0`
    - `n_finite_cuts < 0`
    - `order <= 0`
    - `r <= 1`
    - `d ≠ 2`
"""
function make_branched_manifold_cset(
    npoints::Int64,
    n_vertical_cuts::Int64,
    n_finite_cuts::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    tolerance::Float64=1e-12,
    type::Type{T} = Float32
)::Tuple{CausalSets.BitArrayCauset, Vector{Tuple{T,Vararg{T}}}, Tuple{Vector{Tuple{T,Vararg{T}}}, Vector{Tuple{Tuple{T,Vararg{T}}, Tuple{T,Vararg{T}}}}},Matrix{T}} where {T<:Number}

    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end

    if n_vertical_cuts < 0
        throw(ArgumentError("n_vertical_cuts must be larger than 0, is $n_vertical_cuts"))
    end

    if n_finite_cuts < 0
        throw(ArgumentError("n_finite_cuts must be larger than 0, is $n_finite_cuts"))
    end

    if order <= 0
        throw(ArgumentError("order must be greater than 0, got $order"))
    end

    if r <= 1
        throw(
            ArgumentError(
                "r must be greater than 1 for exponential convergence of the Chebyshev series, got $r",
            ),
        )
    end

    if d != 2
        throw(ArgumentError("Currently, only 2D is supported, got $d"))
    end

    # Generate a matrix of random Chebyshev coefficients that decay exponentially with base r
    # it has to be a (order x order)-matrix because we describe a function of two variables
    chebyshev_coefs = zeros(Float64, order, order)
    for i = 1:order
        for j = 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * Random.randn(rng)
        end
    end

    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order - 1)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)

    # Define the square box boundary in 2D -- this works only in 2D and with square box boundary at the moment
    boundary = CausalSets.BoxBoundary{d}(((-1.0, -1.0), (1.0, 1.0)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints)

    # Randomly promote nbranchpoints of the sprinkling points to branch points
    branch_point_info = generate_random_branch_points(n_vertical_cuts, n_finite_cuts)

    # Remove points on sprinkling on cuts
    branched_sprinkling = filter_sprinkling_near_cuts(sprinkling, branch_point_info; tolerance = tolerance)

    # Construct the causal set from the manifold and sprinkling
    cset = BranchedManifoldCauset(polym, branched_sprinkling, branch_point_info)

    return CausalSets.BitArrayCauset(cset), branched_sprinkling, branch_point_info, type.(chebyshev_coefs)
end