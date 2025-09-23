"""
    are_colinear_overlapping(seg1, seg2; tolerance=1e-12)

Check whether two 2D segments are both colinear and overlapping, using a numerical tolerance.

# Arguments
- `seg1`, `seg2`: Each a tuple of two 2D points (e.g., `((t1, x1), (t2, x2))`), representing line segments in 2D.
- `tolerance`: (Keyword, default `1e-12`) Tolerance for numerical stability in colinearity and overlap checks.

# Returns
- `Bool`: `true` if the segments are colinear (within `tolerance`) and overlap (in the x-direction); `false` otherwise.
"""
function are_colinear_overlapping(seg1::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}, seg2::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}; tolerance::Float64=1e-12)
    u = (seg1[2][1]-seg1[1][1], seg1[2][2]-seg1[1][2])
    v = (seg2[2][1]-seg2[1][1], seg2[2][2]-seg2[1][2])
    w = (seg2[1][1]-seg1[1][1], seg2[1][2]-seg1[1][2])

    cross(a,b) = a[1]*b[2] - a[2]*b[1]

    return abs(cross(u,v)) < tolerance && abs(cross(u,w)) < tolerance && 
    max(min(seg1[1][1], seg1[2][1]), min(seg2[1][1], seg2[2][1])) <= min(max(seg1[1][1], seg1[2][1]), max(seg2[1][1], seg2[2][1])) + tolerance &&
    max(min(seg1[1][2], seg1[2][2]), min(seg2[1][2], seg2[2][2])) <= min(max(seg1[1][2], seg1[2][2]), max(seg2[1][2], seg2[2][2])) + tolerance
end

function is_colinear_overlapping_with_segments(seg::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}, single_branch_points::Vector{CausalSets.Coordinates{2}}, branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}}; tmax = 1., tolerance::Float64=1e-12)
    # Check colinearity with any existing tuple
    for tuple in branch_point_tuples
        if are_colinear_overlapping(seg, tuple)
            return true
        end
    end
    # Check colinearity with any vertical cut from single_points
    for bp in single_branch_points
        v_end = (tmax, bp[2])
        if are_colinear_overlapping(seg, (bp, v_end))
            return true
        end
    end
    return false
end

"""
    interpolate_point(x, y, interm, idx_in)

Linearly interpolate along the segment from `x` to `y` (each `Coordinates{d}`) to the position
where the `idx_in`-th coordinate equals `interm`. Returns the full interpolated point.

# Arguments
- `x, y::Coordinates{d}`: Endpoints of the segment.
- `interm::Float64`: Target value along the `idx_in`-th coordinate.
- `idx_in::Int`: Index of the coordinate (1-based).

# Returns
- `Coordinates{d}`: The interpolated point.

# Throws
- `ArgumentError` if the `idx_in`-th coordinates of `x` and `y` are (numerically) equal.
"""
function interpolate_point(x::CausalSets.Coordinates{N},
                           y::CausalSets.Coordinates{N},
                           interm::Float64,
                           idx_in::Int64;
                           idx_out::Union{Nothing,Int64}=nothing)::Union{CausalSets.Coordinates{N}, Float64} where {N}
    Δ = y[idx_in] - x[idx_in]
    if abs(Δ) < 1e-12
        throw(ArgumentError("Cannot interpolate: segment is parallel to axis $idx_in."))
    end
    α = (interm - x[idx_in]) / Δ
    if isnothing(idx_out)
        return CausalSets.Coordinates{N}(ntuple(i -> x[i] + α*(y[i]-x[i]), N))
    else
        return x[idx_out] + α*(y[idx_out]-x[idx_out])
    end
end

# Helper: check if two segments [a,b], [c,d] in 2D intersect
function segments_intersect(seg1::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}, seg2::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}; tolerance::Float64=1e-12)
    ts1 = [point[1] for point in seg1]
    xs1 = [point[2] for point in seg1]
    ts2 = [point[1] for point in seg2]
    xs2 = [point[2] for point in seg2]
    
    if min(ts1...) > max(ts2...) || min(ts2...) > max(ts1...) || min(xs1...) > max(xs2...) || min(xs2...) > max(xs1...) 
        return false, nothing
    end
    
    # Represent as: x1 + α*(y1-x1), x2 + β*(y2-x2), s,t in [0,1]
    # Solve: x1 + α*(y1-x1) = x2 + β*(y2-x2)
    # => 2x2 system for s,t
    dx1 = xs1[2]-xs1[1]
    dx2 = xs2[2]-xs2[1]
    dt1 = ts1[2]-ts1[1]
    dt2 = ts2[2]-ts2[1]
    det = dt1 * dx2 - dt2 * dx1

    if abs(det) < tolerance
        # Colinear: check 1D overlap in both coordinates
        if max(min(ts1...), min(ts2...)) <= min(max(ts1...), max(ts2...)) + tolerance && 
            max(min(xs1...), min(xs2...)) <= min(max(xs1...), max(xs2...)) + tolerance
            return true, nothing
        else
            return false, nothing
        end
    end

    α = (dt2 * (xs1[1] - xs2[1]) + dx2 * (ts2[1] - ts1[1])) / det
    β = (dt1 * (xs1[1] - xs2[1]) + dx1 * (ts2[1] - ts1[1])) / det

    if -tolerance <= α <= 1 + tolerance && -tolerance <= β <= 1 + tolerance
            t_int = ts1[1] + α * (ts1[2] - ts1[1])
            x_int = xs1[1] + α * (xs1[2] - xs1[1])
            return true, Coordinates{2}((t_int, x_int))
    else
        return false, nothing
    end
end

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

    # Generate tuples of random points in [-1,1]^N iteratively with colinearity check
    tuple_points = Tuple{CausalSets.Coordinates{d}, CausalSets.Coordinates{d}}[]
    while length(tuple_points) < nTuples
        a = ntuple(_ -> rand(rng, -1.0:0.0001:1.0), d)
        b = ntuple(_ -> rand(rng, -1.0:0.0001:1.0), d)
        seg = a[1] <= b[1] ? (a, b) : (b, a)

        # Check colinearity and overlap
        if !is_colinear_overlapping_with_segments(seg, single_points, tuple_points)
            push!(tuple_points, seg)
        end
    end
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
    next_intersection(manifold, branch_point_tuples, x, y, slope) -> Union{Tuple{Coordinates{2}, Int}, Nothing}

Compute the earliest intersection between a null ray (with slope ±1) starting from `ray_origin` and any branch cut segment in `branch_point_tuples`, restricted to occur before the event `y`.

# Arguments
- `manifold`: The background manifold (used for causal ordering).
- `x::Coordinates{2}`: The starting point of the null ray.
- `y::Coordinates{2}`: The event restricting the search; only intersections in the causal past of `y` are considered.
- `slope::Float64`: Slope of the null ray; should be ±1.
- `branch_point_tuples`: Vector of cut segments, each as a tuple `(p, q)` of `Coordinates{2}`.

# Returns
- Either `nothing` if no valid intersection is found, or a tuple `(intersection::Coordinates{2}, index::Int)` where `intersection` is the earliest intersection point and `index` is the 1-based index of the cut segment intersected.
"""
function next_intersection(
    manifold::CausalSets.AbstractManifold,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
    slope::Float64;
    null_separated::Bool=false,
    tolerance::Float64 = 1e-12
)::Union{Tuple{CausalSets.Coordinates{2},Int}, Nothing}
    best_intersection = nothing
    best_index = nothing
    best_time = nothing

    backward = y[1] < x[1]
    dir_sign = backward ? -1. : 1

    # Compute the endpoint of the ray segment as (y[1], ray_origin[2] + slope * (y[1] - ray_origin[1]))
    ray_end = CausalSets.Coordinates{2}((y[1], x[2] + slope * (y[1] - x[1])))
    ray_segment = (x, ray_end)

    for (idx, cut) in enumerate(branch_point_tuples)
        ok, pt = segments_intersect(ray_segment, cut; tolerance=tolerance)
        if ok && pt !== nothing && dir_sign * x[1] - tolerance <= dir_sign * pt[1] <= dir_sign * y[1] + tolerance &&
            (null_separated ? true : (backward ? in_past_of(manifold, y, pt) : in_past_of(manifold, pt, y)))
            if best_intersection === nothing ||
                (dir_sign * pt[1] < dir_sign * best_time) 
                best_intersection, best_index = pt, idx
                best_time = pt[1]
            end
        end
    end

    return isnothing(best_intersection) ? nothing : (best_intersection, best_index)
end

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
function cut_crosses_diamond(manifold::CausalSets.AbstractManifold, x::CausalSets.Coordinates{2}, y::CausalSets.Coordinates{2}, cut::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}; tolerance::Float64=1e-12)
    (b1, b2) = cut

    # Check if endpoints are on different sides of y[2] and x[2]
    if (b1[2] - y[2]) * (b2[2] - y[2]) > 0 && (b1[2] - x[2]) * (b2[2] - x[2]) > 0 && (b1[2] - x[2]) * (b1[2] - y[2]) > 0
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

    # Cut crosses diamond if crossing time is between x[1] (lower corner) and y[1] (upper corner)
    return segments_intersect(cut, (x, y); tolerance=tolerance)[1]
end


"""
    intersected_cut_crosses_diamond(
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2},
        cut1::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
        cut2::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
        intersection::CausalSets.Coordinates{2};
        tolerance::Float64=1e-12,
        corners::Union{Nothing,Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}}=nothing
    ) -> Bool

Given two intersecting cuts `cut1` and `cut2` and their intersection point `intersection`,
split them into four sub-segments at the intersection. Then check whether any of these
sub-segments alone obstructs the diamond between `x` and `y` (i.e. crosses both the left-
moving and right-moving null rays in the uncut geometry).

# Keyword Arguments
- `tolerance::Float64`: Numerical tolerance for intersection checks (default: `1e-12`).
- `corners::Union{Nothing,Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}`:
    Optionally provide the diamond corners `(left_corner, right_corner)` to use instead of recomputing them from `x` and `y`.
    If not provided, they are computed by `diamond_corners(x, y)`.

Returns `true` if such an obstruction is found, `false` otherwise.
"""
function intersected_cut_crosses_diamond(
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
    cut1::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
    cut2::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
    intersection::CausalSets.Coordinates{2};
    tolerance::Float64=1e-12,
    corners::Union{Nothing,Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}}=nothing
)::Bool
    # Split each cut into two pieces at the intersection
    c1a = (cut1[1], intersection)
    c1b = (intersection, cut1[2])
    c2a = (cut2[1], intersection)
    c2b = (intersection, cut2[2])

    # Diamond corners (left, right)
    left_corner, right_corner =
        isnothing(corners) ? diamond_corners(x, y) : corners
    
    # For simplicity, treat each edge as a segment: (x, left_corner), (left_corner, y), (x, right_corner), (right_corner, y)
    left_edges = [(x, left_corner), (left_corner, y)]
    right_edges = [(x, right_corner), (right_corner, y)]

    # All subsegments of cut1 and cut2
    c1subs = [c1a, c1b]
    c2subs = [c2a, c2b]

    # For each pairing: (c1X, left_edge), (c2Y, right_edge), and vice versa
    # Each subsegment must intersect one edge, and the other subsegment the other edge, simultaneously
    # Try all combinations for both left/right edge splits
    for (c1seg, c2seg) in Iterators.product(c1subs, c2subs)
        for (l_edge, r_edge) in Iterators.product(left_edges, right_edges)
            # (c1seg with left edge) and (c2seg with right edge)
            if segments_intersect(c1seg, l_edge; tolerance=tolerance)[1] &&
               segments_intersect(c2seg, r_edge; tolerance=tolerance)[1]
                return true
            end
            # (c2seg with left edge) and (c1seg with right edge)
            if segments_intersect(c2seg, l_edge; tolerance=tolerance)[1] &&
               segments_intersect(c1seg, r_edge; tolerance=tolerance)[1]
                return true
            end
        end
    end
    return false
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
    y::CausalSets.Coordinates{2},
    slope::Float64,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}};
    corners::Union{Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},Nothing}=nothing,
    tolerance::Float64=1e-12    
)::Vector{CausalSets.Coordinates{2}}
    dir_sign = y[1] < x[1] ? -1. : 1.
    local_branch_point_tuples = copy(branch_point_tuples)
    pos = x
    path = [x]
    tfin = y[1]
    tol = 1e-12
    while dir_sign * pos[1] < dir_sign * tfin
        hit = next_intersection(manifold, local_branch_point_tuples, pos, y, slope)
        if isnothing(hit)
            # No more intersections: propagate straight to tfin
            Δt = tfin - pos[1]
            pos = CausalSets.Coordinates{2}((tfin, pos[2] + slope*Δt))
            push!(path, pos)
            continue
        end
        intersection, idx = hit
        (p, q) = local_branch_point_tuples[idx]
        # Move to intersection point
        push!(path, intersection)

        # --- Classify cut type and choose endpoint accordingly
        conformal_proper_time = (p[1] - q[1])^2 - (p[2] - q[2])^2
        # Now select candidate endpoint
        candidate = nothing
        if conformal_proper_time <= tolerance # spacelike or null cut
            # Opposite in spatial direction to slope
            if dir_sign * slope > 0
                candidate = (p[2] > q[2]) ? q : p
            else
                candidate = (p[2] < q[2]) ? q : p
            end
        else # timelike cut
            # Always select future (past) endpoint, larger (smaller) time coordinate
            candidate = (dir_sign * p[1] > dir_sign * q[1]) ? p : q
        end

        # Check if candidate lies within causal diamond of (x,y)
        if point_in_diamond(manifold, candidate, x, y)
            pos = candidate
            push!(path, pos)
            local_branch_point_tuples = deleteat!(local_branch_point_tuples, idx)
            continue
        end

        # --- Simplified handling when candidate is outside the causal diamond
        # 1. Compute diamond corners
        left_corner, right_corner = isnothing(corners) ? diamond_corners(x, y) : corners
        # 2. Define four boundary edges
        diamond_edges = [
            (x, left_corner), (left_corner, y),
            (x, right_corner), (right_corner, y)
        ]
        # 3. Find intersection point of (intersection, candidate) with these edges
        found = false
        for (eidx, edge) in enumerate(diamond_edges)
            ok, pt = segments_intersect((intersection, candidate), edge)
            if ok && pt !== nothing && pt[1] > intersection[1]
                # 4. Push that point to path
                push!(path, pt)
                # 5. If intersection is on the same side as the ray slope, continue; else break
                # dir_sign * Slope < 0 → left edges (edges 1,2), dir_sign * Slope > 0 → right edges (edges 3,4)
                if (dir_sign * slope < 0 && eidx in (1,2)) || (dir_sign * slope > 0 && eidx in (3,4))
                    pos = pt
                else
                    break
                end
            end
        end
        local_branch_point_tuples = deleteat!(local_branch_point_tuples, idx)
    end
    return path
end

"""
    cut_intersections(branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}; tolerance::Float64=1e-12)

Given a vector of branch cut segments, returns a vector of index pairs (i, j) such that branch_point_tuples[i] intersects branch_point_tuples[j], with i < j.

# Arguments
- `branch_point_tuples`: Vector of finite branch cut segments, each as a tuple of two coordinates.
- `tolerance`: Tolerance for intersection detection (default 1e-12).

# Returns
- Vector of pairs (i, j) with i < j, where the i-th and j-th segments intersect.
"""
function cut_intersections(branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}; tolerance::Float64=1e-12)
    intersections = Vector{Tuple{Tuple{Int,Int}, Union{Nothing,CausalSets.Coordinates{2}}}}()
    n = length(branch_point_tuples)
    for i in 1:n-1
        for j in i+1:n
            intersect, point = segments_intersect(branch_point_tuples[i], branch_point_tuples[j]; tolerance=tolerance)
            if intersect
                if point === nothing
                    # colinear intersection: warn and skip
                    @warn "Colinear overlapping cuts ($i,$j) skipped in cut_intersections."
                    continue
                end
                push!(intersections, ((i, j), point))
            end
        end
    end
    return intersections
end


"""
    in_wedge_of(manifold, branch_point_tuples, x, y) -> Bool

Checks whether the event `y` lies inside the wedge (causal future) of `x` in the presence of branch cuts.
This is done by propagating both the left- and right-moving null rays from `x` to the coordinate time of `y`,
accounting for deflections at each cut (using `propagate_ray`). The wedge is considered open if, at all times up to `y[1]`,
the left-moving ray remains to the left of (or coincident with) the right-moving ray.

# Arguments
- `manifold`: The background manifold (typically a 2D spacetime).
- `branch_point_tuples::Vector{Tuple{Coordinates{2}, Coordinates{2}}}`: Vector of branch cut segments (each as a tuple of two coordinates).
- `x::Coordinates{2}`: The initial event (wedge apex).
- `y::Coordinates{2}`: The final event (target for the wedge, only the time coordinate is used).

# Returns
- `Bool`: `true` if the wedge remains open until `y`, i.e., the left and right null rays do not cross before reaching `y[1]`;
  `false` if the wedge collapses (left ray overtakes right ray).

# Notes
- Internally, both left- and right-moving null rays are propagated from `x` to the time of `y` using `propagate_ray`,
  which handles deflections at each cut by continuing from the appropriate endpoint of the cut.
- At each segment endpoint and at each time where one ray changes direction, the function interpolates the position of the other ray
  and checks for wedge collapse by comparing the x-coordinates of the left and right rays.
- If at any such time the left ray is to the right of the right ray (i.e., `x_left > x_right`), the wedge is considered closed and the function returns `false`.
  
  **Clarification:** Only the left-moving null rays are propagated and checked here, because if the left-moving ray can reach `y`, then the wedge is open; if it cannot, then no right-moving propagation can rescue it. Thus, propagating the right-moving null ray is not necessary for determining wedge openness, and only the left-moving rays are checked. This ensures correctness and efficiency.
"""
function in_wedge_of(
    manifold::CausalSets.AbstractManifold,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
)::Bool
    # The wedge is open if the left-moving null ray from x to y can reach y, and
    # the left-moving null ray from y backward to x can reach x, and their paths intersect.
    # This checks if a continuous null curve can connect x to y along the left-moving direction,
    # i.e., the wedge is open.
    left_forward = propagate_ray(manifold, x, y, -1.0, branch_point_tuples)
    if left_forward[end][2] > y[2]
        return false
    end

    left_backward = propagate_ray(manifold, y, x, 1.0, branch_point_tuples)
    if left_backward[end][2] > x[2]
        return false
    end

    # Check if the two piecewise paths intersect (i.e., share any segment intersection)
    for i in 1:length(left_forward)-1
        seg1 = (left_forward[i], left_forward[i+1])
        for j in 1:length(left_backward)-1
            seg2 = (left_backward[j], left_backward[j+1])
            if segments_intersect(seg1, seg2)[1]
                return true
            end
        end
    end
    return false
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
- `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`: Tuple containing single branch points (vertical cuts) and branch cut segments (finite cuts). Every pair of points is assumed to be ordered by coordinate time.

# Returns
- `Bool`: `true` if `x` is causally in the past of `y` and no branch cut obstructs the path; `false` otherwise.

# Throws
- `ArgumentError` if `x` or `y` lies directly on a branch cut (vertical or finite segment).
"""
function CausalSets.in_past_of(
    manifold::CausalSets.ConformallyTimesliceableManifold{N},
    branch_point_info::Tuple{
        Vector{CausalSets.Coordinates{N}},
        Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}
    },
    x::CausalSets.Coordinates{N},
    y::CausalSets.Coordinates{N};
    tolerance::Float64=1e-12
)::Bool where N

    if N != 2
        throw(ArgumentError("Currently, nontrivial topologies are only implemented for dimensionality N=2, is $(N)."))
    end
    
    # Check standard causal relation
    CausalSets.in_past_of(manifold, x, y) || return false

    single_branch_points, branch_point_tuples = branch_point_info

    # Prune irrelevant finite cuts
    corners = diamond_corners(x,y)
    branch_point_tuples = [
        (p,q) for (p,q) in branch_point_tuples
        if y[1] > p[1] && q[1] > x[1] && min(p[2],q[2]) < corners[2][2] && max(p[2],q[2]) > corners[1][2]
    ]
    # Prune irrelevant constant-time cuts
    single_branch_points = [
        b for b in single_branch_points
        if b[1] < y[1] && corners[1][2] < b[2] < corners[2][2]
    ]

    # Check for branch cuts that obstruct causality
    for b in single_branch_points
        if min(x[2], y[2]) < b[2] < max(x[2], y[2])
            CausalSets.in_past_of(manifold, x, b) || return false
        end
    end

    # Check whether causal diamond edges are inhibited by cuts
    past_intersections   = Vector{Union{Nothing, Tuple{CausalSets.Coordinates{2}, Int}}}(undef, 2) # save in case we can reuse this in in_wedge_of
    future_intersections = Vector{Union{Nothing, Tuple{CausalSets.Coordinates{2}, Int}}}(undef, 2) # save in case we can reuse this in in_wedge_of
    for i in 1:2 
        past_intersections[i] = next_intersection(manifold, branch_point_tuples, x, corners[i], (-1.)^i; null_separated = true)
        future_intersections[i] = next_intersection(manifold, branch_point_tuples, y, corners[i], (-1.)^(i+1); null_separated = true)
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

    all_cuts = vcat(branch_point_tuples,[(single_branch_points[i], CausalSets.Coordinates{2}((y[1] + tolerance, single_branch_points[i][2]))) for i in 1:length(single_branch_points)])

    # Reorder each tuple so that p[1] <= q[1]
    all_cuts = [(p[1] <= q[1] ? (p, q) : (q, p)) for (p, q) in all_cuts]
    # Sort by the time coordinate of the first element of each tuple
    all_cuts = sort(all_cuts, by = t -> t[1][1])

    # check intersections between all cuts
    intersections = cut_intersections(all_cuts)
    for ((i, j), intersection_point) in intersections
        # Compute intersection point between all_cuts[i] and all_cuts[j]
        seg1 = all_cuts[i]
        seg2 = all_cuts[j]
        # Check for obstruction
        if intersected_cut_crosses_diamond(x, y, seg1, seg2, intersection_point; corners=corners)
            return false
        end
    end

    return in_wedge_of(manifold, all_cuts, x, y)
end

struct BranchedManifoldCauset{N, M} <: CausalSets.AbstractCauset where {N, M<:CausalSets.AbstractManifold}
    atom_count::Int64
    manifold::M
    branch_point_info::Tuple{
        Vector{CausalSets.Coordinates{N}},
        Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}
    }
    sprinkling::Vector{CausalSets.Coordinates{N}}
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
    branch_point_info::Tuple{
    Vector{CausalSets.Coordinates{N}},
    Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}
    },
    sprinkling::Vector{CausalSets.Coordinates{N}}
)::BranchedManifoldCauset{N, M} where {N, M<:CausalSets.AbstractManifold{N}}
    return BranchedManifoldCauset{N, M}(length(sprinkling), manifold, branch_point_info, sprinkling)
end

"""
    in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int) -> Bool

Returns `true` if element `i` is in the past of element `j`, based on both spacetime and branch relations.
"""
function CausalSets.in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int)::Bool
    x = causet.sprinkling[i]
    y = causet.sprinkling[j]
    return CausalSets.in_past_of(causet.manifold, causet.branch_point_info, x, y)
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
            if CausalSets.in_past_of(causet.manifold, causet.branch_point_info, causet.sprinkling[i], causet.sprinkling[j]; tolerance=tolerance)
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
)::Tuple{CausalSets.BitArrayCauset, Vector{Tuple{T,Vararg{T}}}, Tuple{Vector{Tuple{T,Vararg{T}}}, Vector{Tuple{Tuple{T,Vararg{T}}, Tuple{T,Vararg{T}}}}},Matrix{T}} where {T<:Number} #)::Tuple{CausalSets.BitArrayCauset, Vector{CausalSets.Coordinates{2}}, Tuple{Vector{CausalSets.Coordinates{2}}, Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}},Matrix{T}} where {T<:Number}

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
    cset = BranchedManifoldCauset(polym, branch_point_info, branched_sprinkling)

    return CausalSets.BitArrayCauset(cset), branched_sprinkling, branch_point_info, type.(chebyshev_coefs)
end