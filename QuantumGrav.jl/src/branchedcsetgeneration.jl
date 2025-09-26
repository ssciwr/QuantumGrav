"""
    are_colinear_overlapping(
        seg1::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}},
        seg2::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}};
        tolerance::Float64=1e-12
    ) -> Bool

Determine whether two N-dimensional line segments are colinear (within a numerical tolerance) and overlap along all coordinates.

This function checks if the two segments are colinear—i.e., they lie along the same straight line within a specified `tolerance`—
and also have overlapping intervals in every coordinate axis.

# Arguments
- `seg1::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}`: The first segment, as a tuple of two N-dimensional coordinates.
- `seg2::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}`: The second segment, as a tuple of two N-dimensional coordinates.

# Keyword Arguments
- `tolerance::Float64=1e-12`: Numerical tolerance for colinearity and overlap checks. Used to handle floating point errors when comparing linear dependence and interval overlaps.

# Returns
- `Bool`: Returns `true` if the two segments are colinear (within `tolerance`) and their intervals overlap in all coordinates; `false` otherwise.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- Colinearity is determined by checking that the direction vectors of the two segments, and the vector connecting their starting points, are all linearly dependent (i.e., the rank of the matrix formed by these vectors is ≤ 1, within `tolerance`). This is implemented by checking that all 2×2 minors are less than `tolerance` in magnitude.
- Overlap is checked by verifying that the projections of the two segments onto every coordinate axis overlap, within a numerical tolerance.

# Example
```julia
seg1 = (CausalSets.Coordinates{3}((0.0, 0.0, 0.0)), CausalSets.Coordinates{3}((1.0, 1.0, 1.0)))
seg2 = (CausalSets.Coordinates{3}((0.5, 0.5, 0.5)), CausalSets.Coordinates{3}((1.5, 1.5, 1.5)))
are_colinear_overlapping(seg1, seg2)  # returns true
```
"""
function are_colinear_overlapping(
    seg1::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}},
    seg2::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}};
    tolerance::Float64=1e-12
)::Bool where N
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    # Overlap: intervals overlap along every coordinate axis
    for i in 1:N
        min1 = min(seg1[1][i], seg1[2][i])
        max1 = max(seg1[1][i], seg1[2][i])
        min2 = min(seg2[1][i], seg2[2][i])
        max2 = max(seg2[1][i], seg2[2][i])
        if max(min1, min2) > min(max1, max2) + tolerance
            return false
        end
    end

    # Direction vectors
    u = ntuple(i -> seg1[2][i] - seg1[1][i], N)
    v = ntuple(i -> seg2[2][i] - seg2[1][i], N)
    w = ntuple(i -> seg2[1][i] - seg1[1][i], N)

    # Colinearity helper: all 2x2 minors of two vectors are zero iff they are linearly dependent.
    function all_2x2_minors_zero(a, b, tol)
        for i in 1:length(a)-1
            for j in i+1:length(a)
                minor = a[i]*b[j] - a[j]*b[i]
                if abs(minor) > tol
                    return false
                end
            end
        end
        return true
    end

    # u and v are linearly dependent, and u and w are linearly dependent
    return all_2x2_minors_zero(u, v, tolerance) && all_2x2_minors_zero(u, w, tolerance)
end

"""
    is_colinear_overlapping_with_cuts(
        seg::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}},
        single_branch_points::Vector{CausalSets.Coordinates{N}},
        branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}}};
        tmax = 1.::FLoat64,
        tolerance::Float64=1e-12
    ) -> Bool

Check whether a proposed cut segment `seg` is colinear and overlapping with any existing cut segments, including both finite cuts (`branch_point_tuples`) and timelike boundary-connecting cuts (those extending upward from `single_branch_points`).

# Arguments
- `seg::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}`: The candidate segment to check (as a tuple of two N-dimensional coordinates).
- `single_branch_points::Vector{CausalSets.Coordinates{N}}`: Vector of single branch points, each representing the start of a vertical cut extending from the given point upward in the t-direction (up to `tmax`).
- `branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}}}`: Vector of existing finite cut segments, each as a tuple of two N-dimensional coordinates.

# Keyword Arguments
- `tmax::Float64=1.`: The maximum t-coordinate (used as the endpoint for vertical cuts from `single_branch_points`).
- `tolerance::Float64 = 1e-12`: Numerical tolerance for colinearity and overlap checks.

# Returns
- `Bool`: Returns `true` if `seg` is colinear (within `tolerance`) and overlapping with any existing cut segment (finite or vertical); returns `false` otherwise.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- Colinearity and overlap are checked using `are_colinear_overlapping`.
- Vertical cuts are represented as segments from each single branch point to `(tmax, bp[2])`.

# Example
```julia
using CausalSets
seg = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
single_branch_points = [CausalSets.Coordinates{2}((0.5, 1.0))]
branch_point_tuples = [(CausalSets.Coordinates{2}((0.2, 0.0)), CausalSets.Coordinates{2}((0.8, 0.0)))]
tmax = 1.0
is_colinear_overlapping_with_cuts(seg, single_branch_points, branch_point_tuples; tmax=tmax)
# returns true, since seg overlaps with the vertical cut at x=1.0
```
"""
function is_colinear_overlapping_with_cuts(
    seg::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}, 
    single_branch_points::Vector{CausalSets.Coordinates{N}}, 
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}}}; 
    tmax::Float64 = 1., 
    tolerance::Float64=1e-12
    )::Bool where N
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    # Check colinearity with any existing tuple
    for tuple in branch_point_tuples
        if are_colinear_overlapping(seg, tuple; tolerance = tolerance)
            return true
        end
    end
    # Check colinearity with any vertical cut from single_points
    for bp in single_branch_points
        v_end = (tmax, bp[2])
        if are_colinear_overlapping(seg, (bp, v_end); tolerance = tolerance)
            return true
        end
    end
    return false
end

"""
    interpolate_point(x::CausalSets.Coordinates{N},
                      y::CausalSets.Coordinates{N},
                      interm::Float64,
                      idx_in::Int64;
                      idx_out::Union{Nothing,Int64}=nothing,
                      tolerance::Float64 = 1e-12) -> Union{CausalSets.Coordinates{N}, Float64}

Linearly interpolate along the segment from `x` to `y` (each of type `Coordinates{N}`) to the position where the `idx_in`-th coordinate equals `interm`.

By default, returns the full interpolated coordinate as a `Coordinates{N}` object. If the optional keyword argument `idx_out` is provided, returns only the value of the `idx_out`-th coordinate at the interpolated point.

# Purpose
This function is useful for determining either the full spacetime point or a specific coordinate where a segment between two points reaches a given value along one of its coordinates. It is used in geometric constructions, such as finding intersection points with vertical or horizontal lines, or for propagating null rays in causal set geometry.

# Arguments
- `x::CausalSets.Coordinates{N}`: The starting point of the segment (N-dimensional).
- `y::CausalSets.Coordinates{N}`: The ending point of the segment (N-dimensional).
- `interm::Float64`: The target value for the `idx_in`-th coordinate at which to interpolate.
- `idx_in::Int64`: The coordinate index (1-based) to interpolate along. The function finds the parameter value where `x[idx_in] + α*(y[idx_in]-x[idx_in]) == interm`.

# Keyword Arguments
- `idx_out::Union{Nothing,Int64}=nothing`: If provided as an integer index, only the `idx_out`-th coordinate of the interpolated point is returned as a `Float64`. If `nothing`, the full interpolated point is returned as `Coordinates{N}`
- `tolerance::Float64 = 1e-12`: Numerical tolerance for colinearity and overlap checks.

# Returns
- `CausalSets.Coordinates{N}`: The interpolated point, if `idx_out` is `nothing`.
- `Float64`: The value of the `idx_out`-th coordinate at the interpolated point, if `idx_out` is provided.

# Throws
- `ArgumentError`: If the `idx_in`-th coordinates of `x` and `y` are (numerically) equal, meaning the segment is parallel to the coordinate axis and interpolation is not possible.
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- The interpolation is linear: the result is
  ```
  α = (interm - x[idx_in]) / (y[idx_in] - x[idx_in])
  interpolated[i] = x[i] + α * (y[i] - x[i])  for i = 1:N
  ```
- If `α` is outside `[0,1]`, the target value lies outside the segment.
- The function is robust to floating point errors, treating the segment as parallel if the difference is less than `1e-12`.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((1.0, 2.0))
# Find the point along the segment from x to y where the first coordinate (t) equals 0.5
p = interpolate_point(x, y, 0.5, 1)  # returns Coordinates{2}((0.5, 1.0))
# Find the x-coordinate at t = 0.5 along the segment
xval = interpolate_point(x, y, 0.5, 1; idx_out=2)  # returns 1.0
```
"""
function interpolate_point(x::CausalSets.Coordinates{N},
                           y::CausalSets.Coordinates{N},
                           interm::Float64,
                           idx_in::Int64;
                           idx_out::Union{Nothing,Int64}=nothing,
                           tolerance::Float64 = 1e-12)::Union{CausalSets.Coordinates{N}, Float64} where {N}
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    Δ = y[idx_in] - x[idx_in]
    if abs(Δ) < tolerance
        throw(ArgumentError("Cannot interpolate: segment is parallel to axis $idx_in."))
    end
    α = (interm - x[idx_in]) / Δ
    if isnothing(idx_out)
        return CausalSets.Coordinates{N}(ntuple(i -> x[i] + α*(y[i]-x[i]), N))
    else
        return x[idx_out] + α*(y[idx_out]-x[idx_out])
    end
end

"""
    segments_intersect(
        seg1::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}},
        seg2::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}};
        tolerance::Float64=1e-12
    ) -> Tuple{Bool, Union{Nothing, CausalSets.Coordinates{2}}}

Determine whether two 2D line segments intersect (including colinear overlaps) and, if so, return the intersection point.

# Purpose
Checks whether the two segments in 2D intersect at a point (including at endpoints) or are colinear and overlapping, within a numerical tolerance. Returns a tuple indicating intersection and the intersection point (if unique).

# Arguments
- `seg1::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}`: The first segment, as a tuple of two 2D coordinates.
- `seg2::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}`: The second segment, as a tuple of two 2D coordinates.

# Keyword Arguments
- `tolerance::Float64=1e-12`: Numerical tolerance for detecting colinearity and overlaps.

# Returns
- `Tuple{Bool, Union{Nothing, CausalSets.Coordinates{2}}}`:
    - The first element is `true` if the segments intersect (including colinear overlap), `false` otherwise.
    - The second element is:
        - a `Coordinates{2}` object representing the intersection point, if the intersection is unique,
        - `nothing` if there is no (unique) intersection point.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- Colinear, overlapping segments are considered as intersecting, but the intersection point is returned as `nothing`.
- The function solves for intersection by representing each segment parametrically and solving a 2×2 linear system.
- If the determinant is near zero (within `tolerance`), the segments are parallel or colinear; overlap is checked in both coordinates.

# Example
```julia
seg1 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
seg2 = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 0.0)))
segments_intersect(seg1, seg2)
# returns (true, Coordinates{2}((0.5, 0.5)))

seg3 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 0.0)))
seg4 = (CausalSets.Coordinates{2}((0.5, 0.0)), CausalSets.Coordinates{2}((1.5, 0.0)))
segments_intersect(seg3, seg4)
# returns (true, nothing)  # colinear overlap
```
"""
function segments_intersect(
    seg1::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}, 
    seg2::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}; 
    tolerance::Float64=1e-12
    )::Tuple{Bool, Union{Nothing, CausalSets.Coordinates{2}}}
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end


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
            return true, CausalSets.Coordinates{2}((t_int, x_int))
    else
        return false, nothing
    end
end

"""
    generate_random_branch_points(
        nPoints::Int,
        nTuples::Int;
        consecutive_intersections::Bool = false,
        rng::AbstractRNG = Random.default_rng(),
        tolerance::Float64=1e-12
    ) -> Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}

Generate random branch points for a branched manifold causal set. Produces two collections:
- `nPoints` single branch points (each induces a vertical cut extending upward in the time direction),
- `nTuples` finite cut segments (each is a pair of points joined by a straight line).

The branch points are chosen uniformly at random from the coordinate hypercube defined by coordinate_hypercube_edges.  
Finite cut segments are generated iteratively to avoid colinear overlaps and, if requested, to limit intersections.

# Arguments
- `nPoints::Int`: Number of single branch points to generate (must be ≥ 0).
- `nTuples::Int`: Number of finite topological cut segments to generate (must be ≥ 0).

# Keyword Arguments
- `consecutive_intersections::Bool = false`: If `false`, reject new segments that intersect more than one existing segment to within tolerance. Colinear overlaps are always rejected.
- `coordinate_hypercube_edges::Tuple{Tuple{CausalSets.Coordinates{d}},Tuple{CausalSets.Coordinates{d}}}=(CausalSets.Coordinates{d}(-1 .* ones(d)),CausalSets.Coordinates{d}(ones(d)))`: coordinates of the edges of the hypercube within which the points are sampled. The dimensionality is read off this argument.
- `rng::AbstractRNG=Random.default_rng()`: Random number generator.
- `tolerance::Float64=1e-12`: Numerical tolerance.

# Returns
- `Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`:
  - A vector of `nPoints` coordinates, sorted by time (first coordinate).
  - A vector of `nTuples` ordered pairs `(a,b)`, each representing a finite topological cut segment with `a[1] ≤ b[1]`.  
    The endpoints of each segment are time ordered. The list is sorted by time of the first, i. e., earlier point.

# Throws
- `ArgumentError`: 
    - If `nPoints < 0` or `nTuples < 0`.
    - If `tolerance <= 0`.

# Notes
- Colinear overlaps with existing cuts are always rejected using `is_colinear_overlapping_with_cuts`.
- If `consecutive_intersections=false`, only segments with ≤ 1 intersection are allowed; those with ≥ 2 intersections are rejected.
- Random coordinates are drawn from a given interval on each coordinate axis using the provided `rng`.

# Example
```julia
using Random, CausalSets
rng = MersenneTwister(42)
points, cuts = generate_random_branch_points(3, 2; rng=rng)
# `points` contains 3 random coordinates
# `cuts` contains 2 random cut segments
"""
function generate_random_branch_points(
    nPoints::Int,
    nTuples::Int;
    consecutive_intersections::Bool = false,
    coordinate_hypercube_edges::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}} = (CausalSets.Coordinates{2}((-1., -1.)), CausalSets.Coordinates{2}((1., 1.))),
    rng::Random.AbstractRNG = Random.default_rng(),
    tolerance::Float64=1e-12,
)::Tuple{Vector{CausalSets.Coordinates{N}}, Vector{Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}}} where N
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end


    if nPoints < 0
        throw(ArgumentError("nPoints must be at least 0, got $(nPoints)."))
    end

    if nTuples < 0
        throw(ArgumentError("nTuples must be at least 0, got $(nTuples)."))
    end

    if isnothing(coordinate_hypercube_edges)
        coordinate_hypercube_edges = (CausalSets.Coordinates{d}(-1 .* ones(d)), CausalSets.Coordinates{d}(ones(d)))
    end

    rand_point() = ntuple(i -> rand(rng) * (coordinate_hypercube_edges[2][i] - coordinate_hypercube_edges[1][i]) + coordinate_hypercube_edges[1][i], N)

    # Generate random points in [-1,1]^N
    single_points = [rand_point() for _ in 1:nPoints]

    # Generate tuples of random points in [-1,1]^N iteratively with colinearity and (optionally) intersection check
    tuple_points = [(bound_cuts, CausalSets.Coordinates{N}((coordinate_hypercube_edges[2][1], bound_cuts[2]))) for bound_cuts in single_points]
    
    if !consecutive_intersections
        num_intersections = zeros(nPoints)
    end
    
    while length(tuple_points) < nPoints + nTuples
        a = rand_point()
        b = rand_point()
        seg = a[1] <= b[1] ? (a, b) : (b, a)

        # Check colinearity and overlap
        if is_colinear_overlapping_with_cuts(seg, single_points, tuple_points; tolerance = tolerance, tmax = coordinate_hypercube_edges[2][1])
            continue
        end

        # If no_intersections is requested, check for intersections (excluding colinear overlaps already handled above)
        if !consecutive_intersections
            num_intersections_new = copy(num_intersections)
            push!(num_intersections_new, 0)  # reserve slot for the new candidate
            for (i, existing_segs) in enumerate(tuple_points)
                intersects, _ = segments_intersect(seg, existing_segs; tolerance = tolerance)
                # Count intersections
                if intersects
                    # True intersection (not colinear/overlap): increase intersection count for both involved cuts.
                    num_intersections_new[i] += 1
                    num_intersections_new[end] += 1
                end
                if any(x -> x > 1, num_intersections_new) # No need to continue counting once a cut has more than 1 intersection.
                    break
                end
            end
            if any(x -> x > 1, num_intersections_new) # Reject for more than one intersection.
                continue
            end
            num_intersections = copy(num_intersections_new)
        end

        push!(tuple_points, seg)
    end
    tuple_points = tuple_points[(nPoints+1):end]

    # Sort single_points by coordinate time
    single_points_sorted = sort(single_points, by = x -> x[1])
    # Sort tuple_points by the time of the first point in each tuple
    tuple_points_sorted = sort(tuple_points, by = t -> t[1][1])
    return single_points_sorted, tuple_points_sorted
end

"""
    point_segment_distance(
        p::CausalSets.Coordinates{N},
        seg::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}};
        tolerance::Float64=1e-12
    ) -> Float64

Compute the minimal Euclidean distance between a point `p` and a line segment from `a` to `b` in `N`-dimensional coordinates.

# Purpose
This function calculates the shortest (minimal) Euclidean distance from a given point to a finite line segment in `N`-dimensional Euclidean space.
It is primarily used for filtering sprinkled points that are too close to topological cuts, as well as for argument validation (e.g., to ensure points do not lie directly on a cut).

# Arguments
- `p::CausalSets.Coordinates{N}`: The point from which the distance is measured, as an N-dimensional coordinate.
- `seg::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}}`: The segment characterized by the coordinates of its endpoints.

# Keyword Arguments
- `tolerance::Float64=1e-12`: Numerical tolerance for detecting degenerate (zero-length) segments.

# Returns
- `Float64`: The minimal Euclidean distance from point `p` to the segment `[a, b]`. If the segment is degenerate (i.e., `a ≈ b` within `tolerance`), returns the distance from `p` to `a`.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- The function treats all coordinates as points in N-dimensional Euclidean space, regardless of their interpretation as spacetime coordinates.
- If the segment `[a, b]` is degenerate (the endpoints are equal within `tolerance`), the distance from `p` to `a` is returned.
- The distance is computed by projecting `p` onto the line defined by `a` and `b`, clamping the projection to the segment, and then returning the Euclidean distance between `p` and this closest point.
- Used for cut filtering: e.g., to remove points sprinkled "too close" to a topological cut.

# Example
```julia
using CausalSets
p = CausalSets.Coordinates{2}((0.5, 0.5))
a = CausalSets.Coordinates{2}((0.0, 0.0))
b = CausalSets.Coordinates{2}((1.0, 0.0))
dist = point_segment_distance(p, (a, b))
# returns 0.5, since the closest point on the segment is (0.5, 0.0)
```

# Generalization
- This function works for arbitrary dimension `N`.
"""
function point_segment_distance(
    p::CausalSets.Coordinates{N},
    seg::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}};
    tolerance::Float64=1e-12,
)::Float64 where N
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    a, b = seg
    # Compute vectorized differences
    ab = ntuple(i -> b[i] - a[i], N)
    ap = ntuple(i -> p[i] - a[i], N)
    # Compute squared norm of ab
    denom = sum(ab[i]^2 for i in 1:N)
    if denom < tolerance
        # Degenerate segment: return distance from p to a
        return sqrt(sum((p[i] - a[i])^2 for i in 1:N))
    end
    # Compute projection scalar s = ((p-a) ⋅ (b-a)) / |b-a|^2, clamp to [0,1]
    dot_ap_ab = sum(ap[i]*ab[i] for i in 1:N)
    s = clamp(dot_ap_ab / denom, 0.0, 1.0)
    # Compute projected point
    proj = ntuple(i -> a[i] + s*ab[i], N)
    # Distance from p to projection
    return sqrt(sum((p[i] - proj[i])^2 for i in 1:N))
end

"""
    filter_sprinkling_near_cuts(
        sprinkling::Vector{CausalSets.Coordinates{2}},
        branch_point_info::Tuple{Vector{CausalSets.Coordinates{2}}, Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}};
        tolerance::Float64 = 1e-12
    ) -> Vector{CausalSets.Coordinates{2}}

Filter out sprinkled points that lie too close to any topological cut (finite or vertical).

# Purpose
Removes points from the input `sprinkling` that are within a given minimal Euclidean distance (`tolerance`) of any topological cut in the geometry. 
Topological cuts are either finite (segments between two points) or vertical (extending upward from a single branch point to the maximal time in the sprinkling). This is essential for ensuring that no sprinkled point lies directly on or dangerously close to a topological cut, which could otherwise cause ambiguity or artifacts in the causal set structure.

# Arguments
- `sprinkling::Vector{Coordinates{2}}`: The list of sprinkled points (atoms) in 2D spacetime, typically sorted by coordinate time (first coordinate).
- `branch_point_info::Tuple{Vector{Coordinates{2}}, Vector{Tuple{Coordinates{2}, Coordinates{2}}}}`: Tuple describing the topological cuts:
    - The first element is a vector of single branch points, each representing the start of a vertical cut (extending upward in coordinate time).
    - The second element is a vector of finite topological cut segments, each as a tuple `(a, b)`.

# Keyword Arguments
- `tolerance::Float64=1e-12`: Minimal allowed Euclidean distance from any sprinkled point to any cut. Points closer than this will be removed.

# Returns
- `Vector{Coordinates{2}}`: The filtered list of sprinkled points, with all points near any cut (within `tolerance`) removed.

# Throws
- `ArgumentError`: 
    - If `tolerance` is not positive.
    - If `sprinkling` is not sorted by coordinate time (first coordinate).

# Notes
- **Finite cuts** are represented as tuples `(a, b)` and checked directly.
- **Vertical cuts** are represented as segments from each single branch point `b` to `(tmax, b[2])`, where `tmax` is the largest time coordinate in the sprinkling.
- The distance check uses Euclidean distance in the coordinate space.
- The function is robust to floating-point errors and will not remove points farther than `tolerance` from any cut.
- Used as a preprocessing step after generating topological cuts and before constructing the causal set.

# Example
```julia
using CausalSets
sprinkling = [
    CausalSets.Coordinates{2}((0.1, 0.0)),
    CausalSets.Coordinates{2}((0.5, 0.5)),
    CausalSets.Coordinates{2}((0.9, 1.0))
]
single_branch_points = [CausalSets.Coordinates{2}((0.3, 0.5))]
branch_point_tuples = [(CausalSets.Coordinates{2}((0.6, 0.0)), CausalSets.Coordinates{2}((0.6, 1.0)))]
filtered = filter_sprinkling_near_cuts(
    sprinkling, 
    (single_branch_points, branch_point_tuples); 
    tolerance=0.11
)
# The point at (0.5, 0.5) will be removed if it is within 0.11 of the vertical cut at x=0.5 or the finite cut at x=0.6.
```

# Role in Filtering
This function ensures the integrity of the causal set by removing points that would otherwise be ambiguous due to their proximity to topological cuts. It is especially important in branched or topology-changing spacetimes, where the presence of cuts can affect causal relations.
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
        for cut in branch_point_tuples
            if point_segment_distance(p, cut; tolerance=tolerance) < tolerance
                too_close = true
                break
            end
        end
        # check vertical cuts
        if !too_close
            for b in single_branch_points
                v_end = CausalSets.Coordinates{2}((tmax,b[2]))
                if point_segment_distance(p, (b, v_end); tolerance=tolerance) < tolerance
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
    next_intersection(
        manifold::CausalSets.AbstractManifold,
        branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2},
        slope::Float64;
        null_separated::Bool = false,
        tolerance::Float64 = 1e-12
    ) -> Union{Tuple{CausalSets.Coordinates{2}, Int}, Nothing}

Find the earliest intersection (in coordinate time) between a null ray starting at `x` with slope `slope` (±1) and any topological cut segment in `branch_point_tuples`, restricting to intersections that occur before the event `y`.

# Purpose
This function is used to propagate a null ray (with slope ±1, i.e., left- or right-moving if future-directed) forward or backward in time (detected automatically) through a background with possible topological cuts (such as in a trousers topology or arbitrary cuts), and to determine the first cut segment that the ray would intersect before reaching the event `y`. This is a key geometric operation in constructing causal relations in spacetimes with nontrivial topology.

# Arguments
- `manifold::CausalSets.AbstractManifold`: The background manifold, used for causal ordering (e.g., to check if an intersection is in the causal past/future of `y`).
- `branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}}`: Vector of topological cut segments, each as a tuple `(p, q)` of 2D coordinates. Each segment represents a finite cut in spacetime that can obstruct causal curves.
- `x::CausalSets.Coordinates{2}`: The starting point (origin) of the null ray.
- `y::CausalSets.Coordinates{2}`: The endpoint restricting the search; only intersections with coordinate time between `x[1]` and `y[1]` (according to the direction of propagation) are considered.
- `slope::Float64`: The slope of the null ray; should be either +1 (right-moving) or -1 (left-moving).

# Keyword Arguments
- `null_separated::Bool = false`: If `true`, disables the check that the intersection is strictly in the (causal) past of `y` (useful for null-separated events for which there can be numerical artifacts).
- `tolerance::Float64 = 1e-12`: Numerical tolerance for intersection and ordering checks.

# Returns
- `Union{Tuple{CausalSets.Coordinates{2}, Int}, Nothing}`:
    - If a valid intersection is found, returns a tuple `(intersection, index)`, where `intersection` is the coordinate of the intersection point (as `Coordinates{2}`), and `index` is the 1-based index of the cut segment in `branch_point_tuples` that is intersected.
    - If no valid intersection is found before `y`, returns `nothing`.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- The function considers only intersections that occur "between" `x` and `y` in coordinate time, according to the direction of propagation (i.e., forward or backward).
- If `null_separated` is `false` (default), the function requires that the intersection is strictly in the causal past/future of `y` as determined by the manifold's causal structure. This is important for avoiding spurious intersections at endpoints or for null-separated events.
- Vertical cuts (cuts parallel to the time axis) can be included in `branch_point_tuples` as segments with constant spatial coordinate.
- The function ignores colinear or overlapping intersections (i.e., where the ray lies along the cut).
- Only the earliest valid intersection (in coordinate time) is returned, even if multiple cuts are intersected.

# Example
```julia
using CausalSets
# Define a right-moving null ray from (0.0, 0.0) to (1.0, 1.0)
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((1.0, 1.0))
slope = 1.0
# Define a topological cut segment crossing the ray
cut = (CausalSets.Coordinates{2}((0.5, -0.5)), CausalSets.Coordinates{2}((0.5, 0.5)))
manifold = CausalSets.MinkowskiManifold{2}()
result = next_intersection(manifold, [cut], x, y, slope)
# returns (Coordinates{2}((0.5, 0.5)), 1)
```

# See Also
- [`segments_intersect`](@ref): for computing intersections between segments.
- [`propagate_ray`](@ref): for propagating null rays through cuts.

# Related Concepts
- "Topological cut" refers to any finite segment representing a region of spacetime that cannot be traversed by causal curves (e.g., in topology change scenarios).
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
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

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
            (null_separated ? true : (backward ? CausalSets.in_past_of(manifold, y, pt) : CausalSets.in_past_of(manifold, pt, y)))
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
    diamond_corners(
    x::CausalSets.Coordinates{2}, 
    y::CausalSets.Coordinates{2}; 
    check_causal_relation::Bool=true)
        -> Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}

Compute the left and right corners of the causal diamond defined by two events `x` (past) and `y` (future).

# Purpose
Given two causally related events `x` and `y` with `x[1] < y[1]`, this function determines the intersection points
of the (unobstructed) future lightcone of `x` with the (unobstructed) past lightcone of `y`. These two intersection points define the spatial
extent of the causal diamond between `x` and `y`.

# Arguments
- `x::Coordinates{2}`: The past event (lower corner of the diamond).
- `y::Coordinates{2}`: The future event (upper corner of the diamond).

# Keyword Arguments
- `check_causal_relation::Bool = true`: If false, turns off check whether x is in past of y.

# Returns
- `Tuple{Coordinates{2}, Coordinates{2}}`: A pair `(left_corner, right_corner)` of 2D coordinates, representing
  the left- and right-moving intersections of the lightcones of `x` and `y`.

# Throws
- `ArgumentError`: If `y` is not in the causal future of `x`, not taking into account topological obstructions.

# Notes
- Assumes a 2D Minkowski-type geometry with null rays moving along slopes ±1.
- The left corner is reached by the left-moving null ray from `x` intersecting the right-moving null ray into `y`.
- The right corner is reached by the right-moving null ray from `x` intersecting the left-moving null ray into `y`.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
left, right = diamond_corners(x, y)
# left  = Coordinates{2}((1.0, -1.0))
# right = Coordinates{2}((1.0,  1.0))
"""
function diamond_corners(
    manifold::CausalSets.AbstractManifold, 
    x::CausalSets.Coordinates{2}, 
    y::CausalSets.Coordinates{2}; 
    check_causal_relation::Bool=true)::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}

    if check_causal_relation && !CausalSets.in_past_of(manifold, x, y)
        throw(ArgumentError("y must be in the causal past of x, but is not."))
    end

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

"""
    point_in_diamond(
        manifold::CausalSets.AbstractManifold,
        p::CausalSets.Coordinates{2},
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2};
        check_causal_relation::Bool = true,
    ) -> Bool

Check whether a point `p` lies inside the causal diamond defined by two events `x` (past) and `y` (future) neglecting topological obstructions.

# Purpose
Tests if `p` is causally between `x` and `y`, i.e. inside the causal diamond spanned by their lightcones.  
This requires that `x` is in the causal past of `y`, and that `p` lies causally between them.

# Arguments
- `manifold::AbstractManifold`: The background manifold defining causal relations.
- `p::Coordinates{2}`: The point to test.
- `x::Coordinates{2}`: The past event (lower tip of the diamond).
- `y::Coordinates{2}`: The future event (upper tip of the diamond).

# Keyword Arguments
- `check_causal_relation::Bool = true`: If false, turns off check whether x is in past of y.

# Returns
- `Bool`: `true` if `p` lies inside the causal diamond `[x, y]`; `false` otherwise.

# Throws
- `ArgumentError`: If `y` is not in the causal future of `x`, not taking into account topological obstructions.

# Notes
- A point `p` is in the diamond iff both `x ≺ p` and `p ≺ y`.  
- Works for arbitrary manifolds that implement `in_past_of`.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
p1 = CausalSets.Coordinates{2}((1.0, 0.5))
p2 = CausalSets.Coordinates{2}((3.0, 0.0))

point_in_diamond(CausalSets.MinkowskiManifold{2}(), p1, x, y)  # true
point_in_diamond(CausalSets.MinkowskiManifold{2}(), p2, x, y)  # false
"""
function point_in_diamond(
    manifold::CausalSets.AbstractManifold, 
    p::CausalSets.Coordinates{2}, 
    x::CausalSets.Coordinates{2}, 
    y::CausalSets.Coordinates{2}; 
    check_causal_relation::Bool = true,)::Bool

    if check_causal_relation && !CausalSets.in_past_of(manifold, x, y)
        throw(ArgumentError("y must be in the causal future of x but is not."))
    end
    
    return CausalSets.in_past_of(manifold, x, p) && CausalSets.in_past_of(manifold, p, y)
end

"""
    cut_crosses_diamond(
        manifold::CausalSets.AbstractManifold,
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2},
        cut::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}};
        check_causal_relation::Bool = true,
        tolerance::Float64 = 1e-12
    ) -> Bool

Check whether a finite topological cut segment obstructs the causal diamond between two events `x` (past) and `y` (future).

# Purpose
Determines if the cut segment lies across the causal diamond defined by `x` and `y`, thereby obstructing null rays that would otherwise connect them.  
This models the effect of finite topological cuts (e.g., in trousers-type topology change) on causal relations.

# Arguments
- `manifold::AbstractManifold`: Background manifold used for causal ordering.
- `x::Coordinates{2}`: The past event (lower tip of the diamond).
- `y::Coordinates{2}`: The future event (upper tip of the diamond).
- `cut::Tuple{Coordinates{2}, Coordinates{2}}`: The finite topological cut segment to test.

# Keyword Arguments
- `check_causal_relation::Bool = true`: If false, turns off check whether x is in past of y.
- `tolerance::Float64` = 1e-12: Numerical tolerance for detecting intersections.

# Returns
- `Bool`: 
    - `true` if the cut crosses and obstructs the causal diamond between `x` and `y`.
    - `false` otherwise.

# Throws
- `ArgumentError`: 
    - If `y` is not in the causal future of `x`, not taking into account topological obstructions.
    - If `tolerance <= 0`.

# Notes
- The function rejects cuts that extend purely in coordinate time or are entirely contained within the diamond.
- A cut is considered obstructing if it intersects both null boundaries of the diamond or passes across its interior before `y`.
- Endpoints lying strictly inside the diamond do **not** count as crossing; only transverse cuts are considered.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
cut1 = (CausalSets.Coordinates{2}((1., -1.5)), CausalSets.Coordinates{2}((1.0, 1.5)))
cut2 = (CausalSets.Coordinates{2}((0.5, -0.5)), CausalSets.Coordinates{2}((0.5, 0.3)))
manifold = CausalSets.MinkowskiManifold{2}()

cut_crosses_diamond(manifold, x, y, cut1)  # true, cut passes through diamond
cut_crosses_diamond(manifold, x, y, cut2)  # false, cut endpoint lies inside diamond
"""
function cut_crosses_diamond(
    manifold::CausalSets.AbstractManifold, 
    x::CausalSets.Coordinates{2}, 
    y::CausalSets.Coordinates{2}, 
    cut::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}; 
    check_causal_relation::Bool = true,
    tolerance::Float64=1e-12)
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end


    if check_causal_relation && !CausalSets.in_past_of(manifold, x, y)
        throw(ArgumentError("y must be in the causal future of x but is not."))
    end

    (b1, b2) = cut

    # Check if endpoints are on different sides of both y[2] and x[2], return false if true.
    if (b1[2] - y[2]) * (b2[2] - y[2]) > 0 && (b1[2] - x[2]) * (b2[2] - x[2]) > 0 && (b1[2] - x[2]) * (b1[2] - y[2]) > 0
        return false
    end

    # Check whether cut is vertical
    Δx = b2[2] - b1[2]
    if abs(Δx) < 1e-12
        return false
    end

    # Check whether at least one endpoint is inside the diamond
    if point_in_diamond(manifold, b1, x, y; check_causal_relation = check_causal_relation) || point_in_diamond(manifold, b2, x, y; check_causal_relation = check_causal_relation)
        return false
    end

    # Cut crosses diamond if crossing time is between x[1] (lower corner) and y[1] (upper corner)
    return segments_intersect(cut, (x, y); tolerance=tolerance)[1]
end


"""
    intersected_cut_crosses_diamond(
        manifold::AbstractManifold,
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2},
        cut1::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}},
        cut2::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}},
        intersection::CausalSets.Coordinates{2};
        corners::Union{Nothing, Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}} = nothing,
        check_causal_relation::Bool = true,
        tolerance::Float64 = 1e-12,
    ) -> Bool

Check whether two intersecting topological cuts jointly obstruct the causal diamond between `x` (past) and `y` (future).

# Purpose
When two finite cut segments intersect inside spacetime, the intersection splits each cut into two subsegments.  
This function checks whether some combination of these subsegments blocks both the left- and right-moving null rays 
that define the causal diamond `[x, y]`. If so, no causal curve can cross the diamond, and the cuts together 
form a full obstruction.

# Arguments
- `manifold::AbstractManifold`: Background manifold used for causal ordering.
- `x::Coordinates{2}`: The past event (lower tip of the diamond).
- `y::Coordinates{2}`: The future event (upper tip of the diamond).
- `cut1::Tuple{Coordinates{2}, Coordinates{2}}`: First finite topological cut segment.
- `cut2::Tuple{Coordinates{2}, Coordinates{2}}`: Second finite topological cut segment.
- `intersection::Coordinates{2}`: The intersection point of `cut1` and `cut2`.

# Keyword Arguments
- `corners::Union{Nothing, Tuple{Coordinates{2}, Coordinates{2}}} = nothing`: Optional precomputed diamond corners `(left, right)`.  
  If `nothing`, they are recomputed from `(x, y)` using `diamond_corners`.
- `check_causal_relation::Bool = true`: If false, turns off check whether x is in past of y.
- `tolerance::Float64 = 1e-12`: Numerical tolerance for detecting intersections.

# Returns
- `Bool`: 
    - `true` if some combination of the subsegments of `cut1` and `cut2` obstructs the causal diamond between `x` and `y`.
    - `false` otherwise.

# Throws
- `ArgumentError`: 
    - If `y` is not in the causal future of `x`, not taking into account topological obstructions.
    - If `tolerance <= 0`.

# Notes
- Each cut is split at `intersection`, producing two subsegments per cut.
- The function tests all pairings of subsegments of `cut1` and `cut2` against the four diamond boundary edges.
- An obstruction occurs if one subsegment intersects the left boundary and another intersects the right boundary simultaneously.
- Colinear overlaps are not treated specially here; they are assumed to have been filtered earlier.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
cut1 = (CausalSets.Coordinates{2}((1.0, -1.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
cut2 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((2.0, 0.0)))
intersection = CausalSets.Coordinates{2}((1.0, 0.0))
intersected_cut_crosses_diamond(x, y, cut1, cut2, intersection)  # returns true
"""
function intersected_cut_crosses_diamond(
    manifold::CausalSets.AbstractManifold,
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
    cut1::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
    cut2::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
    intersection::CausalSets.Coordinates{2};
    corners::Union{Nothing,Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}}=nothing,
    check_causal_relation::Bool = true,
    tolerance::Float64=1e-12,
)::Bool
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end
    
    if check_causal_relation && !CausalSets.in_past_of(manifold, x, y)
        throw(ArgumentError("y must be in the causal future of x but is not."))
    end

    # Split each cut into two pieces at the intersection
    c1a = (cut1[1], intersection)
    c1b = (intersection, cut1[2])
    c2a = (cut2[1], intersection)
    c2b = (intersection, cut2[2])

    # Diamond corners (left, right)
    left_corner, right_corner =
        isnothing(corners) ? diamond_corners(manifold, x, y; check_causal_relation = false) : corners
    
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
    propagate_ray(
        manifold::CausalSets.AbstractManifold,
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2},
        slope::Float64,
        branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}};
        corners::Union{Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}, Nothing} = nothing,
        tolerance::Float64 = 1e-12
    ) -> Vector{CausalSets.Coordinates{2}}

Propagate a null ray from `x` towards `y` (forward or backward in time) with slope ±1 through a background containing finite topological cuts.  
The function returns the full piecewise path of the ray, deflecting it at each encountered cut.

# Purpose
This routine follows the trajectory of a left- or right-moving null ray in 2D spacetime in the presence of topological cuts.  
Whenever the ray intersects a cut, it continues from an endpoint of the cut according to geometric rules:
- For **spacelike or null cuts**, the ray continues from the endpoint opposite to its approach side (spatially opposite to the incoming slope).
- For **timelike cuts**, the ray continues from the future (or past, if propagating backward) endpoint of the cut.
- If the chosen endpoint lies outside the causal diamond `[x, y]`, the ray is clipped against the diamond boundary edges.

# Arguments
- `manifold::AbstractManifold`: Background manifold defining causal order (used to test causal relations and the diamond boundaries).
- `x::Coordinates{2}`: Starting point of the ray.
- `y::Coordinates{2}`: Endpoint in time; the ray is propagated until `y[1]`.
- `slope::Float64`: ±1, slope of the null ray (right-moving if +1, left-moving if -1).
- `branch_point_tuples::Vector{Tuple{Coordinates{2}, Coordinates{2}}}`: Finite cut segments, each a tuple `(p, q)`.

# Keyword Arguments
- `corners::Union{Vector{Tuple{Coordinates{2}, Coordinates{2}}}, Nothing} = nothing`: Optional precomputed diamond boundary edges. If `nothing`, they are recomputed from `(x, y)`.
- `check_causal_relation::Bool = true`: If false, turns off check whether x is in past of y.
- `tolerance::Float64 = 1e-12`: Numerical tolerance for classifying cut type and detecting intersections.

# Returns
- `Vector{Coordinates{2}}`: Ordered sequence of points along the piecewise ray, starting at `x` and ending at either `y` or a boundary point of the diamond.

# Throws
- `ArgumentError`: 
    - If `y` is not in the causal future of `x`, not taking into account topological obstructions.
    - If `tolerance <= 0`.

# Notes
- Uses [`next_intersection`](@ref) to locate the earliest intersection with cuts.
- For each intersection, the intersected cut is removed from the active list to avoid re-traversal.
- Rays can contain kinks where they are deflected at cuts; the returned path is suitable for later interpolation or wedge checks.
- The algorithm assumes 2D conformally Minkowski-type propagation with null slopes ±1.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
cut = (CausalSets.Coordinates{2}((1.0, -1.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
manifold = CausalSets.MinkowskiManifold{2}()
path = propagate_ray(manifold, x, y, +1.0, [cut])
# returns a path like [(0.0,0.0), (1.0,1.0), (2.0,0.0)]
"""
function propagate_ray(
    manifold::CausalSets.AbstractManifold,
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
    slope::Float64,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}};
    corners::Union{Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},Nothing}=nothing,
    check_causal_relation::Bool = true,
    tolerance::Float64=1e-12    
)::Vector{CausalSets.Coordinates{2}}
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    backward = y[1] < x[1]
    dir_sign = backward ? -1. : 1.
    if check_causal_relation && !CausalSets.in_past_of(manifold, backward ? y : x, backward ? x : y)
        throw(ArgumentError("y must be in the causal future of x but is not."))
    end
    local_branch_point_tuples = copy(branch_point_tuples)
    pos = x
    path = [x]
    tfin = y[1]
    while dir_sign * pos[1] < dir_sign * tfin
        hit = next_intersection(manifold, local_branch_point_tuples, pos, y, slope; tolerance = tolerance)
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
        is_timelike = conformal_proper_time > tolerance

        # Now select candidate endpoint
        candidate = nothing
        if is_timelike # timelike cut
            # Always select future (past) endpoint, larger (smaller) time coordinate
            candidate = (dir_sign * p[1] > dir_sign * q[1]) ? p : q
        else # spacelike or null cut
            # Opposite in spatial direction to slope
            if dir_sign * slope > 0
                candidate = (p[2] > q[2]) ? q : p
            else
                candidate = (p[2] < q[2]) ? q : p
            end
        end

        # Check if candidate lies within causal diamond of (x,y)
        if point_in_diamond(manifold, candidate, backward ? y : x, backward ? x : y; check_causal_relation = false)
            pos = candidate
            push!(path, pos)
            local_branch_point_tuples = deleteat!(local_branch_point_tuples, idx)
            continue
        end

        # Compute diamond corners
        left_corner, right_corner = isnothing(corners) ? diamond_corners(manifold, backward ? y : x, backward ? x : y; check_causal_relation = false) : corners
        # Define opposite boundary edges
        opposite_diamond_edges = (backward && slope > 0.) || (!backward && slope < 0.) ? [(x, right_corner), (right_corner, y)] : [(x, left_corner), (left_corner, y)]
        
        # Find intersection point of (intersection, candidate) with these edges if existent
        for edge in opposite_diamond_edges # If intersection is on the side opposite to the ray slope, ray ends.
            intersect, pt = segments_intersect((intersection, candidate), edge, tolerance = tolerance)
            if intersect && pt !== nothing
                push!(path, pt) # Push endpoint to path
                return path # Ray ends
            end
        end
        pos = dir_sign * candidate[1] > dir_sign * tfin ? interpolate_point(intersection, pt, tfin, 1; tolerance = tolerance) : candidate # No intersection with opposite side -> continue propagation
        push!(path, pos)
            
        local_branch_point_tuples = deleteat!(local_branch_point_tuples, idx)
    end
    return path
end   


"""
    cut_intersections(
        branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}};
        tolerance::Float64=1e-12
    ) -> Vector{Tuple{Tuple{Int,Int}, CausalSets.Coordinates{2}}}

Given a vector of finite tological cut segments, return a vector of all pairs of indices `(i, j)` (with `i < j`) such that the `i`-th and `j`-th segments intersect, together with the intersection points. Colinear overlapping (degenerate) intersections are warned and skipped.

# Purpose
This function identifies all pairs of finite topolical cut segments that intersect in the 2D coordinate plane, excluding cases where the segments are colinear and overlapping (which do not yield a unique intersection point and are not considered valid intersections for most geometric purposes).

# Arguments
- `branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}`: Vector of finite topological cut segments, each as a tuple `(a, b)` of 2D coordinates.

# Keyword Arguments
- `tolerance::Float64`: Numerical tolerance for intersection detection (default: `1e-12`). Used to handle floating point errors in geometric predicates.

# Returns
- `Vector{Tuple{Tuple{Int,Int}, CausalSets.Coordinates{2}}}`: A vector of pairs, where each element is a tuple `((i, j), point)` with `i < j`, and `point` is the intersection coordinate (`Coordinates{2}`) of the `i`-th and `j`-th segments. Colinear overlaps are not included.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- Only pairs where the segments cross at a unique point are included. If two segments are colinear and overlapping (i.e., they lie on the same line and share a nontrivial interval), the function issues a warning and skips them.
- The function is symmetric: for each pair `(i, j)`, only `i < j` is reported.
- The intersection points are computed using [`segments_intersect`](@ref), which returns `nothing` for colinear overlaps.

# Example
```julia
using CausalSets
cuts = [
    (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 1.0))),
    (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 0.0))),
    (CausalSets.Coordinates{2}((0.5, 0.5)), CausalSets.Coordinates{2}((1.5, 1.5)))
]
intersections = cut_intersections(cuts)
# returns [((1, 2), Coordinates{2}((0.5, 0.5)))]
# The third segment is colinear and overlapping with the first, so it is skipped with a warning.
```

# Treatment of Colinear Overlaps
- If two segments are colinear and overlap (i.e., share a nontrivial interval), they are not reported as intersecting. Instead, the function issues a warning of the form:
    ```
    [ Warn: Colinear overlapping cuts (i,j) skipped in cut_intersections. ]
    ```
  These cases are not included in the returned list, as there is no unique intersection point.
"""
function cut_intersections(
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}}; 
    tolerance::Float64=1e-12
    )::Vector{Tuple{Tuple{Int64,Int64}, CausalSets.Coordinates{2}}}
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

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
    in_wedge_of(
        manifold::CausalSets.AbstractManifold,
        branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}},
        x::CausalSets.Coordinates{2},
        y::CausalSets.Coordinates{2};
        check_causal_relation::Bool = true,
        tolerance::Float64 = 1e-12,
    ) -> Bool

Check whether the causal wedge of `x` (bounded by null rays) remains open up to the time of `y` in the presence of finite topological cuts.

# Purpose
Determines if the null wedge spanned by `x` survives up to `y` without collapsing due to obstructions from topological cuts.  
This is done by propagating left-moving null rays both forward from `x` and backward from `y`, accounting for deflections at each cut.  
The wedge is open if these rays meet without the left ray overtaking the right ray, ensuring causal connectivity between `x` and `y`.

# Arguments
- `manifold::AbstractManifold`: Background spacetime defining causal order.
- `branch_point_tuples::Vector{Tuple{Coordinates{2}, Coordinates{2}}}`: Vector of finite topological cut segments.
- `x::Coordinates{2}`: Apex of the wedge (past event).
- `y::Coordinates{2}`: Target event (future, only its time coordinate matters).

# Keyword Arguments
- `check_causal_relation::Bool = true`: If false, turns off check whether x is in (unobstructed) past of y.
- `tolerance::Float64 = 1e-12`: Numerical tolerance for intersection detection. Used to handle floating point errors in geometric predicates.

# Returns
- `Bool`: 
    - `true` if the wedge remains open up to `y` (causal connection possible),
    - `false` if the wedge collapses before reaching `y`.

# Throws
- `ArgumentError`: If `tolerance <= 0`.

# Notes
- Uses [`propagate_ray`](@ref) to propagate left-moving null rays with slope -1 forward from `x` and with slope +1 backward from `y`.
- Wedge closure occurs if either ray propagates past the spatial position of the opposite corner, or if their piecewise paths fail to intersect.
- Only left-moving rays are explicitly propagated. If the left-moving rays cannot connect `x` and `y`, no right-moving propagation can rescue the wedge.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
cuts = [(CausalSets.Coordinates{2}((1.0, -1.1)), CausalSets.Coordinates{2}((1.0, 1.1)))]
manifold = CausalSets.MinkowskiManifold{2}()
in_wedge_of(manifold, cuts, x, y)  # returns false because the cut closes the wedge
"""
function in_wedge_of(
    manifold::CausalSets.AbstractManifold,
    branch_point_tuples::Vector{Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}},
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2};
    check_causal_relation::Bool = true,
    tolerance::Float64 = 1e-12,
)::Bool
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    if check_causal_relation && !CausalSets.in_past_of(manifold, x, y)
        return false
    end    

    # The wedge is open if the left-moving null ray from x to y can reach y, and
    # the left-moving null ray from y backward to x can reach x, and their paths intersect.
    # This checks if a continuous null curve can connect x to y along the left-moving direction,
    # i.e., the wedge is open.
    left_forward = propagate_ray(manifold, x, y, -1.0, branch_point_tuples; check_causal_relation = false, tolerance = tolerance)
    if left_forward[end][2] > y[2] || left_forward[end][1] < y[1]
        return false
    end

    right_forward = propagate_ray(manifold, x, y, 1.0, branch_point_tuples; check_causal_relation = false, tolerance = tolerance)
    if right_forward[end][2] < y[2] || left_forward[end][1] < y[1]
        return false
    end

    for i in 1:length(left_forward)-1
        for j in 1:length(right_forward)-1
            intersect, pt = segments_intersect(
                (left_forward[i], left_forward[i+1]), 
                (right_forward[j], right_forward[j+1]); 
                tolerance = tolerance)
            if intersect && pt != x && pt != y
                return false
            end
        end
    end

    left_backward = propagate_ray(manifold, y, x, 1.0, branch_point_tuples; check_causal_relation = false, tolerance = tolerance)
    if left_backward[end][2] > x[2] || left_backward[end][1] > x[1]
        return false
    end

    # Check if the two piecewise paths intersect (i.e., share any segment intersection)
    for i in 1:length(left_forward)-1
        seg1 = (left_forward[i], left_forward[i+1])
        for j in 1:length(left_backward)-1
            seg2 = (left_backward[j], left_backward[j+1])
            if segments_intersect(seg1, seg2; tolerance = tolerance)[1]
                return true
            end
        end
    end
end

"""
    in_past_of(
        manifold::ConformallyTimesliceableManifold{N},
        branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}},
        x::Coordinates{N},
        y::Coordinates{N};
        tolerance::Float64 = 1e-12
    ) -> Bool

Determine whether `x` lies in the causal past of `y` in a conformally timesliceable manifold with topological cuts, which can be either timelike and intersect the boundary (trouser-geometry like) or timelike or spacelike but finite.

# Purpose
This extends the standard causal relation by incorporating **topological obstructions**.  
Besides the manifold’s lightcone structure, causal paths may be blocked by:
- **Timelike boundary-connecting cuts** from single branch points (extending at constant spatial coordinate position up to the boundary).
- **Finite cuts** (segments between two points).
- **Intersections or common effects of multiple cuts** that can jointly close causal diamonds. At the moment, only cut intersections between two cuts are implemented.

# Arguments
- `manifold::ConformallyTimesliceableManifold{N}`: Background spacetime (currently only `N = 2` supported).
- `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`:
  - First element: single branch points (vertical cuts).
  - Second element: finite cut segments.
- `x::Coordinates{N}`: Candidate past event.
- `y::Coordinates{N}`: Candidate future event.

# Keyword Arguments
- `tolerance::Float64 = 1e-12`: Numerical tolerance for distance and intersection checks.

# Returns
- `Bool`:  
  - `true` if `x ≺ y` and no topological obstruction blocks the causal path.  
  - `false` otherwise.

# Throws
- `ArgumentError`:  
  - If `N ≠ 2` (unsupported dimension).  
  - If `tolerance <= 0`.

# Notes
1. First checks the manifold’s **unobstructed causal relation** (`x ≺ y`).
2. Prunes irrelevant cuts outside the causal diamond `[x, y]`.
3. Tests single branch points: if any timelike, boundary-connecting cut lies between `x` and `y` and its endpoint is not in the future of `x`, the path is blocked.
4. Tests whether any of the causal-diamond edges is unobstructed.
5. Tests finite cuts: if any cut segment crosses the diamond entirely, return `false`.
6. Considers intersections of cuts: if two cuts combine to obstruct the diamond, return `false`.
7. If no obstruction is found, the function falls back to [`in_wedge_of`](@ref) to check wedge closure. This can become important when several cuts conspire to block ray propagation without intersecting.

# Example
```julia
using CausalSets
x = CausalSets.Coordinates{2}((0.0, 0.0))
y = CausalSets.Coordinates{2}((2.0, 0.0))
single_branch_points = [CausalSets.Coordinates{2}((1.0, 0.5))]
finite_cuts = [(CausalSets.Coordinates{2}((1.0,-1.1)), CausalSets.Coordinates{2}((1.0,1.1)))]
branch_info = (single_branch_points, finite_cuts)

manifold = CausalSets.MinkowskiManifold{2}()
in_past_of(manifold, branch_info, x, y)  # returns false (finite cut obstructs the path)
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
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end


    if N != 2
        throw(ArgumentError("Currently, nontrivial topologies are only implemented for dimensionality N=2, is $(N)."))
    end
    
    # Check standard causal relation
    CausalSets.in_past_of(manifold, x, y) || return false

    single_branch_points, branch_point_tuples = branch_point_info

    # Prune irrelevant finite cuts
    corners = diamond_corners(manifold, x,y; check_causal_relation = false)
    branch_point_tuples = [
        (p,q) for (p,q) in branch_point_tuples
        if y[1] > p[1] && q[1] > x[1] && min(p[2],q[2]) < corners[2][2] && max(p[2],q[2]) > corners[1][2]
    ]
    # Prune irrelevant constant-time cuts
    single_branch_points = [
        b for b in single_branch_points
        if b[1] < y[1] && corners[1][2] < b[2] < corners[2][2]
    ]

    # Check for topological cuts that obstruct causality
    for b in single_branch_points
        if min(x[2], y[2]) < b[2] < max(x[2], y[2])
            CausalSets.in_past_of(manifold, x, b) || return false
        end
    end

    # Efficient obstruction test for single cuts intersecting whole diamond
    for cut in branch_point_tuples
        if cut_crosses_diamond(manifold, x, y, cut; check_causal_relation = false, tolerance = tolerance)
            return false
        end
    end

    vertical_cuts = Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}}[
    (bp, (y[1] + tolerance, bp[2]))
    for bp in single_branch_points
    ]
    all_cuts = vcat(branch_point_tuples, vertical_cuts)
    # Reorder each tuple so that p[1] <= q[1]
    all_cuts = [(p[1] <= q[1] ? (p, q) : (q, p)) for (p, q) in all_cuts]
    # Sort by the time coordinate of the first element of each tuple
    all_cuts = sort(all_cuts, by = t -> t[1][1])

    # Check whether causal diamond edges are inhibited by cuts
    past_intersections   = Vector{Union{Nothing, Tuple{CausalSets.Coordinates{2}, Int}}}(undef, 2) # save in case we can reuse this in in_wedge_of, to be implemented
    future_intersections = Vector{Union{Nothing, Tuple{CausalSets.Coordinates{2}, Int}}}(undef, 2) # save in case we can reuse this in in_wedge_of, to be implemented
    for i in 1:2 
        past_intersections[i] = next_intersection(manifold, all_cuts, x, corners[i], (-1.)^i; null_separated = true, tolerance = tolerance)
        future_intersections[i] = next_intersection(manifold, all_cuts, corners[i], y, (-1.)^(i+1); null_separated = true, tolerance = tolerance)
        if isnothing(past_intersections[i]) && isnothing(future_intersections[i])
            return true
        end
    end

    # check intersections between all cuts
    intersections = cut_intersections(all_cuts; tolerance = tolerance)
    for ((i, j), intersection_point) in intersections
        # Compute intersection point between all_cuts[i] and all_cuts[j]
        seg1 = all_cuts[i]
        seg2 = all_cuts[j]
        # Check for obstruction
        if intersected_cut_crosses_diamond(manifold, x, y, seg1, seg2, intersection_point; corners=corners, check_causal_relation = false, tolerance = tolerance)
            return false
        end
    end

    return in_wedge_of(manifold, all_cuts, x, y; check_causal_relation = false, tolerance = tolerance)
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
- `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`: Tuple containing information about branch points and branch cuts.
    - The first element, `single_branch_points`, is a vector of single branch points, each inducing a vertical cut to the boundary (extending in coordinate time).
    - The second element, `branch_point_tuples`, is a vector of finite branch cut segments, each as a tuple `(p, q)` of coordinates.
- `sprinkling::Vector{Coordinates{N}}`: List of sprinkled spacetime points (atoms), typically sorted by coordinate time.

# Returns
- A `BranchedManifoldCauset{N, M}` instance with the following fields:
    - `atom_count::Int64`: Number of atoms (points) in the causet.
    - `manifold::M`: The background manifold.
    - `branch_point_info::Tuple{Vector{Coordinates{N}}, Vector{Tuple{Coordinates{N}, Coordinates{N}}}}`: Tuple containing single branch points and branch cut segments.
    - `sprinkling::Vector{Coordinates{N}}`: The list of sprinkled points.

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
function CausalSets.BitArrayCauset(causet::BranchedManifoldCauset{N}; tolerance::Float64 = 1e-12)::CausalSets.BitArrayCauset where {N}
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    return convert(CausalSets.BitArrayCauset, causet; tolerance = tolerance)
end

"""
    convert(BitArrayCauset, causet::BranchedManifoldCauset)

Computes the causal matrix for a `BranchedManifoldCauset` and returns a `BitArrayCauset`. Note: this function uses all available threads.
"""
function Base.convert(::Type{CausalSets.BitArrayCauset}, causet::BranchedManifoldCauset{N}; tolerance::Float64=1e-12)::CausalSets.BitArrayCauset where {N}
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    atom_count = causet.atom_count

    future_relations = Vector{BitVector}(undef, atom_count)
    past_relations = Vector{BitVector}(undef, atom_count)

    for i in 1:atom_count
        future_relations[i] = falses(atom_count)
        past_relations[i] = falses(atom_count)
    end

    Threads.@threads for i in 1:atom_count
        for j in i+1:atom_count
            if CausalSets.in_past_of(causet.manifold, causet.branch_point_info, causet.sprinkling[i], causet.sprinkling[j]; tolerance = tolerance)
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
        rng::AbstractRNG,
        order::Int64,
        r::Float64;
        d::Int64 = 2,
        tolerance::Float64 = 1e-12,
        type::Type{T} = Float32
    ) -> Tuple{
        CausalSets.BitArrayCauset,
        Vector{Coordinates{2}},
        Tuple{Vector{Coordinates{2}}, Vector{Tuple{Coordinates{2}, Coordinates{2}}}},
        Matrix{T}
    }

Generate a branched causal set in a random polynomial manifold with timelike boundary-connecting and finite topological cuts.

# Purpose
This function builds a toy model of a branched spacetime:
1. A random 2D polynomial manifold is generated from Chebyshev coefficients with exponential decay.
2. A sprinkling of `npoints` events is placed into the manifold.
3. Random **vertical cuts** (from single branch points extending upward in time) and **finite cuts** (segments between two points) are introduced.
4. Points lying too close to cuts are filtered out.
5. A [`BranchedManifoldCauset`](@ref) and its causal matrix (`BitArrayCauset`) are constructed.

# Arguments
- `npoints::Int64`: Number of sprinkled points. Must be > 0.
- `n_vertical_cuts::Int64`: Number of vertical cuts (≥ 0).
- `n_finite_cuts::Int64`: Number of finite cuts (≥ 0).
- `rng::AbstractRNG`: Random number generator.
- `order::Int64`: Order of the Chebyshev expansion (must be > 0).
- `r::Float64`: Decay base for Chebyshev coefficients (must be > 1).
- `d::Int64`: Dimension of the spacetime. Only `d = 2` is supported (default).
- `tolerance::Float64`: Minimal distance for filtering points too close to cuts and for computing spacetime distances (default: `1e-12`).
- `type::Type{T}`: Numeric type for the returned Chebyshev coefficient matrix (default: `Float32`).

# Returns
A 4-tuple `(cset, sprinkling, branch_point_info, chebyshev_coefs)`:
- `cset::BitArrayCauset`: The causal set with branch cuts encoded.
- `sprinkling::Vector{Coordinates{2}}`: The filtered sprinkled points.
- `branch_point_info::Tuple{Vector{Coordinates{2}}, Vector{Tuple{Coordinates{2}, Coordinates{2}}}}`:  
  - Single branch points (for timelike boundary-connecting cuts),  
  - Finite cut segments.  
- `chebyshev_coefs::Matrix{T}`: Chebyshev coefficients of the random polynomial manifold.

# Throws
- `ArgumentError` if any of:
  - `npoints ≤ 0`
  - `n_vertical_cuts < 0`
  - `n_finite_cuts < 0`
  - `order ≤ 0`
  - `r ≤ 1`
  - `d ≠ 2`
  - `tolerance ≤ 0`

# Example
```julia
using Random, CausalSets
rng = MersenneTwister(1234)
cset, sprinkling, branch_info, coefs =
    make_branched_manifold_cset(50, 2, 3, rng, 5, 2.0)
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
    
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
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
    branch_point_info = generate_random_branch_points(n_vertical_cuts, n_finite_cuts; tolerance = tolerance)

    # Remove points on sprinkling on cuts
    branched_sprinkling = filter_sprinkling_near_cuts(sprinkling, branch_point_info; tolerance = tolerance)

    # Construct the causal set from the manifold and sprinkling
    cset = BranchedManifoldCauset(polym, branch_point_info, branched_sprinkling)

    return CausalSets.BitArrayCauset(cset; tolerance = tolerance), branched_sprinkling, branch_point_info, type.(chebyshev_coefs)
end