function integrate_box_faces_2D(integrand, boundary, rtol, atol, maxevals)
    lower, upper = validate_box_quadrature_2D(boundary, rtol, atol, maxevals)
    function face(axis, side)
        fixed = side == -1 ? lower[axis] : upper[axis]
        other = 3 - axis
        function along_face(u)
            point = CausalSets.Coordinates{2}(axis == 1 ? (fixed, u) : (u, fixed))
            return integrand(point, axis, side)
        end
        value, error = HCubature.hquadrature(
            along_face,
            lower[other],
            upper[other];
            rtol,
            atol = atol / 4,
            maxevals,
        )
        return (integral = value, error = error)
    end
    faces = (
        t_lower = face(1, -1),
        t_upper = face(1, 1),
        x_lower = face(2, -1),
        x_upper = face(2, 1),
    )
    return (
        integral = sum(f.integral for f in faces),
        error = sum(f.error for f in faces),
        faces = faces,
        joint = 0.0,
    )
end

"""
    boundary_curvature_invariant_2D(invariant, coefs, epsilon; boundary=..., kwargs...)

Integrate the boundary completion of a built-in invariant for the regularized
2D metric ``g = (P^2 + \\epsilon) \\eta`` on a coordinate box. Bounds,
derivation matrices, `rtol`, `atol`, and `maxevals` follow
[`integrated_curvature_invariant_2D`](@ref); the evaluation limit applies to each
face. Return `(integral, error, faces, joint)` as a named tuple. Each face has
its own `integral` and estimated `error`; the total error sums face estimates.

Use outward unit normals, ``s=n^a n_a``, and ``K=\\nabla_a n^a``. The boundary
integrand is ``2sK`` for Einstein-Hilbert, ``4sRK`` for the 2D Kretschmann
action ``R^2``, and ``6sR^2K`` for the 2D Goroff-Sagnotti action ``R^3``.
These actions fix the induced metric, and additionally ``R`` for the latter
two. All expressions include the induced measure and omit coupling constants.

For ``R\\Box R``, fix the induced metric, ``R``, and ``\\Box R``. The auxiliary
scalar prescription gives the boundary integrand
``s(4K\\Box R - R n^a\\nabla_a R)``. The second term subtracts the Green-identity
flux; it must accompany the direct bulk integral of ``R\\Box R``. This is one
specified variational problem, not a completion for every boundary condition.

The Weyl actions vanish identically in 2D and have zero completions here.
Corners use the real Lorentzian boost-angle convention: all four box joints
are orthogonal and their contribution is zero. Imaginary signature-flip
constants and reference-subtraction counterterms are excluded. See the
regularized-geometry documentation for the derivation and references.
"""
function boundary_curvature_invariant_2D(
    invariant::Function,
    coefs::Array{Float64,2},
    epsilon::Real;
    boundary::CausalSets.BoxBoundary{2} = CausalSets.BoxBoundary{2}((
        (-1.0, -1.0),
        (1.0, 1.0),
    )),
    rtol::Real = 1e-6,
    atol::Real = 1e-10,
    maxevals::Int64 = 1_000_000,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)
    if !(
        invariant in (
            Ricci_scalar_2D,
            Kretschmann_scalar_2D,
            Goroff_Sagnotti_scalar_2D,
            Ricci_box_Ricci_2D,
            Weyl_squared_2D,
            Weyl_box_Weyl_2D,
        )
    )
        throw(ArgumentError("No boundary completion is defined for this invariant"))
    end
    derivatives, epsilon, _, _ = regularized_curvature_data_2D(
        coefs,
        epsilon;
        higher_derivatives = invariant === Ricci_box_Ricci_2D,
        derivation_matrix1,
        derivation_matrix2,
    )
    function integrand(point, axis, side)
        if invariant === Weyl_squared_2D || invariant === Weyl_box_Weyl_2D
            return 0.0
        end
        f = regularized_density_derivatives_2D(derivatives, point, epsilon)
        causal_sign = axis == 1 ? -1 : 1
        f_normal = axis == 1 ? f.f_t : f.f_x
        # s K sqrt(|h|), with n^axis = side / sqrt(F).
        k_measure = causal_sign * side * f_normal / (2 * f.f)
        if invariant === Ricci_scalar_2D
            return 2 * k_measure
        elseif invariant === Ricci_box_Ricci_2D
            curvature = regularized_ricci_derivatives_2D(derivatives, point, epsilon)
            R_normal = axis == 1 ? curvature.R_t : curvature.R_x
            flux = causal_sign * side * curvature.R * R_normal
            return 4 * curvature.box_R * k_measure - flux
        end
        R = (f.f * (f.f_tt - f.f_xx) - f.f_t^2 + f.f_x^2) / f.f^3
        return invariant === Kretschmann_scalar_2D ? 4 * R * k_measure : 6 * R^2 * k_measure
    end
    return integrate_box_faces_2D(integrand, boundary, rtol, atol, maxevals)
end

"""
    ricci_box_ricci_boundary_flux_2D(coefs, epsilon; boundary=..., kwargs...)

Compute ``\\oint_{\\partial B}s R n^a\\nabla_a R\\,\\mathrm{d}\\Sigma`` for
``g=(P^2+\\epsilon)\\eta``. This is the Green-identity flux in
``\\int R\\Box R\\,\\mathrm{d}V = -\\int (\\nabla R)^2\\,\\mathrm{d}V + \\mathrm{flux}``,
not the complete variational boundary term. Arguments and the named-tuple
result follow [`boundary_curvature_invariant_2D`](@ref).
"""
function ricci_box_ricci_boundary_flux_2D(
    coefs::Array{Float64,2},
    epsilon::Real;
    boundary::CausalSets.BoxBoundary{2} = CausalSets.BoxBoundary{2}((
        (-1.0, -1.0),
        (1.0, 1.0),
    )),
    rtol::Real = 1e-6,
    atol::Real = 1e-10,
    maxevals::Int64 = 1_000_000,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)
    derivatives, epsilon, _, _ = regularized_curvature_data_2D(
        coefs,
        epsilon;
        higher_derivatives = true,
        derivation_matrix1,
        derivation_matrix2,
    )
    function integrand(point, axis, side)
        curvature = regularized_ricci_derivatives_2D(derivatives, point, epsilon)
        causal_sign = axis == 1 ? -1 : 1
        R_normal = axis == 1 ? curvature.R_t : curvature.R_x
        return causal_sign * side * curvature.R * R_normal
    end
    return integrate_box_faces_2D(integrand, boundary, rtol, atol, maxevals)
end

"""
    curvature_action_2D(invariant, coefs, epsilon; kwargs...)

Compute a built-in bulk invariant plus its specified boundary completion.
Return a named tuple with `bulk`, `boundary`, `joint`, `total`, `error`,
`bulk_error`, `boundary_error`, and `faces`. The error adds the estimated bulk
and face errors. Keywords follow [`integrated_curvature_invariant_2D`](@ref).
Boundary conditions and corner conventions follow
[`boundary_curvature_invariant_2D`](@ref). Coupling constants are not included.
"""
function curvature_action_2D(
    invariant::Function,
    coefs::Array{Float64,2},
    epsilon::Real;
    kwargs...,
)
    # Validate that a completion exists before performing bulk quadrature.
    boundary = boundary_curvature_invariant_2D(invariant, coefs, epsilon; kwargs...)
    bulk, bulk_error =
        integrated_curvature_invariant_2D(invariant, coefs, epsilon; kwargs...)
    return (
        bulk = bulk,
        boundary = boundary.integral,
        joint = boundary.joint,
        total = bulk + boundary.integral + boundary.joint,
        error = bulk_error + boundary.error,
        bulk_error = bulk_error,
        boundary_error = boundary.error,
        faces = boundary.faces,
    )
end

"""
    einstein_hilbert_action_2D(coefs, epsilon; kwargs...)

Compute Einstein-Hilbert plus Gibbons-Hawking-York and the real box joints,
without a cosmological constant or the overall ``1/(16\\pi G)`` factor.
This is `curvature_action_2D(Ricci_scalar_2D, coefs, epsilon; kwargs...)`.
For a 2D conformal coordinate box, the bulk and GHY terms cancel analytically;
the returned `total` measures numerical cancellation within the estimated error.
"""
function einstein_hilbert_action_2D(coefs::Array{Float64,2}, epsilon::Real; kwargs...)
    return curvature_action_2D(Ricci_scalar_2D, coefs, epsilon; kwargs...)
end
