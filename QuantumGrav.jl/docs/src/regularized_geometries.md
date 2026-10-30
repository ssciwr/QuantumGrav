# Regularized polynomial geometries

Regularized geometries use a strictly positive volume density before sprinkling.
Let ``P`` be the random polynomial represented by tensor-product Chebyshev
coefficients. The new generator sets

```math
F=P^2+\varepsilon,\qquad
g_{ab}=F^{2/d}\eta_{ab},\qquad
\mathrm{d}V_g=F\,\mathrm{d}^d x,\qquad \varepsilon>0.
```

The original generator still uses ``F=P^2``. The regularized generator draws
the same coefficients for ``P`` from the same RNG state, then samples from the
modified density. Its return value has the same structure as the original.
The offset changes the geometry; it is not applied to an existing point cloud.

```julia
using QuantumGrav, Random

epsilon = 0.1
cset, points, coefs = make_regularized_polynomial_manifold_cset(
    256, Xoshiro(42), 3, 2.0, epsilon; type=Float64,
)
R = Ricci_scalar_2D_of_sprinkling(coefs, points, epsilon)
```

The returned coefficients describe ``P``, so retain `epsilon` alongside them.
Use `type=Float64` when computing continuum invariants from the returned
coefficients, to retain the coefficients used by the sampler. A manifold can
also be built directly with `regularized_polynomial_manifold(coefs, epsilon)`.
Generation supports arbitrary positive dimension; the curvature, integration,
and action methods described below are specific to two dimensions.

## Continuum integrals

The integral API includes the proper volume measure and needs no sprinkling.
Each of the six existing invariant functions accepts an additional positional
`epsilon` for the metric ``g=(P^2+\varepsilon)\eta``. Omitting that argument
retains the original metric convention. Polynomial derivatives are prepared
once for each integral or sprinkling evaluation, including fourth derivatives
for ``R\Box R``. The square root of ``F`` is never approximated by a polynomial.

```julia
using CausalSets

box = BoxBoundary{2}(((-1.0, -1.0), (1.0, 1.0)))
integral, error = integrated_curvature_invariant_2D(
    Kretschmann_scalar_2D, coefs, epsilon;
    boundary=box, rtol=1e-7, atol=1e-10, maxevals=1_000_000,
)
```

Box bounds must lie within ``[-1,1]^2``. Adaptive cubature returns an integral
and an estimated absolute error, and can stop at `maxevals` before reaching
the requested accuracy. Check `error <= max(atol, rtol*abs(integral))`; the
error is an estimate, not a rigorous bound. Small offsets can produce narrow
curvature peaks and require tighter tolerances or more evaluations. Expanded
Taylor coefficients in the CausalSets sampler can also suffer cancellation at
high polynomial orders despite mathematical positivity.

## Boundary conventions and actions

Boundary completions depend on the boundary data held fixed.
The API uses outward unit normals ``n^a``, ``s=n^a n_a``, extrinsic curvature
``K=\nabla_a n^a``, and signature ``(-,+)``. Thus ``s=-1`` on constant-time
faces and ``s=+1`` on constant-space faces. Coupling constants, a cosmological
constant, and reference-subtraction counterterms are not included.

For an ``f(R)`` action, the implemented boundary term is

```math
B_f=2\oint_{\partial B}s\,f'(R)K\,\mathrm{d}\Sigma.
```

The induced metric is fixed, and ``R`` is fixed for the higher-curvature
actions. This prescription gives GHY for ``f(R)=R``, ``4sRK`` for the 2D
Kretschmann action ``R^2``, and ``6sR^2K`` for the 2D Goroff-Sagnotti action
``R^3``. These are completions of the dimensionally reduced 2D actions, not
boundary terms for arbitrary higher-dimensional curvature contractions.
The ``f(R)`` variational prescription is described by
[Guarnizo, Castañeda, and Tejeiro](https://arxiv.org/abs/1002.0617).

Real Lorentzian joint angles vanish at every corner of a conformal coordinate
box. Adjacent face normals remain orthogonal under a conformal rescaling, so
their boost angle ``\operatorname{arsinh}(n\cdot s)`` is zero. The API reports
`joint=0.0` with this convention. It excludes imaginary signature-flip constants;
it does not apply the Euclidean corner-angle prescription. See
[Lehner, Myers, Poisson, and Sorkin](https://arxiv.org/abs/1609.00207), section II.D.

Boundary results expose all four faces and their numerical errors.
`boundary_curvature_invariant_2D(invariant, coefs, epsilon; boundary=box)`
returns a named tuple with `integral`, `error`, `faces`, and `joint`. The face
names are `t_lower`, `t_upper`, `x_lower`, and `x_upper`; each holds an
`integral` and `error`. The evaluation limit applies separately to each face.
Face errors are added, so cancellations can leave a total whose relative
error is larger than the requested face tolerance.

```julia
action = curvature_action_2D(Kretschmann_scalar_2D, coefs, epsilon; boundary=box)
eh = einstein_hilbert_action_2D(coefs, epsilon; boundary=box)
# action.bulk, action.boundary, action.joint, action.total, action.error
# action.faces.t_lower.integral, action.faces.x_upper.error
```

Einstein-Hilbert and GHY cancel analytically for these 2D coordinate boxes.
Writing ``F=P^2+\varepsilon`` gives

```math
RF=\partial_t^2\log F-\partial_x^2\log F.
```

The four signed GHY integrands are ``F_t/F`` on the lower-time face,
``-F_t/F`` on the upper-time face, ``-F_x/F`` on the lower-space face, and
``F_x/F`` on the upper-space face. Their sum cancels the bulk integral.
Consequently, the combined EH+GHY action is zero under the real joint
convention, within quadrature error, for every conformal factor in this family.
The separate bulk and boundary values can still vary between geometries.

## The derivative-curvature action

The ``R\Box R`` completion uses a specified auxiliary-scalar variational
problem. Introduce independent fields ``\lambda`` and ``\phi`` and consider

```math
S_{\mathrm{aux}}=
\int_B\bigl[\lambda(R-\phi)-\nabla_a\phi\nabla^a\phi\bigr]\,\mathrm{d}V
+2\oint_{\partial B}s\lambda K\,\mathrm{d}\Sigma.
```

Fixing the induced metric, ``\lambda``, and ``\phi`` removes the boundary
variations. The auxiliary equations give ``\phi=R`` and
``\lambda=2\Box R``. Substitution and Green's identity then yield

```math
S_{\mathrm{aux}}=
\int_B R\Box R\,\mathrm{d}V
+\oint_{\partial B}s\bigl(4K\Box R-Rn^a\nabla_aR\bigr)\,\mathrm{d}\Sigma.
```

This derivation specifies fixed induced metric, ``R``, and ``\Box R`` as the
boundary data. The API implements that completion, including the flux
subtraction. It does not claim that the same completion applies when different
boundary data are fixed. The broader auxiliary-field approach to
higher-derivative boundary terms is discussed by
[Teimouri, Talaganis, Edholm, and Mazumdar](https://arxiv.org/abs/1606.01911);
the scalar reduction above states the particular prescription used here.

The Green-identity flux is also available separately.
`ricci_box_ricci_boundary_flux_2D(coefs, epsilon; boundary=box)` returns

```math
\mathcal F=\oint_{\partial B}sRn^a\nabla_aR\,\mathrm{d}\Sigma,
\qquad
\int_B R\Box R\,\mathrm{d}V
=-\int_B(\nabla R)^2\,\mathrm{d}V+\mathcal F.
```

This flux is not the complete variational boundary term. Both Weyl actions
and their completions vanish identically in the reduced 2D theory.

## API

```@docs
make_regularized_polynomial_manifold_cset
regularized_polynomial_manifold
integrated_curvature_invariant_2D
boundary_curvature_invariant_2D
ricci_box_ricci_boundary_flux_2D
curvature_action_2D
einstein_hilbert_action_2D
```
