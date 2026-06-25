"""
    OffsetCausalDiamondBoundary{N}(duration::Float64, center::Coordinates{N}) <: AbstractBoundary{N}
A generalization of the CausalDiamondBoundary type which allows for the specification of a center point
"""
struct OffsetCausalDiamondBoundary{N} <: CausalSets.AbstractBoundary{N}
    duration::Float64
    center::CausalSets.Coordinates{N}
end

# NOTE: This is an incomplete set of methods, e.g. sprinkling would not work in the current state
"""
    get_tips(boundary::OffsetCausalDiamondBoundary{N}, manifold::MinkowskiManifold{N})
Obtain the tips of a causal diamond based on the centerpoint and the duration
"""
function get_tips(boundary::OffsetCausalDiamondBoundary{N}, manifold::CausalSets.MinkowskiManifold{N}) where {N}
    past_tip = (boundary.center[1]-boundary.duration/2, boundary.center[2:end]...)
    future_tip = (boundary.center[1]+boundary.duration/2, boundary.center[2:end]...)
    (past_tip, future_tip)
end

"""
    CausalSets.is_in_boundary(manifold::MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N}, coord::Coordinates{N})::Bool
Method for the is_in_boundary function for an OffsetCausalDiamondBoundary
"""
function CausalSets.is_in_boundary(manifold::CausalSets.MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N}, coord::CausalSets.Coordinates{N})::Bool where {N}
    past_tip, future_tip = get_tips(boundary, manifold)
    
    if coord[1] > boundary.center[1]
        # future half of causal diamond:
        return CausalSets.in_past_of(manifold, coord, future_tip)
    else
        # past half of causal diamond:
        return CausalSets.in_past_of(manifold, past_tip, coord)
    end
end

"""
    CausalSets.boundary_volume(manifold::MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N})
Compute the volume of an OffsetCausalDiamondBoundary
"""
function CausalSets.boundary_volume(manifold::CausalSets.MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N}) where {N}
    return pi^((N-1)/2) / (N*2^(N-1)*SpecialFunctions.gamma((N+1)/2)) * boundary.duration^N
end

"""
    causal_diamond_duration_from_volume(manifold::MinkowskiManifold{N}, center::Coordinates{N}, target_volume::Float64)
Inverse volume computation to obtain the duration of an OffsetCausalDiamond required for it to have a target volume
"""
function causal_diamond_duration_from_volume(manifold::CausalSets.MinkowskiManifold{N}, center::CausalSets.Coordinates{N}, target_volume::Float64) where {N}
    return (N*2^(N-1)*SpecialFunctions.gamma((N+1)/2) / (pi^((N-1)/2)) * target_volume)^(1/N)
end

"""
    count_elements_in_boundary(manifold_causet::ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N})
Get the number of points of a ManifoldCauset lyin strictly within in a specific OffsetCausalDiamondBoundary
"""
function count_elements_in_boundary(manifold_causet::CausalSets.ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}) where {N}
    sum(CausalSets.is_in_boundary.(Ref(manifold_causet.manifold), Ref(boundary), manifold_causet.sprinkling))
end

"""
    boundary_from_contained_element_count(manifold_causet::ManifoldCauset{N}, center::Coordinates{N}, target_count::Int64, sprinkling_density::Float64)
Obtain a OffsetCausalDiamondBoundary that contains a specified amount of points in a ManifoldCauset, centered on a specified point. 
Requires the sprinkling density to give efficient estimates of sizes.
"""
function boundary_from_contained_element_count(manifold_causet::CausalSets.ManifoldCauset{N}, center::CausalSets.Coordinates{N}, target_count::Int64, sprinkling_density::Float64) where {N}
    volume_estimate = target_count/sprinkling_density
    upper_bound = Inf
    lower_bound = 0
    while true
        duration_estimate = causal_diamond_duration_from_volume(manifold_causet.manifold, center, max(volume_estimate, eps()))
        if duration_estimate>=upper_bound || duration_estimate<=lower_bound
            #Density based calculation failed, moving on to bisecting calculation"
            break
        end
        boundary = OffsetCausalDiamondBoundary{N}(duration_estimate, center)
        count = count_elements_in_boundary(manifold_causet, boundary)
        diff = target_count-count
        if diff>0
            lower_bound=duration_estimate
        elseif diff<0
            upper_bound=duration_estimate
        else
            return OffsetCausalDiamondBoundary{N}(duration_estimate, center)
        end

        volume_estimate = volume_estimate + diff/sprinkling_density
    end

    while true
        duration_estimate = (upper_bound + lower_bound)/2
        boundary = OffsetCausalDiamondBoundary{N}(duration_estimate, center)
        count = count_elements_in_boundary(manifold_causet, boundary)
        diff = target_count-count
        if diff>0
            lower_bound=duration_estimate
        elseif diff<0
            upper_bound=duration_estimate
        else
            return OffsetCausalDiamondBoundary{N}(duration_estimate, center)
        end
    end
end

# Based on CausalSets.jl/src/causets/sampler.jl
#=
function transitive_closure!(causet::CausalSets.BitArrayCauset)
    for i in 1:causet.atom_count-1
        causet.future_relations[i][i] = 1
        for j in i:causet.atom_count
            if causet.future_relations[i][j]
                for k in j+1:causet.atom_count
                    if causet.future_relations[j][k]
                        causet.future_relations[i][k] = 1
                        causet.past_relations[k][i] = 1
                    end
                end
            end
        end
        causet.future_relations[i][i] = 0
    end
end
=#

# Optimized by walking backwards, so that due to natural labeling, only the nearest future points have to be completed, the rest follows from them already having been transitively completed
"""
    transitive_closure!(causet::BitArrayCauset)
Modifies a BitArrayCauset in place by adding relations so that any relation implied by transitive completeness also appears in the adjacency matrix
"""
function transitive_closure!(causet::CausalSets.BitArrayCauset)
    n = causet.atom_count
    seeds = BitVector(undef, n)
    @inbounds for i in n-1:-1:1
        copyto!(seeds, causet.future_relations[i])
        j = findfirst(seeds)
        while j !== nothing
            causet.future_relations[i] .|= causet.future_relations[j]
            j = findnext(seeds, j+1)
        end
    end

    @inbounds for i in 1:n
        k = findfirst(causet.future_relations[i])
        while k !== nothing
            causet.past_relations[k][i] = true
            k = findnext(causet.future_relations[i], k+1)
        end
    end
end

"""
    bool_mul2(A::BitMatrix, B::BitMatrix) -> BitMatrix
Boolean Matrix multiplication (i.e. 1 if any "paths" exist, rather than normal matrix multiplication which would give integers, and is of course much slower)
Code by Benoit Pasquier on StackOverflow (https://stackoverflow.com/questions/64939193/boolean-matrix-multiplication-in-julia)"""
function bool_mul2(A::BitMatrix, B::BitMatrix)
    mA, nA = size(A)
    mB, nB = size(B)
    nA ≠ mB && error()
    AB = BitArray(undef, mA, nB)
    for i in 1:mA, j in 1:nB
        AB[i,j] = any(A[i,k] && B[k,j] for k in 1:nA)
    end
    AB
end

"""
    generateKRPosetAdjacencyMatrix(n_elements::Integer) -> BitMatrix
Generate a KR poset, with the counts of elements in the layers and the links between layers randomly sampled. Returns an adjacency matrix.
"""
function generate_KR_poset_adjacency_matrix(n_elements::Integer; rng::Random.AbstractRNG = Random.default_rng())
    # Using this approach rather than simply sampling 
    layer_counts = rand(rng, Distributions.Multinomial(n_elements, [0.25, 0.5, 0.25]));
    # NOTE: Should be equivalent to connecting from the bottom, but am not totaly certain
    bottom_to_middle = Random.bitrand(rng, layer_counts[1], layer_counts[2]);
    middle_to_top = Random.bitrand(rng, layer_counts[2], layer_counts[3]);
    bottom_to_top = bool_mul2(bottom_to_middle, middle_to_top);
    A = falses(n_elements, n_elements);
    A[1:layer_counts[1], (layer_counts[1]+1):(layer_counts[1]+layer_counts[2])] = bottom_to_middle;
    A[(layer_counts[1]+1):(layer_counts[1]+layer_counts[2]),(layer_counts[1]+layer_counts[2]+1):end] = middle_to_top;
    A[1:layer_counts[1], (layer_counts[1]+layer_counts[2]+1):end] = bottom_to_top;
    A
end

"""
    BitArrayCauset(A::BitMatrix)
Create a BitArrayCauset from an adjacency matrix A. 
"""
function CausalSets.BitArrayCauset(A::BitMatrix)
    n = size(A, 1);
    future_relations = [A[i,:] for i in 1:n];
    past_relations   = [A[:,j] for j in 1:n];
    CausalSets.BitArrayCauset(n, future_relations, past_relations);
end

"""
    generate_KR_poset(n_elements::Integer) -> BitArrayCauset
Generate a KR poset, with the counts of elements in the layers and the links between layers randomly sampled. Returns an adjacency matrix.
"""
function generate_KR_poset(n_elements::Integer; rng::Random.AbstractRNG = Random.default_rng())
    CausalSets.BitArrayCauset(generate_KR_poset_adjacency_matrix(n_elements; rng = rng))
end

"""
    place_causet_in_manifold_causet(manifold_causet::ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}, small_causet::BitArrayCauset)
Places a causet inside a specified boundary of another causet by replacing connection within that region, maintating external connections based on natural labeling, and performing transitive completion.
"""
function place_causet_in_manifold_causet(manifold_causet::CausalSets.ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}, small_causet::CausalSets.BitArrayCauset) where {N}
    # findall returns indices ordered
    in_boundary_indices = findall(CausalSets.is_in_boundary.(Ref(manifold_causet.manifold), Ref(boundary), manifold_causet.sprinkling))
    out_causet = CausalSets.BitArrayCauset(manifold_causet.manifold, manifold_causet.sprinkling)
    # Both the elements in the manifold_causet and the small_causet are already ordered by time coordinate, so this will 
    # match the elements of the n-element top layer with the 
    for (i,idx) in enumerate(in_boundary_indices)
        out_causet.future_relations[idx][in_boundary_indices] = small_causet.future_relations[i];
        out_causet.past_relations[idx][in_boundary_indices] = small_causet.past_relations[i];
    end
    transitive_closure!(out_causet)
    out_causet
end

"""
    place_causet_in_manifold_causet(manifold_causet::ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}, small_causet::BitArrayCauset)
Places a causet inside a specified boundary of another causet by replacing connections within that boundary, maintating external connections based on natural labeling, and performing transitive completion.
"""
function is_subset(test_boundary::OffsetCausalDiamondBoundary, reference_boundary::CausalSets.AbstractBoundary, manifold::CausalSets.MinkowskiManifold)
    return all(CausalSets.is_in_boundary.(Ref(manifold), Ref(reference_boundary), get_tips(test_boundary, manifold)))
end

"""
    replace_region_with_KR_poset(manifold_causet::ManifoldCauset{N}, sprinkling_boundary::AbstractBoundary, element_count::Int64, sprinkling_density::Float64; require_region_fully_in_boundary::Bool=true, return_KR_poset=false)
Generates a random KR poset and places it in a randomly selected boundary of matching element count. By default, the selected region may not overlap with the boundary of the ManifoldCauset. Optionaly returns the generated KR poset as well.
"""
function replace_region_with_KR_poset(manifold_causet::CausalSets.ManifoldCauset{N}, sprinkling_boundary::CausalSets.AbstractBoundary, element_count::Int64, sprinkling_density::Float64; require_region_fully_in_boundary::Bool=true, return_KR_poset=false, rng::Random.AbstractRNG = Random.default_rng()) where {N}
    KR_poset = generate_KR_poset(element_count; rng = rng)
    if require_region_fully_in_boundary
        while true
            center = CausalSets.generate_sprinkling(manifold_causet.manifold, sprinkling_boundary, 1; rng = rng)[1]
            inner_boundary = boundary_from_contained_element_count(manifold_causet, center, element_count, sprinkling_density)
            if is_subset(inner_boundary, sprinkling_boundary, manifold_causet.manifold)
                break
            end
        end
    else
        center = CausalSets.generate_sprinkling(manifold_causet.manifold, sprinkling_boundary, 1; rng = rng)[1]
        inner_boundary = boundary_from_contained_element_count(manifold_causet, center, element_count, sprinkling_density)
    end
    if return_KR_poset
        return place_causet_in_manifold_causet(manifold_causet, inner_boundary::OffsetCausalDiamondBoundary{N}, KR_poset), KR_poset
    else
        return place_causet_in_manifold_causet(manifold_causet, inner_boundary::OffsetCausalDiamondBoundary{N}, KR_poset)
    end
end

"""
    place_causet_in_manifold_causet(manifold_causet::ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}, small_causet::BitArrayCauset)
Generate a random n-element causet with a defect of size m
"""
function generate_causet_with_KR_defect(n::Int64, m::Int64, sprinkling_boundary::CausalSets.AbstractBoundary, manifold::CausalSets.AbstractManifold; rng::Random.AbstractRNG = Random.default_rng())
    sprinkling_density = n/CausalSets.boundary_volume(manifold, sprinkling_boundary)
    manifold_causet = CausalSets.ManifoldCauset(manifold, CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng))
    combined_causet = replace_region_with_KR_poset(manifold_causet, sprinkling_boundary, m, sprinkling_density; rng = rng)
    return combined_causet
end

# example
#=
manifold = MinkowskiManifold{4}()
sprinkling_boundary = CausalDiamondBoundary{4}(1.)

generate_causet_with_KR_defect(2048, 256, sprinkling_boundary, manifold)
=#
