using TestItems

@testsnippet importModules begin
    import QuantumGrav
    import CausalSets
    import SparseArrays
    import Distributions
end

@testsnippet makeData begin
    import CausalSets
    import QuantumGrav
    import SparseArrays
    import Distributions

    function MockData(n)
        manifold = CausalSets.MinkowskiManifold{2}()
        boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
        sprinkling = CausalSets.generate_sprinkling(
            manifold, boundary, Int(n))
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset
    end

    cset_empty = MockData(0)

    cset_links = MockData(100)
end
