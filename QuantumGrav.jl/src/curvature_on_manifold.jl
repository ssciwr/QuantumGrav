function chebyshev_derivation_matrix(
max_cheb_order::Int64,
derivative_order::Int64
)
    transformation_matrix = zeros(max_cheb_order + 1, max_cheb_order + 1)

    for n in 0 : max_cheb_order

        for k in mod(n-derivative_order, 2) : 2 : n - derivative_order
            prefac = k == 0 ? 0.5 : 1

            transformation_matrix[k+1, n+1] += 2^derivative_order * n * prefac * 
                                                binomial(Int64((n + derivative_order - k) / 2 - 1), Int64((n - derivative_order - k) / 2)) * 
                                                factorial(Int64((n + derivative_order + k) / 2 - 1)) / 
                                                factorial(Int64((n - derivative_order + k) / 2))
        end
    end

    return transformation_matrix
end