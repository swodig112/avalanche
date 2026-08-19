module utils
using LinearAlgebra
using SparseArrays
using Random

function random_network(N, p; directed=false)
    p /= N
    if !directed
        p /= 2
    end
    A = sprand(Float64, N, N, p)
    # A = round
    A[diagind(A)] .= 0
    if !directed
        A = triu(A, 1)
        A = A + A'
    end
    return A
end
end
