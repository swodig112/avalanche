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

function erdos_reyni(N, p)
    A = zeros(Float64, N, N)
    for i in 2:N
        for j in 1:i-1
            if rand() < p
                A[i, j] = 1
                A[j, i] = 1
            end
        end
    end
    A
end

end
