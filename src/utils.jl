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

function barabasi_albert(N::Int, m::Int; rng=Random.default_rng())
    @assert N > m ≥ 1

    A = zeros(Int8, N, N)
    @inbounds for i in 1:(m + 1)
        for j in (i + 1):(m + 1)
            A[i, j] = 1
            A[j, i] = 1
        end
    end

    targets = Int[]
    @inbounds for i in 1:(m + 1)
        for _ in 1:m
            push!(targets, i)
        end
    end

    @inbounds for new_node in (m + 2):N
        selected = Set{Int}()
        while length(selected) < m
            target = rand(rng, targets)
            push!(selected, target)
        end
        for target in selected
            A[new_node, target] = 1
            A[target, new_node] = 1
            push!(targets, target)
            push!(targets, new_node)
        end
    end

    return A
end
end
