module AvalancheGPU
using Random
using KernelAbstractions
import oneAPI: oneArray, synchronize
using PhiloxRNG
using Atomix
include("utils.jl")
using .utils

SEED = 0

@inline function grand(a, b)
    randu01_f32(UInt64(a), UInt64(b), UInt64(0))[1]
end

@inline function calp!(a,b,c,N)
    for i=1:N
        c[i] = 0
        for j=1:N
            c[i] += a[i][j] * b[j]
        end
    end
end

@inline function matvec!(A, x, i, I, N)
    s = 0.0f0
    for j in 1:N
        s += A[i, j] * x[j]
    end
    return s
end

@inline function sumons(nodes, I, N)
    s = 0f0
    for i=1:N
        s += nodes[i, I]
    end
    s
end

@inline function sample_idx(p, r, I, N)
    total = 0f0
    for i=1:N
        total += p[i, I]
    end
    target = r * total
    cumsum = 0f0
    for i=1:N
        cumsum += p[i, I]
        if cumsum >= target
            return i
        end
    end
    return N
end

@kernel function gillespie_sample_avalanche_size(adjmat, α, nodes, p, avchsize)
    I = @index(Global)
    N = length(α)
    for i=1:N
        if nodes[i,I] == 1
            p[i,I] = α[i]
        else
            p[i,I] = matvec!(adjmat, nodes, i, I, N)
        end
    end
    J = 0
    while sumons(nodes, I, N) > 0f0 && J < 100
        idx = sample_idx(p, grand(I,J), I, N)
        J += 1
        if nodes[idx, I] == 1f0
            nodes[idx,I] = 0f0
            for i in 1:N
                if i != idx && nodes[i,I] == 0f0
                    p[i,I] -= adjmat[i,idx]
                end
            end

            p[idx,I] = matvec!(adjmat, nodes, idx, I, N)
            # p[idx,I] = matvec!(adjmat, nodes, idx, I, N)
            # for i in 1:N
                # if nodes[i, I] == 0f0
                    # p[i,I] -= adjmat[i,idx]
                # end
            # end
        else
            nodes[idx,I] = 1f0
            avchsize[I] += UInt64(1)
            p[idx,I] = α[idx]
            for i in 1:N
                if nodes[i, I] == 0f0
                    p[i,I] += adjmat[i,idx]
                end
            end
        end
    end
end

function simulate(N, P)
    # P = 1000
    # N = 100
    nodes = zeros(Float32, (N,P))
    for i=1:P
        nodes[rand(1:N),i] = 1f0
    end
    nodes = oneArray(nodes)
    adjmat = oneArray(Float32.(utils.barabasi_albert(N,8)))
    # adjmat = oneArray(Float32.(utils.erdos_reyni(N,0.1)))
    α = oneArray(ones(Float32, N) .* 100)
    p = oneArray(zeros(Float32,(N,P)))
    avchsize = oneArray(ones(UInt64, P))
    backend = get_backend(nodes)
    s = 1
    # III = UInt32(0)
    while s > 0f0
        synchronize()
        s = sum(Array(nodes))
        println(s > 0f0)
        gillespie_sample_avalanche_size(backend, 256)(adjmat, α, nodes,p,avchsize; ndrange=P)
        # III += UInt32(1)
    end
    synchronize()
    return Array(avchsize)
end

@kernel function gillespie_sample_avalanche_size2(adjmat, α, nodes, p, avchsize, active, counter)
    I = @index(Global)
    if active[I] != UInt8(0)
    N = length(α)
    for i=1:N
        if nodes[i,I] == 1
            p[i,I] = α[i]
        else
            p[i,I] = matvec!(adjmat, nodes, i, I, N)
        end
    end
    J = 0
    while sumons(nodes, I, N) > 0f0 && J < 100
        idx = sample_idx(p, grand(I,J), I, N)
        J += 1
        if nodes[idx, I] == 1f0
            nodes[idx,I] = 0f0
            for i in 1:N
                if i != idx && nodes[i,I] == 0f0
                    p[i,I] -= adjmat[i,idx]
                end
            end

            p[idx,I] = matvec!(adjmat, nodes, idx, I, N)
        else
            nodes[idx,I] = 1f0
            avchsize[I] += UInt64(1)
            p[idx,I] = α[idx]
            for i in 1:N
                if nodes[i, I] == 0f0
                    p[i,I] += adjmat[i,idx]
                end
            end
        end
    end
    if sumons(nodes, I, N) == 0
        active[I] = UInt8(0)
        Atomix.@atomic counter[1] -= Int32(1)
    end
    end
end

function simulate2(N, P)
    nodes = zeros(UInt8, (N,P))
    for i=1:P
        nodes[rand(1:N),i] = one(UInt8)
    end
    nodes    = nodes                                |> oneArray
    backend  = get_backend(nodes)
    α        = ones(Float32, N) .* 100              |> oneArray
    p        = zeros(Float32,(N,P))                 |> oneArray
    avchsize = ones(UInt64, P)                      |> oneArray
    active   = ones(UInt8, P)                       |> oneArray
    adjmat   = Float32.(utils.barabasi_albert(N,8)) |> oneArray
    counter  = Int32[P]                             |> oneArray
    # adjmat   = Float32.(utils.erdos_reyni(N,0.1))   |> oneArray
    while Array(counter)[1] > Int32(0)
        gillespie_sample_avalanche_size2(backend, 256)(adjmat, α, nodes, p, avchsize, active, counter; ndrange=P)
        synchronize()
    end
    Array(avchsize)
end
end
