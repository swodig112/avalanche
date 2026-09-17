module AvalancheGPU
using Random
using KernelAbstractions
import oneAPI: oneArray, synchronize
using PhiloxRNG
using Atomix
using ProgressMeter
include("utils.jl")
using .utils

SEED = 0

@inline function grand(a, b, c)
    randu01_f32(UInt64(a), UInt64(b), UInt64(c))[1]
end

@inline function matvec!(A, x, i, I, N)
    s = 0.0f0
    for j in 1:N
        s += A[i, j] * x[j,I]
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

@kernel function gillespie_sample_avalanche_size(adjmat, α, nodes, p, avchsize, active, counter, randstep, SEED)
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
    while sumons(nodes, I, N) > UInt64(0) && J < 100
        idx = sample_idx(p, grand(I,J + 100 * randstep, SEED), I, N)
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
        # p .= adjmat * nodes[:,I]
        # p[nodes == 1] .= α[nodes == 1]
    end
    if sumons(nodes, I, N) == 0
        active[I] = UInt8(0)
        Atomix.@atomic counter[1] -= Int32(1)
    end
    end
end

function gpu_usage_example(N, P)
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
    adjmat   = Float32.(utils.barabasi_albert(N,50)) |> oneArray
    counter  = Int32[P]                             |> oneArray
    # adjmat   = Float32.(utils.erdos_reyni(N,0.1))   |> oneArray

    # prog = Progress(P)
    randstep = 0
    while true
        gillespie_sample_avalanche_size(backend, 64)(adjmat, α, nodes, p, avchsize, active, counter, randstep, SEED; ndrange=P)
        synchronize()
        remaining = Int(Array(counter)[1])
        completed = P - remaining

        # update!(prog, completed)

        remaining == 0 && break
        randstep += 1
    end
    # finish!(prog)
    Array(avchsize)
end

function gpu_sample_avalanche(nodes, α, adjmat)
    backend  = get_backend(nodes)
    N, P     = size(nodes)
    p        = zeros(Float32, (N, P)) |> oneArray
    avchsize = sum(nodes, dims=1)     |> oneArray
    active   = ones(UInt8, P)         |> oneArray
    counter  = Int32[P]               |> oneArray

    prog = Progress(P)
    randstep = 0
    while true
        gillespie_sample_avalanche_size(backend, 512)(adjmat, α, nodes, p, avchsize, active, counter, randstep, SEED; ndrange=P)
        synchronize()
        remaining = Int(Array(counter)[1])
        completed = P - remaining

        update!(prog, completed)

        remaining == 0 && break
        randstep += 1
    end
    finish!(prog)
    Array(avchsize)
end

function gpu_sample_fully_connected_network_avalanche_size(batchsize, N, R, initial, gpuarrf, backend)
    avchsize = zeros(UInt64, batchsize)
    avchsize .= initial
    A = deepcopy(avchsize)
    avchsize = avchsize |> gpuarrf
    A = A |> gpuarrf
    # while true
    gpu_sample_fully_connected_network_avalanche_size_kernel(backend, 512)(avchsize, A, N, R, initial, SEED; ndrange=batchsize)
    synchronize()
    # end
    Array(avchsize)
end

gfulav = gpu_sample_fully_connected_network_avalanche_size

@kernel function gpu_sample_fully_connected_network_avalanche_size_kernel(avchsize, A, N, R, initial, seed)
    I = @index(Global)
    # turnons = initial
    J = 0
    while A[I] > 0# && J < 100
        if grand(J, I, seed) < N*inv(Float32(R * (N - A[I]) + N))
            A[I] -= UInt64(1)
        else
            avchsize[I] += UInt64(1)
            A[I] += UInt64(1)
        end
        J += 1
    end
end
end
