module Avalanche
using LinearAlgebra
using Random
using ProgressMeter
using Distributed
using PyPlot
using StatsBase
using LsqFit
using Statistics

function evolve_nodes(nodes, weight, α_param, dt=0.01)
    n_nodes = length(nodes)
    A = sum(nodes)
    nodes_new = copy(nodes)
    turnons = 0
    for i in 1:n_nodes
        p_flip = ((weight / n_nodes * A * (1 - nodes[i])) + (α_param * nodes[i])) * dt
        if rand() < p_flip
            nodes_new[i] = 1 - nodes_new[i]
        end
    end
    for i in 1:n_nodes
        if nodes_new[i] == 1 && nodes[i] == 0
            turnons += 1
        end
    end
    return nodes_new, turnons
end

function direct_sample_avalanche_size(N, weight, α_param, initial, dt)
    nodes = zeros(Int, N)
    nodes[1:initial] .= 1
    avchsize = initial
    while sum(nodes) > 0
        nodes, new_turnons = evolve_nodes(nodes, weight, α_param, dt)
        avchsize += new_turnons
    end
    return avchsize
end

function gillespie_sample_avalanche_size(N, R0, initial)
    q(i) = (N)/(R0 * (N - i) + N)
    A = initial
    turnons = 0
    while A > 0
        if rand() < q(A) || A >= N
            A -= 1
        else
            A += 1
        end
        turnons += 1
    end
    turnons
end

function logspace(a, b, length=50)
    10 .^ range(log10(a), log10(b), length=length)
end

function probability_from_samples(avch, length)
    min_size = minimum(filter(s -> s > 0, avch))
    max_size = maximum(avch)
    bin_edges = logspace(min_size, max_size, length)
    h_raw = fit(Histogram, avch, bin_edges)
    h_norm = normalize(h_raw, mode=:probability)

    counts = h_norm.weights
    bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

    valid_indices = findall(counts .> 0)
    x_data = bin_centers[valid_indices]
    y_data = counts[valid_indices]
    x_data, y_data
end

function generate_time_evolution_data(N, weight, α_param, initial, dt, T)
    nodes = zeros(Int, N)
    nodes[1:initial] .= 1
    tdata = zeros(Int, Int(T/dt))
    for i=1:Int(T/dt)
        nodes, _ = evolve_nodes(nodes, weight, α_param, dt)
        tdata[i] = sum(nodes)
    end
    tdata
end

function build_tridiag(f1,f2,f3,N)
    d1 = [f1(i) for i in 1:N-1]
    d2 = [f2(i) for i in 1:N]
    d3 = [f3(i) for i in 1:N-1]
    Tridiagonal(d1, d2, d3)
end


function exactp(R0, N)
    # Using this function isn't recommended unless you only
    # want to calculate P for a few values. If you want to plot
    # (e.g. need to calculate many Ps) use probability_sequence
    # below
    q(i) = (N)/(R0 * (N - i) + N)
    A = build_tridiag(i -> 1-q(i), i->0, i -> q(i+1), N)
    p0 = zeros(N)
    p0[1] = 1
    function P(k)
        # q(1) * p0' * A^(2*(floor(k)-1)) * p0
        q(1) * p0' * matrix_power(A,2*(floor(k)-1))  * p0
    end
end

function probability_sequence(R0, N, x)
    # Here I calculate P as i'm exponentiating A
    # so it's greatly faster than exactp
    q(i) = (N)/(R0 * (N - i) + N)
    A = build_tridiag(i -> 1-q(i), i->0, i -> q(i+1), N)
    p0 = zeros(Float64, N)
    p0[1] = 1
    A2 = A^2
    xp = 1
    res = zeros(Float64, length(x))
    for i in 1:length(x)
        p0 = A2^(floor(x[i])-xp) * p0
        xp = floor(x[i])
        res[i] = q(1) * p0[1]
    end
    res
end


end
