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


end
