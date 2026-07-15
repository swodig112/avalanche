ENV["DISPLAY"]=":0"
using LinearAlgebra
using Distributed
using PyPlot
using StatsBase
using LsqFit
using Statistics


const N_AVCH = 1000000
const N = 800
const α = 1.0
const w = α / N


if nprocs() == 1
    addprocs(Sys.CPU_THREADS, exeflags="--project")
end
# %% Initializing processes
@everywhere begin
    using Random
    using ProgressMeter

    function evolve_nodes(nodes, weight, α_param, dt=0.01)
        n_nodes = length(nodes)
        A = sum(nodes)
        nodes_new = copy(nodes)
        turnons = 0
        for i in 1:n_nodes
            p_flip = ((weight * A * (1 - nodes[i])) + (α_param * nodes[i])) * dt
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
end

# %% Running the Simulation
println("Starting simulation with $(nworkers()) workers for $(N_AVCH) avalanches...")
avch = @showprogress pmap(i -> direct_sample_avalanche_size(N, w, α, 1, 0.01), 1:N_AVCH)
println("Simulation complete. Analyzing and plotting results...")

# %% Plotting
# Binning
min_size = minimum(filter(s -> s > 0, avch))
max_size = maximum(avch)
bin_edges = 10 .^ range(log10(min_size), stop=log10(max_size), length=200)
h_raw = fit(Histogram, avch, bin_edges)
h_norm = normalize(h_raw, mode=:probability)

counts = h_norm.weights
bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

valid_indices = findall(counts .> 0)
x_data = bin_centers[valid_indices]
y_data = counts[valid_indices]

# Plotting
fx_plot = 10 .^ range(log10(minimum(x_data)), stop=log10(maximum(x_data)), length=100)
using SpecialFunctions
fy_plot = fx_plot .^ (-3/2) / zeta(3/2)

plot(x_data, y_data)
xscale("log")
yscale("log")
scatter(x_data, y_data, label="Simulated Data")
xlabel("Avalanche Size")
ylabel("Probability Function")
title("Simulation of Avalanches")
plot(fx_plot, fy_plot, label=L"Power Law: $P(n)=\zeta^{-1}(3/2)n^{-3/2}$", linewidth=2)
legend()
# rmprocs(workers())
