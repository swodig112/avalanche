# %% Initialization

using GLMakie

include("avalib.jl")
include("vislib.jl")
av = Avalanche

# %% Constants
BATCHSIZE = 1_000_000
HISTSIZE  = 30

bin_edges = av.logspace(1, 10000, 16)
bin_edges = unique(round.(Int, bin_edges))
bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])

N = [200, 400, 800, 1600]#, 500, 500]
R = [1.0, 1.0, 1.0, 1.0]#, 1.06, 1.1]

# %% Simulation
samples = []

for i=eachindex(N)
    sample = av.negbin_sample_avalanche(N[i], R[i], BATCHSIZE)
    push!(samples, sample)
end

# %% Calculating PDF
resx = []
resy = []
for i=eachindex(N)
    res = av.probability_from_samples(samples[i], HISTSIZE)
    push!(resx, res[1])
    push!(resy, res[2])
end

# %% Plot
fig = vis.avalanche_plot(N, resx, resy, 1.5, 1.0)
