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

# N = [1500, 900, 500, 300, 100]
# R = [1.1, 1.1318701771910662, 1.1836170746889176,   1.2445496208820592, 1.4814879672151118]
N = [100, 200, 400, 800, 1500]
R = [1.0461524997991367, 1.0297881371325264, 1.020999827682283, 1.0130849440603604, 1.01]

# IDX = 1
# IDY = 5
# b = log(R[5] / R[IDX]) / log(N[IDX] / N[5])
# R_new = R[IDX] .* (N ./ N[IDX]).^(-b)
# R = R_new

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
fig1 = vis.avalanche_plot(N, resx, resy, 1.5, 1.0)
fig2 = Figure()
ax = Axis(
    fig2[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = "N",
    ylabel = "1/R",
)
# for i=eachindex(N)

scatterlines!(
    ax,
    N,
    1 ./ R,
    # marker = :x,
)
# screen1 = GLMakie.Screen()
screen2 = GLMakie.Screen()

# GLMakie.display(screen1, fig1)
GLMakie.display(screen2, fig2)
# display(fig2)
# end
