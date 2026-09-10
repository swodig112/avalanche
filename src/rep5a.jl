# %% Imports
include("./utils.jl")
include("./avalib.jl")
include("./avagpulib.jl")

import .utils
import .Avalanche as av
import .AvalancheGPU as ag
using oneAPI
using ProgressMeter
using GLMakie

# %% Constants
R         = 1
NBATCH     = 1
BATCHSIZE = 100000
N         = 800
w         = 1
p         = Float32[1.0, 0.5, 0.1]
histsize  = 30

# %%
samples = zeros(UInt64, (length(p), NBATCH * BATCHSIZE))

for i=1:length(p)
    println("Running simulation for p=",p[i])
    α      = zeros(Float32, N)
    α     .= p[i] * N * w / R 
    α      = α                            |> oneArray
    for j=1:NBATCH
        println("Running Batch", j)
        nodes = zeros(UInt64, (N,BATCHSIZE))
        for j=1:BATCHSIZE
            nodes[rand(1:N),j] = one(UInt8)
        end
        nodes  = nodes                        |> oneArray
        adjmat = utils.erdos_reyni(N, p[i])  .|> Float32 |> oneArray
        # println("Running batch ", j)
        res = ag.gpu_sample_avalanche(nodes, α, adjmat)
        samples[i,(j-1)*BATCHSIZE+1:j*BATCHSIZE] .= res[1, :]
    end
end

# %% Prepare for Plotting
x = []
y = []
for i=1:size(samples)[1]
    xt, yt = Avalanche.probability_from_samples(samples[i,:], histsize)
    push!(x, xt)
    push!(y, yt)
end


# %% Plot
fig = Figure()
ax = Axis(
    fig[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = "Avalanche Size",
    ylabel = "Probability",
    title  = "Erdos-Reyni constant R"
)
fx_plot = Avalanche.logspace(1, maximum(samples[end,:]), histsize)
using SpecialFunctions
fy_plot = fx_plot .^ (-3/2) / zeta(3/2)
lines!(
    ax,
    fx_plot, fy_plot,
    label = "Power Law"
)
for i=1:length(x)
    scatter!(
        ax, x[i], y[i],
        marker = :x, markersize=15,
        label = "p = $(p[i])"
    )
end
axislegend(ax)
display(fig)
