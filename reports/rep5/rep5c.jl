# %% Imports & Initializations
include("./utils.jl")
include("./avalib.jl")
include("./avagpulib.jl")

import .utils
import .Avalanche as av
import .AvalancheGPU as ag
using oneAPI
using ProgressMeter
using Distributed
using GLMakie

function initialize_workers()
    if nprocs() == 1
        addprocs(Sys.CPU_THREADS, exeflags="--project")
    end
    @sync for p in workers()
        @async remotecall_wait(p) do
            include("avalib.jl")
        end
    end
end

function distributed_gillespie_sampling(N,R,initial,NAVCH)
    initialize_workers()
    avch = @showprogress pmap(i -> Avalanche.gillespie_sample_avalanche_size(N,R,initial), 1:NAVCH)
end

# %% Constants
R         = 1
NBATCH     = 10
BATCHSIZE = 10000
N         = 800
NAVCH     = 100000
w         = 1
histsize  = 50
backend = oneAPIBackend()
p         = 1.0

# %% Simulations

samples = zeros(Int64, (3, NAVCH))

samples[1,:] .= distributed_gillespie_sampling(N,R  ,1,NAVCH)

α      = zeros(Float32, N)
α     .= p * N * w / R 
α      = α                            |> oneArray
for j=1:NBATCH
    println("Running Batch", j)
    nodes = zeros(UInt64, (N,BATCHSIZE))
    for j=1:BATCHSIZE
        nodes[rand(1:N),j] = one(UInt8)
    end
    nodes  = nodes                        |> oneArray
    adjmat = utils.erdos_reyni(N, p)  .|> Float32 |> oneArray
    res = ag.gpu_sample_avalanche(nodes, α, adjmat)
    samples[2,(j-1)*BATCHSIZE+1:j*BATCHSIZE] .= res[1, :]
end

@showprogress for j=1:NBATCH
    res = ag.gfulav(BATCHSIZE, N, 1, 1, oneArray, backend)
    samples[3,(j-1)*BATCHSIZE+1:j*BATCHSIZE] .= res[:]
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
    title  = "A Comparison of Simulation Methods to Check Their Validity"
)
fx_plot = Avalanche.logspace(1, maximum(samples[end,:]), histsize)
using SpecialFunctions
fy_plot = fx_plot .^ (-3/2) / zeta(3/2)
lines!(
    ax,
    fx_plot, fy_plot,
    label = "Power Law",
    color = "grey",
    linestyle = :dash
)
for i=1:length(x)
    scatter!(
        ax, x[i], y[i],
        marker = :x, markersize=15
    )
end
axislegend(ax)
display(fig)
