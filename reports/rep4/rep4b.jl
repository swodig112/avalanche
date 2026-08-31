# %% Initializing
ENV["DISPLAY"] = ":0"
using ProgressMeter
using Distributed
using GLMakie
using Makie

include("avalib.jl")
include("utils.jl")
using .Avalanche
using .utils

function initialize_workers()
    if nprocs() == 1
        addprocs(Sys.CPU_THREADS, exeflags="--project")
    end
    @sync for p in workers()
        @async remotecall_wait(p) do
            include("avalib.jl")
            include("utils.jl")
        end
    end
end

function distributed_gillespie_sampling(N,α,p,initial,NAVCH)
    avch = @showprogress pmap(i -> begin
                                  A = utils.erdos_reyni(N, p);
                                  Avalanche.gillespie_sample_avalanche_size(A,α .* 100, initial);
                              end, 1:NAVCH)
end
initialize_workers()
# %% Set Constants
N = 800
p = [0.01, 0.05]#, 0.1, 0.99]
#p = [1]
samples = []
α = ones(N)
initial = 1
N_AVCH = 100_000
histsize = 50

# %% Run gillespie Simulation
@time for i=p
    # a1 = distributed_gillespie_sampling(N,subc,1,N_AVCH)
    # a2 = distributed_gillespie_sampling(N,1  ,1,N_AVCH)
    # a3 = distributed_gillespie_sampling(N,supc,1,N_AVCH)
    println("Simulating erdos reyni for p =", i)
    push!(samples, distributed_gillespie_sampling(N, α, i, () -> 1,N_AVCH))
end
# %% Prepare for Plotting
x = []
y = []
for i=samples
    xt, yt = Avalanche.probability_from_samples(i, histsize)
    push!(x, xt)
    push!(y, yt)
end
# x1, y1 = Avalanche.probability_from_samples(a1, histsize)
# x2, y2 = Avalanche.probability_from_samples(a2, histsize)
# x3, y3 = Avalanche.probability_from_samples(a3, histsize)

# %% Plot
fig = Figure()
ax = Axis(
    fig[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = "Avalanche Size",
    ylabel = "Probability"
)
fx_plot = Avalanche.logspace(1, maximum(samples[end]), histsize)
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
