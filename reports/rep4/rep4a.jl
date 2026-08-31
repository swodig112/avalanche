# %% Initializing
ENV["DISPLAY"] = ":0"
using ProgressMeter
using Distributed
using GLMakie
using Makie

include("avalib.jl")
using .Avalanche

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
    # initialize_workers()
    avch = @showprogress pmap(i -> Avalanche.gillespie_sample_avalanche_size(N,R,initial), 1:NAVCH)
end
initialize_workers()
# %% Set Constants
N = [1e1, 1e2, 1e3, 1e4, 1e8]
samples = []
α = 1
initial = 1
N_AVCH = 1_000_000
histsize = 50

# %% Run gillespie Simulation
@time for i=N
    # a1 = distributed_gillespie_sampling(N,subc,1,N_AVCH)
    # a2 = distributed_gillespie_sampling(N,1  ,1,N_AVCH)
    # a3 = distributed_gillespie_sampling(N,supc,1,N_AVCH)
    println("Simulating for N =", i)
    push!(samples, distributed_gillespie_sampling(i,1  ,1,N_AVCH))
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
for i=1:length(N)
    scatter!(
        ax, x[i], y[i],
        marker = :x, markersize=15,
        label = "N = 1e$(Int(log10(N[i])))"
    )
end
axislegend(ax)
display(fig)
