# %% 
using GLMakie
using oneAPI
include("optlib.jl")
include("avalib.jl")
include("avagpulib.jl")

ao = avagpuopt
av = Avalanche
ag = AvalancheGPU

BATCHSIZE = 1_000_000
BACKEND   = oneAPIBackend()
HISTSIZE  = 30

bin_edges = av.logspace(1, 10000, 16)
bin_edges = unique(round.(Int, bin_edges))
bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])


# ref_samp  = ag.gfulav(BATCHSIZE, REFN, REFR, INITIAL, GPUARRF, BACKEND)
# ref_pdf   = av.probability_from_samples(ref_samp, bin_edges)[1]
# %%
N = [2000, 500, 500, 500]
R = Float32[1.01f0, 1.01f0, 1.06f0, 1.1f0]
totres = []

for i=eachindex(N)
    ag.SEED += 1
    res = ag.gfulav(BATCHSIZE, N[i], R[i], 1, oneArray, BACKEND)
    push!(totres,res)

end

# %% Prepare for plot
resx = []
resy = []
for i=eachindex(N)
    res = av.probability_from_samples(totres[i], HISTSIZE)
    push!(resx, res[1])
    push!(resy, res[2])
end
# %% Plot
fig = Figure()
ax = Axis(
    fig[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = "Avalanche Size",
    ylabel = "Probability",
)
# fx_plot = Avalanche.logspace(1, maximum(samples[end,:]), HISTSIZE)
# using SpecialFunctions
# fy_plot = fx_plot .^ (-3/2) / zeta(3/2)
# lines!(
    # ax,
    # fx_plot, fy_plot,
    # label = "Power Law"
# )
for i=eachindex(resx)
    scatter!(
        ax, resx[i], resy[i],
        marker = :x, markersize=15,
        label = "N=$(N[i]), R=$(R[i])"
    )
end
axislegend(ax)
display(fig)

