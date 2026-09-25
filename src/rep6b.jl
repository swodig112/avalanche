# %%
include("utils.jl")
include("avalib.jl")
using .utils: geometric_bernoulli
import .Avalanche
av = Avalanche
using GLMakie
using ProgressMeter
HISTSIZE = 30

function sample_avalanche(N, R, P)
    avch = zeros(P)
    @showprogress for i=1:P
        D = 1
        size = 1
        for level=1:N-1
            p = N / (R * (N - level) + N)
            Dp = 0
            for j=1:D
                Dp += geometric_bernoulli(p)
            end
            D = Dp
            size += D
            
        end
        avch[i] = size
    end
    avch
end
N = [2000, 500, 500, 500]
R = Float32[1.01f0, 1.01f0, 1.06f0, 1.1f0]
totres = []

for i=eachindex(N)
    res = sample_avalanche(N[i], R[i], 1000000)
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















