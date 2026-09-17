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
R         = 1.0f0
NBATCH     = 10
BATCHSIZE = 1000000
N         = [100, 200, 300]
w         = 1
histsize  = 30
backend = oneAPIBackend()

# %%
samples = zeros(UInt64, (length(N), NBATCH * BATCHSIZE))

for i=1:length(N)
    println("Running simulation for N = ",N[i])
    @showprogress for j=1:NBATCH
        # println("Running Batch ", j)
        res = ag.gfulav(BATCHSIZE, N[i], R, 1, oneArray, backend)
        # println(size(res))

        
        samples[i,(j-1)*BATCHSIZE+1:j*BATCHSIZE] .= res[:]
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
    ylabel = L"\text{Probability} \times s^{3/2}",
    title  = "Finite Size Effect for R=$(R)",
    # autolimitmargin = (0.1, 0.1)
)


τ = Observable(1.5)
D = Observable(0.0)

# xp = @lift [x[i] ./ (N[i] ^ ($(D) * 0)) for i=1:length(x)]
# yp = @lift [y[i] .* (x[i] .^ $(τ)) for i=1:length(x)]
# sl_τ = Textbox(fig[2, 1], placeholder = "τ",
#                validator = Float64, tellwidth = false)
# sl_D = Textbox(fig[3, 1], placeholder = "D",
#                validator = Float64, tellwidth = false)
# on(sl_τ.stored_string) do s
#     τ[] = parse(Float64, s)
# end
# on(sl_D.stored_string) do s
#     D[] = parse(Float64, s)
# en


# sl_τ = Slider(fig[2,1], range = 0.0:0.1:5,
              # start=1.5,
              # width = 400
              # )
# sl_D = Slider(fig[3,1], range = -5.0:0.1:5,
              # start=0.0,
              # width = 400
              # )
sg = SliderGrid(fig[2, 1],
                (label = "τ", range = 0:0.001:10, startvalue = 1.5),
                (label = "D", range = 0:0.001:5, startvalue = 0),
)

on(sg.sliders[1].value) do val
    τ[] = val
end
on(sg.sliders[2].value) do val
    D[] = val
end


fx_plot = Avalanche.logspace(1, maximum(samples[end,:]), histsize)
using SpecialFunctions
fy_plot = fx_plot .^ (-3/2) / zeta(3/2)
fx = @lift [fx_plot[i] / (N[1]^$(D)) for i in eachindex(fx_plot)]
fy = @lift [fy_plot[i] * (fx_plot[i] ^ $(τ)) for i in eachindex(fy_plot)]
lines!(
    ax, fx, fy,
    label = "Power Law",
    color = "grey",
    linestyle = :dash
)


for i in eachindex(x)
    x_i = @lift [ x[i][j] / (N[i] ^ $(D)) for j in eachindex(x[i]) ]
    y_i = @lift [ y[i][j] * (x[i][j] ^ $(τ)) for j in eachindex(y[i]) ]
    scatterlines!(
        ax,
        x_i,
        y_i,
        marker = :x,
        label  = "N=$(N[i])"
    )
end

axislegend(ax)
display(fig)
