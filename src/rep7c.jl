# %% Initialization

using GLMakie
import SymbolicRegression: SRRegressor, machine, fit!, predict, report

include("avalib.jl")
include("vislib.jl")
av = Avalanche

# %% Constants
N = Float64[100, 200, 400, 800, 1500]
R = [1.0461524997991367, 1.0297881371325264, 1.020999827682283, 1.0130849440603604, 1.01]

# N = log.(N)
# N = @. N - 1
R = @. R - 1
# R = log.(R)
# %% Fitting

model = SRRegressor(
    niterations=50,
    binary_operators=[+, -, *, /],
    unary_operators=[exp, log],
)
mach = machine(model, N, R)
fit!(mach)

# %% Prepare for Plot
X = range(minimum(N), maximum(N), 100) |> Array
Y = predict(mach, (data = X, idx=4))

# %% Plot
# fig1 = vis.avalanche_plot(N, resx, resy, 1.5, 1.0)
fig = Figure()
ax = Axis(
    fig[1, 1],
    # xscale = log10,
    # yscale = log10,
    xlabel = "N",
    ylabel = "1/R",
)


scatterlines!(
    ax,
    N,
    R,
    # marker = :x,
    label = "Original Data",
)

for i=3:6
    println("Eq$(i): R = $(replace(mach.report.equation_strings[i],
        "x1" => "N"))")
    lines!(ax,
           X,
           predict(mach, (data=X, idx=i)),
           label = "Eq$(i)"
           )
end

axislegend()
display(fig)
