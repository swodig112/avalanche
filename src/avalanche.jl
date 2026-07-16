using ProgressMeter
using Distributed
using PyPlot

include("avalib.jl")
using .Avalanche

function distributed_direct_sampling(N, weight, α, initial, dt, N_AVCH)
    if nprocs() == 1
        addprocs(Sys.CPU_THREADS, exeflags="--project")
    end
    @sync for p in workers()
        @async remotecall_wait(p) do
            include("avalib.jl")
        end
    end
    avch = @showprogress pmap(i -> Avalanche.direct_sample_avalanche_size(N, weight, α, 1, 0.01), 1:N_AVCH)
end

# %% Set Constants
T = 10
N = 800
α = 1
initial = 1
dt = 0.01
N_AVCH = 100000
histsize = 500

### First Graph

# %% Run Simulation
a1 = distributed_direct_sampling(N, α, α, initial, dt, N_AVCH)
a2 = distributed_direct_sampling(N, 1.1 * α, α, initial, dt, N_AVCH)
a3 = distributed_direct_sampling(N, 0.9 * α, α, initial, dt, N_AVCH)

# %% Prepare for Plotting
x1, y1 = Avalanche.probability_from_samples(a1, histsize)
x2, y2 = Avalanche.probability_from_samples(a2, histsize)
x3, y3 = Avalanche.probability_from_samples(a3, histsize)

# %% Plot
xscale("log")
yscale("log")
fx_plot = Avalanche.logspace(1, 10e2, histsize)
using SpecialFunctions
fy_plot = fx_plot .^ (-3/2) / zeta(3/2)
plot(fx_plot, fy_plot, color="pink", label=L"Power Law: $P(n)=\zeta^{-1}(3/2)n^{-3/2}$", linewidth=2)
plot(x1, y1, color="green", marker="x", linestyle="", label="Critical R=1")
plot(x2, y2, color="red", marker="x", linestyle="", label="Supercritical R>1")
plot(x3, y3, color="blue", marker="x", linestyle="", label="Subcritical R<1")
xlabel("Avalanche Size")
ylabel("Probability Function")
title("Simulation of Avalanches")
legend()
# rmprocs(workers())

### Second Graph
# %% Simulate

f(A, R, t) = (R*α*A/N) * (N - A) - α  * A
A0 = 0.25 * N
tspan = (0, T)
prob = DE.ODEProblem(f, A0, tspan, 1)
sol4 = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)
prob = DE.ODEProblem(f, A0, tspan, 1.3)
sol5 = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)
prob = DE.ODEProblem(f, A0, tspan, 0.7)
sol6 = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)
y4 = Avalanche.generate_time_evolution_data(N, α,α,200,dt,T)
y5 = Avalanche.generate_time_evolution_data(N, 1.3* α,α,200,dt,T)
y6 = Avalanche.generate_time_evolution_data(N, 0.7*α,α,200,dt,T)
td = range(0, T, Int(T/dt))
plot(td, y4, "g", label="Critical Simulation")
plot(td, y5, "r", label="Supercritical Simulation")
plot(td, y6, "b", label="Subcritical Simulation")
plot(sol4.t,sol4.u, color="g", linestyle="--", label="Critical ODE")
plot(sol5.t,sol5.u, color="r", linestyle="--", label="Supercritical ODE")
plot(sol6.t,sol6.u, color="b", linestyle="--", label="Subcritical ODE")
xlabel("Time")
ylabel("Active Neurons")
title("Time Evolution of The System")
legend()
savefig("fig3.png")
