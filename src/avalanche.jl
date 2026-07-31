# %% Initializing
ENV["DISPLAY"] = ":0"
using ProgressMeter
using Distributed
using PyPlot

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

function distributed_direct_sampling(N, weight, α, initial, dt, N_AVCH)
    initialize_workers()
    avch = @showprogress pmap(i -> Avalanche.direct_sample_avalanche_size(N, weight, α, 1, 0.01), 1:N_AVCH)
end

function distributed_gillespie_sampling(N,R,initial,NAVCH)
    initialize_workers()
    avch = @showprogress pmap(i -> Avalanche.gillespie_sample_avalanche_size(N,R,initial), 1:NAVCH)
end

# %% Set Constants
T = 10
N = 800
α = 1
initial = 1
dt = 0.01
N_AVCH = 1000000
histsize = 800
subc = 0.5
supc = 1.1

### First Graph

# %% Run Direct Simulation
# a1 = distributed_direct_sampling(N, 0.9*α, α, initial, dt, N_AVCH)
# a2 = distributed_direct_sampling(N, 1 * α, α, initial, dt, N_AVCH)
# a3 = distributed_direct_sampling(N, 1.1 * α, α, initial, dt, N_AVCH)

# %% Run gillespie Simulation
@time begin
    a1 = distributed_gillespie_sampling(N,subc,1,N_AVCH)
    a2 = distributed_gillespie_sampling(N,1  ,1,N_AVCH)
    a3 = distributed_gillespie_sampling(N,supc,1,N_AVCH)
end
# %% Prepare for Plotting
x1, y1 = Avalanche.probability_from_samples(a1, histsize)
x2, y2 = Avalanche.probability_from_samples(a2, histsize)
x3, y3 = Avalanche.probability_from_samples(a3, histsize)

# %% Plot
xscale("log")
yscale("log")
fx_plot = Avalanche.logspace(1, maximum(x2), histsize)
using SpecialFunctions
fy_plot = fx_plot .^ (-3/2) / zeta(3/2)
plot(fx_plot, fy_plot, color="pink", label=L"Power Law: $P(n)=\zeta^{-1}(3/2)n^{-3/2}$", linewidth=2)
plot(x2, y2, color="green", marker="x", linestyle="", label="Critical R=1")
plot(x3, y3, color="red", marker="x", linestyle="", label="Supercritical R>1")
plot(x1, y1, color="blue", marker="x", linestyle="", label="Subcritical R<1")
xlabel("Avalanche Size")
ylabel("Probability Function")
title("Simulation of Avalanches")
# legend()
# rmprocs(workers())

# %% Exact Solution
x4 = unique(floor.(Int, x1))
x5 = unique(floor.(Int, x2))
x6 = unique(floor.(Int, x3))
@time begin
    y4 = Avalanche.probability_sequence(subc, N, x4)
    y5 = Avalanche.probability_sequence(1, N, x5)
    y6 = Avalanche.probability_sequence(supc, N, x6)
end

# %% Exact Solution's Plot
plot(x4, y4, color="black", label="Exact Solution")
plot(x5, y5, color="black")#, label="Exact Solution")
plot(x6, y6, color="black")#, label="Exact Solution")
### Second Graph
# %% Simulate

import DifferentialEquations as DE
f(A, R, t) = (R*α*A/N) * (N - A) - α  * A
A0 = 0.25 * N
tspan = (0, T)
prob = DE.ODEProblem(f, A0, tspan, 1)
sol4 = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)
prob = DE.ODEProblem(f, A0, tspan, 1.5)
sol5 = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)
prob = DE.ODEProblem(f, A0, tspan, 0.5)
sol6 = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)
avtime = 10000
y7 = Float64.(Avalanche.generate_time_evolution_data(N, α,α,200,dt,T))
for i=1:avtime
    y7 .+= Avalanche.generate_time_evolution_data(N, α,α,200,dt,T)
end
y8 = Float64.(Avalanche.generate_time_evolution_data(N, 1.5* α,α,200,dt,T))
for i=1:avtime
    y8 .+= Avalanche.generate_time_evolution_data(N, 1.5* α,α,200,dt,T)
end
y9 = Float64.(Avalanche.generate_time_evolution_data(N, 0.5*α,α,200,dt,T))
for i=1:avtime
    y9 .+= Avalanche.generate_time_evolution_data(N, 0.5*α,α,200,dt,T)
end
y7 ./= avtime+1
y8 ./= avtime+1
y9 ./= avtime+1
td = range(0, T, Int(T/dt))
plot(td, y7, "g", label="Critical Simulation")
plot(td, y8, "r", label="Supercritical Simulation")
plot(td, y9, "b", label="Subcritical Simulation")
plot(sol4.t,sol4.u, color="g", linestyle="--", label="Critical ODE")
plot(sol5.t,sol5.u, color="r", linestyle="--", label="Supercritical ODE")
plot(sol6.t,sol6.u, color="b", linestyle="--", label="Subcritical ODE")
xlabel("Time")
ylabel("Active Neurons")
title("Time Evolution of The System")
legend()
# savefig("fig3.png")
