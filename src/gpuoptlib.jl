module avagpuopt
using oneAPI
using Optimization
using Optim
using Optimization
using OptimizationOptimJL
using Optim
using ADTypes


include("avagpulib.jl")
include("avalib.jl")

import .Avalanche
import .AvalancheGPU

av = Avalanche
ag = AvalancheGPU

BATCHSIZE = 1_000_000
REFR      = 1.01f0
REFN      = 2000
INITIAL   = 1
GPUARRF   = oneArray
BACKEND   = oneAPIBackend()

bin_edges = av.logspace(1, 10000, 20)
bin_edges = unique(round.(Int, bin_edges))
bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])

ref_samp  = ag.gfulav(BATCHSIZE, REFN, REFR, INITIAL, GPUARRF, BACKEND)
ref_pdf   = av.probability_from_samples(ref_samp, bin_edges)[2]

function convert_xy_with_τD(x, y, τ, D, N)
    return [x ./ N^D, y .* x.^τ]
end

function error_from_reference(x, N)
    # println(x)
    # R,τ, D = x
    R = x[1]
    τ, D, N = N
    # println(N)
    samples = ag.gfulav(BATCHSIZE, N, R, INITIAL, GPUARRF, BACKEND)
    pdf = av.probability_from_samples(samples, bin_edges)[2]
    # println(x)
    x1, y1 = convert_xy_with_τD(bin_centers, ref_pdf, τ, D, REFN)
    x2, y2 = convert_xy_with_τD(bin_centers, pdf, τ, D, N)

    # mask = (y1 .> 0) .& (y2 .> 0)

# dx = log.(x1[mask]) .- log.(x2[mask])
# dy = log.(y1[mask]) .- log.(y2[mask])

# println(typeof(dx), "  ", eltype(dx))
# println(typeof(dy), "  ", eltype(dy))

# println("ex = ", sum(abs2, dx))
# println("ey = ", sum(abs2, dy))
# println("total = ", sum(abs2, dx) + sum(abs2, dy))

# println("min abs2 dx = ", minimum(abs2.(dx)))
#     println("min abs2 dy = ", minimum(abs2.(dy)))



    
    error_y = sum((sqrt.(y1) .- sqrt.(y2)).^2)

    # x-axis discrepancy
    error_x = sum(abs2, log.(x1) .- log.(x2))

    return Float32(error_x + error_y)
    # return Float32((sum(abs2, x1 - x2))/1f15 + (sum(abs2, y1 - y2))/1f15)
    # return sum(abs2, log.(x1[mask]) .- log.(x2[mask])) + sum(abs2, log.(y1[mask]) .- log.(y2[mask]))
end


function callback(state, loss)
    println(
        "iter = $(state.iter), ",
        "error = $(loss), ",
        "x = $(state.u)"
    )
    return false
end

function testopt()

    # callback = function (state, loss)
#     direct = error_from_reference(state.u, 1500)

#     println("iter          = ", state.iter)
#     println("callback loss = ", loss)
#     println("state.objective= ", state.objective)
#     println("direct         = ", direct)
#     println("u              = ", state.u)
#     # println("gradient       = ", state.gradient)

#     false
# end
    x0 = Float32[1.0]#, 1.0, 1.5]
    # println(x0)

    optf = OptimizationFunction(
        error_from_reference,
        AutoFiniteDiff(
            fdtype = Val(:forward),
            relstep = 1f-3
        )
    )

    prob = OptimizationProblem(
        optf, x0, [0, 0, 1000],
        # lb = [0f0],
        # ub = [5f0],
        # lb=[0f0, 0f0, 0f0],
        # ub=[2f0, 5f0, 5f0]
    )

    # sol = solve(
    #     prob,
    #     Optim.NelderMead();
    #     maxiters = 10,
    #     callback=callback
    # )

    sol = optimize(R -> error_from_reference(Float32(R), [0,0,1000]), 0f0, 1.1f0, Brent())
    # println("Solution: ", sol.u)
    # println("Error:    ", sol.objective)
    # println("Status:   ", sol.retcode)
    return sol
end

end
