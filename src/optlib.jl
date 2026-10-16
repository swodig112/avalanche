module optlib
using Optimization
using Optim
using OptimizationOptimJL
# using ADTypes

include("avalib.jl")
av = Avalanche

BATCHSIZE = 1_000_000
REFR      = 1.1
REFN      = 2000

bin_edges = av.logspace(1, 10000, 20)
bin_edges = unique(round.(Int, bin_edges))
bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])

ref_samp  = av.negbin_sample_avalanche(REFN, REFR, BATCHSIZE)
ref_pdf   = av.probability_from_samples(ref_samp, bin_edges)[2]

function convert_xy_with_τD(x, y, τ, D, N)
    return [x ./ N^D, y .* x.^τ]
end

function error_from_reference(x, N)
    # println(x)
    R, τ, D = x
    # R = x[1]
    # τ, D, N = N
    # samples = ag.gfulav(BATCHSIZE, N, R, INITIAL, GPUARRF, BACKEND)
    samples = av.negbin_sample_avalanche(N, R, BATCHSIZE)
    pdf = av.probability_from_samples(samples, bin_edges)[2]
    x1, y1 = convert_xy_with_τD(bin_centers, ref_pdf, τ, D, REFN)
    x2, y2 = convert_xy_with_τD(bin_centers, pdf, τ, D, N)

    error_y = sum((sqrt.(y1) .- sqrt.(y2)).^2)
    error_x = sum(abs2, log.(x1) .- log.(x2))

    return error_x + error_y
end

function callback(state, loss)
    println(
        "iter = $(state.iter), ",
        "error = $(loss), ",
        "x = $(state.u)"
    )
    return false
end

function opt()

    x0 = [1, 1, 1.5]
    optf = OptimizationFunction(
        error_from_reference,
        AutoFiniteDiff(
            fdtype = Val(:forward),
            relstep = 1e-3
        )
    )

    prob = OptimizationProblem(
        optf, x0, 1000,
        # lb = [0f0],
        # ub = [5f0],
        lb=[1, 0, 0],
        ub=[1.15, 5, 5]
    )

    sol = solve(
        prob,
        Optim.NelderMead();
        maxiters = 10,
        callback=callback
    )
    return sol

end

function opt()
    x0 = [1, 1, 1.5]    
    optf = OptimizationFunction(error_from_reference)

prob = OptimizationProblem(
    optf, x0, 1000;
    lb = [1.0, 0.0, 0.0],
    ub = [1.15, 5.0, 5.0]
)

sol = solve(
    prob,
    Optim.NelderMead();
    maxiters = 10,
    callback = callback
)
end
# [1.0491408806972033, 0.2821044169599918, 0.04451304354652397]
function opt()
    x0 = [1.2, 1.0, 1.5]

    optf = OptimizationFunction(error_from_reference)

    prob = OptimizationProblem(
        optf, x0, 1000;
        lb = [1.1, 0.0, 0.0],
        ub = [1.3, 5.0, 5.0]
    )

    solve(
        prob,
        Optim.SAMIN();
        maxiters = 100,
        callback = callback
    )
end
end
