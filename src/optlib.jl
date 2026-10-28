module optlib
using Optimization
using Optim
using OptimizationOptimJL
using Interpolations
# using ADTypes

include("avalib.jl")
av = Avalanche

BATCHSIZE = 1_000_000
REFR      = 1.01
REFN      = 1500
HISTSIZE  = 30

# bin_edges = av.logspace(1, 10000, 20)
# bin_edges = unique(round.(Int, bin_edges))
# bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])

ref_samp  = av.negbin_sample_avalanche(REFN, REFR, BATCHSIZE)
ref_bin_centers, ref_pdf   = av.probability_from_samples(ref_samp, HISTSIZE)

function convert_xy_with_τD(x, y, τ, D, N)
    return [x ./ N^D, y .* x.^τ]
end

function error_from_reference(x, N)
    # println(x)
    # R, τ, D = x
    R = x[1]
    τ, D, N = N
    # samples = ag.gfulav(BATCHSIZE, N, R, INITIAL, GPUARRF, BACKEND)
    samples = av.negbin_sample_avalanche(N, R, BATCHSIZE)
    bin_centers, pdf = av.probability_from_samples(samples, HISTSIZE)
    x1, y1 = convert_xy_with_τD(ref_bin_centers, ref_pdf, τ, D, REFN)
    x2, y2 = convert_xy_with_τD(bin_centers, pdf, τ, D, N)

    # println(x2, y2)
    
    x1log = log.(x1)
    y1log = log.(y1)
    x2log = log.(x2)
    y2log = log.(y2)

    # println(x1log, y1log)

    itp1 = interpolate((x1log, ), y1log, Gridded(Linear()))
    itp2 = interpolate((x2log, ), y2log, Gridded(Linear()))
    rmin = max(x1log[1], x2log[1])
    rmax = min(x1log[end], x2log[end])
    
    # error_y = sum((sqrt.(y1) .- sqrt.(y2)).^2)
    # error_x = sum(abs2, log.(x1) .- log.(x2))

    err = [(itp1(x) - itp2(x))^2 for x in range(rmin, rmax, HISTSIZE)]
    
    return sum(err)
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
    x0 = [1.05]

    # optf = OptimizationFunction((a, b) -> @run error_from_reference(a,b))
    optf = OptimizationFunction(error_from_reference)
    
    prob = OptimizationProblem(
        optf, x0, (1.5, 1.0, 100);
        lb = [1.0],
        ub = [1.15]
    )

    solve(
        prob,
        Optim.SAMIN();
        maxiters = 100,
        callback = callback
    )
end
end
