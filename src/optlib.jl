module optlib
using Optimization
using Optim
using OptimizationOptimJL
using Interpolations

include("avalib.jl")
av = Avalanche

BATCHSIZE = 1_000_000
REFR      = 1.01
REFN      = 1500
HISTSIZE  = 30

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
    samples = av.negbin_sample_avalanche(N, R, BATCHSIZE)
    bin_centers, pdf = av.probability_from_samples(samples, HISTSIZE)
    x1, y1 = convert_xy_with_τD(ref_bin_centers, ref_pdf, τ, D, REFN)
    x2, y2 = convert_xy_with_τD(bin_centers, pdf, τ, D, N)
    
    x1log = log.(x1)
    y1log = log.(y1)
    x2log = log.(x2)
    y2log = log.(y2)

    itp1 = interpolate((x1log, ), y1log, Gridded(Linear()))
    itp2 = interpolate((x2log, ), y2log, Gridded(Linear()))
    rmin = max(x1log[1], x2log[1])
    rmax = min(x1log[end], x2log[end])
    
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
    x0 = [1.05]
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
