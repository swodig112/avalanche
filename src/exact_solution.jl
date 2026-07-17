### Don't use this file. It's only here for archival purposes
using LinearAlgebra

N = 800
R0 = 1
q(i) = (N)/(R0 * (N - i) + N)
initial = zeros(N)
initial[1] = 1

function rp(i, j)
    if i > 1 && j == 0
        0
    elseif i == 1 && j == 0
        1
    elseif i == 2 && j == 1
        1 - q(1)
    elseif i == 0 && j == 1
        q(1)
    elseif i == 1
        rp(2, j-1) * q(2)
    elseif 1 < i && i < N
        rp(i-1, j-1)*(1-q(i-1)) + rp(i+1,j-1)*q(i+1)
    elseif i==N
        rp(N-1, j-1) * (1 - q(N-1))
    end
end

function vecp(l)
    rp.(1:N,l)
end


function build_tridiag(f1,f2,f3,N)
    d1 = [f1(i) for i in 1:N-1]
    d2 = [f2(i) for i in 1:N]
    d3 = [f3(i) for i in 1:N-1]
    Tridiagonal(d1, d2, d3)
end

A = build_tridiag(i -> 1-q(i), i->0, i -> q(i+1), N)
p0 = zeros(N)
p0[1] = 1

function matrix_power(M, m)
    m = floor(Int, m)
    # Ensure m is non-negative
    @assert m >= 0 "Power must be a non-negative integer."
    
    # Get size and create identity matrix of the same type
    N = size(M, 1)
    result = Matrix{eltype(M)}(I, N, N)
    
    base = M
    
    # The algorithm
    while m > 0
        if isodd(m)
            result = result * base
        end
        base = base * base
        m >>= 1 # Integer division by 2
    end
    return result
end

function exactp(R0, N)
    # Using this function isn't recommended unless you only
    # want to calculate P for a few values. If you want to plot
    # (e.g. need to calculate many Ps) use probability_sequence
    # below
    q(i) = (N)/(R0 * (N - i) + N)
    A = build_tridiag(i -> 1-q(i), i->0, i -> q(i+1), N)
    p0 = zeros(N)
    p0[1] = 1
    function P(k)
        # q(1) * p0' * A^(2*(floor(k)-1)) * p0
        q(1) * p0' * matrix_power(A,2*(floor(k)-1))  * p0
    end
end

function probability_sequence(R0, N, x)
    # Here I calculate P as i'm exponentiating A
    # so it's greatly faster than exactp
    q(i) = (N)/(R0 * (N - i) + N)
    A = build_tridiag(i -> 1-q(i), i->0, i -> q(i+1), N)
    p0 = zeros(Float64, N)
    p0[1] = 1
    A2 = A^2
    xp = 1
    res = zeros(Float64, length(x))
    for i in 1:length(x)
        p0 = A2^(floor(x[i])-xp) * p0
        xp = floor(x[i])
        res[i] = q(1) * p0[1]
    end
    res
end

