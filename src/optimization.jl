using Revise
using Plots, LinearAlgebra, Distributions

f(X) = X[1]^2 + X[1]*X[2] + X[2]^2
𝛁f(X) = [ 2*X[1]+X[2], X[1]+2*X[2] ]

# we want to minimize f(X)
function minimizev1()
    X = [9, 34]
    y = f(X)
    η = 1e-3

    println(X, " ", y)
    it = 0
    while norm(𝛁f(X)) > 1e-3
        it += 1
        X = X - η*𝛁f(X)
        y = f(X)
        println("it: $it, $X $y")
    end
end

function bisection(X, grad_X)
    # bisection to find the best η
    αu = rand(Uniform(1e-3, 5e-1))
    αl = 0
    αm = (αu + αl) / 2
    h̄ = 𝛁f(X + αm*grad_X)'*grad_X

    while αu - αl < 1e-3
        if h̄ > 0
            αu = αm
        elseif  h̄ < 0
            αl = αm
        end
        αm = (αu + αl) / 2
        h̄ = 𝛁f(X + αm*grad_X)'*grad_X
    end
    return αm
end

function minimizev2()
    X = [9, 34]
    y = f(X)
    it = 0

    while norm(𝛁f(X)) > 1e-3
        η = bisection(X, 𝛁f(X))
        X = X - η*𝛁f(X)
        y = f(X)
        println("it: $it, $X $y")
        it += 1
    end
end