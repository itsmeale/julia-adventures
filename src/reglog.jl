module RegLog


include("metrics.jl")

using CSV, DataFrames, LinearAlgebra, Plots, .Metrics

export logisticregression

#=
Codifica a variavel alvo multiclasse para um vetor de dimensão
K onde, K é o número de classes.

Se o conjunto de classes é {1, 2, 3}, uma instância pode ser
codificada como [1, 0, 0], [0, 1, 0] ou [0, 0, 1].
=#
function preparey(y::Vector)::Matrix
    n = size(y)[1]
    Y = zeros(n, 3)
    𝓒 = unique(y)
    for (idx, class) ∈ enumerate(y)
        classindex = findfirst(c -> c == class, 𝓒)
        Y[idx, classindex] = 1
    end
    return Y
end

#= Le o dataset iris =#
function readiris()
    path = "data/raw/iris.m"
    df = DataFrame(CSV.File(path, header=false))
    X = Matrix(df[:, 1:4])
    Y = Vector(df[:, 5])
    return X, Y
end

function softmax(𝓢)
    k = size(𝓢)[2]
    return exp.(𝓢) ./ (sum(exp.(𝓢), dims=2) * ones(1, k))
end

function logisticregression(X, Y)

    X, Y = readiris()
    Yₘ = preparey(Y)
    n, 𝓓 = size(X)  # number of instances and features
    k = size(Yₘ)[2]  # number of classes

    # auxiliar functions
    total_error(Y, Ŷ) = -sum(Y .* log.(Ŷ))
    𝛁g(E, X) = E' * X

    # add bias and create weights vector
    X = hcat(X, ones(n))
    𝓦 = rand(k, 𝓓 + 1)

    it = 0
    itmax = 1000
    η = 1e-3
    ϵ = 1e-3
    errors = Vector{Float64}()
    grad_norm = Inf

    while (grad_norm > ϵ) & (it < itmax)
        𝓢 = X * 𝓦'
        Ŷ = softmax(𝓢)
        𝛁 = 𝛁g(Ŷ - Yₘ, X)
        𝓦 -= η * 𝛁
        𝓔 = total_error(Yₘ, Ŷ)
        grad_norm = norm(𝛁)

        println("it $it, E=$𝓔")
        push!(errors, 𝓔)
        it += 1
    end

    # evaluation metrics
    Metrics.multiclass_report(Yₘ, Ŷ)
    plot(errors)

    return Ŷ, 𝓦, errors
end


end