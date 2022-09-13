module RegLog


include("metrics.jl")

using CSV, DataFrames, LinearAlgebra, Plots, Distributions, .Metrics
using .Datasets

export logisticregression


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


function softmaxregression(X, Y)

    X, Y = readiris()
    X = hcat(X, ones(size(X)[1]))
    Yₘ = preparey(Y)
    n, 𝓓 = size(X)  # number of instances and features
    k = size(Yₘ)[2]  # number of classes

    # auxiliar functions
    cross_entropy(Yₘ, Ŷ) = -sum(Yₘ .* log.(Ŷ))

    # weights vector
    𝓦 = rand(k, 𝓓)

    it = 0
    itmax = 1000
    ϵ = 1e-3
    errors = Vector{Float64}()
    𝛁 = ones(k, 𝓓)

    while (norm(𝛁) > ϵ) & (it < itmax)
        Ŷ = softmax(X * 𝓦')
        𝛁 = (Ŷ - Yₘ)' * X
        𝓔 = cross_entropy(Yₘ, Ŷ)
        
        αu = rand(Uniform(1e-3, .99))
        αl = 0
        αm = (αu + αl) / 2

        while αu - αl > 1e-9
            
            Wi = 𝓦 - αm * 𝛁  # Dou o passo com a escala do alfa atual que foi chutado
            Yi = softmax(X * Wi')  # Calculo a matriz com as probabilidades
            𝛁α = (Yi - Yₘ)' * X  # Obtenho o gradiente de 𝛁f(x + αd)
            h̄ = 𝛁α[:]' * 𝛁[:]  # Calculo h'(α) = 𝛁f(x + αd)'d

            if (h̄ > 0)
                αu = αm
            elseif (h̄ < 0)
                αl = αm
            end
            αm = (αu + αl) / 2
        end
        
        𝓦 = 𝓦 - αm * 𝛁
        println("it $it, E=$𝓔, α=$αm")
        push!(errors, 𝓔)
        it += 1
    end

    plot(
        errors,
        xlabel="it",
        ylabel="error",
        title="Error convergence",
        color=:black,
        linewidth=2
    )

    return Ŷ, 𝓦, errors
end


end