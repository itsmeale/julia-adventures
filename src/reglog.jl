module RegLog


include("metrics.jl")

using CSV, DataFrames, LinearAlgebra, Plots, Distributions, .Metrics

export preparey, readiris, softmax, ĥ, h, bisection, softmaxregression


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

function readiris(path)
    df = DataFrame(CSV.File(path, header=false))
    X = Matrix(df[:, 1:4])
    Y = Vector(df[:, 5])
    return X, Y
end

function softmax(𝓢)
    k = size(𝓢)[2]
    return exp.(𝓢) ./ (sum(exp.(𝓢), dims=2) * ones(1, k))
end

function ĥ(α, 𝓦, 𝛁, X, Yₘ)
    W = 𝓦 + α * (-𝛁)
    Y = softmax(X * W')
    𝛁α = (Y - Yₘ)' * X
    𝛁α[:]' * -𝛁[:]
end

function h(α, 𝓦, 𝛁, X, Yₘ)
    W = 𝓦 - α * 𝛁
    Y = softmax(X * W')
    -sum(Yₘ .* log.(Y))
end

function bisection(𝓦, 𝛁, X, Yₘ)
    αl = 0
    αu = let
        α = rand()
        while ĥ(α, 𝓦, 𝛁, X, Yₘ) < 0
            α = rand()
        end
        α
    end
    ᾱ = (αl + αu) / 2

    hl = ĥ(ᾱ, 𝓦, 𝛁, X, Yₘ)

    while abs(hl) > 1e-5
        if hl > 0
            αu = ᾱ
        elseif hl < 0
            αl = ᾱ
        end
        ᾱ = (αl + αu) / 2
        hl = ĥ(ᾱ, 𝓦, 𝛁, X, Yₘ)
    end

    ᾱ
end

function softmaxregression()

    X, Y = readiris("data/raw/iris.m")
    X = hcat(X, ones(size(X)[1]))
    Yₘ = preparey(Y)
    n, 𝓓 = size(X)  # number of instances and features
    k = size(Yₘ)[2]  # number of classes

    # weights vector
    θ = rand(k, 𝓓)

    it = 0
    itmax = 1000
    ϵ = 2e-2
    errors = Vector{Float64}()
    𝛁 = ones(k, 𝓓)
    norm_𝛁 = norm(𝛁)

    while (norm_𝛁 > ϵ) & (it < itmax)
        Ŷ = softmax(X * θ')
        𝛁 = (Ŷ - Yₘ)' * X
        𝛁ₙ = 𝛁/norm(𝛁)
        𝓔 = -sum(Yₘ .* log.(Ŷ))
        η = bisection(θ, 𝛁ₙ, X, Yₘ)
        θ = θ - η * 𝛁ₙ

        norm_∇ = norm(𝛁)
        println("it $it, E=$𝓔, α=$η, norm(𝛁)=$norm_∇")
        push!(errors, 𝓔)
        it += 1
    end

    plot!(
        errors,
        xlabel="it",
        ylabel="error",
        title="Error convergence",
        color=:blue,
        linewidth=3
    )

    return Ŷ, 𝓦, errors 
end


end