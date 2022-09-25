module RegLog


include("metrics.jl")

using Revise
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
    # estabiliza a softmax para evitar overflow
    𝓢 = 𝓢 .- findmax(𝓢, dims=2)[1]
    return exp.(𝓢) ./ (sum(exp.(𝓢), dims=2) * ones(1, k))
end

function ĥ(α, 𝓦, 𝛁, X, Yₘ)
    W = 𝓦 + α * (-𝛁)
    Y = softmax(X * W')
    𝛁α = (Y - Yₘ)' * X
    𝛁α[:]' * -𝛁[:]
end

function bisection(𝓦, 𝛁, X, Yₘ)
    αl = 0
    αu = let
        α = rand()
        while ĥ(α, 𝓦, 𝛁, X, Yₘ) < 0
            α *= 2
        end
        α
    end
    ᾱ = (αl + αu) / 2

    hl = ĥ(ᾱ, 𝓦, 𝛁, X, Yₘ)

    while (abs(hl) > 1e-5)
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

function ⊗(A::Matrix, B::Matrix)::Matrix
    return kron(A, B)
end

function hessian(X, Ŷ)
    𝓘 = Matrix(I, k, k)
    Ymul = Ŷ'*Ŷ
    𝓗 = (𝓘 - Ymul) ⊗ (X'*X)
end

function newton_direction(X, Ŷ, 𝛁)
    𝓗 = hessian(X, Ŷ)
    λ, _ = eigen(𝓗)
    if minimum(λ) < 0
        # positivate hessian matrix
        𝓗 = 𝓗 - (Matrix(I, size(𝓗)) .* 1.001*((minimum(λ))))
    end
    𝓗_inv = inv(𝓗)
    reshape(𝓗_inv * (𝛁[:]), (3, 5))
end

function softmaxregression()
    newton_opt = true
    use_bissection = true

    X, Y = readiris("data/raw/iris.m")
    X = hcat(X, ones(size(X)[1]))
    Yₘ = preparey(Y)
    n, 𝓓 = size(X)  # number of instances and features
    k = size(Yₘ)[2]  # number of classes
    η = 1e-3  # fixed learning rate

    # weights vector
    θg = rand(k, 𝓓) 
    θ = θg

    it = 0
    itmax = 2000
    ϵ = 2e-2
    loss_values = Vector{Float64}()
    𝛁 = ones(k, 𝓓)

    while (norm(𝛁) > ϵ) & (it < itmax)
        Ŷ = softmax(X * θ')
        𝛁 = d = (Ŷ - Yₘ)' * X
        loss = -sum(Yₘ .* log.(Ŷ))
        
        if newton_opt
            d = newton_direction(X, Ŷ, 𝛁)
        end

        if use_bissection
            η = bisection(θ, d, X, Yₘ)
        end

        θ = θ - η * d

        push!(loss_values, loss)
        println("it: $it, loss: $loss")
        it += 1
    end

    plot!(loss_values, label="Multinomial RegLog + bissection + newton", linewidth=3)

    return θ, loss_values 
end

end