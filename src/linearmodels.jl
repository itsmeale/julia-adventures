module LinearModels

include("datasets.jl")

using .Datasets, Plots

function addbias(X)
    hcat(X, ones(size(X)[1]))
end

#= Funciona apenas para classificadores binários =#
function signal(y)
    classes = ones(size(y))
    for (i, y) ∈ enumerate(y)
        classes[i] = (y > 0) ? 1 : 0
    end
    classes
end

function regression(X, y⃗)
    X = addbias(X)
    X⁻¹ = inv(X' * X) * X'
    ω⃗ = X⁻¹ * y⃗
    ω⃗
end

function regressionclassifier(X, y⃗)
    ω⃗ = regression(X, y⃗)
    ℋ = addbias(X) * ω⃗
    ŷ = signal(ℋ)
    ω⃗, ŷ
end

function acc(y, ŷ)
    n = length(y)
    tp = 0
    for (i, j) in zip(y, ŷ)
        tp += (i == j) ? 1 : 0
    end
    tp/n
end

end

