module LinearModels

include("datasets.jl")

using .Datasets, Plots

function addbias(X)
    return hcat(X, ones(size(X)[1]))
end

function regression(X, y⃗)
    X = addbias(X)
    X⁻¹ = inv(X' * X) * X'
    w⃗ = X⁻¹ * y⃗
    return w⃗
end

end
