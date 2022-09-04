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
    ypred = X * w⃗

    mse = (ypred-y⃗).^2
    scatter(X[:, 1], mse)
    ylims!(-3, 3)

    scatter(X[:, 1], y⃗)
    plot!(X[:,  1], ypred, linewidth=4)
end

end
