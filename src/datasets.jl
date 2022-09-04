module Datasets

using DataFrames, CSV, Plots, Distributions

export readiris, readlinear, addbias


function preparey(y::Vector)::Matrix
    n = size(y)[1]
    Y = zeros(n, 3)
    C = unique(y)

    for (idx, class) ∈ enumerate(y)
        classindex = findfirst(c -> c == class, C)
        Y[idx, classindex] = 1
    end

    return Y
end

function readiris()
    path::String = "data/raw/iris.data"
    df = DataFrame(CSV.File(path, header = false))
    X = Matrix(df[:, 1:4])
    Y = Vector(df[:, 5])

    ymulticlass = preparey(Y)
    return X, ymulticlass
end

function readlinear()
    x⃗ = rand(100)
    y⃗ = [
        (xᵢ * 2) + rand(Uniform(-0.2, 0.2))
        for xᵢ ∈ x⃗
    ]
    scatter(x⃗, y⃗)
    return x⃗, y⃗
end

end
