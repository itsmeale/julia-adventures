module sin5016

include("linearmodels.jl")
include("datasets.jl")

using .LinearModels
using .Datasets

X, y = toyclassification()
regressionclassifier(X, y)

end
