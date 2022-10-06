#=
Implementação de algoritmos de otimização

- Gradiente - OK
- Newton - OK
- DFP - OK
- BFGS - OK
- Gradiente Conjugado - OK
- Secante - OK
=#

using Revise
using Plots, LinearAlgebra


#= Função a ser minimizada =#
function f(x₁, x₂)
    return 3*(x₁ - 1)^2 + (x₂ - 2)^2
end

#= Gradiente da função =#
function 𝛁f(x₁, x₂)
    return [ 6*(x₁ - 1), 2*(x₂ - 2) ]
end

#= Hessiana da função =#
function 𝓗f(x₁, x₂)
    return [
        6 0
        0 2
    ]
end


function h̄(direction_f, 𝛁f, 𝓗f, X, d⃗, η)
    X = X - η * d⃗    
    𝛁new = direction_f(f, 𝛁f, 𝓗f, X)
    return 𝛁new' ⋅ (-d⃗)
end


function bissection(𝛁f, 𝓗f, direction_f, X, d⃗)
    αl = 0
    αu = let
        α = rand()
        while h̄(direction_f, 𝛁f, 𝓗f, X, d⃗, α) < 0
            α *= 2
        end
        α
    end

    αm = (αl + αu) / 2
    h = h̄(direction_f, 𝛁f, 𝓗f, X, d⃗, αm)

    while abs(h) > 1e-3
        if h > 0
            αu = αm
        elseif h < 0
            αl = αm
        end
        αm = (αl + αu) / 2
        h = h̄(direction_f, 𝛁f, 𝓗f, X, d⃗, αm)
    end

    return αm
end


# Algoritmos de otimização
function gradient_descent(f, 𝛁f, 𝓗f, X)
    return 𝛁f(X[1], X[2])
end


function newton(f, 𝛁f, 𝓗f, X)
    return inv(𝓗f(X[1], X[2])) * 𝛁f(X[1], X[2])
end


function optimize(f, 𝛁f, 𝓗f, direction_f)
    X = [8 -8]'
    η = 1e-1

    ps = Matrix{Float64}(undef, 0, 2)
    ps = vcat(ps, X')

    while norm(𝛁f(X[1], X[2])) > 1e-3
        d⃗ = direction_f(f, 𝛁f, 𝓗f, X)
        η = bissection(𝛁f, 𝓗f, direction_f, X, d⃗)
        X = X - η * d⃗
        ps = vcat(ps, X')
    end

    ps
end

#=======================#
function h̄_dfd(X, d, α, 𝛁f)
    Xi = X + α * d
    gradi = 𝛁f(Xi[1], Xi[2])
    return gradi' * d
end


function bissection_dfd(X, d, 𝛁f)
    al = 0
    au = let 
        a = rand()
        while h̄_dfd(X, d, a, 𝛁f) < 0
            a = a * 2
        end
        a
    end

    α = (al + au) / 2
    hl = h̄_dfd(X, d, α, 𝛁f)

    while abs(hl) > 1e-3
        if hl > 0
            au = α
        elseif hl < 0
            al = α
        end
        α = (al + au)/2
        hl = h̄_dfd(X, d, α, 𝛁f)
    end

    α
end


function minimize_dfp(f, 𝛁f)
    X = [8 -8]'

    grad = 𝛁f(X[1], X[2])
    d = grad

    M = Matrix{Float64}(I, (2, 2))

    ps = Matrix{Float64}(undef, 0, 2)
    ps = vcat(ps, X')

    i = 1
    while norm(grad) >= 1e-3
        d = -M * grad
        
        if (i % 2 == 0)
            d = -grad
            M = Matrix{Float64}(I, (2, 2))
        end

        η = bissection_dfd(X, d, 𝛁f)
        X = X + η * d

        ps = vcat(ps, X')

        ngrad = 𝛁f(X[1], X[2])

        p = η*d
        q = ngrad - grad
        M = M + ((p * p') / (p' * q)) - ((M*q*q'*M)/(q'*M*q))
        grad = ngrad

        i += 1
    end

    ps
end

function minimize_bfgs()
    X = [8 -8]'

    grad = 𝛁f(X[1], X[2])
    d = grad

    M = Matrix{Float64}(I, (2, 2))

    ps = Matrix{Float64}(undef, 0, 2)
    ps = vcat(ps, X')

    i = 1
    while norm(grad) >= 1e-3
        d = -M * grad
        
        if (i % 2 == 0)
            d = -grad
            M = Matrix{Float64}(I, (2, 2))
        end

        η = bissection_dfd(X, d, 𝛁f)
        X = X + η * d
        ps = vcat(ps, X')

        ngrad = 𝛁f(X[1], X[2])

        p = η*d
        q = ngrad - grad

        m₁ = ((p * p') / (p' * q)) * (1 + (q'*M*q)/(p'*q))
        m₂ = (M*q*p' + p*q'*M) / (p'*q)

        M = M + m₁ - m₂
        grad = ngrad

        i += 1
    end

    ps
end

function minimize_oss()
    X = [8 -8]'

    grad = 𝛁f(X[1], X[2])
    p = q = [0, 0]
    A = B = 0
    ps = Matrix{Float64}(undef, 0, 2)
    ps = vcat(ps, X')

    i = 1
    while norm(grad) >= 1e-3
        d = -grad + A*p + B*q
        
        if (i % 2 == 0)
            d = -grad
        end

        η = bissection_dfd(X, d, 𝛁f)
        X = X + η * d

        ps = vcat(ps, X')

        ngrad = 𝛁f(X[1], X[2])

        p = η*d
        q = ngrad - grad

        B = (p'*grad)/(p'*q)
        A = -(1 + ((q'*q)/(p'*q))) * B + (q'*grad)/(p'*q)

        grad = ngrad
        i += 1
    end

    ps
end

function minimize_pr()
    X = [8 -8]'

    d = -𝛁f(X[1], X[2])

    i = 1
    ps = Matrix{Float64}(undef, 0, 2)
    ps = vcat(ps, X')

    while norm(d) > 1e-3
        η = bissection_dfd(X, d, 𝛁f)
        X = X + η * d

        ps = vcat(ps, X')

        gᵢ = -𝛁f(X[1], X[2])

        if (i % 2 != 0)
            β = gᵢ'*(gᵢ-d) / (d'*d)
            d = gᵢ + β * d
        else
            d = gᵢ
        end
        i += 1
    end

    ps
end

function minimize_fr()
    X = [8 -8]'

    d = -𝛁f(X[1], X[2])

    ps = Matrix{Float64}(undef, 0, 2)
    ps = vcat(ps, X')

    i = 1
    while norm(d) > 1e-3
        η = bissection_dfd(X, d, 𝛁f)
        X = X + η * d

        ps = vcat(ps, X')

        gᵢ = -𝛁f(X[1], X[2])

        if (i % 2 != 0)
            β = (gᵢ'*gᵢ) / (d'*d)
            d = gᵢ + β * d
        else
            d = gᵢ
        end
        i += 1
    end

   ps
end

gradient_positions = optimize(f, 𝛁f, 𝓗f, gradient_descent)
newton_positions = optimize(f, 𝛁f, 𝓗f, newton)
dfp_positions = minimize_dfp(f, 𝛁f)
bfgs_positions = minimize_bfgs()
oss_positions = minimize_oss()
pr_positions = minimize_pr()
fr_positions = minimize_fr()

x1 = -10:0.2:10
x2 = -10:0.2:10

plot(x1, x2, f, st=:contourf)
plot!(
    gradient_positions[:, 1],
    gradient_positions[:, 2],
    f,
    markersize=3,
    markershape=:circle,
    color=:white,
    label="Gradient Descent"
)
plot!(
    newton_positions[:, 1],
    newton_positions[:, 2],
    f,
    markersize=3,
    markershape=:cross,
    color=:yellow,
    label="Newton"
)
plot!(
    oss_positions[:, 1],
    oss_positions[:, 2],
    f,
    markersize=3,
    markershape=:diamond,
    color=:yellow,
    label="OSS"
)
plot!(
    dfp_positions[:, 1],
    dfp_positions[:, 2],
    f,
    markersize=3,
    markershape=:diamond,
    color=:yellow,
    label="DPF"
)
plot!(
    bfgs_positions[:, 1],
    bfgs_positions[:, 2],
    f,
    markersize=3,
    markershape=:diamond,
    color=:yellow,
    label="BGFS"
)
plot!(
    pr_positions[:, 1],
    pr_positions[:, 2],
    f,
    markersize=3,
    markershape=:diamond,
    color=:yellow,
    label="Polak-Ribiére"
)
plot!(
    fr_positions[:, 1],
    fr_positions[:, 2],
    f,
    markersize=3,
    markershape=:diamond,
    color=:yellow,
    label="Fletcher-Reeves"
)