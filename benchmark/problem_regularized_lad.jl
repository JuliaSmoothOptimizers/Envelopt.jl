using ProximalOperators, ManualNLPModels, NLPModels
using RegularizedProblems, RegularizedOptimization
using Random, LinearAlgebra

using Envelopt

# regularized least absolute deviations
# min_x \|Ax-b\|_1 + g(x)

Random.seed!(123) # seed for reproducibility

nvar = 100
na = Int(ceil(0.6*nvar))
A = randn(na, nvar)
b = randn(na)

h = ProximalOperators.NormL1()

# composition function (linear)
evalc!(cx, x) = begin
  @assert length(cx) == na
  @assert length(x) == nvar
  cx .= A*x - b
  cx
end

jprod!(jv, x, v) = begin
  @assert length(jv) == na
  @assert length(x) == nvar
  @assert length(v) == nvar
  jv .= A*v
  jv
end

jtprod!(jtv, x, v) = begin
  @assert length(jtv) == nvar
  @assert length(x) == nvar
  @assert length(v) == na
  jtv .= A' * v
  jtv
end

zeroobj(x) = 0.0
zerograd!(gx, x) = begin
  gx .= 0.0
  gx
end

# initial guess
x0 = randn(nvar)

# models
model = NLPModel(x0, zeroobj, grad = zerograd!)

Fmodel = NLPModel(
  x0,
  zeroobj,
  grad = zerograd!,
  cons = (evalc!, zeros(na), zeros(na)),
  jprod = jprod!,
  jtprod = jtprod!,
)

println("nvar $(model.meta.nvar)")
println("ncon $(model.meta.ncon)")
println("na $(Fmodel.meta.ncon)")

# regularizer
lambda = 0.01
g = ProximalOperators.NormL1(lambda)
#g = ProximalOperators.NormL0(lambda)

# problem objective
LADobjective(x) = h(A*x-b) + g(x)

# Envelopt model
env_model = EnveloptNLPModel(model, Fmodel, h, g = g)

# tolerance
TOL = 1e-5

# call Envelopt
stats, status, u, inner_iter =
  envelopt(env_model, ptol_min = TOL, dtol_min = TOL, subsolver = R2EnveloptSubSolver(env_model))
ENV_x = stats.solution
ENV_obj = LADobjective(ENV_x)

using ProximalAlgorithms, ProximalOperators

# solve with Douglas-Rachford, at least for convex g

# problem functions
fs = (g, h)
idxs = ((1:nvar,), ((nvar + 1):(nvar + na),))
phi1 = ProximalOperators.SlicedSeparableSum(fs, idxs)

AA = zeros(na, nvar+na)
AA[:, 1:nvar] .= A
for i = 1:na
  AA[i, nvar + i] = -1.0
end
phi2 = ProximalOperators.IndAffine(AA, b)

# initial guess
xz0 = zeros(nvar+na)
xz0[1:nvar] .= x0

# call solver
DR_solver = ProximalAlgorithms.DouglasRachford(tol = TOL, gamma = 1.0, maxit = 100000)
DR_solution, DR_iter = DR_solver(x0 = xz0, f = phi1, g = phi2)
DR_x = DR_solution[1:nvar]
DR_obj = LADobjective(DR_x)