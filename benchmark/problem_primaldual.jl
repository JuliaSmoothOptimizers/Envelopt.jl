using ProximalOperators, ManualNLPModels, NLPModels
using ShiftedProximalOperators
using RegularizedProblems, RegularizedOptimization
using ProximalOperators, ProximalAlgorithms
using Random, LinearAlgebra
using DataFrames, CSV
using Zygote
using DifferentiationInterface: AutoZygote

using Envelopt

Random.seed!(123) # seed for reproducibility

# min    1  \|x-xhat\|^2 + g(x) + h(Ax)
#  x    2 λ

nvar = 10
na = Int(ceil(0.5*nvar))
xhat = randn(nvar)
A = randn(na, nvar)
h = ProximalOperators.NormL1()

# initial guess
x0 = randn(nvar)
y0 = zeros(na)

# anchor point
xhat = randn(nvar)

# options
TOL = 1e-5
MAXITER = 10_000

problem_objective(x, g, lambda) = (norm(x - xhat)^2) / (2*lambda) + g(x) + h(A*x)

function proxcompsum_afba(g, lambda)
  f = ProximalAlgorithms.AutoDifferentiable(x -> (norm(x - xhat)^2) / (2*lambda), AutoZygote())
  beta_f = 1/lambda
  solver = ProximalAlgorithms.AFBA(tol = TOL, maxit = MAXITER, verbose = true)
  (x, y), iter = solver(x0 = x0, y0 = y0, f = f, g = g, h = h, L = A, beta_f = beta_f)
  obj = problem_objective(x, g, lambda)
  flag = iter < MAXITER
  return iter, obj, flag
end

# composition function (linear)
evalc!(cx, x) = begin
  cx .= A*x
  cx
end

jprod!(jv, x, v) = begin
  jv .= A*v
  jv
end

jtprod!(jtv, x, v) = begin
  jtv .= A' * v
  jtv
end

# zero objective
zeroobj(x) = 0.0
zerograd!(gx, x) = begin
  gx .= 0.0
  gx
end

Fmodel = NLPModel(
  x0,
  zeroobj,
  grad = zerograd!,
  cons = (evalc!, zeros(na), zeros(na)),
  jprod = jprod!,
  jtprod = jtprod!,
)

function proxcompsum_envelopt(g, lambda, subsolvermaker)
  # quadratic objective
  ls_obj(x) = (norm(x - xhat)^2) / (2*lambda)
  ls_grad!(gx, x) = begin
    gx .= (x - xhat) ./ lambda
    gx
  end
  # models for Envelopt
  model = NLPModel(x0, ls_obj, grad = ls_grad!)
  # Envelopt model
  env_model = EnveloptNLPModel(model, Fmodel, h, g = g)
  # call Envelopt
  subsolver = subsolvermaker(env_model)
  stats, status, u, inner_iter =
    envelopt(env_model, ptol_min = TOL, dtol_min = TOL, subsolver = subsolver)
  x = stats.solution
  obj = problem_objective(x, g, lambda)
  iter = inner_iter
  flag = stats.status == :first_order
  return iter, obj, flag
end

# ############################################################
ntrials = 49
lambda_vec = 10 .^ range(-3, 3, length = ntrials)
res = (
  envelopt_r2 = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
  envelopt_nmpg = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
  afba = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
)

gmakers =
  [ProximalOperators.NormL1, ShiftedProximalOperators.RootNormLhalf, ProximalOperators.NormL0]

for gmaker in gmakers
  for i = 1:ntrials
    lambda = lambda_vec[i]
    g = gmaker(lambda)

    # AFBA
    iter, obj, flag = proxcompsum_afba(g, lambda)
    res.afba.iter[i] = iter
    res.afba.obj[i] = obj
    res.afba.flag[i] = flag

    # Envelopt R2
    subsolvermaker = R2EnveloptSubSolver
    iter, obj, flag = proxcompsum_envelopt(g, lambda, subsolvermaker)
    res.envelopt_r2.iter[i] = iter
    res.envelopt_r2.obj[i] = obj
    res.envelopt_r2.flag[i] = flag

    # Envelopt NMPG
    subsolvermaker = NMPGEnveloptSubSolver
    iter, obj, flag = proxcompsum_envelopt(g, lambda, subsolvermaker)
    res.envelopt_nmpg.iter[i] = iter
    res.envelopt_nmpg.obj[i] = obj
    res.envelopt_nmpg.flag[i] = flag
  end

  # save results
  println("Problem with g=$(gmaker)")
  for k in keys(res)
    filename = "$(gmaker)_$(k).csv"
    mat = zeros(ntrials, 4)
    mat[:, 1] .= lambda_vec
    mat[:, 2] .= res[k].flag
    mat[:, 3] .= res[k].iter
    mat[:, 4] .= res[k].obj
    df = DataFrame(mat, :auto)
    CSV.write(filename, df, writeheader = false)

    println("$(k) solved $(Int(sum(res[k].flag))) out of $(ntrials)")
  end
end
