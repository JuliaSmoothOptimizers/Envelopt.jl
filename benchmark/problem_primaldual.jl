using ProximalOperators, ManualNLPModels, NLPModels
using ShiftedProximalOperators
using RegularizedProblems, RegularizedOptimization
using ProximalOperators, ProximalAlgorithms
using Random, LinearAlgebra
using DataFrames, CSV, Statistics
using Zygote
using DifferentiationInterface: AutoZygote

using Envelopt

Random.seed!(123) # seed for reproducibility

# min    1  \|x-xi\|^2 + g(x) + h(Ax)
#  x    2 λ

nvar = 10
na = 10
xi = randn(nvar)
A = randn(na, nvar)

# initial guess
x0 = randn(nvar)

# options
TOL = 1e-5
MAXITER = 100_000

problem_objective(x, g, h, lambda) = (norm(x - xi)^2) / (2*lambda) + g(x) + h(A*x)

function proxcompsum_afba(g, h, lambda)
  f = ProximalAlgorithms.AutoDifferentiable(x -> (norm(x - xi)^2) / (2*lambda), AutoZygote())
  beta_f = 1/lambda
  y0 = zeros(na)
  solver = ProximalAlgorithms.AFBA(tol = TOL, maxit = MAXITER, verbose = false)
  (x, y), iter = solver(x0 = x0, y0 = y0, f = f, g = g, h = h, L = A, beta_f = beta_f)
  obj = problem_objective(x, g, h, lambda)
  flag = iter < MAXITER
  return iter, obj, flag
end

function proxcompsum_vucondat(g, h, lambda)
  f = ProximalAlgorithms.AutoDifferentiable(x -> (norm(x - xi)^2) / (2*lambda), AutoZygote())
  beta_f = 1/lambda
  y0 = zeros(na)
  solver = ProximalAlgorithms.VuCondat(tol = TOL, maxit = MAXITER, verbose = false)
  (x, y), iter = solver(x0 = x0, y0 = y0, f = f, g = g, h = h, L = A, beta_f = beta_f)
  obj = problem_objective(x, g, h, lambda)
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

function proxcompsum_envelopt(g, h, lambda, subsolvermaker)
  # quadratic objective
  ls_obj(x) = (norm(x - xi)^2) / (2*lambda)
  ls_grad!(gx, x) = begin
    gx .= (x - xi) ./ lambda
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
  obj = problem_objective(x, g, h, lambda)
  iter = inner_iter
  flag = stats.status == :first_order
  return iter, obj, flag
end

ntrials = 100
lambda_vec = 10 .^ range(-3, 3, length = ntrials)
gmakers = [NormL1, RootNormLhalf, NormL0]
hmakers = [NormL1, NormL2]

res = (
  envelopt_r2 = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
  envelopt_r2dh = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
  envelopt_nmpg = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
  afba = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
  vucondat = (flag = zeros(ntrials), iter = zeros(ntrials), obj = zeros(ntrials)),
)
dfstats = DataFrame(
  solver = String[],
  h = String[],
  g = String[],
  solved = Int[],
  itermax = Int[],
  itermedian = Real[],
)

for hmaker in hmakers
  h = hmaker(1.0)
  for gmaker in gmakers
    g = gmaker(1.0)
    for i = 1:ntrials
      lambda = lambda_vec[i]

      # Vu-Condat
      iter, obj, flag = proxcompsum_vucondat(g, h, lambda)
      res.vucondat.iter[i] = iter
      res.vucondat.obj[i] = obj
      res.vucondat.flag[i] = flag

      # AFBA
      iter, obj, flag = proxcompsum_afba(g, h, lambda)
      res.afba.iter[i] = iter
      res.afba.obj[i] = obj
      res.afba.flag[i] = flag

      # Envelopt R2
      subsolvermaker = R2EnveloptSubSolver
      iter, obj, flag = proxcompsum_envelopt(g, h, lambda, subsolvermaker)
      res.envelopt_r2.iter[i] = iter
      res.envelopt_r2.obj[i] = obj
      res.envelopt_r2.flag[i] = flag

      # Envelopt NMPG
      subsolvermaker = NMPGEnveloptSubSolver
      iter, obj, flag = proxcompsum_envelopt(g, h, lambda, subsolvermaker)
      res.envelopt_nmpg.iter[i] = iter
      res.envelopt_nmpg.obj[i] = obj
      res.envelopt_nmpg.flag[i] = flag

      # Envelopt R2DH
      subsolvermaker = R2DHEnveloptSubSolver
      iter, obj, flag = proxcompsum_envelopt(g, h, lambda, subsolvermaker)
      res.envelopt_r2dh.iter[i] = iter
      res.envelopt_r2dh.obj[i] = obj
      res.envelopt_r2dh.flag[i] = flag
    end

    # save results
    println("Problem with h=$(hmaker) and g=$(gmaker)")
    for k in keys(res)
      filename = "$(hmaker)_$(gmaker)_$(k).csv"
      mat = zeros(ntrials, 4)
      mat[:, 1] .= lambda_vec
      mat[:, 2] .= res[k].flag
      mat[:, 3] .= res[k].iter
      mat[:, 4] .= res[k].obj
      df = DataFrame(mat, :auto)
      CSV.write(filename, df, writeheader = false)

      # statistics (only solved)
      idx = BitArray(undef, ntrials)
      for i = 1:ntrials
        idx[i] = res[k].flag[i] == 0.0 ? false : true
      end
      tmp = res[k].iter[idx]
      itermedian = median(tmp)
      itermax = maximum(tmp)
      solved = Int(sum(res[k].flag))

      println("$(k) solved $(solved) out of $(ntrials): iter max $(itermax), median $(itermedian)")

      push!(dfstats, ("$(k)", "$(hmaker)", "$(gmaker)", solved, itermax, itermedian))
    end
  end
end

filenamestats = "primaldual_stats$(ntrials).csv"
CSV.write(filenamestats, dfstats, writeheader = false)
