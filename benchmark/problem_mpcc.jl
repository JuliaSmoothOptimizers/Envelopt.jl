using ProximalOperators, ADNLPModels
using Random, LinearAlgebra
using NCL, MadNLP

using Envelopt

# min (x₁ - 1)² + (x₂ - 1)² + \|x\|_1   s.t. x₁ * x₂ = 0, x₁ ≥ 0, x₂ ≥ 0.

function which_expected_solution(x, tol = 1e-2)
  x00 = [0, 0]
  x05 = [0, 0.5]
  x50 = [0.5, 0]
  if norm(x - x00) <= tol
    return 0
  elseif norm(x - x05) <= tol
    return 1
  elseif norm(x - x50) <= tol
    return 2
  else
    @warn "Unexpected solution"
    return -1
  end
end

Random.seed!(123)

ntrials = 100
res = (
  madnlp = zeros(ntrials),
  envelopt = zeros(ntrials),
  ncl = zeros(ntrials),
  ncl_envelopt = zeros(ntrials),
  ind_envelopt = zeros(ntrials),
)

h = NormL1(1.0)

for i = 1:ntrials
  x0 = 10 * randn(2)

  model = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 1)^2,
    x0,
    [0.0, 0.0],
    [Inf, Inf],
    x -> [x[1] * x[2]],
    [0.0],
    [0.0],
  )

  fullmodel = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 1)^2 + x[1] + x[2],
    x0,
    [0.0, 0.0],
    [Inf, Inf],
    x -> [x[1] * x[2]],
    [0.0],
    [0.0],
  )

  # with MadNLP
  madnlp_solver = MadNLP.MadNLPSolver(
    fullmodel,
    hessian_approximation = MadNLP.CompactLBFGS,
    print_level = MadNLP.INFO,
    tol = 1e-6,
  )
  stats = MadNLP.solve!(madnlp_solver)
  if stats.status == MadNLP.Status(1)
    res.madnlp[i] = which_expected_solution(stats.solution)
  else
    res.madnlp[i] = -2
  end
  display(stats.solution)
  display(stats.objective)

  # with Envelopt
  env_model = EnveloptNLPModel(model, h)
  stats, status, u = envelopt(env_model, verbose = true)
  if status == "first_order"
    res.envelopt[i] = which_expected_solution(stats.solution)
  else
    res.envelopt[i] = -2
  end
  display(stats.solution)
  display(stats.objective)

  # with NCL
  stats = NCLSolve(fullmodel)
  if stats.status == :first_order
    res.ncl[i] = which_expected_solution(stats.solution)
  else
    res.ncl[i] = -2
  end
  display(stats.solution)
  display(stats.objective)

  # with NCL+Envelopt
  ncl_model = NCLModel(model)
  env_ncl_model = EnveloptNLPModel(ncl_model, h)
  stats, status, u = envelopt(env_ncl_model, verbose = true)
  if status == "first_order"
    res.ncl_envelopt[i] = which_expected_solution(stats.solution[1:2])
  else
    res.ncl_envelopt[i] = -2
  end
  display(stats.solution)
  display(stats.objective)

  # with Envelopt on different formulation (to obtain NCL-like regularization)
  model_ind = ADNLPModel(x -> (x[1] - 1)^2 + (x[2] - 1)^2, x0, [0.0, 0.0], [Inf, Inf])
  eval_F!(Fx, x) = begin
    Fx .= 0.0
    Fx[1:2] .= x[1:2]
    Fx[3] = x[1] * x[2]
    Fx
  end
  Fmodel_ind = ADNLPModel!(x -> 0.0, zeros(2), eval_F!, zeros(3), zeros(3))
  h_ind = SlicedSeparableSum((h, IndZero()), ((1:2,), (3,)))
  env_model_ind = EnveloptNLPModel(model_ind, Fmodel_ind, h_ind)
  stats, status, u_ind = envelopt(env_model_ind, verbose = true, max_outer = 100)
  if status == "first_order"
    res.ind_envelopt[i] = which_expected_solution(stats.solution)
  else
    res.ind_envelopt[i] = -2
  end
  display(stats.solution)
  display(stats.objective)
end

for k in keys(res)
  println(
    "$(k): $(sum(res[k] .> 0)) global sol / $(sum(res[k] .== 0)) local max / $(sum(res[k] .== -1)) unexpected sol/ $(sum(res[k] .== -2)) failed",
  )
end
