using ProximalOperators, ADNLPModels
using Random, LinearAlgebra, Statistics
using NCL, MadNLP, NLPModelsIpopt

using Envelopt

include("normL1nonnegative.jl")

# min (x₁ - 1)² + (x₂ - 1)² + \|x\|_1   s.t. x₁ * x₂ ≤ 0, x₁ ≥ 0, x₂ ≥ 0.

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

function process_output(stats)
  is_solved = if stats.status == :first_order
    true
  elseif stats.status == MadNLP.Status(1)
    true
  else
    false
  end
  flag = if is_solved
    which_expected_solution(stats.solution[1:2])
  else
    -2
  end
  display(stats.solution)
  display(stats.objective)
  flag
end

Random.seed!(123) # seed for reproducibility

TOL = 1e-6
ntrials = 100

res = (
  envelopt_madnlp = (flag = zeros(ntrials), iter = zeros(ntrials)),
  envelopt_ipopt = (flag = zeros(ntrials), iter = zeros(ntrials)),
  ncl_envelopt_madnlp = (flag = zeros(ntrials), iter = zeros(ntrials)),
  ncl_envelopt_ipopt = (flag = zeros(ntrials), iter = zeros(ntrials)),
  madnlp = (flag = zeros(ntrials), iter = zeros(ntrials)),
  ipopt = (flag = zeros(ntrials), iter = zeros(ntrials)),
  alps = (flag = zeros(ntrials), iter = zeros(ntrials)),
  ncl = (flag = zeros(ntrials), iter = zeros(ntrials)),
  envelopt_madnlp_bnd = (flag = zeros(ntrials), iter = zeros(ntrials)),
  envelopt_ipopt_bnd = (flag = zeros(ntrials), iter = zeros(ntrials)),
  envelopt_madnlp_unc = (flag = zeros(ntrials), iter = zeros(ntrials)),
  envelopt_ipopt_unc = (flag = zeros(ntrials), iter = zeros(ntrials)),
)

h = NormL1(1.0)

eval_F!(Fx, x) = begin
  Fx .= 0.0
  Fx[1:2] .= x[1:2]
  Fx[3] = x[1] * x[2]
  Fx
end
Fmodel_bnd = ADNLPModel!(x -> 0.0, zeros(2), eval_F!, zeros(3), zeros(3))
h_bnd = SlicedSeparableSum((h, IndNonpositive()), ((1:2,), (3,)))
h_unc = SlicedSeparableSum((NormL1Nonnegative(1.0), IndNonpositive()), ((1:2,), (3,)))

for i = 1:ntrials
  x0 = 100 * randn(2)

  model = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 1)^2,
    x0,
    [0.0, 0.0],
    [Inf, Inf],
    x -> [x[1] * x[2]],
    [-Inf],
    [0.0],
  )

  model_bnd = ADNLPModel(x -> (x[1] - 1)^2 + (x[2] - 1)^2, x0, [0.0, 0.0], [Inf, Inf])

  model_unc = ADNLPModel(x -> (x[1] - 1)^2 + (x[2] - 1)^2, x0)

  fullmodel = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 1)^2 + x[1] + x[2],
    x0,
    [0.0, 0.0],
    [Inf, Inf],
    x -> [x[1] * x[2]],
    [-Inf],
    [0.0],
  )

  # with MadNLP
  madnlp_solver = MadNLP.MadNLPSolver(fullmodel, print_level = MadNLP.INFO, tol = TOL)
  stats = MadNLP.solve!(madnlp_solver)
  res.madnlp.flag[i] = process_output(stats)
  res.madnlp.iter[i] = stats.iter

  # with Ipopt
  stats = ipopt(fullmodel, warm_start_init_point = "yes", tol = TOL)
  res.ipopt.flag[i] = process_output(stats)
  res.ipopt.iter[i] = stats.iter

  # with Envelopt+MadNLP
  env_model = EnveloptNLPModel(model, h)
  stats, status, u, nlp_iter = envelopt(env_model, ptol_min = TOL, dtol_min = TOL)
  res.envelopt_madnlp.flag[i] = process_output(stats)
  res.envelopt_madnlp.iter[i] = nlp_iter

  # with Envelopt+Ipopt
  env_model = EnveloptNLPModel(model, h)
  stats, status, u, nlp_iter = envelopt(
    env_model,
    ptol_min = TOL,
    dtol_min = TOL,
    subsolver = IPOPTEnveloptSubSolver(env_model),
  )
  res.envelopt_ipopt.flag[i] = process_output(stats)
  res.envelopt_ipopt.iter[i] = nlp_iter

  # TODO add AL from RegularizedOptimization
  # stats = AL(model, h, atol = TOL, ctol = TOL)
  # res.alps.flag[i] = process_output(stats)
  # res.alps.iter[i] = stats.iter

  # with NCL
  stats = NCLSolve(fullmodel, opt_tol = TOL, feas_tol = TOL)
  res.ncl.flag[i] = process_output(stats)
  res.ncl.iter[i] = stats.iter

  # with NCL+Envelopt+MadNLP
  ncl_model = NCLModel(model)
  env_ncl_model = EnveloptNLPModel(ncl_model, h)
  stats, status, u, nlp_iter = envelopt(env_ncl_model, ptol_min = TOL, dtol_min = TOL)
  res.ncl_envelopt_madnlp.flag[i] = process_output(stats)
  res.ncl_envelopt_madnlp.iter[i] = nlp_iter

  # with NCL+Envelopt+Ipopt
  ncl_model = NCLModel(model)
  env_ncl_model = EnveloptNLPModel(ncl_model, h)
  stats, status, u, nlp_iter = envelopt(
    env_ncl_model,
    ptol_min = TOL,
    dtol_min = TOL,
    subsolver = IPOPTEnveloptSubSolver(env_ncl_model),
  )
  res.ncl_envelopt_ipopt.flag[i] = process_output(stats)
  res.ncl_envelopt_ipopt.iter[i] = nlp_iter

  #=====================================================#
  # BND formulation with Envelopt+MadNLP
  env_model_bnd = EnveloptNLPModel(model_bnd, Fmodel_bnd, h_bnd)
  stats, status, u, nlp_iter = envelopt(env_model_bnd, ptol_min = TOL, dtol_min = TOL)
  res.envelopt_madnlp_bnd.flag[i] = process_output(stats)
  res.envelopt_madnlp_bnd.iter[i] = nlp_iter

  # BND formulation with Envelopt+Ipopt
  env_model_bnd = EnveloptNLPModel(model_bnd, Fmodel_bnd, h_bnd)
  stats, status, u, nlp_iter = envelopt(
    env_model_bnd,
    ptol_min = TOL,
    dtol_min = TOL,
    subsolver = IPOPTEnveloptSubSolver(env_model_bnd),
  )
  res.envelopt_ipopt_bnd.flag[i] = process_output(stats)
  res.envelopt_ipopt_bnd.iter[i] = nlp_iter

  #=====================================================#
  # UNC formulation with Envelopt+MadNLP
  env_model_unc = EnveloptNLPModel(model_unc, Fmodel_bnd, h_unc)
  stats, status, u, nlp_iter = envelopt(env_model_unc, ptol_min = TOL, dtol_min = TOL)
  res.envelopt_madnlp_unc.flag[i] = process_output(stats)
  res.envelopt_madnlp_unc.iter[i] = nlp_iter

  # UNC formulation with Envelopt+Ipopt
  env_model_unc = EnveloptNLPModel(model_unc, Fmodel_bnd, h_unc)
  stats, status, u, nlp_iter = envelopt(
    env_model_unc,
    ptol_min = TOL,
    dtol_min = TOL,
    subsolver = IPOPTEnveloptSubSolver(env_model_unc),
  )
  res.envelopt_ipopt_unc.flag[i] = process_output(stats)
  res.envelopt_ipopt_unc.iter[i] = nlp_iter
end

for k in keys(res)
  tmp_flag = res[k].flag
  tmp_iter = res[k].iter
  println("$(k)")
  println(
    "   $(sum(tmp_flag .> 0)*100/ntrials) global sol / $(sum(tmp_flag .== 0)*100/ntrials) local max",
  )
  if sum(tmp_flag .< 0) > 0
    @warn "unexpected behaviour"
    println(
      "   $(sum(tmp_flag .== -1)*100/ntrials) unexpected sol/ $(sum(tmp_flag .== -2)*100/ntrials) failed",
    )
  end
  println("   NLP iterations: median $(median(tmp_iter)), max $(maximum(tmp_iter))")
end
