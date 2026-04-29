export IPOPTEnveloptSubSolver

using NLPModelsIpopt

# IPOPT subproblem solver
mutable struct IPOPTEnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::IpoptSolver
  stats::GenericExecutionStats
  name::String
end

# ... constructor
function IPOPTEnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing IPOPT subproblem solver"
  solver = IpoptSolver(env_model)
  stats = GenericExecutionStats(env_model)
  return IPOPTEnveloptSubSolver(solver, stats, "IPOPT")
end

const ipopt_fixed_options = Dict(
  :sb => "yes",  # options that are always used
  :print_level => 0,
  :max_iter => 100,
  :dual_inf_tol => 0.1,  # IPOPT stops when relative AND absolute tolerances are met.
  :constr_viol_tol => 0.1,
  :compl_inf_tol => 0.1,
)

# TODO: need smarter initialization
function compute_mu_init(outer_iter::Int)
  mu_init = 1.0e-1
  if 2 <= outer_iter < 4
    mu_init = 1e-4
  elseif 4 <= outer_iter < 6
    mu_init = 1e-5
  elseif 6 <= outer_iter < 8
    mu_init = 1e-6
  elseif 8 <= outer_iter < 10
    mu_init = 1e-7
  elseif outer_iter >= 10
    mu_init = 1e-8
  end
  mu_init
end

# ... solve
# IPOPT will automatically use LBFGS Hessian approximations because
# the EnveloptNLPModel has hess_available = false.
function (sub::IPOPTEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  outer_iter::Int;
  kwargs...,
)

  # prepare for warm start
  # TODO: try solver.mu from the previous solve
  mu_init = compute_mu_init(outer_iter)

  # obtain multipliers for warm start
  y0 = sub.stats.multipliers
  zL0 = sub.stats.multipliers_L
  zU0 = sub.stats.multipliers_U
  return NLPModelsIpopt.solve!(
    sub.solver,
    env_model,
    sub.stats;
    warm_start_init_point = outer_iter > 0 ? "yes" : "no",
    x0 = x0,
    y0 = y0,
    zL0 = zL0,
    zU0 = zU0,
    mu_init = mu_init,
    ipopt_fixed_options...,
    kwargs...,
  )
end
