module EnveloptKnitroExt

using Envelopt
using NLPModels
using NLPModelsKnitro
using SolverCore

# Knitro subproblem solver
mutable struct KnitroEnveloptSubSolver <: Envelopt.AbstractEnveloptSubSolver
  solver::KnitroSolver
  stats::GenericExecutionStats
  name::String
  z::Vector{Float64}  # KNITRO wants a single vector of dual variables
end

# ... constructor
function Envelopt.KnitroEnveloptSubSolver(env_model::EnveloptNLPModel; linear_api::Bool = true)
  eltype(env_model) == Float64 || error("KnitroEnveloptSubSolver only supports Float64 problems")
  @debug "initializing Knitro subproblem solver"
  solver = KnitroSolver(env_model; linear_api = linear_api)
  stats = GenericExecutionStats(env_model)
  z = fill!(similar(get_x0(env_model)), 0)
  return KnitroEnveloptSubSolver(solver, stats, "Knitro", z)
end

const knitro_fixed_options = Dict(
  :nlp_algorithm => 1,       # Interior/Direct
  :bar_directinterval => 0,  # Only use direct linear algebra
  :bar_initpt => 3,          # Center initial guess wrt two-sided bounds
  :bar_murule => 1,          # Monotone
  :outlev => 0,
  :maxit => 100,
)

# TODO: need smarter initialization
function compute_mu_init(outer_iter::Int)
  mu_init = 1.0e-1
  if 2 <= outer_iter < 4
    mu_init = 1e-3
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
# Knitro will automatically use LBFGS Hessian approximations because
# the EnveloptNLPModel has hess_available = false.
function (sub::KnitroEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  outer_iter::Int;
  tol::Float64 = 1.0e-6,
  kwargs...,
)

  # prepare for warm start
  # TODO: try solver.mu from the previous solve
  mu_init = compute_mu_init(outer_iter)
  bar_slackboundpush = mu_init

  # obtain multipliers for warm start
  y0 = sub.stats.multipliers
  if has_bounds(env_model)
    zL0 = sub.stats.multipliers_L
    zU0 = sub.stats.multipliers_U
    sub.z .= zL0 .- zU0
  end

  setparams!(
    sub.solver;
    x0 = x0,
    y0 = y0,
    z0 = sub.z,
    bar_initmu = mu_init,
    bar_slackboundpush = bar_slackboundpush,
    opttol = tol,
    feastol = tol,
    knitro_fixed_options...,
    kwargs...,
  )

  return NLPModelsKnitro.solve!(sub.solver, env_model, sub.stats)
end

end
